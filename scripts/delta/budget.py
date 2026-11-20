#!/usr/bin/env python3
"""Bound Delta submissions by a cumulative allowance and retain their job IDs.

Only ``submit`` allocates resources. Accounting and status use login-node Slurm
queries; an uncertain submission or missing accounting record blocks another job.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

HEADROOM_SECONDS = 300
MAX_BUDGET_SU = Decimal("26")
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"}


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def fields(text: str) -> dict[str, str]:
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)\s*=\s*(\S+)", text))


def live_rate(account: str, cpus: int) -> Decimal:
    """Verify the supported one-A100 billing contract before allocating."""
    partition = fields(command(["scontrol", "show", "partition", "gpuA100x4", "--oneliner"]))
    config = fields(command(["scontrol", "show", "config"]))
    if (config.get("PriorityFlags") != "MAX_TRES" or config.get("OverTimeLimit") != "0"
            or partition.get("OverTimeLimit", "NONE") not in {"NONE", "0"}):
        raise ValueError("Require MAX_TRES billing and OverTimeLimit=0")
    if not 0 <= int(config["KillWait"]) < HEADROOM_SECONDS:
        raise ValueError("Slurm KillWait exceeds the reserved billing headroom")
    if partition.get("State") != "UP":
        raise ValueError("The A100 partition is not UP")
    weights = dict(item.split("=", 1) for item in partition["TRESBillingWeights"].split(","))
    if set(weights) != {"CPU", "Mem", "GRES/gpu"} or not weights["Mem"].endswith("G"):
        raise ValueError("Unrecognized Delta billing weights")
    qos_names = {partition["QoS"], account}
    rows = command(["sacctmgr", "--noheader", "--parsable2", "show", "qos", ",".join(sorted(qos_names)), "format=Name,UsageFactor"]).splitlines()
    factors = dict(row.split("|") for row in rows)
    if len(rows) != len(qos_names) or set(factors) != qos_names:
        raise ValueError("Cannot verify both partition and job QoS billing factors")
    if any(Decimal(factor) != 1 for factor in factors.values()):
        raise ValueError("Require partition and job QoS billing factors of one")
    rate = max(Decimal(cpus) * Decimal(weights["CPU"]), Decimal(60) * Decimal(weights["Mem"][:-1]), Decimal(weights["GRES/gpu"])) / 1000
    if not rate.is_finite() or rate <= 0 or rate > 1:
        raise ValueError("The requested allocation exceeds one SU per hour")
    if account != "bhvw-delta-gpu" or not 1 <= cpus <= 16:
        raise ValueError("Unexpected account or CPU request")
    return rate


def atomic_json(path: Path, value: dict, *, replace: bool = True) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".budget-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def submission_lock(data: Path):
    with (data / ".submission.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another submission is being prepared for this dataset") from exc
        yield


def load_budget(data: Path, allowance: Decimal, account: str, *, manifest: Path | None = None) -> dict:
    if not allowance.is_finite() or allowance <= 0:
        raise ValueError("Budget must be finite and positive")
    if manifest is None and allowance > MAX_BUDGET_SU:
        raise ValueError("Dataset budget must be at most 26 additional SU")
    manifest_path = manifest if manifest is not None else data / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    path = data / "budget.json"
    if path.exists():
        ledger = json.loads(path.read_text())
        if (ledger.get("schema_version") != 1 or ledger.get("manifest_sha256") != manifest_hash
                or ledger.get("account") != account or Decimal(ledger["allowance_su"]) != allowance):
            raise ValueError("Existing budget does not match the manifest, account or allowance")
        return ledger
    if list((data / "submissions").glob("job-*.json")):
        raise ValueError("Job receipts exist but budget.json is missing; reconcile before continuing")
    return {"schema_version": 1, "budget_id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest_hash, "account": account, "allowance_su": str(allowance), "jobs": []}


def accounting(ledger: dict) -> dict:
    """Use allocation records, including cancelled/failed jobs, never job steps."""
    jobs = ledger["jobs"]
    if any(not job.get("job_id") for job in jobs):
        raise ValueError("Unresolved submission in budget.json; reconcile its Slurm comment before submitting again")
    records = {}
    if jobs:
        ids = [str(job["job_id"]) for job in jobs]
        if len(ids) != len(set(ids)) or any(re.fullmatch(r"[0-9]+", value) is None for value in ids):
            raise ValueError("Invalid or duplicate job IDs in the budget")
        result = command(["sacct", "--allocations", "--noheader", "--parsable2", "--jobs", ",".join(ids),
                          "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,Account,Partition,End"])
        for line in result.splitlines():
            parts = line.split("|")
            if len(parts) != 7 or parts[0] not in ids or parts[0] in records:
                raise ValueError("Unexpected or duplicate allocation accounting record")
            records[parts[0]] = parts
        if set(records) != set(ids):
            raise ValueError("Slurm accounting is incomplete; do not submit another job yet")
    used = Decimal(0)
    raw_used = Decimal(0)
    active = []
    summaries = []
    for job in jobs:
        job_id, state, elapsed, tres, account, partition, end = records[str(job["job_id"]) ]
        if account != ledger["account"] or partition != "gpuA100x4":
            raise ValueError("Recorded job belongs to a different account or partition")
        elapsed_seconds = int(elapsed)
        resources = dict(item.split("=", 1) for item in tres.split(",") if item)
        if elapsed_seconds < 0 or (elapsed_seconds > 0 and "billing" not in resources):
            raise ValueError("Missing allocation billing information")
        billing = Decimal(resources.get("billing", "0"))
        if not billing.is_finite() or billing < 0 or billing > 1000 or (elapsed_seconds > 0 and billing == 0):
            raise ValueError("Unexpected allocation billing rate")
        charge = billing / 1000 * Decimal(elapsed_seconds) / 3600
        # Reserve upward to the accounting display precision for each job;
        # repeated small jobs must not accumulate hidden rounding overspend.
        budget_charge = charge.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        raw_used += charge
        used += budget_charge
        is_terminal = state.split()[0].rstrip("+") in TERMINAL and end not in {"Unknown", "None", ""}
        if not is_terminal:
            active.append(job_id)
        summaries.append({"job_id": job_id, "state": state, "elapsed_seconds": elapsed_seconds,
                          "charge_su": str(charge), "budget_charge_su": str(budget_charge), "requested_wall_seconds": job["wall_seconds"]})
    remaining = max(Decimal(0), Decimal(ledger["allowance_su"]) - used)
    return {"budget_id": ledger["budget_id"], "allowance_su": ledger["allowance_su"],
            "used_su": str(used), "raw_charge_su": str(raw_used), "remaining_su": str(remaining), "active_job_ids": active, "jobs": summaries}


def wall_time(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def remaining_worker_seconds(requested: int, margin: int) -> int:
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.isdigit():
        deadline = int(end)
    else:
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise ValueError("Cannot determine the Slurm allocation deadline")
        end_text = fields(command(["scontrol", "show", "job", job_id, "--oneliner"]))["EndTime"]
        deadline = int(datetime.fromisoformat(end_text).timestamp())
    remaining = min(requested, deadline - int(time.time()) - margin)
    if remaining <= 0:
        raise ValueError("No worker time remains before the allocation shutdown margin")
    return remaining


def dataset_submission_record(args, scheduler, worker, runtime, wall):
    return command([str(args.python), str(Path(__file__).with_name("record_submission.py")),
                    str(args.code_root), str(args.data_root), str(args.python), str(args.parameters), str(args.run_parameters),
                    str(runtime), str(wall), str(args.shutdown_margin_seconds), *scheduler, "--", *worker])


def submit(args, *, manifest=None, batch_file="gaussian_amplitudes.batch.sh", record_intent=dataset_submission_record) -> None:
    data = args.data_root
    allowance = Decimal(args.budget_su)
    split = args.arguments.index("--")
    scheduler, worker = args.arguments[:split], args.arguments[split + 1:]
    supported = {"parsable", "account", "qos", "partition", "nodes", "ntasks", "cpus-per-task", "gpus-per-node",
                 "mem", "time", "signal", "no-requeue", "constraint", "export", "job-name", "chdir", "output", "error"}
    options = {}
    for item in scheduler:
        key, _, value = item.removeprefix("--").partition("=")
        if not item.startswith("--") or key not in supported or key in options:
            raise ValueError("Unsupported or duplicate scheduler option")
        options[key] = value
    required = {"partition": "gpuA100x4", "nodes": "1", "ntasks": "1", "gpus-per-node": "1", "mem": "60G",
                "no-requeue": "", "parsable": "", "constraint": "work", "export": "NIL"}
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError("Require one A100 allocation, 60G and no automatic requeue")
    account, cpus = options["account"], int(options["cpus-per-task"])
    if options.get("qos") != account:
        raise ValueError("Require the explicitly verified account QoS")
    if not (60 <= args.shutdown_margin_seconds <= 300 and 0 < args.wall_seconds <= 93300
            and 0 < args.worker_seconds <= min(93000, args.wall_seconds - args.shutdown_margin_seconds)):
        raise ValueError("Invalid wall, worker or shutdown budget")
    if options.get("time") != wall_time(args.wall_seconds):
        raise ValueError("The explicit scheduler wall time must match the requested budget")
    if options.get("signal") != f"USR1@{args.shutdown_margin_seconds}":
        raise ValueError("The scheduler stop signal must match the shutdown margin")
    rate = live_rate(account, cpus)
    # Include legacy jobs that predate this allowance. Avoid duplicate dataset
    # writers and overlapping charges even before a first job ID is recorded.
    if command(["squeue", "--noheader", "--user", getpass.getuser(), "--account", account, "--format=%i"]):
        raise ValueError("Another job for this user and account is queued or running")
    ledger = load_budget(data, allowance, account, manifest=manifest)
    report = accounting(ledger)
    if report["active_job_ids"]:
        raise ValueError("A previous budget job is still active; reconnect to it instead of resubmitting")
    available_seconds = int((Decimal(report["remaining_su"]) / rate * 3600).to_integral_value(rounding=ROUND_FLOOR))
    wall = min(args.wall_seconds, available_seconds - HEADROOM_SECONDS)
    runtime = min(args.worker_seconds, wall - args.shutdown_margin_seconds)
    if wall <= args.shutdown_margin_seconds or runtime <= 0:
        raise ValueError("Insufficient budget after reserving shutdown and billing headroom")
    scheduler = [f"--time={wall_time(wall)}" if item.startswith("--time=") else item for item in scheduler]
    token = f"vpml-{ledger['budget_id']}-{uuid.uuid4().hex[:12]}"
    scheduler.append(f"--comment={token}")
    script = Path(__file__).with_name(batch_file)
    batch = [str(script), str(args.code_root), str(data), str(args.python), str(args.parameters), str(runtime),
             "--shutdown-margin", str(args.shutdown_margin_seconds), *worker]
    for name in list(os.environ):
        if name.startswith("SBATCH_"):
            del os.environ[name]
    if args.action == "test-submit":
        print(command(["sbatch", *scheduler, "--test-only", *batch]))
        return
    (data / "logs").mkdir(exist_ok=True, mode=0o700)
    intent = str(record_intent(args, scheduler, worker, runtime, wall))
    entry = {"job_id": None, "comment": token, "intent": intent, "wall_seconds": wall,
             "billing_rate_su_per_hour": str(rate), "billing_headroom_seconds": HEADROOM_SECONDS}
    ledger["jobs"].append(entry)
    atomic_json(data / "budget.json", ledger)
    # A timeout/error may follow scheduler acceptance. Leave the unresolved record
    # in place until an operator reconciles it; never blindly resubmit.
    result = command(["sbatch", *scheduler, *batch])
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?", result)
    if match is None:
        raise ValueError("Unrecognized sbatch receipt; reconcile the unresolved budget entry")
    entry["job_id"] = match[1]
    entry["accepted_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(data / "budget.json", ledger)
    atomic_json(data / "submissions" / f"job-{match[1]}.json", {"budget_id": ledger["budget_id"], **entry}, replace=False)
    print(json.dumps({"job_id": match[1], "intent": intent, "wall_seconds": wall, "worker_seconds": runtime,
                      "remaining_su_before_submission": report["remaining_su"], "budget": str(data / "budget.json")}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status")
    status.add_argument("data_root", type=Path)
    status.add_argument("budget_su")
    status.add_argument("account")
    deadline = sub.add_parser("worker-seconds")
    deadline.add_argument("requested", type=int)
    deadline.add_argument("margin", type=int)
    for name in ("submit", "test-submit"):
        p = sub.add_parser(name)
        for field in ("code_root", "data_root", "python", "parameters", "run_parameters"):
            p.add_argument(field, type=Path)
        for field in ("worker_seconds", "wall_seconds", "shutdown_margin_seconds"):
            p.add_argument(field, type=int)
        p.add_argument("budget_su")
        p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "status":
            print(json.dumps(accounting(load_budget(args.data_root, Decimal(args.budget_su), args.account)), indent=2))
        elif args.action == "worker-seconds":
            print(remaining_worker_seconds(args.requested, args.margin))
        elif args.action == "submit":
            with submission_lock(args.data_root):
                submit(args)
        else:
            submit(args)
    except (ValueError, InvalidOperation, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f"Delta budget check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
