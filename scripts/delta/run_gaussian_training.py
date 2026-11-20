"""Submit or inspect a single bounded Delta allocation for Gaussian closure training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import subprocess
import sys

import budget
from record_submission import git_identity, publish_intent

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parents[1]
ACCOUNT = "bhvw-delta-gpu"
INITIAL_ALLOWANCE = Decimal("1.0")
SHUTDOWN_SECONDS = 300
sys.path.insert(0, str(CODE_ROOT / "src"))
from vpml.datasets.provenance import source_record
from vpml.io.serialization import file_digest


def absolute_path(name: str) -> Path:
    value = Path(os.environ.get(name, ""))
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return value.resolve()


def training_record(args, scheduler, worker, runtime, wall) -> Path:
    manifest = args.dataset / "manifest.json"
    sources = ("run_gaussian_training.py", "gaussian_training.batch.sh", "record_submission.py", "budget.py")
    return publish_intent(args.data_root, {
        "schema_version": 1, "record_type": "training_submission_intent",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest), "sha256": file_digest(manifest)},
        "normalization": {"path": str(args.parameters), "sha256": file_digest(args.parameters)},
        "code": {"root": str(CODE_ROOT), **git_identity(CODE_ROOT),
                 "source": source_record(CODE_ROOT / "src/vpml"),
                 "launcher_sha256": {name: file_digest(SCRIPT_DIR / name) for name in sources}},
        "scheduler": {"arguments": scheduler, "wall_seconds": wall, "shutdown_margin_seconds": SHUTDOWN_SECONDS},
        "worker": {"python": str(args.python), "arguments": worker, "max_seconds": runtime,
                   "run_root": str(args.data_root)},
    })


def current_allowance(root: Path) -> Decimal:
    ledger = root / "budget.json"
    if ledger.exists():
        return Decimal(json.loads(ledger.read_text())["allowance_su"])
    return INITIAL_ALLOWANCE


def configure_job(args) -> None:
    args.parameters = absolute_path("VPML_NORMALIZATION")
    if not args.parameters.is_file():
        raise ValueError("VPML_NORMALIZATION must name the E10 normalization artifact")
    args.python = Path(sys.executable)  # Keep the venv interpreter's symlink.
    args.code_root = CODE_ROOT
    args.wall_seconds = args.wall_minutes * 60
    args.shutdown_margin_seconds = SHUTDOWN_SECONDS
    args.worker_seconds = args.wall_seconds - SHUTDOWN_SECONDS
    args.arguments = [
        "--parsable", f"--account={ACCOUNT}", f"--qos={ACCOUNT}", "--partition=gpuA100x4",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--gpus-per-node=1", "--mem=60G",
        f"--time={budget.wall_time(args.wall_seconds)}", f"--signal=USR1@{SHUTDOWN_SECONDS}",
        "--no-requeue", "--constraint=work", "--export=NIL", "--job-name=vpml-closure",
        f"--chdir={CODE_ROOT}", f"--output={args.data_root}/logs/slurm-%j.out",
        f"--error={args.data_root}/logs/slurm-%j.err", "--",
        "--dataset", str(args.dataset), "--microbatch-size", str(args.microbatch_size),
    ]
    if args.cache:
        cache = args.cache.resolve()
        for source in (args.dataset, CODE_ROOT):
            if cache == source or cache.is_relative_to(source) or source.is_relative_to(cache):
                raise ValueError("The writable cache must be separate from the dataset and source")
        args.arguments.extend(("--cache", str(cache)))
    for option in ("max_epochs", "max_updates"):
        value = getattr(args, option)
        if value is not None:
            if value < 1:
                raise ValueError("Training targets must be positive")
            args.arguments.extend(("--" + option.replace("_", "-"), str(value)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("budget", help="report recorded jobs, charges and remaining allowance")
    extend = actions.add_parser("extend-budget", help="set a higher cumulative allowance for this same run")
    extend.add_argument("total_su", type=Decimal, help="new cumulative total, including charges already incurred")
    for name in ("test-submit", "submit"):
        action = actions.add_parser(name)
        action.add_argument("--microbatch-size", type=int, choices=[n for n in range(1, 49) if 48 % n == 0], default=4)
        action.add_argument("--cache", type=Path, help="optional shared trajectory cache")
        action.add_argument("--wall-minutes", type=int, default=60)
        action.add_argument("--max-epochs", type=int, help="absolute completed-epoch target")
        action.add_argument("--max-updates", type=int, help="absolute optimizer update target")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        args.dataset = absolute_path("VPML_REFERENCE_ROOT")
        args.data_root = absolute_path("VPML_RUN_ROOT")
        for source in (args.dataset, CODE_ROOT):
            if args.data_root == source or args.data_root.is_relative_to(source) or source.is_relative_to(args.data_root):
                raise ValueError("The training run directory must be separate from the dataset and source")
        if "%" in str(args.data_root) or "\n" in str(args.data_root):
            raise ValueError("The Slurm log directory cannot contain percent or newline characters")
        manifest = args.dataset / "manifest.json"
        if not manifest.is_file():
            raise ValueError("The reference dataset manifest is missing")
        if args.action == "budget":
            ledger = budget.load_budget(args.data_root, current_allowance(args.data_root), ACCOUNT, manifest=manifest)
            print(json.dumps(budget.accounting(ledger), indent=2))
            return 0
        if args.action == "extend-budget":
            with budget.submission_lock(args.data_root):
                if not (args.data_root / "budget.json").exists():
                    raise ValueError("No training budget to extend")
                previous = current_allowance(args.data_root)
                ledger = budget.load_budget(args.data_root, previous, ACCOUNT, manifest=manifest)
                if not args.total_su.is_finite() or args.total_su <= previous:
                    raise ValueError("The new total SU allowance must be finite and higher than the current allowance")
                if budget.accounting(ledger)["active_job_ids"]:
                    raise ValueError("Wait until all recorded jobs have finished before extending the allowance")
                ledger.setdefault("amendments", []).append({"previous_su": str(previous), "total_su": str(args.total_su),
                                                            "at": datetime.now(timezone.utc).isoformat()})
                ledger["allowance_su"] = str(args.total_su)
                budget.atomic_json(args.data_root / "budget.json", ledger)
                print(f"Cumulative training allowance: {args.total_su} SU")
            return 0
        if not 6 <= args.wall_minutes <= 1555:
            raise ValueError("Wall minutes must be between 6 and 1555")
        configure_job(args)
        args.budget_su = current_allowance(args.data_root)
        options = {"manifest": manifest, "batch_file": "gaussian_training.batch.sh", "record_intent": training_record}
        if args.action == "submit":
            args.data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            with budget.submission_lock(args.data_root):
                budget.submit(args, **options)
        else:
            budget.submit(args, **options)
    except (OSError, KeyError, ValueError, InvalidOperation, subprocess.SubprocessError) as exc:
        print(f"Delta training launch failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
