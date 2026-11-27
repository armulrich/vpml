"""Check cumulative allocation limits without contacting a cluster."""
from __future__ import annotations

from argparse import Namespace
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("delta_budget", ROOT / "scripts/delta/budget.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


class DeltaBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="vpml budget ")
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        (self.data / "manifest.json").write_text('{"cases": []}\n')
        self.commands = []
        self.records = {}
        self.reply = "1001"
        self.partition = ("PartitionName=gpuA100x4 QoS=gpua100x4qos State=UP "
                          "TRESBillingWeights=CPU=62.5,Mem=16G,GRES/gpu=1000")
        self.config = "PriorityFlags = MAX_TRES\nKillWait = 30 sec\nOverTimeLimit = 0 min"
        self.qos = "gpua100x4qos|1.000000\nbhvw-delta-gpu|1.000000"
        self.args = Namespace(
            action="submit", data_root=self.data, code_root=ROOT,
            python=Path("/unused/python"),
            parameters=ROOT / "scripts/delta/gaussian_amplitudes.json",
            run_parameters=ROOT / "scripts/delta/gaussian_amplitudes.params.sh",
            worker_seconds=93000, wall_seconds=93300, shutdown_margin_seconds=300,
            budget_su="26", arguments=[
                "--parsable", "--account=bhvw-delta-gpu", "--partition=gpuA100x4",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--gpus-per-node=1",
                "--mem=60G", "--time=25:55:00", "--signal=USR1@300", "--no-requeue",
                "--constraint=work", "--export=NIL", "--qos=bhvw-delta-gpu", "--", "--stop-on-failure"],
        )

    def command(self, arguments):
        self.commands.append(arguments)
        if arguments[:3] == ["scontrol", "show", "partition"]:
            return self.partition
        if arguments[:3] == ["scontrol", "show", "config"]:
            return self.config
        if arguments[0] == "sacctmgr":
            return self.qos
        if arguments[0] == "sacct":
            ids = arguments[arguments.index("--jobs") + 1].split(",")
            return "\n".join(self.records[job] for job in ids if job in self.records)
        if arguments[0] == "squeue":
            return ""
        if len(arguments) > 1 and Path(arguments[1]).name == "record_submission.py":
            directory = self.data / "submissions"
            directory.mkdir(exist_ok=True)
            intent = directory / "example-intent.json"
            intent.write_text("{}\n")
            return str(intent)
        if arguments[0] == "sbatch":
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply
        self.fail(f"Unexpected external command: {arguments}")

    def submit(self):
        with mock.patch.object(budget, "command", side_effect=self.command):
            budget.submit(self.args)

    def record(self, job="1001", *, state="COMPLETED", elapsed=21600,
               resources="billing=1000,cpu=16,gres/gpu=1,mem=60G,node=1",
               account="bhvw-delta-gpu", end="2026-11-18T18:00:00"):
        self.records[job] = f"{job}|{state}|{elapsed}|{resources}|{account}|gpuA100x4|{end}"

    def ledger(self):
        return json.loads((self.data / "budget.json").read_text())

    def scheduler_calls(self):
        return [args for args in self.commands if args[0] == "sbatch"]

    def test_first_submission_records_acceptance_and_retains_billing_headroom(self):
        self.submit()
        ledger = self.ledger()
        self.assertEqual(ledger["allowance_su"], "26")
        self.assertEqual(ledger["jobs"][0]["job_id"], "1001")
        self.assertEqual(ledger["jobs"][0]["wall_seconds"], 93300)
        self.assertEqual(ledger["jobs"][0]["billing_headroom_seconds"], 300)
        receipt = json.loads((self.data / "submissions/job-1001.json").read_text())
        self.assertEqual(receipt["budget_id"], ledger["budget_id"])
        self.assertEqual(receipt["job_id"], "1001")
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_cancelled_job_consumes_budget_and_continuation_gets_only_remainder(self):
        self.submit()
        identity = self.ledger()["budget_id"]
        self.record(state="CANCELLED by 1000", elapsed=21600)
        self.reply = "1002"
        self.submit()
        self.assertEqual(self.ledger()["budget_id"], identity)
        self.assertEqual(self.ledger()["allowance_su"], "26")
        self.assertEqual(self.ledger()["jobs"][1]["wall_seconds"], 71700)
        second = self.scheduler_calls()[1]
        self.assertIn("--time=19:55:00", second)
        self.assertIn("71400", second)
        self.assertNotIn("--time=25:55:00", second)

    def test_multiple_failed_and_completed_jobs_are_all_charged(self):
        self.submit()
        self.record(state="FAILED", elapsed=3600)
        self.reply = "1002"
        self.submit()
        self.record(job="1002", elapsed=7200)
        with mock.patch.object(budget, "command", side_effect=self.command):
            report = budget.accounting(self.ledger())
        self.assertEqual(Decimal(report["used_su"]), Decimal(3))
        self.assertEqual(Decimal(report["remaining_su"]), Decimal(23))
        self.assertEqual(report["active_job_ids"], [])

    def test_fractional_job_charges_round_up_without_accumulating_free_retries(self):
        self.submit()
        ledger = self.ledger()
        original = ledger["jobs"][0]
        ledger["jobs"] = [dict(original, job_id=str(job)) for job in (1001, 1002, 1003)]
        for job in ("1001", "1002", "1003"):
            self.record(job=job, state="FAILED", elapsed=101)
        with mock.patch.object(budget, "command", side_effect=self.command):
            report = budget.accounting(ledger)
        self.assertEqual(Decimal(report["used_su"]), Decimal("0.09"))
        self.assertEqual(Decimal(report["remaining_su"]), Decimal("25.91"))
        self.assertLess(abs(Decimal(report["raw_charge_su"]) - Decimal(303) / 3600), Decimal("1e-25"))

    def test_active_or_missing_accounting_blocks_another_allocation(self):
        self.submit()
        for state in ("RUNNING", "PENDING", "COMPLETING", "UNRECOGNIZED"):
            with self.subTest(state=state):
                self.record(state=state, end="Unknown")
                with self.assertRaisesRegex(ValueError, "still active"):
                    self.submit()
        self.records.clear()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_terminal_state_without_end_time_still_blocks_continuation(self):
        self.submit()
        self.record(state="COMPLETED", end="Unknown")
        with self.assertRaisesRegex(ValueError, "still active"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_ambiguous_scheduler_reply_is_not_retried(self):
        self.reply = "unrecognized response"
        with self.assertRaisesRegex(ValueError, "Unrecognized sbatch"):
            self.submit()
        self.assertIsNone(self.ledger()["jobs"][0]["job_id"])
        self.reply = "1002"
        with self.assertRaisesRegex(ValueError, "Unresolved submission"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_scheduler_timeout_retains_unresolved_submission(self):
        self.reply = subprocess.TimeoutExpired("sbatch", 30)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.submit()
        self.assertIsNone(self.ledger()["jobs"][0]["job_id"])
        self.reply = "1002"
        with self.assertRaisesRegex(ValueError, "Unresolved submission"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_exhausted_allowance_and_last_shutdown_reserve_prevent_submission(self):
        self.submit()
        for elapsed in (26 * 3600, 26 * 3600 - 300):
            with self.subTest(elapsed=elapsed):
                self.record(elapsed=elapsed)
                with self.assertRaisesRegex(ValueError, "Insufficient budget"):
                    self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_new_allowance_or_changed_manifest_cannot_reset_existing_budget(self):
        self.submit()
        self.record(elapsed=3600)
        self.args.budget_su = "25"
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.submit()
        self.args.budget_su = "26"
        (self.data / "manifest.json").write_text('{"cases": ["changed"]}\n')
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_missing_ledger_with_existing_receipt_blocks_reset(self):
        self.submit()
        (self.data / "budget.json").unlink()
        with self.assertRaisesRegex(ValueError, "budget.json is missing"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_wrong_account_or_missing_billing_is_not_counted_as_free(self):
        self.submit()
        for options in ({"account": "another-project"}, {"resources": "cpu=16,gres/gpu=1"}):
            with self.subTest(options=options):
                self.record(**options)
                with self.assertRaises(ValueError):
                    self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_pending_cancellation_without_allocation_costs_zero(self):
        self.submit()
        self.record(state="CANCELLED", elapsed=0, resources="")
        with mock.patch.object(budget, "command", side_effect=self.command):
            report = budget.accounting(self.ledger())
        self.assertEqual(Decimal(report["used_su"]), 0)
        self.assertEqual(Decimal(report["remaining_su"]), 26)

    def test_live_higher_billing_or_changed_shutdown_policy_prevents_submission(self):
        original = self.partition
        self.partition = original.replace("GRES/gpu=1000", "GRES/gpu=2000")
        with self.assertRaisesRegex(ValueError, "exceeds one SU"):
            self.submit()
        self.partition = original
        self.qos = "gpua100x4qos|2.0\nbhvw-delta-gpu|1.0"
        with self.assertRaisesRegex(ValueError, "QoS billing factors"):
            self.submit()
        self.qos = "gpua100x4qos|1.0\nbhvw-delta-gpu|1.0"
        self.config = self.config.replace("OverTimeLimit = 0", "OverTimeLimit = 10")
        with self.assertRaisesRegex(ValueError, "OverTimeLimit"):
            self.submit()
        self.assertEqual(self.scheduler_calls(), [])
        self.assertFalse((self.data / "budget.json").exists())

    def test_partition_overtime_cannot_override_the_global_stop_limit(self):
        self.partition += " OverTimeLimit=10"
        with self.assertRaisesRegex(ValueError, "OverTimeLimit"):
            self.submit()
        self.assertEqual(self.scheduler_calls(), [])

    def test_positive_elapsed_with_zero_billing_is_rejected(self):
        self.submit()
        self.record(elapsed=3600, resources="billing=0,cpu=16,gres/gpu=1")
        with self.assertRaisesRegex(ValueError, "billing rate"):
            self.submit()
        self.assertEqual(len(self.scheduler_calls()), 1)

    def test_repeated_job_id_is_not_counted_as_two_independent_jobs(self):
        self.submit()
        ledger = self.ledger()
        ledger["jobs"].append(dict(ledger["jobs"][0]))
        self.record()
        with mock.patch.object(budget, "command", side_effect=self.command), \
                self.assertRaisesRegex(ValueError, "duplicate job IDs"):
            budget.accounting(ledger)

    def test_allowance_above_authorized_limit_never_submits(self):
        self.args.budget_su = "26.01"
        with self.assertRaisesRegex(ValueError, "at most 26"):
            self.submit()
        self.assertEqual(self.scheduler_calls(), [])
        self.assertFalse((self.data / "budget.json").exists())

    def test_dry_run_preserves_existing_dataset_and_creates_no_budget(self):
        self.args.action = "test-submit"
        before = {p.name: p.read_bytes() for p in self.data.iterdir()}
        self.submit()
        self.assertEqual({p.name: p.read_bytes() for p in self.data.iterdir()}, before)
        self.assertIn("--test-only", self.scheduler_calls()[0])

    def test_worker_deadline_accounts_for_time_already_spent_in_setup(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_END_TIME": "1500"}), \
                mock.patch.object(budget.time, "time", return_value=1000):
            self.assertEqual(budget.remaining_worker_seconds(93000, 300), 200)
            self.assertEqual(budget.remaining_worker_seconds(100, 300), 100)
        with mock.patch.dict(os.environ, {"SLURM_JOB_END_TIME": "1200"}), \
                mock.patch.object(budget.time, "time", return_value=1000), \
                self.assertRaisesRegex(ValueError, "No worker time"):
            budget.remaining_worker_seconds(93000, 300)

    def test_concurrent_submission_lock_is_exclusive(self):
        with budget.submission_lock(self.data):
            with self.assertRaisesRegex(ValueError, "Another submission"):
                with budget.submission_lock(self.data):
                    self.fail("A second submitter acquired the same dataset lock")
        with budget.submission_lock(self.data):
            pass  # Lock is released without needing to remove a lock file.


if __name__ == "__main__":
    unittest.main()
