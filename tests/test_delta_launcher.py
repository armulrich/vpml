"""Exercise launcher boundaries with fake Python and scheduler executables.

No test contacts Slurm, starts JAX, installs packages or reserves hardware.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/delta/run_gaussian_amplitudes_v1.sh"


class DeltaLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vpml launcher ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.log = self.directory / "calls.jsonl"
        self.data = self.directory / "data with spaces $(unused)"
        self.python = self.bin / "mock python"
        self.python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["MOCK_CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"program": "python", "args": sys.argv[1:],
                             "backend": os.environ.get("JAX_PLATFORMS"),
                             "ld_path": os.environ.get("LD_LIBRARY_PATH")}) + "\\n")
if os.environ.get("MOCK_FAIL_VALIDATE") and "validate" in sys.argv:
    sys.exit(3)
''')
        self.python.chmod(0o700)
        scheduler = self.bin / "sbatch"
        scheduler.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["MOCK_CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"program": "sbatch", "args": sys.argv[1:],
                             "sbatch_env": [name for name in os.environ if name.startswith("SBATCH_")]}) + "\\n")
print("hypothetical" if "--test-only" in sys.argv else "mock-submitted")
''')
        scheduler.chmod(0o700)
        self.environment = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin",
                                VPML_CODE_ROOT=str(ROOT), VPML_DATA_ROOT=str(self.data),
                                VPML_PYTHON=str(self.python), MOCK_CALL_LOG=str(self.log),
                                LD_LIBRARY_PATH="/unrelated/cuda", SBATCH_ARRAY_INX="0-1000",
                                SBATCH_EXCLUSIVE="1")

    def run_action(self, *args, environment=None):
        return subprocess.run(["/bin/bash", str(LAUNCHER), *args],
                              env=environment or self.environment, text=True,
                              capture_output=True, check=False)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_no_action_unknown_action_and_help_never_submit(self):
        self.assertEqual(self.run_action().returncode, 2)
        self.assertEqual(self.run_action("unknown").returncode, 2)
        self.assertEqual(self.run_action("--help").returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.data.exists())

    def test_metadata_actions_never_call_scheduler(self):
        for action in ("prepare", "validate", "status"):
            with self.subTest(action=action):
                result = self.run_action(action)
                self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["program"] for call in calls], ["python"] * 3)
        self.assertTrue(all(call["backend"] == "cpu" and call["ld_path"] is None for call in calls))
        self.assertEqual(calls[0]["args"][1:], ["prepare", "--manifest", str(self.data / "manifest.json"),
                                              "--parameters", str(ROOT / "scripts/delta/gaussian_amplitudes_v1.json")])
        self.assertEqual(calls[2]["args"][1:], ["status", "--out", str(self.data / "reference"),
                                               "--manifest", str(self.data / "manifest.json")])

    def test_test_submit_is_only_scheduler_test_and_quoted_paths_survive(self):
        result = self.run_action("test-submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["program"] for call in calls], ["python", "sbatch"])
        self.assertEqual(calls[0]["args"][1], "validate")
        arguments = calls[1]["args"]
        self.assertEqual(calls[1]["sbatch_env"], [])
        for argument in ("--test-only", "--account=bhvw-delta-gpu", "--partition=gpuA100x4",
                         "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--gpus-per-node=1",
                         "--mem=60G", "--time=01:00:00", "--signal=USR1@180", "--no-requeue",
                         "--constraint=work", "--export=NIL"):
            self.assertIn(argument, arguments)
        self.assertIn(str(self.data), arguments)
        self.assertIn(str(self.python), arguments)
        self.assertFalse(self.data.exists())

    def test_explicit_submit_is_the_only_action_without_test_only(self):
        result = self.run_action("submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["program"] for call in calls], ["python", "sbatch"])
        self.assertNotIn("--test-only", calls[1]["args"])
        self.assertTrue((self.data / "logs").is_dir())
        self.assertEqual((self.data / "logs").stat().st_mode & 0o777, 0o700)

    def test_failed_validation_and_bad_paths_block_scheduler(self):
        environment = dict(self.environment, MOCK_FAIL_VALIDATE="1")
        self.assertEqual(self.run_action("test-submit", environment=environment).returncode, 3)
        self.assertFalse(any(call["program"] == "sbatch" for call in self.calls()))
        for path in ("relative/path", "/", "/tmp/has%j", "/tmp/has\nnewline"):
            environment = dict(self.environment, VPML_DATA_ROOT=path)
            self.assertEqual(self.run_action("submit", environment=environment).returncode, 2)
        environment = dict(self.environment, VPML_PYTHON="python")
        self.assertEqual(self.run_action("submit", environment=environment).returncode, 2)
        self.assertFalse(any(call["program"] == "sbatch" for call in self.calls()))

    def test_batch_entrypoint_refuses_execution_without_allocation(self):
        environment = dict(self.environment)
        environment.pop("SLURM_JOB_ID", None)
        batch = ROOT / "scripts/delta/gaussian_amplitudes_v1.batch.sh"
        result = subprocess.run(["/bin/bash", str(batch)], env=environment, text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_dependency_lock_pins_direct_and_transitive_requirements(self):
        lock = (ROOT / "environments/delta/requirements.lock").read_text()
        for requirement in ("jax==0.10.0", "jaxlib==0.10.0", "jax-cuda12-plugin==0.10.0",
                            "numpy==2.4.4", "scipy==1.17.1", "matplotlib==3.10.9"):
            self.assertIn(requirement, lock)
        requirements = [line for line in lock.splitlines() if line and line[0].isalnum()]
        self.assertGreater(len(requirements), 20)
        self.assertTrue(all("==" in line for line in requirements))
        for section in lock.split("\n"):
            if section and section[0].isalnum():
                remaining = lock.split(section, 1)[1]
                self.assertTrue(remaining.lstrip().startswith("--hash=sha256:"))


if __name__ == "__main__":
    unittest.main()
