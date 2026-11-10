"""Exercise launcher boundaries with fake Python and scheduler executables.

No test contacts Slurm, starts JAX, installs packages or reserves hardware.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/delta/run_gaussian_amplitudes.sh"
PARAMETERS = ROOT / "scripts/delta/gaussian_amplitudes.json"
RUN_PARAMETERS = ROOT / "scripts/delta/gaussian_amplitudes.params.sh"
MANIFEST_IDS = ["case_" + f"{index:024x}" for index in range(5)]


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
from pathlib import Path
if Path(sys.argv[1]).name == "record_submission.py":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
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
from pathlib import Path
with open(os.environ["MOCK_CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"program": "sbatch", "args": sys.argv[1:],
                             "sbatch_env": [name for name in os.environ if name.startswith("SBATCH_")],
                             "submission_intents": sorted(str(path) for path in
                                 (Path(os.environ["VPML_DATA_ROOT"]) / "submissions").glob("*.json"))}) + "\\n")
print("hypothetical" if "--test-only" in sys.argv else "mock-submitted")
''')
        scheduler.chmod(0o700)
        self.environment = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin",
                                VPML_CODE_ROOT=str(ROOT), VPML_DATA_ROOT=str(self.data),
                                VPML_PYTHON=str(self.python), MOCK_CALL_LOG=str(self.log),
                                LD_LIBRARY_PATH="/unrelated/cuda", SBATCH_ARRAY_INX="0-1000",
                                SBATCH_EXCLUSIVE="1")
        self.environment.pop("VPML_RUN_PARAMS", None)
        self.environment.pop("VPML_SOURCE_COMMIT", None)

    def run_action(self, *args, environment=None):
        return subprocess.run(["/bin/bash", str(LAUNCHER), *args],
                              env=environment or self.environment, text=True,
                              capture_output=True, check=False)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def parameter_environment(self, extra):
        parameters = self.directory / "explicit parameters.sh"
        parameters.write_text(f"source {shlex.quote(str(RUN_PARAMETERS))}\n{extra}\n")
        return dict(self.environment, VPML_RUN_PARAMS=str(parameters))

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
                                              "--parameters", str(PARAMETERS)])
        self.assertEqual(calls[1]["args"][1:], ["validate", "--manifest", str(self.data / "manifest.json"),
                                              "--parameters", str(PARAMETERS)])
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
                         "--constraint=work", "--export=NIL", "--job-name=vpml-gaussian"):
            self.assertIn(argument, arguments)
        self.assertIn(str(self.data), arguments)
        self.assertIn(str(self.python), arguments)
        self.assertEqual(arguments[-6:], [str(ROOT), str(self.data), str(self.python),
                                         str(PARAMETERS), "3300", "--stop-on-failure"])
        self.assertNotIn("--case-id", arguments)
        self.assertFalse(self.data.exists())

    def test_canonical_scientific_parameters_and_explicit_default_match(self):
        parameters = json.loads(PARAMETERS.read_text())
        self.assertEqual(parameters["numerics"], {
            "nx": 2048, "nv": 8192, "dt": 0.01, "final_time": 120.0,
            "vmin": -8.0, "vmax": 8.0, "output_nx": 128, "save_dt": 0.025,
            "block_steps": 400, "storage_dtype": "float64",
        })
        result = self.run_action("test-submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.calls()
        self.log.unlink()
        result = self.run_action("test-submit", environment=dict(
            self.environment, VPML_RUN_PARAMS=str(RUN_PARAMETERS)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), expected)
        self.assertFalse(self.data.exists())

    def test_explicit_selection_is_validated_and_forwarded_exactly(self):
        selected = ["case_" + f"{index:024x}" for index in range(5)]
        environment = self.parameter_environment(
            "VPML_CASE_IDS=(" + " ".join(shlex.quote(value) for value in selected) + ")\n"
            "VPML_STOP_ON_FAILURE=1\nVPML_SLURM_JOB_NAME=vpml-gaussian-selected")
        result = self.run_action("test-submit", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["program"] for call in calls], ["python", "sbatch"])
        expected = [item for case_id in selected for item in ("--case-id", case_id)]
        self.assertEqual(calls[0]["args"][-len(expected):], expected)
        self.assertNotIn("--stop-on-failure", calls[0]["args"])
        batch_arguments = calls[1]["args"]
        self.assertEqual(batch_arguments[-len(expected)-1:], [*expected, "--stop-on-failure"])
        self.assertIn("--job-name=vpml-gaussian-selected", batch_arguments)
        self.assertFalse(self.data.exists())

    def test_selected_case_validation_failure_never_calls_scheduler(self):
        environment = self.parameter_environment("VPML_CASE_IDS=(unknown unknown)\nVPML_STOP_ON_FAILURE=1")
        environment["MOCK_FAIL_VALIDATE"] = "1"
        result = self.run_action("submit", environment=environment)
        self.assertEqual(result.returncode, 3)
        self.assertEqual([call["program"] for call in self.calls()], ["python"])
        self.assertEqual(self.calls()[0]["args"][-4:], ["--case-id", "unknown", "--case-id", "unknown"])
        self.assertFalse(self.data.exists())

    def test_run_parameter_path_and_resource_limits_fail_before_scheduler(self):
        for value in ("relative.params.sh", str(self.directory / "missing.params.sh")):
            with self.subTest(path=value):
                result = self.run_action("submit", environment=dict(self.environment, VPML_RUN_PARAMS=value))
                self.assertEqual(result.returncode, 2, result.stderr)
        for setting in ("VPML_SLURM_TIME=02:00:00", "VPML_SLURM_TIME=00:60:00",
                        "VPML_MAX_SECONDS=3600", "VPML_MAX_SECONDS=0",
                        "VPML_SLURM_TIME=00:30:00", "VPML_SLURM_CPUS=32",
                        "VPML_SLURM_MEMORY=120G", "VPML_SLURM_PARTITION=gpuA40x4",
                        "VPML_SHUTDOWN_MARGIN_SECONDS=59", "VPML_SHUTDOWN_MARGIN_SECONDS=181",
                        "VPML_SHUTDOWN_MARGIN_SECONDS=invalid",
                        "VPML_SLURM_TIME=00:05:00\nVPML_MAX_SECONDS=241\nVPML_SHUTDOWN_MARGIN_SECONDS=60",
                        "VPML_SLURM_TIME=00:01:00\nVPML_MAX_SECONDS=1\nVPML_SHUTDOWN_MARGIN_SECONDS=60",
                        "VPML_STOP_ON_FAILURE=maybe"):
            with self.subTest(setting=setting):
                result = self.run_action("submit", environment=self.parameter_environment(setting))
                self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.data.exists())

    def test_bounded_comparison_jobs_fit_within_fifteen_minutes(self):
        wall_seconds = 0
        for wall, worker in (("00:05:00", 240), ("00:10:00", 540)):
            with self.subTest(wall=wall):
                environment = self.parameter_environment(
                    f"VPML_SLURM_TIME={wall}\nVPML_MAX_SECONDS={worker}\n"
                    "VPML_SHUTDOWN_MARGIN_SECONDS=60")
                result = self.run_action("test-submit", environment=environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                scheduler = self.calls()[-1]
                self.assertIn(f"--time={wall}", scheduler["args"])
                self.assertIn("--signal=USR1@60", scheduler["args"])
                self.assertEqual(scheduler["args"][-2], str(worker))
                self.assertEqual(scheduler["args"][-1], "--stop-on-failure")
                wall_seconds += int(wall[3:5]) * 60
        self.assertEqual(wall_seconds, 900)
        self.assertFalse(self.data.exists())

    def test_explicit_submit_is_the_only_action_without_test_only(self):
        self.data.mkdir(mode=0o700)
        manifest = self.data / "manifest.json"
        manifest.write_text(json.dumps({"cases": [{"case_id": case_id} for case_id in MANIFEST_IDS]}))
        result = self.run_action("submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["program"] for call in calls], ["python", "sbatch"])
        self.assertNotIn("--test-only", calls[1]["args"])
        self.assertTrue((self.data / "logs").is_dir())
        self.assertEqual((self.data / "logs").stat().st_mode & 0o777, 0o700)
        records = calls[1]["submission_intents"]
        self.assertEqual(len(records), 1)
        path = Path(records[0])
        payload = path.read_bytes()
        self.assertEqual(path.stem, hashlib.sha256(payload).hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        record = json.loads(payload)
        self.assertEqual(record["record_type"], "submission_intent")
        self.assertEqual(record["manifest"]["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(record["scientific_parameters"]["text"], PARAMETERS.read_text())
        self.assertEqual(record["launcher_parameters"]["text"], RUN_PARAMETERS.read_text())
        self.assertEqual(record["selected_case_ids"], MANIFEST_IDS)
        self.assertEqual(record["scheduler"]["wall_seconds"], 3600)
        self.assertEqual(record["scheduler"]["shutdown_margin_seconds"], 180)
        self.assertEqual(record["worker"]["max_seconds"], 3300)
        self.assertTrue(record["worker"]["stop_on_failure"])
        self.assertEqual(record["worker"]["arguments"], ["--stop-on-failure"])
        self.assertIn("--gpus-per-node=1", record["scheduler"]["arguments"])
        self.assertEqual(record["code"]["root"], str(ROOT))
        self.assertIn("vpml/datasets/runner.py", record["code"]["source"]["files"])
        self.assertRegex(record["code"]["source"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(record["code"]["launcher_sha256"][LAUNCHER.name],
                         hashlib.sha256(LAUNCHER.read_bytes()).hexdigest())
        # A later submit records its own intent without overwriting earlier evidence.
        result = self.run_action("submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(path.parent.glob("*.json"))), 2)
        self.assertEqual(path.read_bytes(), payload)

    def test_explicit_parameter_snapshots_survive_later_edits(self):
        self.data.mkdir(mode=0o700)
        manifest = self.data / "manifest.json"
        manifest.write_text(json.dumps({"cases": [{"case_id": case_id} for case_id in MANIFEST_IDS]}))
        scientific = self.directory / "custom science.json"
        scientific.write_bytes(PARAMETERS.read_bytes())
        environment = self.parameter_environment(
            f"VPML_PARAMETERS_FILE={shlex.quote(str(scientific))}\n"
            "VPML_CASE_IDS=()\nVPML_STOP_ON_FAILURE=0\n"
            "VPML_MAX_SECONDS=1200\nVPML_SLURM_TIME=00:30:00")
        launcher_snapshot = Path(environment["VPML_RUN_PARAMS"]).read_text()
        scientific_snapshot = scientific.read_text()
        result = self.run_action("submit", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = next((self.data / "submissions").glob("*.json"))
        before = path.read_bytes()
        scientific.write_text("{}")
        Path(environment["VPML_RUN_PARAMS"]).write_text("# changed later\n")
        manifest.write_text("{}")
        record = json.loads(path.read_bytes())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(record["scientific_parameters"]["text"], scientific_snapshot)
        self.assertEqual(record["launcher_parameters"]["text"], launcher_snapshot)
        self.assertEqual(record["selected_case_ids"], MANIFEST_IDS)
        self.assertEqual(record["scheduler"]["wall_seconds"], 1800)
        self.assertEqual(record["worker"]["max_seconds"], 1200)
        self.assertFalse(record["worker"]["stop_on_failure"])
        self.assertEqual(record["worker"]["arguments"], [])

    def test_missing_manifest_blocks_submission_intent_and_scheduler(self):
        result = self.run_action("submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call["program"] for call in self.calls()], ["python"])
        self.assertFalse((self.data / "submissions").exists())

    def test_failed_record_write_leaves_no_published_or_temporary_intent(self):
        self.data.mkdir(mode=0o700)
        manifest = self.data / "manifest.json"
        manifest.write_text(json.dumps({"cases": [{"case_id": case_id} for case_id in MANIFEST_IDS]}))
        path = ROOT / "scripts/delta/record_submission.py"
        spec = importlib.util.spec_from_file_location("delta_record_submission", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arguments = [str(path), str(ROOT), str(self.data), str(self.python),
                     str(PARAMETERS), str(RUN_PARAMETERS), "3300", "3600", "180",
                     "--gpus-per-node=1", "--"]
        # Fail after the payload has been written but before publication. This
        # must not leave a truncated or writable intent under its final name.
        with mock.patch.object(sys, "argv", arguments), \
                mock.patch.dict(os.environ, self.environment, clear=True), \
                mock.patch.object(module.os, "fchmod", side_effect=OSError("permission failure")):
            with self.assertRaisesRegex(OSError, "permission failure"):
                module.main()
        self.assertEqual(list((self.data / "submissions").iterdir()), [])

    def test_archive_commit_declaration_is_separate_from_observed_git(self):
        path = ROOT / "scripts/delta/record_submission.py"
        spec = importlib.util.spec_from_file_location("delta_record_submission", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        commit = "A" * 40
        with mock.patch.dict(os.environ, {"VPML_SOURCE_COMMIT": commit}):
            identity = module.git_identity(self.directory)
        self.assertEqual(identity, {"git_commit": None, "git_dirty": None,
                                    "declared_source_commit": commit.lower()})
        with mock.patch.dict(os.environ, {"VPML_SOURCE_COMMIT": "not-a-commit"}):
            with self.assertRaisesRegex(ValueError, "40 hexadecimal"):
                module.git_identity(self.directory)

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
        batch = ROOT / "scripts/delta/gaussian_amplitudes.batch.sh"
        result = subprocess.run(["/bin/bash", str(batch)], env=environment, text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def batch_command(self):
        scheduler = self.bin / "fake srun"
        scheduler.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["MOCK_CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"program": "srun", "args": sys.argv[1:],
                             "backend": os.environ.get("JAX_PLATFORMS"),
                             "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
''')
        scheduler.chmod(0o700)
        batch = self.directory / "batch.sh"
        source = (ROOT / "scripts/delta/gaussian_amplitudes.batch.sh").read_text()
        # Run the actual parser and command construction; substitute only the
        # system Slurm executable so this test cannot reserve hardware.
        self.assertEqual(source.count("exec /usr/bin/srun "), 1)
        batch.write_text(source.replace("exec /usr/bin/srun ", f"exec {shlex.quote(str(scheduler))} "))
        command = ["/bin/bash", str(batch), str(ROOT), str(self.data), str(self.python),
                   str(ROOT / "scripts/delta/gaussian_amplitudes.json"), "3300"]
        environment = dict(self.environment, SLURM_JOB_ID="fake-job", SLURM_CPUS_PER_TASK="16",
                           CUDA_VISIBLE_DEVICES="allocated-device")
        return command, environment

    def test_batch_forwards_only_selected_cases_and_failure_policy(self):
        command, environment = self.batch_command()
        options = ["--case-id", "case_a", "--case-id", "case_b", "--stop-on-failure"]
        result = subprocess.run([*command, *options], env=environment, text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual([call["program"] for call in calls], ["python", "srun"])
        self.assertEqual(calls[0]["args"][-4:], options[:-1])
        self.assertEqual(calls[1]["args"][-len(options)-2:], ["--max-seconds", "3300", *options])
        self.assertIn("--gpus-per-task=1", calls[1]["args"])
        self.assertEqual(calls[1]["backend"], "cuda")
        self.assertEqual(calls[1]["visible_devices"], "allocated-device")

    def test_batch_rejects_unsupported_options_and_oversized_budget(self):
        command, environment = self.batch_command()
        for options in (["--backend", "cpu"], ["--case-id"],
                        ["--stop-on-failure", "--stop-on-failure"]):
            with self.subTest(options=options):
                result = subprocess.run([*command, *options], env=environment, text=True,
                                        capture_output=True, check=False)
                self.assertEqual(result.returncode, 2, result.stderr)
        result = subprocess.run([*command[:-1], "3301"], env=environment, text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
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
