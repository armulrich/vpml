"""Reject unavailable or altered historical sources before configuring Modal."""

import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest import mock


EVALUATOR = Path(__file__).resolve().parents[1] / "scripts/evaluate_wavenumber_e5.py"


class ArchivedEvaluationSourceTests(unittest.TestCase):
    def load_evaluator(self, source):
        modal = types.ModuleType("modal")
        modal.is_local = lambda: True
        modal.App = mock.Mock(side_effect=AssertionError("Modal must not be configured"))
        with mock.patch.dict(sys.modules, {"modal": modal}), mock.patch.dict(
            os.environ, {"VPML_FROZEN_SOURCE": str(source)}
        ):
            runpy.run_path(str(EVALUATOR))

    def test_missing_snapshot_fails_before_modal_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "original E5 frozen_source"):
                self.load_evaluator(Path(directory) / "missing")

    def test_changed_snapshot_fails_before_modal_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for package in ("model", "vpml"):
                (source / package).mkdir()
                (source / package / "__init__.py").write_text("# changed source\n")
            with self.assertRaisesRegex(ValueError, "pinned training identity"):
                self.load_evaluator(source)


if __name__ == "__main__":
    unittest.main()
