"""Exercise the real loss and the isolated burn-in initializer derivative."""

import json
import unittest
from unittest.mock import mock_open, patch

from model.diagnostics.linear_pair_preflight import main, run_preflight


class LinearPairPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_preflight(backend="cpu")

    def test_shared_seed_and_finite_actual_paired_gradients(self):
        report = self.report
        self.assertTrue(report["passed"], report["checks"])
        self.assertGreater(len(report["shared_parameter_names"]), 10)
        self.assertEqual(report["mismatched_parameter_names"], [])
        for row in report["paired_actual_loss"].values():
            self.assertTrue(row["finite_loss_aux_gradient"])
            self.assertGreater(row["gradient_l2"], 0)

    def test_same_forward_isolates_disconnected_initializer_head(self):
        report = self.report
        actual = report["paired_actual_loss"]["latent6"]["initializer_tensor_gradient_l2"]
        self.assertEqual(len(actual), 3)
        self.assertTrue(all(value == 0 for value in actual.values()))
        counter = report["counterfactual"]
        self.assertGreater(counter["initializer_gradient_l2"], 0)
        self.assertEqual(counter["other_parameter_gradient_l2"], 0)
        self.assertTrue(report["checks"]["counterfactual_forward_trajectory_unchanged"])
        self.assertTrue(report["checks"]["reconstruction_matches_actual_loss"])
        self.assertIn("known_initializer_limitation", report["status"])
        json.dumps(report, allow_nan=False)

    def test_runner_out_contract_inherits_backend(self):
        output = mock_open()
        with patch("model.diagnostics.linear_pair_preflight.run_preflight", return_value=self.report) as probe:
            with patch("pathlib.Path.exists", return_value=False), patch("pathlib.Path.open", output):
                self.assertEqual(main(["--out", "/unused/preflight.json"]), 0)
        probe.assert_called_once_with(backend="auto", archive_dir=None)
        output.assert_called_once_with("x")
        saved = json.loads(output().write.call_args.args[0])
        self.assertTrue(saved["passed"])
        self.assertIn("known_initializer_limitation", saved["status"])


if __name__ == "__main__":
    unittest.main()
