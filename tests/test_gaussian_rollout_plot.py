import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vpml.diagnostics.gaussian_rollout_plot import energy_figure, main


class GaussianRolloutPlotTests(unittest.TestCase):
    def test_plots_only_the_saved_autonomous_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            times = np.array([0., 40., 80., 120.])
            reference = np.array([.105, .02, .028, .01])
            initial = np.array([.105, .034])
            cases = []
            for case_id, strength in (("high", 1.), ("low", .1)):
                cases.append(dict(case_id=case_id, strength=strength,
                                  initial={"first_nonfinite_step": 2}, trained={}))
                trained = reference if strength < 1 else np.array([.105, .02, 1., 1e307])
                np.savez(root / f"{case_id}.npz", times=times, reference_energy=reference,
                         initial_energy=initial, trained_energy=trained)
            (root / "report.json").write_text(json.dumps({"cases": cases, "checkpoint_step": 103}))

            fig = energy_figure(root)
            self.assertEqual(fig.axes[0].get_title(), r"$\lambda = 0.1$")
            for axis in fig.axes:
                self.assertEqual(axis.get_xlabel(), r"$\omega_p t$")
                self.assertEqual(axis.get_yscale(), "log")
                line = next(l for l in axis.lines if l.get_label() == "Untrained model")
                np.testing.assert_array_equal(line.get_xdata(), times[:len(initial)])
                np.testing.assert_array_equal(line.get_ydata(), initial)
                self.assertIn("nonfinite at $t=80$", axis.texts[0].get_text())
                self.assertGreaterEqual(axis.get_ylim()[1], 1e307)

            main([str(root)])
            self.assertTrue((root / "electric-energy.png").read_bytes().startswith(b"\x89PNG"))
