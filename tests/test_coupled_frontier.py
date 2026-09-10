import copy
import unittest

from model.diagnostics.compare_coupled_frontier import compare_frontier, compare_frontier_set


class CoupledFrontierTest(unittest.TestCase):
    def panels(self):
        baseline = {"cases": [dict(case_id=regime + "_ic00", regime=regime,
            bounded_to_final_time=True, minimum_density=0.5, minimum_pressure=0.5,
            epsilon_grow=0.04, epsilon_E=0.1, significant_transitions=[],
            latent_saturation={"saved_state_saturation_fraction": 0},
            full_envelope={"log_envelope_rmse": 1.0}) for regime in
            ("linear_landau", "nonlinear_landau_weak", "nonlinear_landau_strong")]}
        candidate = copy.deepcopy(baseline)
        candidate["cases"][-1].update(epsilon_E=0.08, full_envelope={"log_envelope_rmse": 0.8})
        return baseline, candidate

    def test_joint_improvement_passes_development_gate(self):
        self.assertTrue(compare_frontier(*self.panels())["passed"])

    def test_weaker_baseline_cannot_replace_stronger_checkpoint(self):
        old, new = self.panels()
        stronger = copy.deepcopy(old)
        stronger["cases"][-1]["epsilon_E"] = 0.07
        result = compare_frontier_set({"frozen": old, "specialist": stronger}, new)
        self.assertFalse(result["passed"])
        self.assertTrue(result["comparisons"]["frozen"]["passed"])
        self.assertFalse(result["comparisons"]["specialist"]["passed"])
        self.assertFalse(compare_frontier_set({}, new)["passed"])

    def test_better_average_cannot_hide_metric1_regression(self):
        old, new = self.panels()
        new["cases"][0]["epsilon_grow"] = 0.041
        result = compare_frontier(old, new)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Metric 1" in f for f in result["failures"]))

    def test_missing_envelope_or_saturation_cannot_pass(self):
        for key in ("full_envelope", "latent_saturation"):
            old, new = self.panels()
            del new["cases"][1][key]
            self.assertFalse(compare_frontier(old, new)["passed"])

    def test_later_rebound_regression_is_not_hidden(self):
        old, new = self.panels()
        transitions = [dict(start_time=i * 10, peak_time=i * 10 + 5,
                            teacher_factor=10.0, model_factor=5.0) for i in range(3)]
        old["cases"][-1]["significant_transitions"] = transitions
        new["cases"][-1]["significant_transitions"] = copy.deepcopy(transitions)
        new["cases"][-1]["significant_transitions"][-1]["model_factor"] = 2.0
        self.assertFalse(compare_frontier(old, new)["passed"])


if __name__ == "__main__":
    unittest.main()
