"""Checks for the measured-field to history-matrix construction."""
import numpy as np

from scripts.analyze_gaussian_dataset import (
    common_standardization, effective_rank, history_features, spatial_features,
    primary_cases, refinement_comparisons,
    common_spectral_band_errors,
)


def test_effective_rank_uses_singular_values_not_squares():
    matrix = np.array([[1, .5, .25], [1, -.5, -.25],
                       [-1, .5, -.25], [-1, -.5, .25]])
    result = effective_rank(matrix)
    np.testing.assert_allclose(result["singular_values"], [2, 1, .5])
    p = np.array([4, 2, 1]) / 7
    np.testing.assert_allclose(result["effective_rank"], np.exp(-(p * np.log(p)).sum()))
    np.testing.assert_allclose(effective_rank(np.eye(3))["effective_rank"], 3)
    np.testing.assert_allclose(effective_rank(np.ones((5, 3)))["effective_rank"], 1)
    assert effective_rank(np.zeros((4, 3)))["effective_rank"] == 0


def test_history_blocks_are_three_times_of_one_trajectory():
    theta = 2 * np.pi * np.arange(32) / 32
    t = np.arange(401) / 10
    case = {"times": t,
            "n": 1 + (1 + t[:, None]) * np.cos(theta),
            "u": t[:, None] * np.sin(theta),
            "p": 1 + t[:, None] ** 2 * np.cos(2 * theta)}
    snapshot = spatial_features(case, 4)
    assert snapshot.shape == (401, 24)
    np.testing.assert_allclose(snapshot[:, 0], (1 + t) / 2)
    np.testing.assert_allclose(snapshot[:, 12], -t / 2, atol=1e-12)
    history = history_features(case, 4)
    assert history.shape == (20, 72)
    np.testing.assert_allclose(history[0], np.concatenate([snapshot[0], snapshot[75], snapshot[150]]))
    np.testing.assert_allclose(history[-1], np.concatenate([snapshot[190], snapshot[265], snapshot[340]]))


def test_common_calibration_keeps_between_collection_offsets():
    a = np.array([[0., 0.], [2., 0.]])
    b = np.array([[4., 0.], [6., 0.]])
    aa, bb, mean, scale = common_standardization(a, b)
    np.testing.assert_allclose(mean, [3, 0])
    np.testing.assert_allclose(scale[0], np.sqrt(5))
    assert np.isfinite(aa).all() and np.isfinite(bb).all()
    assert aa[:, 0].mean() < 0 < bb[:, 0].mean()
    np.testing.assert_allclose(np.vstack((aa, bb)).mean(axis=0), 0, atol=1e-15)


def test_refinement_uses_actual_velocity_coordinates_and_excludes_duplicate_cases():
    def case(nx, nv, label):
        x = np.arange(nx) * 2 * np.pi / nx
        v = np.linspace(-8, 8, nv)
        times = np.arange(401) / 10
        variation = np.cos(x)[None, :] * np.exp(-times[:, None] / 20)
        return {"id": "s1_l1_" + label, "family": "gaussian", "seed": 1,
                "strength": 1., "k0": 1., "times": times, "x": x, "v": v,
                "n": 1 + variation, "u": variation, "p": 1 + 2 * variation,
                "E": .3 * variation, "Q": .1 * variation, "g": .2 * variation,
                "final_f": 2 + .1 * v[:, None] + .1 * np.sin(x)[None, :],
                "grid": {"nx": nx, "nv": nv, "dt": .02}, "eta_NL": .2,
                "min_full_distribution": 0., "max_negative_mass_fraction": 0.,
                "qualification_pass": True}
    base, refined = case(16, 16, "base"), case(32, 32, "refined")
    assert len(primary_cases([base, refined])) == 1
    result = refinement_comparisons([base, refined])[0]
    assert result["refinement_checks_pass"]
    assert "dt" not in result["changed_grid_parameters"]
    assert result["final_distribution_relative_L2"] < 1e-15
    assert max(result["relative_L2_errors"].values()) < 1e-15


def test_common_band_is_compared_before_nodal_aliasing():
    times = np.linspace(0, 1, 11)
    x_coarse = 2 * np.pi * np.arange(16) / 16
    x_fine = 2 * np.pi * np.arange(32) / 32
    amplitude = (1 + times)[:, None]
    low = amplitude * np.cos(x_coarse)
    fine = amplitude * (np.cos(x_fine) + .4 * np.cos(10 * x_fine))
    indices = np.arange(len(times))
    errors = common_spectral_band_errors({"E": low, "n": 1 + low},
                                         {"E": fine, "n": 1 + fine},
                                         indices, indices, times, 4)
    assert max(errors.values()) < 1e-14
    assert np.linalg.norm(low - fine[:, ::2]) / np.linalg.norm(fine[:, ::2]) > .1
