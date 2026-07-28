import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.train.interface_flux_data import (
    IC_SPLIT_HELDOUT,
    IC_SPLIT_TRAIN,
    StreamingMoments,
    build_ic_manifest,
    case_shard_is_complete,
    evaluate_manifest_case,
    grouped_history_gather,
    load_ic_manifest,
    load_sharded_reference,
    write_case_shard,
)


class InterfaceFluxDataTests(unittest.TestCase):
    def test_manifest_is_deterministic_stratified_and_normalized(self) -> None:
        first = build_ic_manifest(
            cases_per_regime=8,
            heldout_per_regime=2,
            generation_seed=3,
            split_seed=5,
        )
        second = build_ic_manifest(
            cases_per_regime=8,
            heldout_per_regime=2,
            generation_seed=3,
            split_seed=5,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["cases"]), 24)
        x = np.linspace(0.0, 4.0 * np.pi, 16384, endpoint=False)
        for regime in (
            "linear_landau",
            "nonlinear_landau_weak",
            "nonlinear_landau_strong",
        ):
            cases = [case for case in first["cases"] if case["regime"] == regime]
            self.assertEqual(
                sum(case["split"] == IC_SPLIT_TRAIN for case in cases),
                6,
            )
            self.assertEqual(
                sum(case["split"] == IC_SPLIT_HELDOUT for case in cases),
                2,
            )
            for case in cases:
                perturbation = evaluate_manifest_case(case, x)
                self.assertAlmostEqual(
                    float(np.max(np.abs(perturbation))),
                    float(case["epsilon"]),
                    places=12,
                )
                self.assertAlmostEqual(float(case["relative_phases"][0]), 0.0)

    def test_shard_round_trip_checksum_and_memmap_loading(self) -> None:
        manifest = build_ic_manifest(
            cases_per_regime=2,
            heldout_per_regime=1,
            generation_seed=7,
            split_seed=11,
        )
        case = manifest["cases"][0]
        history = (
            np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
            + 1j
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "cases").mkdir()
            (cache_dir / "snapshots").mkdir()
            write_case_shard(
                cache_dir,
                case,
                history,
                history_times=np.arange(5, dtype=np.float64),
                snapshots={
                    "times": np.arange(5, dtype=np.float64),
                    "energy": np.ones((5,), dtype=np.float64),
                    "E_hat_hist_times": np.arange(5, dtype=np.float64),
                    "E_hat_hist": np.ones((5, 3), dtype=np.complex128),
                    "k_arr": np.arange(3, dtype=np.float64),
                    "perturbation_x": np.zeros((3,), dtype=np.float64),
                    "snapshot_times": np.array([0.0]),
                    "snapshot_f": np.ones((1, 4, 3), dtype=np.float64),
                },
            )
            self.assertTrue(
                case_shard_is_complete(
                    cache_dir,
                    str(case["case_id"]),
                    expected_shape=history.shape,
                )
            )
            single_manifest = dict(manifest)
            single_manifest["cases"] = [case]
            loaded = load_sharded_reference(
                cache_dir,
                single_manifest,
                coeff_key="coeff",
            )
            loaded_history = loaded[str(case["regime"])]["coeff"][0]
            self.assertIsInstance(loaded_history, np.memmap)
            np.testing.assert_array_equal(loaded_history, history)

    def test_grouped_memmap_gather_matches_stacked_indexing_and_order(self) -> None:
        rng = np.random.default_rng(13)
        stacked = (
            rng.normal(size=(4, 20, 7, 5))
            + 1j * rng.normal(size=(4, 20, 7, 5))
        ).astype(np.complex64)
        case_indices = np.array([[3, 3, 3], [0, 0, 0], [3, 3, 3], [1, 1, 1]])
        time_indices = np.array([[8, 9, 10], [2, 1, 0], [4, 5, 6], [7, 8, 9]])
        expected = stacked[
            case_indices,
            time_indices,
            :6,
            :,
        ]
        actual = grouped_history_gather(
            tuple(stacked[index] for index in range(stacked.shape[0])),
            case_indices,
            time_indices,
            hermite_slice=slice(0, 6),
            fourier_slice=slice(None),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_streaming_moments_match_direct_float64_statistics(self) -> None:
        rng = np.random.default_rng(17)
        values = rng.normal(size=(10003, 9))
        moments = StreamingMoments()
        for block in np.array_split(values, 37):
            moments.update(block)
        mean, std = moments.finalize()
        np.testing.assert_allclose(mean, np.mean(values, axis=0), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(std, np.std(values, axis=0), rtol=1e-13, atol=1e-14)

    def test_manifest_file_hash_validation(self) -> None:
        manifest = build_ic_manifest(
            cases_per_regime=2,
            heldout_per_regime=1,
            generation_seed=19,
            split_seed=23,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            import json

            with path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            self.assertEqual(load_ic_manifest(path), manifest)


if __name__ == "__main__":
    unittest.main()
