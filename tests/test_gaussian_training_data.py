"""Gaussian windows preserve the kinetic moment convention and split by shape."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vpml.datasets.gaussian_training import GaussianTrainingData
from vpml.datasets.manifest import Numerics, build_manifest, write_manifest
from vpml.datasets.reader import read_case
from vpml.datasets.runner import run_dataset


def settings():
    return SimpleNamespace(nx=8, dt=0.025, preparation_steps=0, horizon=2,
                           compute_dtype="float64", memory_steps=4,
                           relative_normalization="window_rms")


def manifest(draws, splits, strengths, final_time):
    return build_manifest(
        k0_values=(0.4,), mode_counts=(3,), decays=(2.0,),
        draws_per_combination=draws, split_counts=splits, strengths=strengths,
        numerics=Numerics(nx=16, nv=64, output_nx=8, dt=0.025, save_dt=0.025,
                          final_time=final_time, storage_dtype="float64"),
    )


class GaussianTrainingDataTests(unittest.TestCase):
    def test_training_output_identity_includes_both_splits_and_survives_a_copy(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            definition = manifest(3, (1, 1, 1), (.7,), .25)
            write_manifest(source / "manifest.json", definition)
            for case in definition["cases"]:
                if case["split"] == "test":
                    continue
                path = source / "cases" / case["case_id"] / "progress.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"status": "complete", "outputs": {"sha256": "a" * 64}}))
            identity = GaussianTrainingData(source, base / "cache", settings()).training_outputs_sha256
            for split in ("train", "validation"):
                copy = base / f"copy-{split}"
                shutil.copytree(source, copy)
                data = GaussianTrainingData(copy, base / "cache", settings())
                self.assertEqual(data.training_outputs_sha256, identity)
                case = next(case for case in definition["cases"] if case["split"] == split)
                path = copy / "cases" / case["case_id"] / "progress.json"
                path.write_text(json.dumps({"status": "complete", "outputs": {"sha256": "b" * 64}}))
                restarted = GaussianTrainingData(copy, base / "cache", settings())
                self.assertEqual(data.manifest_sha256, restarted.manifest_sha256)
                self.assertNotEqual(restarted.training_outputs_sha256, identity)

    def test_epoch_covers_each_split_and_balances_strength_even_during_first_hour(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            definition = manifest(5, (3, 1, 1), (.1, .3, .5, .7, 1.), 12.)
            write_manifest(root / "manifest.json", definition)
            data = GaussianTrainingData(root, root / "cache", settings())
            rows = data.epoch_rows(1, 1729)
            np.testing.assert_array_equal(rows, data.epoch_rows(1, 1729))
            self.assertFalse(np.array_equal(rows, data.epoch_rows(2, 1729)))
            self.assertEqual(len({tuple(row) for row in rows}), len(rows))
            self.assertEqual(len(rows), 15 * data.anchors_per_case)
            strengths = np.array([case["strength"] for case in definition["cases"]])
            for offset in range(0, 5 * 48, 48):
                _, counts = np.unique(strengths[rows[offset:offset + 48, 0]], return_counts=True)
                self.assertEqual(sorted(counts), [9, 9, 10, 10, 10])
            _, overall = np.unique(strengths[rows[:240, 0]], return_counts=True)
            np.testing.assert_array_equal(overall, [48] * 5)
            shapes = {}
            for split in ("train", "validation", "test"):
                split_rows = data.epoch_rows(0, 1729, split)
                indices = np.unique(split_rows[:, 0])
                cases = [data.manifest["cases"][i] for i in indices]
                self.assertTrue(all(case["split"] == split for case in cases))
                shapes[split] = {case["shape_id"] for case in cases}
                for i in indices:
                    np.testing.assert_array_equal(
                        np.sort(split_rows[split_rows[:, 0] == i, 1]),
                        np.arange(data.anchors_per_case),
                    )
            self.assertTrue(shapes["train"].isdisjoint(shapes["validation"] | shapes["test"]))
            self.assertTrue(shapes["validation"].isdisjoint(shapes["test"]))

    def test_batch_uses_checked_raw_moments_and_reuses_atomic_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            cache = Path(directory) / "cache"
            definition = manifest(1, (1, 0, 0), (.7,), .25)
            run_dataset(definition, root, backend="cpu", max_seconds=60)
            saved = read_case(root, definition["cases"][0]["case_id"])
            raw = saved["raw_moments"].copy()
            raw[:, (0, 2), :] -= 1
            data = GaussianTrainingData(root, cache, settings())
            self.assertEqual(data.anchors_per_case, 2)
            batch = data.batch(np.array([[0, 0], [0, 1]]))
            np.testing.assert_array_equal(batch["initial"], raw[[0, 8]])
            np.testing.assert_array_equal(batch["targets"], np.stack((raw[1:3], raw[9:11])))
            np.testing.assert_allclose(batch["k_arr"], np.tile(.4 * np.arange(5), (2, 1)))
            np.testing.assert_array_equal(batch["memory"], 0)
            np.testing.assert_array_equal(batch["heat_flux_gradient_history"], 0)
            self.assertFalse(data.state(0).flags.writeable)
            with self.assertRaisesRegex(ValueError, "Anchor"):
                data.batch([[0, 2]])
            with self.assertRaisesRegex(IndexError, "case index"):
                data.batch([[-1, 0]])
            again = GaussianTrainingData(root, cache, settings())
            with patch("vpml.datasets.gaussian_training._read_current_case",
                       side_effect=AssertionError("verified cache should be reused")):
                np.testing.assert_array_equal(again.state(0), raw)
            # A damaged derived file is rebuilt from the checked reference.
            case_file = next(cache.rglob("*.npy"))
            with case_file.open("ab") as handle:
                handle.write(b"corruption")
            rebuilt = GaussianTrainingData(root, cache, settings())
            np.testing.assert_array_equal(rebuilt.state(0), raw)
            self.assertEqual(len(list(cache.rglob("*.npy"))), 1)

    def test_grid_or_cadence_mismatch_is_rejected_before_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root / "manifest.json", manifest(1, (1, 0, 0), (.3,), .25))
            config = settings()
            config.dt = .05
            with self.assertRaisesRegex(ValueError, "must match"):
                GaussianTrainingData(root, root / "cache", config)
