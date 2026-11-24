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

    def test_sampled_rounds_replay_are_balanced_and_use_only_bounded_draws(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            definition = manifest(5, (3, 1, 1), (.1, .3, .5, .7, 1.), 25.)
            write_manifest(root / "manifest.json", definition)
            data = GaussianTrainingData(root, root / "cache", settings())
            draw = lambda step: data.sample_rows(step, 1729, 48)
            first = np.concatenate([draw(step) for step in range(25)])
            self.assertEqual(len(set(map(tuple, first))), 1200)
            self.assertEqual(data.round_updates(48), 25)
            strengths = np.asarray([case["strength"] for case in data.cases])
            self.assertTrue(all(data.cases[i]["split"] == "train" for i in first[:, 0]))
            strength_counts = np.unique(strengths[first[:, 0]], return_counts=True)[1]
            np.testing.assert_array_equal(strength_counts, np.full(5, 240))
            for i in range(25):
                counts = np.unique(strengths[draw(i)[:, 0]], return_counts=True)[1]
                self.assertEqual(sorted(counts), [9, 9, 10, 10, 10])
            next_round = np.concatenate([draw(step) for step in range(25, 50)])
            self.assertTrue(set(map(tuple, first)) & set(map(tuple, next_round)))
            np.testing.assert_array_equal(draw(0), first[:48])
            # Simulate an enormous eligible window space without constructing it.
            data.anchors_per_case = 100_000_000
            far_future = draw(1_000_000)
            self.assertEqual(far_future.shape, (48, 2))
            self.assertTrue((far_future[:, 1] < data.anchors_per_case).all())

    def test_small_corpus_uses_short_final_batch_before_reshuffling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root / "manifest.json", manifest(1, (1, 0, 0), (.3, .5), .25))
            data = GaussianTrainingData(root, root / "cache", settings())
            self.assertEqual(data.round_updates(3), 2)
            first = data.sample_rows(0, 8, 3)
            tail = data.sample_rows(1, 8, 3)
            self.assertEqual(len(tail), 1)
            self.assertEqual(len(set(map(tuple, np.vstack((first, tail))))), 4)
            self.assertEqual(len(data.sample_rows(2, 8, 3)), 3)

    def test_training_index_opens_without_case_metadata_scan_and_pins_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            cache = Path(directory) / "cache"
            definition = manifest(3, (1, 1, 1), (.3, 1.), .25)
            write_manifest(root / "manifest.json", definition)
            for case in definition["cases"]:
                if case["split"] != "test":
                    path = root / "cases" / case["case_id"] / "progress.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({"status": "complete", "outputs": {"sha256": "a"*64}}))
            unindexed = GaussianTrainingData(root, cache, settings())
            indexed = GaussianTrainingData.open_indexed(root, cache, settings())
            self.assertEqual(indexed.training_outputs_sha256, unindexed.training_outputs_sha256)
            for i, plain in enumerate(definition["cases"]):
                self.assertEqual(indexed.cases[i], {k: plain[k] for k in indexed.cases[i]})
            with patch("vpml.datasets.training_index.read_manifest", side_effect=AssertionError), \
                 patch("vpml.datasets.training_index.read_completed_digest", side_effect=AssertionError):
                resumed = GaussianTrainingData.open_indexed(root, cache, settings())
                self.assertEqual(indexed.training_outputs_sha256, resumed.training_outputs_sha256)
            path = root / "cases" / indexed.cases[0]["case_id"] / "progress.json"
            path.write_text(json.dumps({"status": "complete", "outputs": {"sha256": "b"*64}}))
            with self.assertRaisesRegex(ValueError, "published training index"):
                resumed.state(0)

    def test_packed_moments_match_source_and_detect_selected_case_corruption(self):
        from vpml.datasets.gaussian_training import pack_training_cache

        with TemporaryDirectory() as directory:
            base = Path(directory)
            root, cache = base / "reference", base / "packed-cache"
            definition = manifest(2, (1, 1, 0), (.3, .5), .25)
            run_dataset(definition, root, backend="cpu", max_seconds=60)
            direct = GaussianTrainingData(root, base / "comparison-cache", settings())
            reference = direct.batch([[2, 0], [0, 1], [1, 0]])
            target_bytes = 11 * 3 * 8 * 8 * 2
            packed_path = pack_training_cache(root, cache, target_bytes=target_bytes)
            self.assertEqual(len(list(packed_path.glob("*.f64"))), 2)
            self.assertFalse(list(cache.rglob("case_*.npy")))
            with patch("vpml.datasets.gaussian_training._read_current_case", side_effect=AssertionError):
                self.assertEqual(pack_training_cache(root, cache), packed_path)
                indexed = GaussianTrainingData.open_indexed(root, cache, settings())
                batch = indexed.batch([[2, 0], [0, 1], [1, 0]])
                for name in reference:
                    np.testing.assert_array_equal(reference[name], batch[name])
                self.assertFalse(indexed.state(0).flags.writeable)
            with (packed_path / "00000000.f64").open("r+b") as handle:
                first = handle.read(1)[0]
                handle.seek(0)
                handle.write(bytes([first ^ 1]))
            reopened = GaussianTrainingData.open_indexed(root, cache, settings())
            with self.assertRaisesRegex(ValueError, "Packed moment checksum mismatch"):
                reopened.state(0)
