"""Manifest identity, split integrity and immutable publication checks."""
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vpml.datasets.manifest import Numerics, build_manifest, read_manifest, validate_manifest, write_manifest


def small_manifest(**kwargs):
    options = dict(k0_values=(0.3,), mode_counts=(3,), decays=(1.2,),
                   draws_per_combination=3, strengths=(0.1, 1.0), split_counts=(1, 1, 1),
                   numerics=Numerics(nx=16, nv=33, output_nx=8, final_time=0.1, block_steps=2))
    options.update(kwargs)
    return build_manifest(**options)


class ManifestTests(unittest.TestCase):
    def test_default_count_and_grouped_splits(self):
        manifest = build_manifest()
        self.assertEqual(len(manifest["cases"]), 1200)
        self.assertEqual(Counter(case["split"] for case in manifest["cases"]),
                         {"train": 960, "validation": 120, "test": 120})
        self.assertEqual(len({case["shape_id"] for case in manifest["cases"]}), 240)
        self.assertEqual(len({case["case_id"] for case in manifest["cases"]}), 1200)
        shapes = {}
        for case in manifest["cases"]:
            shapes.setdefault(case["shape_id"], []).append(case)
        for cases in shapes.values():
            self.assertEqual(len({case["split"] for case in cases}), 1)
            self.assertEqual({case["strength"] for case in cases}, {0.1, 0.3, 0.5, 0.7, 1.0})
            for case in cases[1:]:
                self.assertEqual(case["coefficients"], cases[0]["coefficients"])

    def test_reproducibility_and_identity_include_scientific_parameters(self):
        baseline = small_manifest()
        self.assertEqual(baseline, small_manifest())
        original = baseline["cases"][0]["case_id"]
        for change in (dict(k0_values=(0.4,)), dict(mode_counts=(5,)), dict(decays=(2.0,)),
                       dict(base_seed=123), dict(amplitudes={"density": .2, "velocity": .25, "temperature": .1})):
            with self.subTest(change=change):
                self.assertNotEqual(original, small_manifest(**change)["cases"][0]["case_id"])

    def test_validates_numerics_and_preserves_noninteger_output_ratio(self):
        numerical = Numerics()
        self.assertEqual(numerical.nsteps, 12000)
        self.assertEqual(numerical.nsamples, 4801)
        self.assertEqual(numerical.save_dt / numerical.dt, 2.5)
        with self.assertRaises(FrozenInstanceError):
            numerical.nx = 8
        for change in (dict(nx=15), dict(nv=True), dict(output_nx=2048), dict(output_nx=7),
                       dict(dt=0), dict(dt=np.nan), dict(final_time=.013), dict(save_dt=.009),
                       dict(save_dt=.031), dict(vmin=8), dict(storage_dtype="float16"), dict(block_steps=0)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Numerics(**change)

    def test_rejects_corrupt_ambiguous_or_unsupported_manifests(self):
        baseline = small_manifest()
        alterations = [
            lambda m: m.update(unexpected=1),
            lambda m: m.update(schema_version=True),
            lambda m: m.update(output_schema="unknown"),
            lambda m: m["numerics"].update(unexpected=1),
            lambda m: m["cases"][0].update(case_id="../../outside"),
            lambda m: m["cases"][0].update(strength=float("nan")),
            lambda m: m["cases"][0].update(mode_count=5),
            lambda m: m["cases"][0]["coefficients"][0][0].__setitem__(0, 99),
            lambda m: m["cases"][0]["coefficients"][0][0].__setitem__(0, True),
            lambda m: m["cases"][0]["coefficients"][0][0].__setitem__(0, "0.1"),
            lambda m: m["cases"][1].update(split="test"),
            lambda m: m["cases"].append(deepcopy(m["cases"][0])),
            lambda m: m["cases"].pop(),
        ]
        for alter in alterations:
            manifest = deepcopy(baseline)
            alter(manifest)
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_atomic_exclusive_roundtrip_and_failure_cleanup(self):
        manifest = small_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, manifest)
            self.assertEqual(read_manifest(path), manifest)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_manifest(path, manifest)
            self.assertEqual(path.read_bytes(), before)
            failed_path = Path(directory) / "failed.json"
            with patch("vpml.datasets.manifest.os.link", side_effect=OSError("injected publication failure")):
                with self.assertRaises(OSError):
                    write_manifest(failed_path, manifest)
            self.assertFalse(failed_path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [path])
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                read_manifest(path)
            path.write_text('{"schema_version":NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                read_manifest(path)

    def test_builder_rejects_ambiguous_choices(self):
        for change in (dict(k0_values=(.3, .3)), dict(strengths=()), dict(draws_per_combination=0),
                       dict(split_counts=(1, 1, 2)), dict(base_seed=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                small_manifest(**change)


if __name__ == "__main__":
    unittest.main()
