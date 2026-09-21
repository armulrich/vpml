"""Verify paired command equivalence and original-cache write isolation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class LinearPairRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "model/train/modal_linear_closure_pair.py"
        spec = importlib.util.spec_from_file_location("test_pair_runner", path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_pair_commands_differ_only_in_model_dimension_and_private_cache(self):
        r = self.runner
        a = r.training_arguments("history", 20, Path("/nonexistent/pair/train"))
        b = r.training_arguments("latent", 20, Path("/nonexistent/pair/train"))
        self.assertEqual(a[a.index("--training-regimes")+1], "linear_landau")
        self.assertEqual(a[a.index("--normalization-checkpoint")+1], str(r.NORMALIZATION))
        self.assertEqual(a[a.index("--grad-clip")+1], "0")
        self.assertEqual(a[a.index("--update-norm-cap")+1], "0")
        self.assertNotIn("--steps-per-epoch", a)
        self.assertNotIn("--resume-run", a)
        differences = [i for i, (x,y) in enumerate(zip(a,b)) if x != y]
        self.assertEqual(differences, [a.index("--reference-cache")+1, a.index("--latent-memory-dim")+1])
        self.assertEqual(len(a), len(b))

    def test_paths_are_distinct_and_not_historical(self):
        r = self.runner
        self.assertNotEqual(r.run_path("history"), r.run_path("latent"))
        for model in ("history", "latent"):
            self.assertNotIn(r.run_path(model).name, r.SOURCE_RUNS)
        with self.assertRaises(ValueError):
            r.run_path("unknown")

    def test_derived_metadata_writes_do_not_touch_original_cache(self):
        r = self.runner
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); original=root/"original"; private=root/"private"
            (original/"derived/targets").mkdir(parents=True)
            (original/"cases").mkdir()
            (original/"metadata.json").write_text('original')
            (original/"derived/targets/metadata.json").write_text('target-original')
            (original/"derived/targets/case.npy").write_bytes(b'array')
            with mock.patch.object(r,"CACHE",original), mock.patch.object(r,"private_cache_path",return_value=private):
                r.prepare_private_cache("history")
            (private/"derived/targets/metadata.json").write_text('private-change')
            self.assertEqual((original/"derived/targets/metadata.json").read_text(),'target-original')
            self.assertTrue((private/"derived/targets/case.npy").is_symlink())
            self.assertEqual((private/"derived/targets/case.npy").read_bytes(),b'array')


if __name__ == "__main__":
    unittest.main()
