"""Metadata-only selection preserves the scientific manifest and its order."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest

from vpml.datasets.manifest import build_manifest
from vpml.datasets.selection import select_cases


class DatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(k0_values=(.5,), mode_counts=(3,), decays=(2.0,),
                                       strengths=(.1, .3, .5), draws_per_combination=1)
        self.ids = [case["case_id"] for case in self.manifest["cases"]]

    def test_subset_retains_manifest_order_and_does_not_mutate_manifest(self):
        original = copy.deepcopy(self.manifest)
        selected = select_cases(self.manifest, [self.ids[2], self.ids[0]])
        self.assertEqual([case["case_id"] for case in selected], [self.ids[0], self.ids[2]])
        self.assertEqual(self.manifest, original)

    def test_omitted_selection_returns_all_cases_in_a_new_list(self):
        selected = select_cases(self.manifest)
        self.assertEqual(selected, self.manifest["cases"])
        self.assertIsNot(selected, self.manifest["cases"])

    def test_explicit_empty_duplicate_unknown_and_malformed_selections_are_rejected(self):
        for selected in ([], (), "", self.ids[0], [self.ids[0], self.ids[0]], ["unknown"],
                         [1], [[self.ids[0]]], {self.ids[0]}, {"case_id": self.ids[0]}):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, "unique known IDs"):
                select_cases(self.manifest, selected)

    def test_import_does_not_load_a_numerical_backend(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; from vpml.datasets.selection import select_cases; "
            "assert 'jax' not in sys.modules; assert 'jaxlib' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=root,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
