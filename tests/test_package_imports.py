"""Metadata-only imports remain usable without a numerical backend."""
import os
import subprocess
import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_metadata_commands_do_not_import_jax_scipy_or_plotting(self):
        code = '''
import importlib.abc
import sys
class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'scipy', 'matplotlib'}:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockNumerics())
from vpml.datasets.manifest import build_manifest
from vpml.datasets.runner import dataset_status
from vpml.io.serialization import json_digest
assert len(build_manifest()['cases']) == 1200
assert len(json_digest({'test': 1})) == 64
'''
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ)

    def test_public_solver_names_keep_their_implementations(self):
        import vpml
        from vpml.solvers.physical_grid import PhysicalGridVlasovPoissonConfig
        self.assertIs(vpml.PhysicalGridVlasovPoissonConfig, PhysicalGridVlasovPoissonConfig)
        self.assertIn("FourierHermiteIMEX", dir(vpml))
        with self.assertRaises(AttributeError):
            getattr(vpml, "missing_public_name")


if __name__ == "__main__":
    unittest.main()
