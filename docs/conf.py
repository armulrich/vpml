"""Sphinx configuration; importing documentation never starts a computation."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" if (ROOT / "src" / "vpml").is_dir() else ROOT
sys.path.insert(0, str(SOURCE))

# Autodoc imports modules for signatures and docstrings, with no GPU selection.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["VPML_JAX_BACKEND"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

project = "VPML"
author = "VPML contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
root_doc = "index"
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_title = "VPML documentation"
html_theme_options = {"description": "Vlasov–Poisson simulation and learned closures"}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
