# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Glasses Detector"
copyright = "2024, Mantas Birškus"
author = "Mantas Birškus"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx_toolbox.collapse",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
}

# TODO: possibly remove when sphinx supports types
autodoc_type_aliases = {
    "FilePath": "str | bytes | os.PathLike",
    "Scalar": "bool | int | float | str | numpy.generic | numpy.ndarray | torch.Tensor",
    "Tensor": "typing.Iterable[bool | int | float | str | numpy.generic | numpy.ndarray | torch.Tensor | typing.Iterable[...]] | PIL.Image.Image",
    "Default": "bool | int | float | str | numpy.generic | numpy.ndarray | torch.Tensor | typing.Iterable[bool | int | float | str | numpy.generic | numpy.ndarray | torch.Tensor | typing.Iterable[...]] | PIL.Image.Image",
    "StandardScalar": "bool | int | float | str",
    "StandardTensor": "list[bool | int | float | str | list[...]]",
    "StandardDefault": "bool | int | float | str | list[bool | int | float | str | list[...]]",
    "NonDefault[T]": "T",
    "Anything": "typing.Any",
}

autodoc_typehints = "both"
navigation_with_keys = False
napoleon_use_param = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# html_css_files = ["css/rtd_dark.css"]
