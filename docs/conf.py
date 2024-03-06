# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

from sphinx.application import Sphinx

sys.path += [
    str(Path(__file__).parent.parent / "src"),
    str(Path(__file__).parent),
]

from helpers import BuildFinished, CustomInvs

DOCS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Glasses Detector"
copyright = "2024, Mantas Birškus"
author = "Mantas Birškus"
release = "v1.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "tqdm": ("https://tqdm.github.io/docs/", "_static/inv/tqdm.inv"),
}

# -- Options for napaleon/autosummary/autodoc output -------------------------
napoleon_use_param = True
autosummary_generate = True
autodoc_typehints = "both"
autodoc_member_order = "bysource"

templates_path = ["_templates"]
bibtex_bibfiles = ["_static/bib/references.bib"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "alt_text": "Glasses Detector - Home",
        "text": f"Glasses Detector {release}",
        "image_light": "_static/img/logo-light.png",
        "image_dark": "_static/img/logo-dark.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mantasu/glasses-detector",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Colab",
            "url": "https://colab.research.google.com/github/mantasu/glasses-detector/blob/main/notebooks/demo.ipynb",
            "icon": "fa-custom fa-colab",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/glasses-detector",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "Zenodo",
            "url": "https://zenodo.org/doi/10.5281/zenodo.8126101",
            "icon": "fa-custom fa-zenodo",
        },
    ],
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 7,
}
html_context = {
    "github_user": "mantasu",
    "github_repo": "glasses-detector",
    "github_version": "main",
    "doc_path": "docs",
}
html_static_path = ["_static"]
html_js_files = ["js/colab-icon.js", "js/pypi-icon.js", "js/zenodo-icon.js"]
html_css_files = ["css/highlights.css", "css/signatures.css", "css/custom.css"]
html_title = f"Glasses Detector {release}"
html_favicon = "_static/img/logo-light.png"

# -- Custom Template Functions -----------------------------------------------
# https://www.sphinx-doc.org/en/master/development/theming.html#defining-custom-template-functions


def setup(app: Sphinx):
    # Add local inventories to intersphinx_mapping
    custom_invs = CustomInvs(static_path=DOCS_DIR / "_static")
    app.config.intersphinx_mapping.update(custom_invs())

    # Add custom build-finished event
    build_finished = BuildFinished(DOCS_DIR / "_static", DOCS_DIR / "conf.yaml")
    app.connect("build-finished", build_finished)
