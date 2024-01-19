# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Tag

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Glasses Detector"
copyright = "2024, Mantas Birškus"
author = "Mantas Birškus"
release = "v1.0.0"

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
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "tqdm": ("https://tqdm.github.io/docs/", "_static/inv/tqdm.inv"),
    "builtin_constants": (
        "https://docs.python.org/3",
        "_static/inv/builtin_constants.inv",
    ),
    "overloads": ("", "_static/inv/overloads.inv"),
}

# -- Options for napaleon/autosummary/autodoc output -------------------------
napoleon_use_param = True
autosummary_generate = True
autodoc_typehints = "both"
autodoc_member_order = "bysource"

templates_path = ["_templates"]
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
    "github_url": "https://github.com/mantasu/glasses-detector",
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
html_css_files = ["css/custom.css"]
html_title = f"Glasses Detector {release}"
html_favicon = "_static/img/logo-light.png"

# -- Custom Template Functions -----------------------------------------------
# https://www.sphinx-doc.org/en/master/development/theming.html#defining-custom-template-functions

TYPE_ALIASES = {
    "FilePath": "glasses_detector.utils.",
    "Scalar": "glasses_detector.components.pred_type.",
    "Tensor": "glasses_detector.components.pred_type.",
    "Default": "glasses_detector.components.pred_type.",
    "StandardScalar": "glasses_detector.components.pred_type.",
    "StandardTensor": "glasses_detector.components.pred_type.",
    "StandardDefault": "glasses_detector.components.pred_type.",
    "NonDefault": "glasses_detector.components.pred_type.",
    "Either": "glasses_detector.components.pred_type.",
}

LONG_SIGNATURE_IDS = [
    "glasses_detector.utils.flatten",
    "glasses_detector.components.pred_type.PredType",
    "glasses_detector.components.pred_interface.PredInterface.process_file",
    "glasses_detector.components.pred_interface.PredInterface.process_dir",
    "glasses_detector.components.base_model.BaseGlassesModel",
    "glasses_detector.components.base_model.BaseGlassesModel.predict",
    "glasses_detector.components.base_model.BaseGlassesModel.process_dir",
    "glasses_detector.components.base_model.BaseGlassesModel.process_file",
]


def keep_only_data(soup: BeautifulSoup):
    def has_children(tag: Tag, txt1: str, txt2: str):
        if tag.name != "dt":
            return False

        # Get the prename and name elements of the signature
        ch1 = tag.select_one("span.sig-prename.descclassname span.pre")
        ch2 = tag.select_one("span.sig-name.descname span.pre")

        return ch1 and ch2 and ch1.string == txt1 and ch2.string == txt2

    for alias, module in TYPE_ALIASES.items():
        if dt := soup.find("dt", id=f"{module}{alias}"):
            # Copy class directive's a
            a = dt.find("a").__copy__()
            dt.parent.decompose()
        else:
            continue

        if dt := soup.find(lambda tag: has_children(tag, module, alias)):
            # ID and a for data directive
            dt["id"] = f"{module}{alias}"
            dt.append(a)
            dt.find("span", class_="sig-prename descclassname").decompose()


def process_in_page_toc(soup: BeautifulSoup):
    for li in soup.find_all("li", class_="toc-h3 nav-item toc-entry"):
        if span := li.find("span"):
            # Modify the toc-nav span element here
            span.string = span.string.split(".")[-1]


def break_long_signatures(soup: BeautifulSoup):
    for id in LONG_SIGNATURE_IDS:
        if not (dt := soup.find("dt", id=id)):
            continue

        for sig_param in dt.find_all("em", class_="sig-param"):
            # Add long-sig to the identified sig-param ems
            sig_param["class"].append("long-sig")

        for dt_sibling in dt.find_next_siblings("dt"):
            for sig_param in dt_sibling.find_all("em", class_="sig-param"):
                # Add long-sig for overrides, i.e., sibling dts, too
                sig_param["class"].append("long-sig")


def edit_html(app, exception):
    if app.builder.format != "html":
        return

    for pagename in app.env.found_docs:
        if not isinstance(pagename, str):
            continue

        with (Path(app.outdir) / f"{pagename}.html").open("r") as f:
            # Parse HTML using BeautifulSoup html parser
            soup = BeautifulSoup(f.read(), "html.parser")

            keep_only_data(soup)
            process_in_page_toc(soup)
            break_long_signatures(soup)

        with (Path(app.outdir) / f"{pagename}.html").open("w") as f:
            # Write back HTML
            f.write(str(soup))


def setup(app):
    app.connect("build-finished", edit_html)
