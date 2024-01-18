# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

from bs4 import BeautifulSoup

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
}

# TODO: possibly remove when sphinx supports types
autodoc_type_aliases = {
    "FilePath": "glasses_detector.utils.FilePath",
    "Scalar": "glasses_detector.components.pred_type.Scalar",
    "Tensor": "glasses_detector.components.pred_type.Tensor",
    "Default": "glasses_detector.components.pred_type.Default",
    "StandardScalar": "glasses_detector.components.pred_type.StandardScalar",
    "StandardTensor": "glasses_detector.components.pred_type.StandardTensor",
    "StandardDefault": "glasses_detector.components.pred_type.StandardDefault",
    "NonDefault[T]": "glasses_detector.components.pred_type.NonDefault",
    "Either": "glasses_detector.components.pred_type.Either",
}

autodoc_long_signature_ids = [
    "glasses_detector.utils.flatten",
    "glasses_detector.components.pred_interface.PredInterface.process_file",
    "glasses_detector.components.pred_interface.PredInterface.process_dir",
    "glasses_detector.components.base_model.BaseGlassesModel",
    "glasses_detector.components.base_model.BaseGlassesModel.predict",
    "glasses_detector.components.base_model.BaseGlassesModel.process_dir",
    "glasses_detector.components.base_model.BaseGlassesModel.process_file",
]

autodoc_typehints = "both"
autodoc_member_order = "bysource"
autosummary_generate = False
napoleon_use_param = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/mantasu/glasses-detector",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# -- Custom Template Functions -----------------------------------------------
# https://www.sphinx-doc.org/en/master/development/theming.html#defining-custom-template-functions


def keep_only_data(soup: BeautifulSoup):
    data_dict, class_dict = {}, {}

    for alias, id in autodoc_type_aliases.items():
        for dl in soup.find_all("dl"):
            if dl["class"] != ["py", "data"] and dl["class"] != ["py", "class"]:
                continue

            # Get the prename and name elements of the signature
            prename = dl.find("span", class_="sig-prename descclassname")
            name = dl.find("span", class_="sig-name descname")

            if prename is None or name is None:
                continue

            prename = prename.find("span", class_="pre")
            name = name.find("span", class_="pre")

            if prename is None or name is None:
                continue

            if prename.string != ".".join(id.split(".")[:-1] + [""]):
                continue

            if name.string != id.split(".")[-1]:
                continue

            if dl["class"] == ["py", "data"]:
                data_dict[alias] = dl
                continue

            if dl["class"] == ["py", "class"]:
                class_dict[alias] = dl
                continue

    for alias, id in autodoc_type_aliases.items():
        if alias not in data_dict or alias not in class_dict:
            continue

        # Get the dt element of the data
        dt = data_dict[alias].find("dt")

        # Add ID to dt
        dt["id"] = id

        # Copy a from class dt to data dt
        dt.append(class_dict[alias].find("a"))

        # Remove class dt
        class_dict[alias].decompose()


def process_in_page_toc(soup: BeautifulSoup):
    for li in soup.find_all("li", class_="toc-h3 nav-item toc-entry"):
        if span := li.find("span"):
            # Modify the toc-nav span element here
            span.string = span.string.split(".")[-1]


def break_long_signatures(soup: BeautifulSoup):
    for id in autodoc_long_signature_ids:
        if not (dt := soup.find("dt", id=id)):
            continue

        for dt in dt.parent.find_all("dt"):
            for sig_param in dt.find_all("em", class_="sig-param"):
                # Add long-sig (for overrides, i.e., sibling dt, too)
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
