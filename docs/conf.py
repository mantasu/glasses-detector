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
    "builtin_constants": (
        "https://docs.python.org/3",
        "_static/inv/builtin_constants.inv",
    ),
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
    "glasses_detector.components.pred_type.PredType",
    "glasses_detector.components.pred_interface.PredInterface.process_file",
    "glasses_detector.components.pred_interface.PredInterface.process_dir",
    "glasses_detector.components.base_model.BaseGlassesModel",
    "glasses_detector.components.base_model.BaseGlassesModel.predict",
    "glasses_detector.components.base_model.BaseGlassesModel.process_dir",
    "glasses_detector.components.base_model.BaseGlassesModel.process_file",
    "glasses_detector.classifier.GlassesClassifier",
    "glasses_detector.classifier.GlassesClassifier.predict",
    "glasses_detector.classifier.GlassesClassifier.process_dir",
    "glasses_detector.classifier.GlassesClassifier.process_file",
    "glasses_detector.detector.GlassesDetector",
    "glasses_detector.detector.GlassesDetector.draw_rects",
    "glasses_detector.detector.GlassesDetector.predict",
    "glasses_detector.detector.GlassesDetector.process_dir",
    "glasses_detector.detector.GlassesDetector.process_file",
    "glasses_detector.segmenter.GlassesSegmenter",
    "glasses_detector.segmenter.GlassesSegmenter.draw_mask",
    "glasses_detector.segmenter.GlassesSegmenter.predict",
    "glasses_detector.segmenter.GlassesSegmenter.process_dir",
    "glasses_detector.segmenter.GlassesSegmenter.process_file",
    "glasses_detector.architectures.tiny_binary_detector.TinyBinaryDetector.forward",
    "glasses_detector.architectures.tiny_binary_detector.TinyBinaryDetector.compute_loss",
]

LONG_PARAMETER_IDS = {
    "glasses_detector.classifier.GlassesClassifier.predict": ["format"],
    "glasses_detector.detector.GlassesDetector.predict": ["format"],
    "glasses_detector.segmenter.GlassesSegmenter.predict": ["format"],
}

CUSTOM_PYTHON_SYNTAX_COLORS = {
    "GlassesClassifier": "dark-python-class",
    "GlassesDetector": "dark-python-class",
    "GlassesSegmenter": "dark-python-class",
    "process_file": "dark-python-function",
    "process_dir": "dark-python-function",
    "run": "dark-python-function",
    "load": "dark-python-function",
    "type": "dark-python-class",
    "format": "dark-python-variable",
    "np": "dark-python-class",
    "subprocess": "dark-python-class",
    "random": "dark-python-class",
    # "rand": "dark-python-variable",
}

CUSTOM_BASH_SYNTAX_COLORS = {
    "dark-bash-keyword": (
        "full",
        ["glasses-detector", "git", "clone", "pip", "install"],
    ),
    "dark-bash-flag": ("start", ["-"]),
    "dark-bash-op": ("full", ["&&"]),
}


def align_rowspans(soup: BeautifulSoup):
    if tds := soup.find_all("td", rowspan=True):
        for td in tds:
            td["valign"] = "middle"


def add_collapse_ids(soup: BeautifulSoup):
    if details := soup.find_all("details"):
        for detail in details:
            if detail.has_attr("name"):
                detail["id"] = "-".join(detail["name"].split())


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
    def break_long_params(id, sig_param):
        if (params := LONG_PARAMETER_IDS.get(id)) is None:
            return

        is_opened = False

        for span in sig_param.find_all("span", class_="pre"):
            if span.string == "[":
                is_opened = True
            elif span.string == "]":
                is_opened = False

            if (
                span.string == "|"
                and not is_opened
                and span.parent.parent.parent.find("span", class_="pre").string
                in params
            ):
                # Add long-sig to spans with |
                span["class"].append("long-sig")

    for id in LONG_SIGNATURE_IDS:
        if not (dt := soup.find("dt", id=id)):
            continue

        for sig_param in dt.find_all("em", class_="sig-param"):
            # Add long-sig to the identified sig-param ems
            sig_param["class"].append("long-sig")
            break_long_params(id, sig_param)

        for dt_sibling in dt.find_next_siblings("dt"):
            for sig_param in dt_sibling.find_all("em", class_="sig-param"):
                # Add long-sig for overrides, i.e., sibling dts, too
                sig_param["class"].append("long-sig")
                break_long_params(id, sig_param)


from bs4 import BeautifulSoup, NavigableString
import re


def customize_code_block_colors(soup):
    # Find all 'span' elements within 'div.highlight-python div.highlight pre'
    spans = soup.select("div.highlight-python div.highlight pre span")

    for span in spans:
        for key, value in CUSTOM_PYTHON_SYNTAX_COLORS.items():
            # If the span's text is 'GlassesClassifier', change its color to green
            if key == span.get_text().strip():
                span["class"].append(value)

    highlight_bash_divs = soup.select("div.highlight-bash div.highlight")

    for div in highlight_bash_divs:
        pre_elements = div.find_all("pre")
        for pre in pre_elements:
            for content in pre.contents:
                if isinstance(content, NavigableString):
                    words = content.split(" ")
                    for i, word in enumerate(words):
                        i_modified = False

                        for key, (rule, value) in CUSTOM_BASH_SYNTAX_COLORS.items():
                            for w in value:
                                detected = False
                                wors = []
                                for wor in word.split("\n"):
                                    if rule == "start" and wor.startswith(w):
                                        wors.append(f'<span class="{key}">{wor}</span>')
                                        detected = True
                                    elif rule == "full" and (
                                        wor == w or wor == "\n" + w
                                    ):
                                        wors.append(f'<span class="{key}">{wor}</span>')
                                        detected = True
                                    else:
                                        wors.append(
                                            f'<span class="dark-bash-value">{wor}</span>'
                                        )

                                if detected:
                                    words[i] = "\n".join(wors)
                                    i_modified = True
                                    break

                        if not i_modified:
                            words[i] = f'<span class="dark-bash-value">{word}</span>'

                    new_content = " ".join(words)
                    content.replace_with(BeautifulSoup(new_content, "html.parser"))
    soup.prettify()


def edit_html(app, exception):
    if app.builder.format != "html":
        return

    for pagename in app.env.found_docs:
        if not isinstance(pagename, str):
            continue

        with (Path(app.outdir) / f"{pagename}.html").open("r") as f:
            # Parse HTML using BeautifulSoup html parser
            soup = BeautifulSoup(f.read(), "html.parser")

            align_rowspans(soup)
            keep_only_data(soup)
            add_collapse_ids(soup)
            process_in_page_toc(soup)
            break_long_signatures(soup)
            customize_code_block_colors(soup)

        with (Path(app.outdir) / f"{pagename}.html").open("w") as f:
            # Write back HTML
            f.write(str(soup))


def setup(app):
    app.connect("build-finished", edit_html)
