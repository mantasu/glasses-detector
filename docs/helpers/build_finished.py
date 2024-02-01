import os
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, NavigableString, Tag
from sphinx.application import Sphinx


class BuildFinished:
    def __init__(self, static_path: str = "_static", conf_path: str = "conf.yaml"):
        # Init inv directory and create it if not exists
        self.inv_dir = os.path.join(static_path, "inv")
        os.makedirs(self.inv_dir, exist_ok=True)

        with open(conf_path) as f:
            # Load conf.yaml and get build_finished section
            self.conf = yaml.safe_load(f)["build-finished"]

    def align_rowspans(self, soup: BeautifulSoup):
        if tds := soup.find_all("td", rowspan=True):
            for td in tds:
                td["valign"] = "middle"

    def add_collapse_ids(self, soup: BeautifulSoup):
        if details := soup.find_all("details"):
            for detail in details:
                if detail.has_attr("name"):
                    detail["id"] = "-".join(detail["name"].split())

    def keep_only_data(self, soup: BeautifulSoup):
        def has_children(tag: Tag, txt1: str, txt2: str):
            if tag.name != "dt":
                return False

            # Get the prename and name elements of the signature
            ch1 = tag.select_one("span.sig-prename.descclassname span.pre")
            ch2 = tag.select_one("span.sig-name.descname span.pre")

            return ch1 and ch2 and ch1.string == txt1 and ch2.string == txt2

        for alias, module in self.conf["TYPE_ALIASES"].items():
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

    def process_in_page_toc(self, soup: BeautifulSoup):
        for li in soup.find_all("li", class_="toc-h3 nav-item toc-entry"):
            if span := li.find("span"):
                # Modify the toc-nav span element here
                span.string = span.string.split(".")[-1]

    def break_long_signatures(self, soup: BeautifulSoup):
        def break_long_params(id, sig_param):
            if (params := self.conf["LONG_PARAMETER_IDS"].get(id)) is None:
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

        for id in self.conf["LONG_SIGNATURE_IDS"]:
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

    def customize_code_block_colors_python(self, soup: BeautifulSoup):
        for span in soup.select("div.highlight-python div.highlight pre span"):
            for name, keyword in self.conf["CUSTOM_SYNTAX_COLORS_PYTHON"].items():
                if span.get_text().strip() in keyword:
                    # Add class of the syntax keyword
                    span["class"].append(name)

    def customize_code_block_colors_bash(self, soup: BeautifulSoup):
        # Select content groups
        pres = soup.select("div.highlight-bash div.highlight pre")
        pres.extend(soup.select("code.highlight-bash"))

        # Define the constants
        KEEP_CLS = {"c1", "w"}
        OP_CLS = "custom-highlight-op"
        START_CLS = "custom-highlight-start"
        DEFAULT_CLS = "custom-highlight-default"

        # Get the starts and flatten the keywords
        starts = self.conf["CUSTOM_SYNTAX_COLORS_BASH"][START_CLS]
        ops = self.conf["CUSTOM_SYNTAX_COLORS_BASH"][OP_CLS] + ["\n"]
        flat_kwds = [
            (cls, kwd)
            for cls, kwds in self.conf["CUSTOM_SYNTAX_COLORS_BASH"].items()
            for kwd in kwds
            if cls not in [START_CLS, DEFAULT_CLS, OP_CLS]
        ]

        for pre in pres:
            for content in pre.contents:
                if (
                    isinstance(content, Tag)
                    and "class" in content.attrs.keys()
                    and not any(cls in content["class"] for cls in KEEP_CLS)
                ):
                    # Only keep the text part, i.e., remove <span></span>
                    content.replace_with(NavigableString(content.get_text()))
                elif isinstance(content, NavigableString) and "\n" in content:
                    # Init the splits
                    sub_contents = []

                    for sub_content in content.split("\n"):
                        if sub_content != "":
                            # No need to add borderline empty strings
                            sub_contents.append(NavigableString(sub_content))

                        # Also add the newline character as NS
                        sub_contents.append(NavigableString("\n"))

                    # Replace the original content with splits
                    content.replace_with(*sub_contents[:-1])

        for pre in pres:
            # Init the starts
            start_idx = 0

            for content in pre.contents:
                if not isinstance(content, NavigableString):
                    # Skip non-navigable strings
                    continue

                if content in ops:
                    # Reset start
                    start_idx = 0

                    # If keyword is an operator, wrap with OP_CLS
                    new_content = f'<span class="{OP_CLS}">{content}</span>'
                    content.replace_with(BeautifulSoup(new_content, "html.parser"))
                    continue

                # Get the start keyword if it exists
                start = [
                    sub_start
                    for start in starts
                    for sub_start_idx, sub_start in enumerate(start.split())
                    if start_idx == sub_start_idx and content == sub_start
                ]

                # Increment start idx
                start_idx += 1

                if len(start) > 0:
                    # If keyword is a start
                    new_content = f'<span class="{START_CLS}">{start[0]}</span>'
                    content.replace_with(BeautifulSoup(new_content, "html.parser"))
                    continue

                # Check if any of the keywords from config matches
                is_kwd = [content.startswith(kwd) for _, kwd in flat_kwds]

                if any(is_kwd):
                    # Add the corresponding keyword class
                    cls, _ = flat_kwds[is_kwd.index(True)]
                    new_content = f'<span class="{cls}">{content}</span>'
                else:
                    # Add the default class if no keyword is found
                    new_content = f'<span class="{DEFAULT_CLS}">{content}</span>'

                # Replace the original content with the new one
                content.replace_with(BeautifulSoup(new_content, "html.parser"))

        # Prettify soup
        soup.prettify()

    def edit_html(self, app: Sphinx):
        if app.builder.format != "html":
            return

        for pagename in app.env.found_docs:
            if not isinstance(pagename, str):
                continue

            with (Path(app.outdir) / f"{pagename}.html").open("r") as f:
                # Parse HTML using BeautifulSoup html parser
                soup = BeautifulSoup(f.read(), "html.parser")

                self.align_rowspans(soup)
                self.keep_only_data(soup)
                self.add_collapse_ids(soup)
                self.process_in_page_toc(soup)
                self.break_long_signatures(soup)
                self.customize_code_block_colors_python(soup)
                self.customize_code_block_colors_bash(soup)

            with (Path(app.outdir) / f"{pagename}.html").open("w") as f:
                # Write back HTML
                f.write(str(soup))

    def __call__(self, app, exception):
        self.edit_html(app)
