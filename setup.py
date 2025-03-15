import setuptools

DESCRIPTION = r"""
<h1 align="center"><img src="https://raw.githubusercontent.com/mantasu/glasses-detector/main/docs/_static/img/logo-light.png" width=27px height=27px> Glasses Detector</h1>

<div align="center">

[![Colab](https://raw.githubusercontent.com/mantasu/glasses-detector/main/docs/_static/svg/colab.svg)](https://colab.research.google.com/github/mantasu/glasses-detector/blob/main/notebooks/demo.ipynb)
[![Docs](https://github.com/mantasu/glasses-detector/actions/workflows/sphinx.yaml/badge.svg)](https://mantasu.github.io/glasses-detector/)
[![PyPI](https://img.shields.io/pypi/v/glasses-detector?color=yellow&logo=data:image/png;base64,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)](https://pypi.org/project/glasses-detector/)
[![Python](https://img.shields.io/badge/python-≥%203.12-blue?logo=data:image/png;base64,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)](https://docs.python.org/3/)
[![CUDA](https://img.shields.io/badge/cuda-yes-5eb304?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![DOI](https://img.shields.io/badge/doi-10.5281/zenodo.8126101-ad55d9?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAACXBIWXMAAAsTAAALEwEAmpwYAAAE9GlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNy4xLWMwMDAgNzkuZWRhMmIzZmFjLCAyMDIxLzExLzE3LTE3OjIzOjE5ICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgMjMuMSAoTWFjaW50b3NoKSIgeG1wOkNyZWF0ZURhdGU9IjIwMjMtMTAtMDRUMTQ6NDQ6MDIrMDI6MDAiIHhtcDpNb2RpZnlEYXRlPSIyMDIzLTEwLTA0VDE0OjU3OjQ2KzAyOjAwIiB4bXA6TWV0YWRhdGFEYXRlPSIyMDIzLTEwLTA0VDE0OjU3OjQ2KzAyOjAwIiBkYzpmb3JtYXQ9ImltYWdlL3BuZyIgcGhvdG9zaG9wOkNvbG9yTW9kZT0iMyIgeG1wTU06SW5zdGFuY2VJRD0ieG1wLmlpZDozZmQ0NmNlYy01YTc3LTRjMjMtYjZiOC1hY2IwNjJiYzliODgiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6M2ZkNDZjZWMtNWE3Ny00YzIzLWI2YjgtYWNiMDYyYmM5Yjg4IiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6M2ZkNDZjZWMtNWE3Ny00YzIzLWI2YjgtYWNiMDYyYmM5Yjg4Ij4gPHhtcE1NOkhpc3Rvcnk+IDxyZGY6U2VxPiA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0iY3JlYXRlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDozZmQ0NmNlYy01YTc3LTRjMjMtYjZiOC1hY2IwNjJiYzliODgiIHN0RXZ0OndoZW49IjIwMjMtMTAtMDRUMTQ6NDQ6MDIrMDI6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCAyMy4xIChNYWNpbnRvc2gpIi8+IDwvcmRmOlNlcT4gPC94bXBNTTpIaXN0b3J5PiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pmx8nAYAAAPbSURBVGiB7ZpfiFVVFMZ/d/6E1oRplhU6hgoaIgpJgSCIEJb5J0MLfDJKJwTfIhQf8iF8iOghEgpBSiGqMUwxfBEV8kUKAxERhlJxGMUc/4Bmztzp82Hfcc7sWfves/fcaze5Hxzu7G+ftfb67lp7n73PnYIkHgY0/dcBVAsNIfWGhpB6Q0NIvaElwE8FxlTBv4A/gAGjrwDMBJ4FJpXal4FLwNn4kST/6lR10S1pdsb/WEnbJPVUsNkm6QkjPvPKNpoldVRZxCB+K42xXlJvhN0tSZsVKaRN0t5qRe6hW9KBUdgfVAUhBQ3fay0Ffo6uzweDA8DKUKcvBOAD4G1gAvYkDeFvoAi8GGFzC/fFXSi124HXgceNewV8CHxqeqqUssjr24hy+Uj2ZB4naWvA5oqkFkkF366aIrZEiFiaw98SSUXPrk/Sx5KerpWQlyNEdET4XWvYX5C0uhZCxkq6mlPExQT/JzwfdyXtkCux+/dVY4vyA/CkwfcbnD1Ry8O3eQSYBzyWJUcrZBOwzOC/Bj4z+OMJY/QY3GRgYpYYjZAFwOcGXwTeAVqNvjsJ41iPgDZgSpZIFdICHAz0rSp9zjD6LiWMZWVkDDDNDygFPwLjDX4XQwKLRv9bQBfQnHOcAWCOwTcDz+B2zII0IRuBFQb/O/Bupm29Z/oyYbwQHs02YktrFrDD4O9SZh9UIzThMnK/EYNDAX4dcDExoBQUyIiAOCE/Ac8b/E7gu+SQ0nAdOJcl8s6R97BL5xywISKAk8AN3EMtFQLO4Krj30Eyj5DJuG/dwpLIIJaRtgRXRJ7SOhbg1+GW0hjf43KMl4RKGdkFTDf43cA3FWytJ3JqSS0COnBLbhH4B/gedygbgPJC1uC2Gj7O47JRCd0GNx84lcM2i3bgqMf1AvvJPKtCpdWO29VaeAX7YefDeje1JYedj0+8tnBL/elhcQTOAF2B88T7EeeIuQEfb0b4eMGwvyZ3SmzL3msZfxEIYG9EAIPXLwFf03PYTpB03rPrl3RU0jxJTeWEvBYY+HiCCCS1Srpt+OuT9EYZu1cl/WXY9UraJC8bkoZN9rXAV4E67Qf2YL+mKYeb2CfFVmAf8Gfp81fc6jMXdwyYbdgIt+ncB9we2TukansgG/WCIxrK8ogsZletwgiV9YNTwPLS31aGhwk5A/TVMJj9uLeYsdgJvIRVTll4KVpeo7K4LOmp0hiLJe2Wm/Dl0ClplV9Coct69zsHWIjL1mj/m6CAy3InbuudxXO4TecU3CJSKH3rPcBh3A9E+QcyhPwv8dD8htgQUm9oCKk3NITUG+4BHP2mwwGKgTwAAAAASUVORK5CYII=)](https://zenodo.org/badge/latestdoi/610509640)
[![License](https://img.shields.io/badge/license-MIT-red?logo=data:image/png;base64,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)](https://opensource.org/licenses/MIT)

![Banner](https://raw.githubusercontent.com/mantasu/glasses-detector/main/docs/_static/img/banner.jpg)

</div>

## About

Package for processing images with different types of glasses and their parts. It provides a quick way to use the pre-trained models for **3** kinds of tasks, each divided into multiple categories, for instance, *classification of sunglasses* or *segmentation of glasses frames*.

<br>

<div align="center">

<table align="center"><tbody>
    <tr><td><strong>Classification</string></td> <td> 👓 <em>transparent</em> 🕶️ <em>opaque</em> 🥽 <em>any</em> ➿<em>shadows</em></td></tr>
    <tr><td><strong>Detection</string></td> <td> 🤓 <em>worn</em> 👓  <em>standalone</em> 👀 <em>eye-area</em></td></tr>
    <tr><td><strong>Segmentation</string></td> <td> 😎 <em>full</em> 🖼️ <em>frames</em> 🦿 <em>legs</em> 🔍 <em>lenses</em> 👥 <em>shadows</em></td></tr>
</tbody></table>

$\color{gray}{\textit{Note: }\text{refer to}}$ [Glasses Detector Features](https://mantasu.github.io/glasses-detector/docs/features.html) $\color{gray}{\text{for visual examples.}}$

</div>

## Installation

> [!IMPORTANT]
> Minimum version of [Python 3.12](https://www.python.org/downloads/release/python-3120/) is **REQUIRED**. Also, you may want to install [Pytorch](https://pytorch.org/get-started/locally/) in advance to select specific configuration for your device and environment.

### Pip Package

If you only need the library with pre-trained models, just install the [pip package](https://pypi.org/project/glasses-detector/) and see **Quick Start** for usage (also check [Glasses Detector Installation](https://mantasu.github.io/glasses-detector/docs/features.html) for more details):

```bash
pip install glasses-detector
```

You can also install it from the source:

```bash
git clone https://github.com/mantasu/glasses-detector
cd glasses-detector && pip install .
```

### Local Project

If you want to train your own models on the given datasets (or on some other datasets), just clone the project and install training requirements, then see **[Running](https://github.com/mantasu/glasses-detector?tab=readme-ov-file#running)** section to see how to run training and testing.

```bash
git clone https://github.com/mantasu/glasses-detector
cd glasses-detector && pip install -r requirements.txt
```

You can create a virtual environment for your packages via [venv](https://docs.python.org/3/library/venv.html), however, if you have conda, then you can simply use it to create a new environment, for example:

```bash
conda create -n glasses-detector python=3.12
conda activate glasses-detector 
```

> To set-up the datasets, refer to **[Data](https://github.com/mantasu/glasses-detector?tab=readme-ov-file#data)** section.

## Quick Start

### Command Line

You can run predictions via the command line. For example, classification of a single image and segmentation of images inside a directory can be performed by running:

```bash
glasses-detector -i path/to/img.jpg -t classification -d cuda -f int # Prints 1 or 0
glasses-detector -i path/to/img_dir -t segmentation -f mask -e .jpg  # Generates masks
```

> [!TIP]
> You can also specify things like `--output-path`, `--size`, `--batch-size` etc. Check the [Glasses Detector CLI](https://mantasu.github.io/glasses-detector/docs/cli.html) and [Command Line Examples](https://mantasu.github.io/glasses-detector/docs/examples.html#command-line) for more details.

### Python Script

You can import the package and its models via the python script for more flexibility. Here is an example of how to classify people wearing sunglasses:

```python
from glasses_detector import GlassesClassifier

# Generates a CSV with each line "<img_name.jpg>,<True|False>"
classifier = GlassesClassifier(size="small", kind="sunglasses")
classifier.process_dir("path/to/dir", "path/to/preds.csv", format="bool")
```

And here is a more efficient way to process a dir for detection task (only single bbox per image is currently supported):

```python
from glasses_detector import GlassesDetector

# Generates dir_preds with bboxes as .txt for each img
detector = GlassesDetector(kind="eyes", device="cuda")
detector.process_dir("path/to/dir", ext=".txt", batch_size=64)
```

> [!TIP]
> Again, there are a lot more things that can be specified, for instance, `output_size` and `pbar`. It is also possible to directly output the results or save them in a variable. See [Glasses Detector API](https://mantasu.github.io/glasses-detector/docs/api.html) and [Python Script Examples](https://mantasu.github.io/glasses-detector/docs/examples.html#python-script) for more details.

### Demo

Feel free to play around with some [demo image files](https://github.com/mantasu/glasses-detector/demo/). For example, after installing through [pip](https://pypi.org/project/glasses-detector/), you can run:

```bash
git clone https://github.com/mantasu/glasses-detector && cd glasses-detector/data
glasses-detector -i demo -o demo_labels.csv --task classification:eyeglasses
```

You can also check out the [demo notebook](https://github.com/mantasu/glasses-detector/notebooks/demo.ipynb) which can be also accessed via [Google Colab](https://colab.research.google.com/github/mantasu/glasses-detector/blob/master/notebooks/demo.ipynb).

## Credits

For references and citation, please see [Glasses Detector Credits](https://mantasu.github.io/glasses-detector/docs/credits.html).
"""

setuptools.setup(
    name="glasses-detector",
    version="1.0.2",
    author="Mantas Birškus",
    author_email="mantix7@gmail.com",
    license="MIT",
    description="Glasses classification, detection, and segmentation.",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/mantasu/glasses-detector",
    project_urls={
        "Documentation": "https://mantasu.github.io/glasses-detector",
        "Bug Tracker": "https://github.com/mantasu/glasses-detector/issues",
    },
    keywords=[
        "python",
        "pytorch",
        "torchvision",
        "computer",
        "vision",
        "image",
        "face",
        "eyes",
        "transparent",
        "opaque",
        "glasses",
        "googles",
        "spectacles",
        "eyeglasses",
        "sunglasses",
        "frames",
        "lenses",
        "legs",
        "shadows",
        "binary",
        "identification",
        "identifier",
        "classification",
        "classifier",
        "segmentation",
        "segmenter",
        "detection",
        "detector",
        "cuda",
        "mps",
        "gpu",
    ],
    install_requires=[
        "tqdm",
        "torch",
        "torchvision",
    ],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Environment :: Console",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Environment :: Other Environment",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "glasses-detector=glasses_detector.__main__:main",
        ]
    },
    python_requires=">=3.12",
)
