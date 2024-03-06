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
> Minimum version of [Python 3.12](https://www.python.org/downloads/release/python-3120/) is required. Also, you may want to install [Pytorch](https://pytorch.org/get-started/locally/) (select *Nightly* for compatibility) in advance to select specific configuration for your device and environment.

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

## Data

Before downloading the datasets, please install `unrar` package, for example if you're using Ubuntu (if you're using Windows, just install [WinRAR](https://www.win-rar.com/start.html?&L=0)):

```bash
sudo apt-get install unrar
```

Also, ensure the scripts are executable:

```bash
chmod +x scripts/*
```

Once you download all the datasets (or some that interest you), process them:

```bash
python scripts/preprocess.py --root data -f -d
```

> [!TIP]
> You can also specify only certain tasks, e.g., `--tasks classification segmentation` would ignore detection datasets. It is also possible to change image size and val/test split fractions: use `--help` to see all the available CLI options.

After processing all the datasets, your `data` directory should have the following structure:

```bash
└── data                    # The data directory (root) under project
    ├── classification
    │   ├── anyglasses      # Datasets with any glasses as positives
    │   ├── eyeglasses      # Datasets with transparent glasses as positives
    │   ├── shadows         # Datasets with visible glasses frames shadows as positives
    │   └── sunglasses      # Datasets with semi-transparent/opaque glasses as positives 
    │
    ├── detection
    │   ├── eyes            # Datasets with bounding boxes for eye area 
    │   ├── solo            # Datasets with bounding boxes for standalone glasses
    │   └── worn            # Datasets with bounding boxes for worn glasses
    │
    └── segmentation
        ├── frames          # Datasets with masks for glasses frames
        ├── full            # Datasets with masks for full glasses (frames + lenses)
        ├── legs            # Datasets with masks for glasses legs (part of frames)
        ├── lenses          # Datasets with masks for glasses lenses
        ├── shadows         # Datasets with masks for eyeglasses frames cast shadows
        └── smart           # Datasets with masks for glasses frames and lenses if opaque
```

Almost every dataset will have `train`, `val` and `test` sub-directories. These splits for _classification_ datasets are further divided to `<category>` and `no_<category>`, for _detection_ - to `images` and `annotations`, and for _segmentation_ - to `images` and `masks` sub-sub-directories. By default, all the images are `256x256`.

> [!NOTE]
> Instead of downloading the datasets manually one-by-one, here is a [Kaggle Dataset](https://www.kaggle.com/datasets/mantasu/glasses-detector) that you could download which already contains everything.

<details>

<summary><b>Download Instructions</b></summary>

Download the following files and _place them all_ inside the cloned project under directory `data` which will be your data `--root` (please note for some datasets you need to have created a free [Kaggle](https://www.kaggle.com/) account):

**Classification** datasets:

1. From [CMU Face Images](http://archive.ics.uci.edu/dataset/124/cmu+face+images) download `cmu+face+images.zip`
2. From [Specs on Faces](https://sites.google.com/view/sof-dataset) download `original images.rar` and `metadata.rar`
3. From [Sunglasses / No Sunglasses](https://www.kaggle.com/datasets/amol07/sunglasses-no-sunglasses) download `archive.zip` and _rename_ to `sunglasses-no-sunglasses.zip`
4. From [Glasses and Coverings](https://www.kaggle.com/datasets/mantasu/glasses-and-coverings) download `archive.zip` and _rename_ to `glasses-and-coverings.zip`
5. From [Face Attributes Grouped](https://www.kaggle.com/datasets/mantasu/face-attributes-grouped) download `archive.zip` and _rename_ to `face-attributes-grouped.zip`
6. From [Face Attributes Extra](https://www.kaggle.com/datasets/mantasu/face-attributes-extra) download `archive.zip` and _rename_ to `face-attributes-extra.zip`
7. From [Glasses No Glasses](https://www.kaggle.com/datasets/jorgebuenoperez/datacleaningglassesnoglasses) download `archive.zip` and _rename_ to `glasses-no-glasses.zip`
8. From [Indian Facial Database](https://drive.google.com/file/d/1DPQQ2omEYPJDLFP3YG2h1SeXbh2ePpOq/view) download `An Indian facial database highlighting the Spectacle.zip`
9. From [Face Attribute 2](https://universe.roboflow.com/heheteam-g9fnm/faceattribute-2) download `FaceAttribute 2.v2i.multiclass.zip` (choose `v2` and `Multi Label Classification` format)
10. From [Glasses Shadows Synthetic](https://www.kaggle.com/datasets/mantasu/glasses-shadows-synthetic) download `archive.zip` and _rename_ to `glasses-shadows-synthetic.zip`

**Detection** datasets:

11. From [AI Pass](https://universe.roboflow.com/shinysky5166/ai-pass) download `AI-Pass.v6i.coco.zip` (choose `v6` and `COCO` format)
12. From [PEX5](https://universe.roboflow.com/pex-5-ylpua/pex5-gxq3t) download `PEX5.v4i.coco.zip` (choose `v4` and `COCO` format)
13. From [Sunglasses Glasses Detect](https://universe.roboflow.com/burhan-6fhqx/sunglasses_glasses_detect) download `sunglasses_glasses_detect.v1i.coco.zip` (choose `v1` and `COCO` format)
14. From [Glasses Detection](https://universe.roboflow.com/su-yee/glasses-detection-qotpz) download `Glasses Detection.v2i.coco.zip` (choose `v2` and `COCO` format)
15. From [Glasses Image Dataset](https://universe.roboflow.com/new-workspace-ld3vn/glasses-ffgqb) download `glasses.v1-glasses_2022-04-01-8-12pm.coco.zip` (choose `v1` and `COCO` format)
16. From [EX07](https://universe.roboflow.com/cam-vrmlm/ex07-o8d6m) download `Ex07.v1i.coco.zip` (choose `v1` and `COCO` format)
17. From [No Eyeglass](https://universe.roboflow.com/doms/no-eyeglass) download `no eyeglass.v3i.coco.zip` (choose `v3` and `COCO` format)
18. From [Kacamata-Membaca](https://universe.roboflow.com/uas-kelas-machine-learning-blended/kacamata-membaca) download `Kacamata-Membaca.v1i.coco.zip` (choose `v1` and `COCO` format)
19. From [Only Glasses](https://universe.roboflow.com/woodin-ixal8/onlyglasses) download `onlyglasses.v1i.coco.zip` (choose `v1` and `COCO` format)

**Segmentation** datasets:

20. From [CelebA Mask HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) download `CelebAMask-HQ.zip` and from [CelebA Annotations](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) download `annotations.zip`
21. From [Glasses Segmentation Synthetic Dataset](https://www.kaggle.com/datasets/mantasu/glasses-segmentation-synthetic-dataset) download `archive.zip` and _rename_ to `glasses-segmentation-synthetic.zip`
22. From [Face Synthetics Glasses](https://www.kaggle.com/datasets/mantasu/face-synthetics-glasses) download `archive.zip` and _rename_ to `face-synthetics-glasses.zip`
23. From [Eyeglass](https://universe.roboflow.com/azaduni/eyeglass-6wu5y) download `eyeglass.v10i.coco-segmentation.zip` (choose `v10` and `COCO Segmentation` format)
24. From [Glasses Lenses Segmentation](https://universe.roboflow.com/yair-etkes-iy1bq/glasses-lenses-segmentation) download `glasses lenses segmentation.v7-sh-improvments-version.coco.zip` (choose `v7` and `COCO` format)
25. From [Glasses Lens](https://universe.roboflow.com/yair-etkes-iy1bq/glasses-lens) download `glasses lens.v6i.coco-segmentation.zip` (choose `v6` and `COCO Segmentation` format)
26. From [Glasses Segmentation Cropped Faces](https://universe.roboflow.com/yair-etkes-iy1bq/glasses-segmentation-cropped-faces) download `glasses segmentation cropped faces.v2-segmentation_models_pytorch-s_1st_version.coco-segmentation.zip` (choose `v2` and `COCO Segmentation` format)
27. From [Spects Segmentation](https://universe.roboflow.com/teamai-wuk2z/spects-segementation) download `Spects Segementation.v3i.coco-segmentation.zip` (choose `v3` and `COCO Segmentation`)
28. From [KINH](https://universe.roboflow.com/fpt-university-1tkhk/kinh) download `kinh.v1i.coco.zip` (choose `v1` and `COCO` format)
29. From [Capstone Mini 2](https://universe.roboflow.com/christ-university-ey6ms/capstone_mini_2-vtxs3) download `CAPSTONE_MINI_2.v1i.coco-segmentation.zip` (choose `v1` and `COCO Segmentation` format)
30. From [Sunglasses Color Detection](https://universe.roboflow.com/andrea-giuseppe-parial/sunglasses-color-detection-roboflow) download `Sunglasses Color detection roboflow.v2i.coco-segmentation.zip` (choose `v2` and `COCO Segmentation` format)
31. From [Sunglasses Color Detection 2](https://universe.roboflow.com/andrea-giuseppe-parial/sunglasses-color-detection-2) download `Sunglasses Color detection 2.v3i.coco-segmentation.zip` (choose `v3` and `COCO Segmentation` format)
32. From [Glass Color](https://universe.roboflow.com/snap-ml/glass-color) download `Glass-Color.v1i.coco-segmentation.zip` (choose `v1` and `COCO Segmentation` format)

The table below shows which datasets are used for which tasks and their categories. Feel free to pick only the ones that interest you.

<div align="center">

| Task           | Category     | Dataset IDs                                                |
| -------------- | ------------ | ---------------------------------------------------------- |
| Classification | `anyglasses` | `1`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `14`, `15`, `16`   |
| Classification | `eyeglasses` | `2`, `4`, `5`, `6`, `11`, `12`, `13`, `14`, `15`           |
| Classification | `sunglasses` | `1`, `2`, `3`, `4`, `5`, `6`, `11`, `12`, `13`, `14`, `15` |
| Classification | `shadows`    | `10`                                                       |
| Detection      | `eyes`       | `14`, `15`, `16`, `17`                                     |
| Detection      | `solo`       | `18`, `19`                                                 |
| Detection      | `worn`       | `11`, `12`, `13`, `14`, `15`, `16`                         |
| Segmentation   | `frames`     | `21`, `23`                                                 |
| Segmentation   | `full`       | `20`, `27`, `28`                                           |
| Segmentation   | `legs`       | `29`, `30`, `31`                                           |
| Segmentation   | `lenses`     | `23`, `24`, `25`, `26`, `30`, `31`, `32`                   |
| Segmentation   | `shadows`    | `21`                                                       |
| Segmentation   | `smart`      | `22`                                                       |

</div>

</details>

## Running

To run custom training and testing, it is first advised to familiarize with how [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) works and briefly check its [CLI documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli). In particular, take into account what arguments are accepted by the [Trainer class](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#trainer) and how to customize your own [optimizer](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html#multiple-optimizers) and [scheduler](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html#multiple-schedulers) via command line. **Prerequisites**:

1. Clone the repository
2. Install the requirements
3. Download and preprocess the data

### Training

You can run simple training as follows (which is the default):
```bash
python scripts/run.py fit --task classification:anyglasses --size medium 
```

You can customize things like `batch-size`, `num-workers`, as well as `trainer` and `checkpoint` arguments:
```bash
python scripts/run.py fit --batch-size 64 --trainer.max_epochs 300 --checkpoint.dirname ckpt
```

It is also possible to overwrite default optimizer and scheduler:
```bash
python scripts/run.py fit --optimizer Adam --optimizer.lr 1e-3 --lr_scheduler CosineAnnealingLR
```

### Testing

To run testing, specify the trained model and the checkpoint to it:
```bash
python scripts/run.py test -t classification:anyglasses -s small --ckpt_path path/to/model.ckpt
```

Or you can also specify the `pth` file to pre-load the model with weights:
```bash
python scripts/run.py test -t classification:anyglasses -s small -w path/to/weights.pth
```

> If you get _UserWarning: No positive samples in targets, true positive value should be meaningless_, increase the batch size.

## Credits

For references and citation, please see [Glasses Detector Credits](https://mantasu.github.io/glasses-detector/docs/credits.html).
