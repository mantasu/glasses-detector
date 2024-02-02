# Glasses Detector

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg?color=purple)](https://colab.research.google.com/github/mantasu/glasses-detector/blob/main/notebooks/demo.ipynb)
[![Docs](https://github.com/mantasu/glasses-detector/actions/workflows/sphinx.yaml/badge.svg)](https://mantasu.github.io/glasses-detector/)
[![PyPI](https://img.shields.io/pypi/v/glasses-detector?color=orange)](https://pypi.org/project/glasses-detector/)
[![Python](https://img.shields.io/badge/python-3.12%20|%203.13-yellow)](https://docs.python.org/3/)
[![CUDA: yes](https://img.shields.io/badge/cuda-yes-green)](https://developer.nvidia.com/cuda-toolkit)
[![DOI](https://raw.githubusercontent.com/mantasu/glasses-detector/main/docs/_static/svg/doi.svg)](https://zenodo.org/badge/latestdoi/610509640)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

<div style="background-color: #882211; border-radius: 0.25em; padding: 0.25em 0.5em;"><b>Warning!</b> The package is currently in development and all the information here is only relevant for the future release <code>v1.0.0</code>. Details and usage of the current active release <code>v0.1.1</code> is available on the <a href="https://mantasu.github.io/glasses-detector/">documentation page</a>.</div>

> **Note**: The main branch is in development - do not clone it yet, instead, clone the [latest release](https://github.com/mantasu/glasses-detector/releases/tag/v0.1.1) for now.

## About

Package for processing images with different types of glasses and their parts. It provides a quick way to use the pre-trained models for **3** kinds of tasks, each being divided to multiple categories, for instance, *classification of sunglasses* or *segmentation of glasses frames*:

* **Classification**: transparent, sunglasses, any glasses
* **Detection**: solo, worn, eye-area
* **Segmentation**: frames, lenses, legs, shadows, full, hybrid

> **Note**: refer to [Glasses Detector Features](https://mantasu.github.io/glasses-detector/docs/features.html) for visual examples.

## Installation

Minimum version of [Python 3.12](https://www.python.org/downloads/release/python-3120/) is required. Also, you may want to install [Pytorch](https://pytorch.org/get-started/locally/) (select *Nightly*) in advance to enable GPU support for your device.

### Pip Package

If you only need the library with pre-trained models, just install the [pip package](https://pypi.org/project/glasses-detector/) and see _Quick Start_ for usage (also check [Glasses Detector Installation](https://mantasu.github.io/glasses-detector/docs/features.html) for more details):

```bash
pip install glasses-detector
```

You can also install it from source:

```bash
git clone https://github.com/mantasu/glasses-detector
cd glasses-detector && pip install .
```

### Local Project

If you want to train your own models on the given datasets (or on some other datasets), just clone the project and install training requirements, then see _Running_ section to see how to run training and testing.

```bash
git clone https://github.com/mantasu/glasses-detector
cd glasses-detector && pip install -r requirements.txt
```

You can create a virtual environment for your packages via [venv](https://docs.python.org/3/library/venv.html), however, if you have conda, then you can simply use it to create a new environment, for example:

```bash
conda create -n glasses-detector python=3.12
conda activate glasses-detector 
```

To set-up the datasets, refer to _Data_ section.

## Quick Start

### Command Line

You can run predictions via the command line. For example, classification of a single image and segmentation of images inside a directory can be performed by running:

```bash
glasses-detector -i path/to/img.jpg -t classification -d cuda -f int # Prints 1 or 0
glasses-detector -i path/to/img_dir -t segmentation -f mask -e .jpg  # Generates masks
```

> **Note**: you can also specify things like `--output-path`, `--size`, `--batch-size` etc. Check the [Glasses Detector CLI](https://mantasu.github.io/glasses-detector/docs/cli.html) and [Command Line Examples](https://mantasu.github.io/glasses-detector/docs/examples.html#command-line) for more details.

### Python Script

You can import the package and its models via the python script for more flexibility. Here is an example of how to classify people wearing sunglasses:

```python
from glasses_detector import GlassesClassifier

# Generates a CSV with each line "<path/to/dir/img.jpg>,<bool>"
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

> **Note**: again, there are a lot more things that can be specified, for instance, `output_size` and `pbar`. It is also possible to directly output the results or save them in a variable. See [Glasses Detector API](https://mantasu.github.io/glasses-detector/docs/api.html) and [Python Script Examples](https://mantasu.github.io/glasses-detector/docs/examples.html#python-script) for more details.

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

> **Tip**: you can also specify only certain tasks, e.g., `--tasks classification segmentation` would ignore detection datasets. It is also possible to change image size and val/test split fractions: use `--help` to see all the available CLI options.

After processing all the datasets, your `data` directory should have the following structure:

```
└── data                    <- The data directory (root) under project
    ├── classification
    |   ├── anyglasses      <- Datasets with any glasses as positives
    │   ├── eyeglasses      <- Datasets with transparent glasses as positives
    |   └── sunglasses      <- Datasets with semi-transparent/opaque glasses as positives 
    │
    ├── detection
    |   ├── eyes            <- Datasets with bounding boxes for eye area 
    |   ├── standalone      <- Datasets with bounding boxes for standalone glasses
    |   └── worn            <- Datasets with bounding boxes for worn glasses
    |
    └── segmentation
        ├── frames          <- Datasets with masks for glasses frames
        ├── full            <- Datasets with masks for full glasses (frames + lenses)
        ├── legs            <- Datasets with masks for glasses legs (part of frames)
        ├── lenses          <- Datasets with masks for glasses lenses
        ├── shadows         <- Datasets with masks for eyeglasses frames cast shadows
        └── smart           <- Datasets with masks for glasses frames and lenses if opaque
```

Almost every dataset will have `train`, `val` and `test` sub-directories. These splits for _classification_ datasets are further divided to `<category>` and `no_<category>`, for _detection_ - to `images` and `annotations`, and for _segmentation_ - to `images` and `masks` sub-sub-directories. By default, all the images are `256x256`.

> **Note**: instead of downloading the datasets manually one-by-one, here is a [Kaggle Dataset](https://www.kaggle.com/datasets/mantasu/glasses-detector) that you could download which already contains everything.

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

**Detection** datasets:

10. From [AI Pass](https://universe.roboflow.com/shinysky5166/ai-pass) download `AI-Pass.v6i.coco.zip` (choose `v6` and `COCO` format)
11. From [PEX5](https://universe.roboflow.com/pex-5-ylpua/pex5-gxq3t) download `PEX5.v4i.coco.zip` (choose `v4` and `COCO` format)
12. From [Sunglasses Glasses Detect](https://universe.roboflow.com/burhan-6fhqx/sunglasses_glasses_detect) download `sunglasses_glasses_detect.v1i.coco.zip` (choose `v1` and `COCO` format)
13. From [Glasses Detection](https://universe.roboflow.com/su-yee/glasses-detection-qotpz) download `Glasses Detection.v2i.coco.zip` (choose `v2` and `COCO` format)
14. From [Glasses Image Dataset](https://universe.roboflow.com/new-workspace-ld3vn/glasses-ffgqb) download `glasses.v1-glasses_2022-04-01-8-12pm.coco.zip` (choose `v1` and `COCO` format)
15. From [EX07](https://universe.roboflow.com/cam-vrmlm/ex07-o8d6m) download `Ex07.v1i.coco.zip` (choose `v1` and `COCO` format)
16. From [No Eyeglass](https://universe.roboflow.com/doms/no-eyeglass) download `no eyeglass.v3i.coco.zip` (choose `v3` and `COCO` format)
17. From [Kacamata-Membaca](https://universe.roboflow.com/uas-kelas-machine-learning-blended/kacamata-membaca) download `Kacamata-Membaca.v1i.coco.zip` (choose `v1` and `COCO` format)
18. From [Only Glasses](https://universe.roboflow.com/woodin-ixal8/onlyglasses) download `onlyglasses.v1i.coco.zip` (choose `v1` and `COCO` format)

**Segmentation** datasets:

19. From [CelebA Mask HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) download `CelebAMask-HQ.zip` and from [CelebA Annotations](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) download `annotations.zip`
20. From [Glasses Segmentation Synthetic Dataset](https://www.kaggle.com/datasets/mantasu/glasses-segmentation-synthetic-dataset) download `archive.zip` and _rename_ to `glasses-segmentation-synthetic.zip`
21. From [Face Synthetics Glasses](https://www.kaggle.com/datasets/mantasu/face-synthetics-glasses) download `archive.zip` and _rename_ to `face-synthetics-glasses.zip`
22. From [Eyeglass](https://universe.roboflow.com/azaduni/eyeglass-6wu5y) download `eyeglass.v10i.coco-segmentation.zip` (choose `v10` and `COCO Segmentation` format)
23. From [Glasses Lenses Segmentation](https://universe.roboflow.com/yair-etkes-iy1bq/glasses-lenses-segmentation) download `glasses lenses segmentation.v7-sh-improvments-version.coco.zip` (choose `v7` and `COCO` format)
24. From [Glasses Lens](https://universe.roboflow.com/yair-etkes-iy1bq/glasses-lens) download `glasses lens.v6i.coco-segmentation.zip` (choose `v6` and `COCO Segmentation` format)
25. From [Glasses Segmentation Cropped Faces](https://universe.roboflow.com/yair-etkes-iy1bq/glasses-segmentation-cropped-faces) download `glasses segmentation cropped faces.v2-segmentation_models_pytorch-s_1st_version.coco-segmentation.zip` (choose `v2` and `COCO Segmentation` format)
26. From [Spects Segmentation](https://universe.roboflow.com/teamai-wuk2z/spects-segementation) download `Spects Segementation.v3i.coco-segmentation.zip` (choose `v3` and `COCO Segmentation`)
27. From [KINH](https://universe.roboflow.com/fpt-university-1tkhk/kinh) download `kinh.v1i.coco.zip` (choose `v1` and `COCO` format)
28. From [Capstone Mini 2](https://universe.roboflow.com/christ-university-ey6ms/capstone_mini_2-vtxs3) download `CAPSTONE_MINI_2.v1i.coco-segmentation.zip` (choose `v1` and `COCO Segmentation` format)
29. From [Sunglasses Color Detection](https://universe.roboflow.com/andrea-giuseppe-parial/sunglasses-color-detection-roboflow) download `Sunglasses Color detection roboflow.v2i.coco-segmentation.zip` (choose `v2` and `COCO Segmentation` format)
30. From [Sunglasses Color Detection 2](https://universe.roboflow.com/andrea-giuseppe-parial/sunglasses-color-detection-2) download `Sunglasses Color detection 2.v3i.coco-segmentation.zip` (choose `v3` and `COCO Segmentation` format)
31. From [Glass Color](https://universe.roboflow.com/snap-ml/glass-color) download `Glass-Color.v1i.coco-segmentation.zip` (choose `v1` and `COCO Segmentation` format)

The table below shows which datasets are used for which tasks and their categories. Feel free to pick only the ones that interest you.

| Task           | Category     | Dataset IDs                                                |
| -------------- | ------------ | ---------------------------------------------------------- |
| Classification | `anyglasses` | `1`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `13`, `14`, `15`   |
| Classification | `eyeglasses` | `2`, `4`, `5`, `6`, `10`, `11`, `12`, `13`, `14`           |
| Classification | `sunglasses` | `1`, `2`, `3`, `4`, `5`, `6`, `10`, `11`, `12`, `13`, `14` |
| Detection      | `eyes`       | `13`, `14`, `15`, `16`                                     |
| Detection      | `standalone` | `17`, `18`                                                 |
| Detection      | `worn`       | `10`, `11`, `12`, `13`, `14`, `15`                         |
| Segmentation   | `frames`     | `20`, `22`                                                 |
| Segmentation   | `full`       | `19`, `26`, `27`                                           |
| Segmentation   | `legs`       | `28`, `29`, `30`                                           |
| Segmentation   | `lenses`     | `22`, `23`, `24`, `25`, `29`, `30`, `31`                   |
| Segmentation   | `shadows`    | `20`                                                       |
| Segmentation   | `smart`      | `21`                                                       |

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

If you get _UserWarning: No positive samples in targets, true positive value should be meaningless_, increase the batch size.

## Credits

For references and citation, please see [Glasses Detector Credits](https://mantasu.github.io/glasses-detector/docs/credits.html)
