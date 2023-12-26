import setuptools

DESCRIPTION = r"""
# Glasses Detector

[![PyPI](https://img.shields.io/pypi/v/glasses-detector?color=orange)](https://pypi.org/project/glasses-detector/)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11-yellow)](https://docs.python.org/3/)
[![CUDA: yes](https://img.shields.io/badge/cuda-yes-green)](https://developer.nvidia.com/cuda-toolkit)
[![Docs: passing](https://img.shields.io/badge/docs-passing-skyblue)](https://mantasu.github.io/glasses-detector/)
[![DOI](https://zenodo.org/badge/610509640.svg)](https://zenodo.org/badge/latestdoi/610509640)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

**Eyeglasses** and **sunglasses** _classifier_ + **glasses** and their **frames** _segmenter_. This project provides a quick way to use the pre-trained models via python script or terminal. Based on selected task, an image or a directory of images will be processed and corresponding labels or masks will be generated.

> **Note**: the project is BETA stage. Currently, only sunglasses classification models (except _huge_) and full glasses segmentation models (except _medium_) are available.

## Installation

Minimum version of [Python 3.10](https://www.python.org/downloads/release/python-3100/) is required. Also, you may want to install [Pytorch](https://pytorch.org/get-started/locally/) in advance for your device to enable GPU support. Note that _CUDA_ is backwards compatible, thus even if you have the newest version of [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), _Pytorch_ should work just fine.

To install the package, simply run:

```bash
pip install glasses-detector
```

You can also install it from source:

```bash
git clone https://github.com/mantasu/glasses-detector
cd glasses-detector && pip install .
```

If you want to train your own models on the same datasets (or your custom ones), check the [GitHub repository](https://github.com/mantasu/glasses-detector).

## Features

There are 2 kinds of classifiers and 2 kinds of segmenters (terminology is a bit off but easier to handle with unique names):
* **Eyeglasses classifier** - identifies only transparent glasses, i.e., prescription spectacles.
* **Sunglasses classifier** - identifies only occluded glasses, i.e., sunglasses.
* **Full glasses segmenter** - segments full glasses, i.e., their frames and actual glasses (regardless of the glasses type).
* **Glasses frames segmenter** - segments glasses frames (regardless of the glasses type).

Each kind has 5 different model architectures with naming conventions set from *tiny* to *huge*.

### Classification

A classifier only identifies whether a corresponding category of glasses (transparent eyeglasses or occluded sunglasses)  is present:

| Model type             | ![eyeglasses](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/eyeglasses.jpg) | ![sunglasses](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/sunglasses.jpg) | ![no_glasses](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses.jpg) |
| ---------------------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| Eyeglasses classifier  | wears                                | doesn't wear                         | doesn't wear                         |
| Sunglasses classifier  | doesn't wear                         | wears                                | doesn't wear                         |
| Any glasses classifier | wears                                | wears                                | doesn't wear                         |

These are the performances of _eyeglasses_ and _sunglasses_ models and their sizes. Note that the joint _glasses_ classifier would have an average accuracy and a combined model size of both _eyeglasses_ and _sunglasses_ models.

<details>

<summary><b>Eyeglasses classification models (performance & weights)</b></summary>

| Model type                   | BCE loss $\downarrow$ | F1 score $\uparrow$ | ROC-AUC score $\uparrow$ | Num params $\downarrow$ | Model size $\downarrow$ |
| ---------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Eyeglasses classifier tiny   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier small  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier medium | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier large  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier huge   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

</details>

<details>

<summary><b>Sunglasses classification models (performance & weights)</b></summary>

| Model type                   | BCE loss $\downarrow$ | F1 score $\uparrow$ | ROC-AUC score $\uparrow$ | Num params $\downarrow$ | Model size $\downarrow$ |
| ---------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Sunglasses classifier tiny   | 0.1149                | 0.9137              | 0.9967                   | **27.53 k**             | **0.11 Mb**             |
| Sunglasses classifier small  | 0.0645                | 0.9434              | 0.9987                   | 342.82 k                | 1.34 Mb                 |
| Sunglasses classifier medium | **0.0491**            | 0.9651              | **0.9992**               | 1.52 M                  | 5.84 Mb                 |
| Sunglasses classifier large  | 0.0532                | **0.9685**          | 0.9990                   | 4.0 M                   | 15.45 Mb                |
| Sunglasses classifier huge   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

</details>

### Segmentation

A full-glasses segmenter generates masks of people wearing corresponding categories of glasses and their frames, whereas frames-only segmenter generates corresponding masks but only for glasses frames:

| Model type                        | ![eyeglasses](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/eyeglasses.jpg) | ![sunglasses](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/sunglasses.jpg) | ![no_glasses](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses.jpg) |
| --------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| Full/frames eyeglasses segmenter  | ![full/frames eyeglasses mask](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/eyeglasses_mask.jpg) | ![black image](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses_mask.jpg)                  | ![black image](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses_mask.jpg) |
| Full/frames sunglasses segmenter  | ![black image](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses_mask.jpg)                 | ![full/frames sunglasses mask](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/sunglasses_mask.jpg)  | ![black image](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses_mask.jpg) |
| Full/frames any glasses segmenter | ![full/frames eyeglasses mask](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/eyeglasses_mask.jpg) | ![full/frames sunglasses mask](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/sunglasses_mask.jpg)  | ![black image](https://raw.githubusercontent.com/mantasu/glasses-detector/main/assets/no_glasses_mask.jpg) |

There is only one model group for each _full-glasses_ and _frames-only_ _segmentation_ tasks. Each group is trained for both _eyeglasses_ and _sunglasses_. Although you can use it as is, it is only one part of the final _full-glasses_ or _frames-only_ _segmentation_ model - the other part is a specific _classifier_, therefore, the accuracy and the model size would be a combination of the generic (base) _segmenter_ and a _classifier_ of a specific glasses category.

<details>

<summary><b>Full glasses segmentation models (performance & weights)</b></summary>

| Model type                    | BCE loss $\downarrow$ | F1 score $\uparrow$ | Dice score $\uparrow$    | Num params $\downarrow$ | Model size $\downarrow$ |
| ----------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Full glasses segmenter tiny   | 0.0580                | 0.9054              | 0.9220                   | **926.07 k**            | **3.54 Mb**             |
| Full glasses segmenter small  | 0.0603                | 0.8990              | 0.9131                   | 3.22 M                  | 12.37 Mb                |
| Full glasses segmenter medium | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Full glasses segmenter large  | **0.0515**            | **0.9152**          | **0.9279**               | 32.95 M                 | 125.89 Mb               |
| Full glasses segmenter huge   | 0.0516                | 0.9147              | 0.9272                   | 58.63 M                 | 224.06 Mb               |

</details>

<details>

<summary><b>Glasses frames segmentation models (performance & weights)</b></summary>

| Model type                      | BCE loss $\downarrow$ | F1 score $\uparrow$ | Dice score $\uparrow$    | Num params $\downarrow$ | Model size $\downarrow$ |
| ------------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Glasses frames segmenter tiny   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Glasses frames segmenter small  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Glasses frames segmenter medium | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Glasses frames segmenter large  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Glasses frames segmenter huge   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

</details>

## Examples

### Command Line

You can run predictions via the command line. For example, classification of a single or multiple images, can be performed via

```bash
glasses-detector -i path/to/img --kind sunglasses-classifier # Prints 1 or 0
glasses-detector -i path/to/dir --kind sunglasses-classifier # Generates CSV
```

Running segmentation is similar, just change the `kind` argument:

```bash
glasses-detector -i path/to/img -k glasses-segmenter # Generates img_mask file
glasses-detector -i path/to/dir -k glasses-segmenter # Generates dir with masks
```

> **Note**: you can also specify things like `--output-path`, `--label-type`, `--size`, `--device` etc. Use `--glasses-detector -h` for more details or check the [documentation page](https://mantasu.github.io/glasses-detector/modules/).

### Python Script

You can import the package and its models via the python script for more flexibility. Here is an example of how to classify people wearing sunglasses (will generate an output file where each line will contain the name of the image and the predicted label, e.g., `some_image.jpg,1`):

```python
from glasses_detector import SunglassesClassifier

classifier = SunglassesClassifier(model_type="small", pretrained=True).eval()

classifier.predict(
    input_path="path/to/dir", 
    output_path="path/to/output.csv",
    label_type="int",
)
```

Using a segmenter is similar, here is an example of using a sunglasses segmentation model:

```python
from glasses_detector import FullSunglassesSegmenter

# model_type can also be a tuple: (classifier size, base glasses segmenter size)
segmenter = FullSunglassesSegmenter(model_type="small", pretrained=True).eval()

segmenter.predict(
    input_path="path/to/dir",
    output_path="path/to/dir_masks",
    mask_type="img",
)
```

> **Note**: there is much more flexibility that you can do with the given models, for instance, you can use only base segmenters without accompanying classifiers, or you can define your own prediction methods without resizing images to `256x256` (as what is done in the background). For more details refer to the [documentation page](https://mantasu.github.io/glasses-detector/modules/), for instance at how segmenter [prediction method](https://mantasu.github.io/glasses-detector/modules/glasses_detector.bases.base_segmenter.html#glasses_detector.bases.base_segmenter.BaseSegmenter.predict) works.

### Demo

Feel free to play around with some [demo image files](https://github.com/mantasu/glasses-detector/demo/). For example, after installing through [pip](https://pypi.org/project/glasses-detector/), you can run:

```bash
git clone https://github.com/mantasu/glasses-detector && cd glasses-detector/data
glasses-detector -i demo -o demo_labels.csv --kind sunglasses-classifier --label str
```

## References

The following model architectures were used from [Torchvision](https://pytorch.org/vision/stable/index.html) library:
* **Classifier small** - [ShuffleNet V2 (x0.5)](https://pytorch.org/vision/stable/models/generated/torchvision.models.shufflenet_v2_x0_5.html#torchvision.models.shufflenet_v2_x0_5) based on [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) paper
* **Classifier medium** - [MobileNet V3 (small)](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small) based on [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper
* **Classifier large** - [EfficientNet B0](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0) based on [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) paper
* **Segmenter small** - [LRASPP](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.lraspp_mobilenet_v3_large) based on [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) paper
* **Segmenter medium** - [FCN (ResNet-50)](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet50.html#torchvision.models.segmentation.fcn_resnet50) based on [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) paper
* **Segmenter large** - [DeepLab V3 (ResNet-101)](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101) based on [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) paper

**Tiny classifiers** and **tiny segmenters** are the custom models created by me with the aim to have as few parameters as possible while still maintaining a reasonable accuracy.

## Citation

```bibtex
@misc{glasses-detector,
  author = {Mantas Birškus},
  title = {Glasses Detector},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mantasu/glasses-detector}},
  doi = {10.5281/zenodo.8126101}
}
```
"""

setuptools.setup(
    name = "glasses-detector",
    version = "0.1.1",
    author = "Mantas Birškus",
    author_email = "mantix7@gmail.com",
    license = "MIT",
    description = f"Eyeglasses and sunglasses detector (classifier and segmenter)",
    long_description = DESCRIPTION,
    long_description_content_type = "text/markdown",
    url = "https://github.com/mantasu/glasses-detector",
    project_urls = {
        "Documentation": "https://mantasu.github.io/glasses-detector",
        "Bug Tracker": "https://github.com/mantasu/glasses-detector/issues",
    },
    keywords = [
        "face",
        "python",
        "pytorch",
        "glasses",
        "frames",
        "eyeglasses",
        "sunglasses",
        "binary",
        "classification",
        "classifier",
        "segmentation",
        "segmenter",
        "detection",
        "detector",
    ],
    install_requires = [
        "tqdm",
        "torch",
        "torchvision",
        "albumentations",
    ],
    classifiers = [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "glasses-detector=glasses_detector.__main__:main",
        ]
    },
    python_requires = ">=3.10"
)
