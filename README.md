# Glasses Detector

**Eyeglasses** and **sunglasses** _classifier_ + **glasses** and their **frames** _segmenter_. This project provides scripts to download the corresponding datasets, train the corresponding models and by itself is a PyPi project that provides a quick way to use the trained models via python script or terminal.

> **Note**: the project is BETA stage. Currently, only full-glasses base segmentation models are provided.

## Installation

Minimum version of [Python 3.10](https://www.python.org/downloads/release/python-3100/) is required. Also, you may want to install [Pytorch](https://pytorch.org/get-started/locally/) in advance for your device to enable GPU support. Note that _CUDA_ is backwards compatible, thus even if you have the newest version of [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), _Pytorch_ should work just fine.

### Pip Package

If you only need the interface, just install the pip package and see _Examples_ section or read the [documentation page](mantasu.github.io/glasses-segmenter/) of how to use it:

```bash
pip install glasses-detector
```

You can also install it from source:

```bash
git clone https://github.com/mantasu/glasses-segmenter
cd glasses-segmenter && pip install .
```

### Local Project

If you want to train your own models on the given datasets (or on some other datasets), just clone the project and install training requirements, then see _Running_ section to see how to run training and testing.

```bash
git clone https://github.com/mantasu/glasses-segmenter
cd glasses-segmenter && pip install -r requirements.txt
```

You can create a virtual environment for your packages via [venv](https://docs.python.org/3/library/venv.html), however, if you have conda, then you can simply use it to create a new environment, for example:

```bash
conda create -n glasses-detector python=3.11
conda activate glasses-detector 
```

To set-up the datasets for the 4 tasks (2 classification and 2 segmentation tasks), refer to _Data_ section.

## Features

There are 3 categories of classifiers and segmenters (terminology is a bit off but easier to handle with unique names):
* **Eyeglasses** - identifies and segments only transparent glasses, i.e., prescription spectacles.
* **Sunglasses** - identifies and segments only occluded glasses, i.e., sunglasses.
* **Glasses** - identifies and segments all types of glasses.

There are the 3 available model groups (1 classification and 2 segmentation groups).

### Classification

A classifier only identifies whether a person is wearing a corresponding category of glasses:
  
| Models / Input images | \<image with eyeglasses\> | \<image with sunglasses\> | \<image without glasses\> |
| --------------------- | ------------------------- | ------------------------- | ------------------------- |
| Eyeglasses classifier | yes                       | no                        | no                        |
| Sunglasses classifier | no                        | yes                       | no                        |
| Glasses classifier    | yes                       | yes                       | no                        |

These are the performances of _eyeglasses_ and _sunglasses_ model performances and their sizes. Note that the joint _glasses_ classifier would have an average accuracy and a combined model size of both _eyeglasses_ and _sunglasses_ models.

| Model type / Test metric     | BCE loss $\downarrow$ | F1 score $\uparrow$ | ROC-AUC score $\uparrow$ | Num params $\downarrow$ | Model size $\downarrow$ |
| ---------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Eyeglasses classifier tiny   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier small  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier medium | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Eyeglasses classifier large  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

| Model type / Test metric     | BCE loss $\downarrow$ | F1 score $\uparrow$ | ROC-AUC score $\uparrow$ | Num params $\downarrow$ | Model size $\downarrow$ |
| ---------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Sunglasses classifier tiny   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Sunglasses classifier small  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Sunglasses classifier medium | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Sunglasses classifier large  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

### Full-glasses segmentation

A segmenter generates masks of people wearing corresponding categories of glasses and their frames:

| Models / Input images | \<image with eyeglasses\> | \<image with sunglasses\> | \<image without glasses\> |
| --------------------- | ------------------------- | ------------------------- | ------------------------- |
| Eyeglasses segmenter  | \<mask with eyeglasses\>  | \<black image\>           | \<black image\>           |
| Sunglasses segmenter  | \<black image\>           | \<mask with sunglasses\>  | \<black image\>           |
| Glasses segmenter     | \<mask with eyeglasses\>  | \<mask with sunglasses\>  | \<black image\>           |

There is only one glasses _segmentation_ model group which is trained for both _eyeglasses_ and _sunglasses_. Although you can use it as is, it is only one part of the final _segmentation_ model - the other part is a specific _classifier_, therefore, the accuracy and the model size would be a combination of the generic (base) _segmenter_ and a _classifier_ of a specific glasses category.

| Model type / Test metric     | BCE loss $\downarrow$ | F1 score $\uparrow$ | Dice score $\uparrow$    | Num params $\downarrow$ | Model size $\downarrow$ |
| ---------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Base segmenter tiny          | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Base segmenter small         | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Base segmenter medium        | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Base segmenter large         | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

### Frames-only segmentation

A frames segmenter generates masks of only the glasses frames for the corresponding categories of glasses that people wear:

| Models / Input images        | \<image with eyeglasses\>        | \<image with sunglasses\>        | \<image without glasses\> |
| ---------------------------- | -------------------------------- | -------------------------------- | ------------------------- |
| Eyeglasses frames segmenter  | \<mask with eyeglasses frames\>  | \<black image\>                  | \<black image\>           |
| Sunglasses frames segmenter  | \<black image\>                  | \<mask with sunglasses frames\>  | \<black image\>           |
| Glasses frames segmenter     | \<mask with eyeglasses frames\>  | \<mask with sunglasses frames\>  | \<black image\>           |

Similarly to the _glasses & frames segmentation_ model group, there is only one _frames-only segmentation_ model group which is trained for both _eyeglasses_ and _sunglasses_. Although you can use it as is, it is only one part of the final _frames segmentation_ model - the other part is a specific _classifier_, therefore, the accuracy and the model size would be a combination of the generic (base) _frames segmenter_ and a _classifier_ of a specific glasses category.

| Model type / Test metric     | BCE loss $\downarrow$ | F1 score $\uparrow$ | Dice score $\uparrow$    | Num params $\downarrow$ | Model size $\downarrow$ |
| ---------------------------- | --------------------- | ------------------- | ------------------------ | ----------------------- | ----------------------- |
| Base frames segmenter tiny   | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Base frames segmenter small  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Base frames segmenter medium | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |
| Base frames segmenter large  | TBA                   | TBA                 | TBA                      | TBA                     | TBA                     |

## Examples

### Command Line

The installed pip package can be can run via the command line. For example, classification of a single or multiple images, can be performed via

```bash
glasses-detector -i path/to/img --task glasses-classification # Prints True/False
glasses-detector -i path/to/dir --task glasses-classification # Generates CSV
```

Running segmentation is similar, just change the task argument:

```bash
glasses-detector -i path/to/img -t glasses-segmentation # Generates img_mask file
glasses-detector -i path/to/dir -t glasses-segmentation # Generates dir with masks
```

> **Note**: you can also specify things like `--output-path`, `--label-type`, `--model`, `--device` etc. Use `--glasses-detector -h` for more details or check the [documentation page]().

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

Using a segmenter is similar, here is an example of using a frames segmentation model for only transparent eyeglasses:

```python
from glasses_detector import EyeglassesSegmenter

segmenter = EyeglassesSegmenter(model_type="small", pretrained=True).eval()

segmenter.predict(
    input_path="path/to/dir",
    output_path="path/to/dir_masks",
    mask_type="img",
)
```

> **Note**: there is much more flexibility that you can do with the given models, for instance, you can use only base segmenters without accompanying classifiers, or you can define your own prediction methods without resizing images to `256x256` (as what is done in the background). For more details refer to the [documentation page](), for instance at how segmenter [prediction method]() works.

## Data

Before downloading the datasets, please install `unrar` package, for example if you're using Ubuntu (if you're using Windows, just install WinRAR):

```bash
sudo apt-get install unrar
```

Also, ensure the scripts are executable:

```bash
chmod +x scripts/*
```

Once you download a specific dataset (instructions given below), run `preprocess.py` script and specify the corresponding task name and the directory with the dataset(-s) for that task. Note that after running the script, the original raw data will be extracted and deleted, thus make backups if needed. E.g., for sunglasses classification, run:

```bash
python scripts/preprocess.py --task sunglasses-classification --root data/classification/sunglasses
```

After processing every dataset group, your `data` directory should have the following structure:

```
└── data                    <- The data directory under project
    ├── classification
    │   ├── eyeglasses      <- Contains sub-folders with eyeglasses images
    |   └── sunglasses      <- Contains sub-folders with sunglasses images
    │
    └── segmentation
        ├── full-glasses    <- Contains sub-folders with full-glasses images/masks
        └── glass-frames    <- Contains sub-folders with glass-frames images/masks
```


### Classification

<details>

<summary><b>Eyeglasses</b></summary>

Coming soon!

</details>

<details>

<summary><b>Sunglasses</b></summary>

Download the following files and _place them all under directory_ `data/classification/sunglasses` (please note for some datasets you need to have created a free [Kaggle](https://www.kaggle.com/) account):

* From [CMU Face Images](http://archive.ics.uci.edu/dataset/124/cmu+face+images) download `cmu+face+images.zip`
* From [Face Attributes Grouped](https://www.kaggle.com/datasets/mantasu/face-attributes-grouped) download `archive.zip` and _rename_ to `face-attributes-grouped.zip`
* From [Face Attributes Extra](https://www.kaggle.com/datasets/mantasu/face-attributes-extra) download `archive.zip` and _rename_ to `face-attributes-extra.zip`
* From [Glasses and Coverings](https://www.kaggle.com/datasets/mantasu/glasses-and-coverings) download `archive.zip` and _rename_ to `glasses-and-coverings.zip`
* From [Specs on Faces](https://sites.google.com/view/sof-dataset) download `whole images.rar` and `metadata.rar`
* From [Sunglasses / No Sunglasses](https://www.kaggle.com/datasets/amol07/sunglasses-no-sunglasses) download `archive.zip` and _rename_ to `sunglasses-no-sunglasses.zip`

After downloading all the datasests and putting them under the specified directory, run the script to extract the data and create splits:

```bash
python scripts/preprocess.py --task sunglasses-classification --root data/classification/sunglasses
```

After running `preprocess.py`, the following subdirectories should be created inside root:

```
└── data/classification/sunglasses
    ├── cmu-face-images 
    |   ├── test
    |   |   ├── no_sunglasses       <- 256x256 images of people without sunglasses
    │   |   └── sunglasses          <- 256x256 images of people with sunglasses
    │   |
    |   ├── train
    │   |   ├── no_sunglasses       <- 256x256 images of people without sunglasses
    │   |   └── sunglasses          <- 256x256 images of people with sunglasses
    │   |
    |   └── val
    │       ├── no_sunglasses       <- 256x256 images of people without sunglasses
    │       └── sunglasses          <- 256x256 images of people with sunglasses
    |
    ├── face-attributes-grouped     <- Same directory tree as for cmu-face-images
    ├── glasses-and-coverings       <- Same directory tree as for cmu-face-images
    ├── specs-on-faces              <- Same directory tree as for cmu-face-images
    └── sunglasses-no-sunglasses    <- Same directory tree as for cmu-face-images
```

</details>

### Segmentation

<details>

<summary><b>Full Glasses</b></summary>

Download the following files and _place them all under directory_ `data/segmentation/full-glasses`:

* From [CelebA Mask HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) download `CelebAMask-HQ.zip`
* From [CelebA Annotations](https://drive.google.com/file/d/1xd-d1WRnbt3yJnwh5ORGZI3g-YS-fKM9/view) download `annotations.zip`

After downloading the files and putting them under the specified directory, run the script to extract the data and create splits:

```bash
python scripts/preprocess.py --task full-glasses-segmentation --root data/segmentation/full-glasses
```

After running `preprocess.py`, the following subdirectories should be created inside root:

```
└── data/segmentation/full-glasses
    └── celeba-mask-hq
        ├── test
        |   ├── images              <- 256x256 images of people with glasses
        |   └── masks               <- 256x256 images of corresponding masks
        |
        ├── train
        |   ├── images              <- 256x256 images of people with glasses
        |   └── masks               <- 256x256 images of corresponding masks
        |
        └── val
            ├── images              <- 256x256 images of people with glasses
            └── masks               <- 256x256 images of corresponding masks
```

</details>

<details>

<summary><b>Glass Frames</b></summary>

Coming soon!

</details>

## Running



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
  doi = {TBA}
}
```
