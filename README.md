# Sunglasses or Not?

|                                      |                                      |                                      |                                      |
| :----------------------------------: | :----------------------------------: | :----------------------------------: | :----------------------------------: |
| ![Example 1](data/demo/example1.jpg) | ![Example 2](data/demo/example2.jpg) | ![Example 3](data/demo/example3.jpg) | ![Example 4](data/demo/example4.jpg) |
| Wears sunglasses                     | No sunglasses                        | No sunglasses                        | Wears sunglasses                     |

## About

A small side project for building a classifier to detect if a person is wearing sunglasses. 4 datasets were used to train [ShuffleNet V2](https://arxiv.org/abs/1807.11164) model (only 1.4M parameters) for binary classification. The final test results:

<center>

| Test metric | Joint Dataset |
| ----------- | ------------- |
| F1 Score    | 0.9658        |
| BCE Loss    | 0.1022        |

</center>

## Setup

The code was built and tested using [Python 3.10.9](https://www.python.org/downloads/release/python-3109/) It is recommended to setup [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:
```bash
conda create -n sunglasses-or-not python=3.10
conda activate sunglasses-or-not
```

The environment uses [Pytorch 1.13](https://pytorch.org/blog/PyTorch-1.13-release/) with [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive). Please, also install the required packages (may take some time):
```bash
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Datasets

The instructions are provided for Linux users. Please enable executive privilages for python scripts. Also, you may want to install the unzipping packages, e.g., for Ubuntu:
```bash
chmod +x ./scripts/*.py
sudo apt-get install unrar unzip
```

Once all the datasets are downloaded and preprocessed, the data structure should look as follows:
```
├── data                            <- The data directory under project
│   ├── cmu-face-images
│   │   └── test
│   │   |   └── no_sunglasses       <- 256x256 images of poeple without sunglasses
│   │   |   └── sunglasses          <- 256x256 images of poeple with sunglasses
│   │   |
|   |   └── train
│   │   |   └── no_sunglasses       <- 256x256 images of poeple without sunglasses
│   │   |   └── sunglasses          <- 256x256 images of poeple with sunglasses
│   │   |
|   |   └── val
│   │   |   └── no_sunglasses       <- 256x256 images of poeple without sunglasses
│   │   |   └── sunglasses          <- 256x256 images of poeple with sunglasses
│   │
│   ├── glasses-and-coverings       <- Same directory tree as cmu-face-images
│   ├── specs-on-faces              <- Same directory tree as cmu-face-images
│   └── sunglasses-no-sunglasses    <- Same directory tree as cmu-face-images

```

> **Note**: if it takes very long to preprocess the datasets, feel free to change the super resolution model to a smaller one by specifying it with an additional command line argument `--sr-model`, e.g., `--sr-model ninasr_b0`. For avalable models, see [torchsr](https://pypi.org/project/torchsr/) package.

<details><summary><h3>CMU Face Images</h3></summary>

1. Download the data form the official **[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/cmu+face+images)** website:
    * Download image archive from [here](http://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces.tar.gz) and place under `data/cmu-faces-images/faces.tar.gz`
2. Extract the data:
    ```bash
    tar zxvf data/cmu-face-images/faces.tar.gz -C data/cmu-face-images/
    ```
3. Preprocess the data:
    ```bash
    python scripts/preprocess.py --data-dir data/cmu-face-images --criteria file/sunglasses --filter _2 _4 .anonr .tar --val-size 0.15 --test-size 0.15 --sr-scale 4 --resize 256 256 --seed 0
    ```
4. Clean up:
    ```bash
    rm -rf data/cmu-face-images/faces data/cmu-face-images/faces.tar.gz
    ```

</details>

<details><summary><h3>Glasses and Coverings</h3></summary>

1. Download the data from **[Kaggle](https://www.kaggle.com/datasets/mantasu/glasses-and-coverings?resource=download)** website (you have to create a free account):
    * Download image archive from [here](https://www.kaggle.com/datasets/mantasu/glasses-and-coverings) and place under `data/glasses-and-sunglasses/archive.zip`
2. Extract the data:
    ```bash
    unzip data/glasses-and-coverings/archive.zip -d data/glasses-and-coverings
    ```
3. Preprocess the data:
    ```bash
    python scripts/preprocess.py --data-dir data/glasses-and-coverings --criteria dir/sunglasses --filter .zip --val-size 0.15 --test-size 0.15 --seed 0
    ```
4. Clean up:
    ```bash
    rm -rf data/glasses-and-coverings/glasses-and-coverings data/glasses-and-coverings/archive.zip
    ```

</details>

<details><summary><h3>Specs on Faces</h3></summary>

1. Download the data form the official **[Specs on Faces (SoF) Dataset](https://sites.google.com/view/sof-dataset)** website:
    * Download image archive from [here](https://drive.google.com/file/d/14ZEyWfinmlOw0kaHXIBXXlzkMY0Ds87Z/view) and place under `data/specs-on-faces/whole images.rar`
    * Download metadata from [here](https://drive.google.com/file/d/0BwO0RMrZJCioaTVURnZoZG5jUVE/view?resourcekey=0-F8-ejyF8NX4GC129ustqLg) and place under `data/specs-on-faces/metadata.rar`
2. Extract the data:
    ```bash
    unrar x data/specs-on-faces/whole\ images.rar data/specs-on-faces
    unrar x data/specs-on-faces/metadata.rar data/specs-on-faces
    ```
3. Preprocess the data:
    ```bash
    python scripts/preprocess.py --data-dir data/specs-on-faces --criteria data/specs-on-faces/metadata/metadata.mat --filter .mat .rar _.jpg _Gn _Gs _Ps _en _em --val-size 0.15 --test-size 0.15 --sr-scale 4 --resize 256 256 --seed 0
    ```
4. Clean up:
    ```bash
    rm -rf data/specs-on-faces/whole\ images data/specs-on-faces/metadata
    rm data/specs-on-faces/whole\ images.rar data/specs-on-faces/metadata.rar
    ```

</details>

<details><summary><h3>Sunglasses / No Sunglasses</h3></summary>

1. Download the data from **[Kaggle](https://www.kaggle.com/datasets/amol07/sunglasses-no-sunglasses?resource=download)** website (you have to create a free account):
    * Download image archive from [here](https://www.kaggle.com/datasets/amol07/sunglasses-no-sunglasses/download?datasetVersionNumber=2) and place under `data/sunglasses-no-sunglasses/archive.zip`
2. Extract the data:
    ```bash
    unzip data/sunglasses-no-sunglasses/archive.zip -d data/sunglasses-no-sunglasses
    ```
3. Preprocess the data:
    ```bash
    python scripts/split.py --data-dir data/sunglasses-no-sunglasses --criteria dir/with_glasses --filter .zip --val-size 0.15 --test-size 0.15 --sr-scale 4 --resize 256 256 --seed 0
    ```
4. Clean up:
    ```bash
    rm -rf data/sunglasses-no-sunglasses/glasses_noGlasses data/sunglasses-no-sunglasses/archive.zip
    ```

</details>

## Running the Code

To train and evaluate the best model checkpoint after training:
```bash
python scripts/train.py -d path/to/data -n <NUM_EPOCHS>
```

To predict whether a person is wearing glasses:
```bash
python scripts/predict.py -i path/to/image -m path/to/my_model.pth
```

> My trained model is available [here](https://drive.google.com/file/d/1_1xq2X9VoezMkW8pgo3Kl-O4rmy_cLc1/view?usp=sharing). If you download it and place it under `checkpoints/sunglasses-classifier-best.pth`, you can try the demo pictures immediatelly:

```bash
python scripts/predict.py -i data/demo/example1.jpg # wears sunglasses [99.99%]
python scripts/predict.py -i data/demo/example2.jpg # no sunglasses [83.72%]
```

## References
1. Mitchell, Tom. "Cmu face images data set." _Data School of Computer Science_. 1999.
2. Afifi, Mahmoud, and Abdelrahman Abdelhamed. "Afif4: Deep gender classification based on adaboost-based fusion of isolated facial features and foggy faces." _Journal of Visual Communication and Image Representation 62_ (2019): 77-86.
3. Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." _Proceedings of the European Conference on Computer Vision (ECCV)._ 2018.
4. Paszke, Adam, et al. "Pytorch: An imperative style, high-performance deep learning library." _Advances in Neural Information Processing Systems 32_ (2019).