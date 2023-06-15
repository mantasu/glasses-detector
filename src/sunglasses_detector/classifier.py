import os
import numpy
import torch
import torch.nn as nn
import PIL.Image as Image

from typing import Any
from ._data import ImageLoaderMixin
from torchvision.models import (
    shufflenet_v2_x0_5,
    mobilenet_v3_small,
    efficientnet_b0,
)

class SunglassesClassifier(nn.Module, ImageLoaderMixin):
    """Classifier to determine if the person is wearing sunglasses.

    A binary classifier that tells whether a person in the image is
    wearing sunglasses. It has 4 different base models ranging from 
    tiny to large - check the table below for more details:

    .. list-table:: Classification backbone model properties
        :header-rows: 1

        * - Backbone name
          - Num parameters ↓
          - Model size ↓
          - BCE loss ↓
          - F1 score ↑
          - ROC-AUC score ↑
        * - TinySungNet (tiny)
          - **27.5 K**
          - **0.110 Mb**
          - 0.1878
          - 0.8461
          - 0.9930
        * - ShuffleNet (small)
          - 342 K
          - TODO
          - TODO
          - TODO
          - TODO
        * - MobileNet (medium)
          - 1.5 M
          - TODO
          - TODO
          - TODO
          - TODO
        * - EfficientNet (large)
          - 4.0 M
          - TODO
          - TODO
          - TODO
          - TODO

    Args:
        base_model (str): The name of the base model to use for 
            classification. These are the available options:

            * "tinysungnet" or "tiny" - The smallest model that is 
                uniquely part of this package. For more information, see
                :class:`.TinySunglassesClassifier`.
            * "shufflenet" or "small" - ShuffleNet V2 model taken from 
                torchvision package. For more information, see 
                :func:`~torchvision.models.shufflenet_v2_x0_5`.
            * "mobilenet" or "medium" - MobileNet V3 model taken from 
                torchvision package. For more information, see 
                :func:`~torchvision.models.mobilenet_v3_small`.
            * "efficientnet" or "large" - EfficientNet B0 model taken 
                from torchvision package. For more information, see 
                :func:`~torchvision.models.efficientnet_b0`.

            Defaults to "mobilenet".
    """
    def __init__(self, base_model: str = "mobilenet"):
        super().__init__()
        self.base_model = self._load_base_model(base_model)

    def _load_base_model(self, model: str):
        match model:
            case "tinysungnet" | "tiny":
                m = TinySunglassesClassifier()
            case "shufflenet" | "small":
                m = shufflenet_v2_x0_5()
                m.fc = nn.Linear(m.fc.in_features, 1)
            case "mobilenet" | "medium":
                m = mobilenet_v3_small()
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
            case "efficientnet" | "large":
                m = efficientnet_b0()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
            case _:
                raise NotImplementedError(f"{model} is not a valid choice!")

        return m
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Predicts raw scores for the given batch of inputs. Scores are 
        unbounded - anything that's less than 0 means sunglasses are 
        unlikely and anything that's above 0 indicates that sunglasses 
        are likely.

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note 
                that pixel values are normalized and squeezed between 
                0 and 1.

        Returns:
            torch.Tensor: An output tensor of shape (N,) indicating 
                whether a person in each nth image is wearing sunglasses 
                or not. The scores are unbounded, thus, to convert to a 
                probability, sigmoid function must be used.
        """
        return self.base_model(x)
    
    @torch.no_grad()
    def predict(self, image: str | Image.Image | numpy.ndarray) -> bool:
        """Predicts whether sunglasses are present in the given image.

        Takes an image or a path to the image and outputs a boolean 
        value indicating whether the person in the image is wearing 
        sunglasses or not.

        Args:
            image (str | Image.Image | numpy.ndarray): The path to the 
                image to generate the prediction for or the image itself
                represented as :class:`Image.Image` or as a 
                :class:`numpy.ndarray`. Note that the image should have 
                values between 0 and 255 and be of RGB format. 
                Normalization is not needed as the channels will be 
                automatically normalized before passing through the 
                network.

        Returns:
            bool: ``True`` if the person in the image is likely to wear 
                sunglasses and ``False`` otherwise.
        """
        # Loads the image properly and predict
        x = self.load_image(image)[None, ...]
        is_sunglasses = (self(x) > 0)[0]

        return is_sunglasses
    
    @torch.no_grad()
    def process_dir(
        self,
        input_dir: str,
        output_file: str | None = None,
        label_map: dict[bool, Any] = {True: 1, False: 0},
        sep=','
    ):
        """Generates a prediction for each image in the directory.

        Goes though all images in the directory and generates a 
        prediction of whether the person in the image is wearing 
        sunglasses or not. Each prediction is then written to an output 
        file line by line, i.e., Each line is of the form::

            <image_name.ext><separator><prediction>
        
        For example::

            my_image.jpg,1

        .. warning::
            Please ensure the directory contains valid images and only 
            image files, otherwise errors may occur.

        Args:
            input_dir (str): The path to the input directory with image 
                files.
            output_file (str | None, optional): The output file path to 
                which to write the predictions. If not specified, the 
                same directory where the ``input_dir`` folder is located 
                will be used and the file in that directory will have 
                the same name as ``input_dir`` basename, just with a 
                suffix of *_sunglasses_info.csv*. Defaults to None.
            label_map (dict[bool, Any], optional): The dictionary 
                mapping the predictions to the values to write in the 
                output file. The dictionary must contain 2 entries: 
                ``True`` (indicating sunglasses are present) and 
                ``False`` (no sunglasses). The values for those entries 
                will be used as predictions that will be written next to 
                image file names. Defaults to {True: 1, False: 0}.
            sep (str, optional): The separator to use to separate image 
                file names and the predictions. Defaults to ','.
        """
        if output_path is None:
            # Create a CSV output file by default
            output_path = input_dir + "_sunglasses_info.csv"
        
        with open(output_file, 'w') as f:
            for file in os.scandir(input_dir):
                # Predict and write the prediction
                is_sunglasses = self.predict(file.path)
                f.write(f"{file.name}{sep}{label_map[is_sunglasses]}\n")

class TinySunglassesClassifier(nn.Module):
    """Tiny classifier to predict whether sunglasses are present.

    This is a custom classifier created with the aim to contain very few
    parameters while maintaining a reasonable accuracy. It only has 
    several sequential convolutional and polling blocks (with 
    batch-norm in between).
    """
    def __init__(self):
        super().__init__()

        # Several convolutional blocks
        self.features = nn.Sequential(
            self._create_block(3, 5, 3),
            self._create_block(5, 10, 3),
            self._create_block(10, 15, 3),
            self._create_block(15, 20, 3),
            self._create_block(20, 25, 3),
            self._create_block(25, 80, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Fully connected layer
        self.fc = nn.Linear(80, 1)

    def _create_block(self, num_in, num_out, filter_size):
        return nn.Sequential(
            nn.Conv2d(num_in, num_out, filter_size, 1, "valid", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_out),
            nn.MaxPool2d(2, 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Predicts raw scores for the given batch of inputs. Scores are 
        unbounded, anything that's less than 0 means, sunglasses are 
        unlikely and anything that's above 0 indicates that sunglasses 
        are likely.

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note 
                that pixel values are normalized and squeezed between 
                0 and 1.

        Returns:
            torch.Tensor: An output tensor of shape (N,) indicating 
                whether a person in each nth image is wearing sunglasses 
                or not. The scores are unbounded, thus, to convert to a 
                probability, sigmoid function must be used.
        """
        return self.fc(self.features(x))