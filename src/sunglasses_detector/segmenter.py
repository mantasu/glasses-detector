import os
import numpy
import torch
import torch.nn as nn
import PIL.Image as Image

from ._data import ImageLoaderMixin
from torchvision.ops import Conv2dNormActivation
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)


class GlassesSegmenter(nn.Module, ImageLoaderMixin):
    """Segmenter to mark the pixels of the glasses.

    A binary segmenter that marks the pixels of eyeglasses or sunglasses 
    wore by a person in the image. It has 4 different base models 
    ranging from tiny to large - check the table below for more details:

    .. list-table:: Segmentation backbone model properties
        :header-rows: 1

        * - Backbone name
          - Num parameters
          - Model size
          - BCE loss
          - F1 score
          - Dice score
        * - TinyGlassNet (tiny)
          - 926 K
          - 3.704 (MB)
          - 0.0584
          - 0.9031
          - 0.9201
        * - LR-ASPP (small)
          - 3.2 M
          - TODO
          - TODO
          - TODO
          - TODO
        * - FCN (medium)
          - 35.3 M
          - TODO
          - TODO
          - TODO
          - TODO
        * - DeepLab (large)
          - 61.0 M
          - TODO
          - TODO
          - TODO
          - TODO

    Args:
        base_model (str): The name of the base model to use for 
            classification. These are the available options:

            * "tinysungnet" or "tiny" - The smallest model that is 
                uniquely part of this package. For more information, see
                :class:`.TinyGlassesSegmenter`.
            * "lraspp" or "small" - LR-ASPP model taken from 
                torchvision package. For more information, see 
                :func:`~torchvision.models.segmentation.lraspp_mobilenet_v3_large`.
            * "fcn" or "medium" - FCN model taken from torchvision 
                package. For more information, see 
                :func:`~torchvision.models.segmentation.fcn_resnet50`.
            * "deeplab" or "large" - DeepLab V3 model taken from 
                torchvision package. For more information, see 
                :func:`~torchvision.models.segmentation.deeplabv3_resnet101`.

            Defaults to "fcn".
    """
    def __init__(self, base_model: str = "fcn"):
        super().__init__()
        self.base_model = self._load_base_model(base_model)
    
    def _load_base_model(self, model: str):
        match model:
            case "tinyglassnet" | "tiny":
                m = TinyGlassesSegmenter()
            case "lraspp" | "small":
                m = lraspp_mobilenet_v3_large()
                m.classifier = LRASPPHead(40, 960, 1, 128)
            case "fcn" | "medium":
                m = fcn_resnet50()
                m.classifier[-1] = nn.Conv2d(512, 1, 1)
                m.aux_classifier = None
            case "deeplab" | "large":
                m = deeplabv3_resnet101()
                m.classifier = DeepLabHead(2048, 1)
                m.aux_classifier = None
            case _:
                raise NotImplementedError(f"{model} is not a valid choice!")
        
        return m
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Predicts raw pixel scores for the given batch of inputs. Scores 
        are unbounded - anything that's less than 0 means glasses 
        belonging to the pixel is unlikely and anything that's above 0 
        indicates that glasses for a particular pixel is likely.

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note 
                that pixel values are normalized and squeezed between 
                0 and 1.

        Returns:
            torch.Tensor: An output tensor of shape (N, 1, H, W) 
                indicating which pixels in the image fall under category
                of 'glasses'. The scores are unbounded, thus, to convert 
                to probabilities, sigmoid function must be used.
        """
        return self.base_model(x)["out"]

    @torch.no_grad()
    def predict(self, image: str | Image.Image | numpy.ndarray) -> numpy.ndarray:
        """Predicts which pixels in the image are 'glasses'.

        Takes an image or a path to the image and outputs black and
        white mask as :class:`numpy.ndarray` of type :attr:`numpy.uint8`
        with values of either 255 (white) indicating pixels under 
        category 'glasses' or 0 (black) indicating the rest of the 
        pixels.

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
            numpy.ndarray: Output mask of shape (H, W) of type 
                :attr:`numpy.uint8` with values of 255 representing 
                *'glasses'* pixels and 0 representing *'no glasses'* 
                pixels.
        """
        # Loads the image properly and predict
        x = self.load_image(image)[None, ...]
        print(x)
        mask = ((self(x) > 0)[0, 0] * 255).numpy().astype(numpy.uint8)

        return mask

    @torch.no_grad()
    def process_dir(
        self,
        input_dir: str,
        output_dir: str | None = None,
    ):
        """Predicts glasses masks for each image in the directory.

        Goes through all the images in the given directory and predicts 
        glasses mask for each image. Each mask is saved as a grayscale 
        image under a separate directory with the same names as the 
        original images.

        .. warning::
            Please ensure the directory contains valid images and only 
            image files, otherwise errors may occur.

        Args:
            input_dir (str): The path to the input directory of images 
                with glasses to generate masks for.
            output_dir (str | None, optional): The output directory to 
                save the masks for the glasses. If not specified, the 
                same path as for ``input_dir`` is used and additionally 
                "_masks" suffix is added to the name. Defaults to None.
        """
        if output_dir is None:
            # Define default out-dir
            output_dir = input_dir + "_masks"
        
        # Create the possibly non-existing dirs
        os.makedirs(output_dir, exist_ok=True)
        
        for file in os.scandir(input_dir):
            # Predict and save mask as gray
            mask = self.predict(file.path)
            image = Image.fromarray(mask, mode='L')
            image.save(os.path.join(output_dir, file.name))

class TinyGlassesSegmenter(nn.Module):
    """Tiny segmenter to mark which parts of the image are glasses.

    This is a custom segmenter created with the aim to contain very few
    parameters while maintaining a reasonable accuracy. It only has 
    several sequential up-convolution and down-convolution layers with 
    residual connections and is very similar to U-Net.

    .. note::
        You can read more about U-Net architecture in the following 
        paper by O. Ronneberger et al.:
        `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_
    """
    class _Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.pool0 = nn.MaxPool2d(2)
            self.conv1 = Conv2dNormActivation(in_channels, out_channels)
            self.conv2 = Conv2dNormActivation(out_channels, out_channels)
        
        def forward(self, x):
            return self.conv2(self.conv1(self.pool0(x)))
        
    class _Up(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            half_channels = in_channels // 2
            self.conv0 = nn.ConvTranspose2d(half_channels, half_channels, 2, 2)
            self.conv1 = Conv2dNormActivation(in_channels, out_channels)
            self.conv2 = Conv2dNormActivation(out_channels, out_channels)

        def forward(self, x1, x2):
            x1 = self.conv0(x1)

            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2))
            
            x = torch.cat([x2, x1], dim=1)

            return self.conv2(self.conv1(x))
    
    def __init__(self):
        super().__init__()

        # Feature extraction layer
        self.first = nn.Sequential(
            Conv2dNormActivation(3, 16),
            Conv2dNormActivation(16, 16),
        )

        # Down-sampling layers
        self.down1 = self._Down(16, 32)
        self.down2 = self._Down(32, 64)
        self.down3 = self._Down(64, 128)
        self.down4 = self._Down(128, 128)

        # Up-sampling layers
        self.up1 = self._Up(256, 64)
        self.up2 = self._Up(128, 32)
        self.up3 = self._Up(64, 16)
        self.up4 = self._Up(32, 16)

        # Pixel-wise classification layer
        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Performs forward pass.

        Predicts raw pixel scores for the given batch of inputs. Scores 
        are unbounded - anything that's less than 0 means glasses 
        belonging to the pixel is unlikely and anything that's above 0 
        indicates that glasses for a particular pixel is likely.

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note 
                that pixel values are normalized and squeezed between 
                0 and 1.

        Returns:
            dict[str, torch.Tensor]: A dictionary with a single "out" 
                entry (for compatibility). The value is an output tensor 
                of shape (N, 1, H, W) indicating which pixels in the 
                image fall under category of 'glasses'. The scores are 
                unbounded, thus, to convert to probabilities, sigmoid 
                function must be used.
        """
        # Extract primary features
        x1 = self.first(x)

        # Downsample features
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Updample features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Predict one channel
        out = self.last(x)

        return {"out": out}
