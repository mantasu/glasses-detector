import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation


class TinyBinarySegmenter(nn.Module):
    """Tiny binary segmenter.

    This is a custom segmenter created with the aim to contain very few
    parameters while maintaining a reasonable accuracy. It only has 
    several sequential up-convolution and down-convolution layers with 
    residual connections and is very similar to U-Net.

    Note:
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
        # self.down1 = self._Down(16, 32)
        # self.down2 = self._Down(32, 64)
        # self.down3 = self._Down(64, 128)
        # self.down4 = self._Down(128, 128)
        self.down1 = self._Down(16, 32)
        self.down2 = self._Down(32, 64)
        self.down3 = self._Down(64, 64)

        # Up-sampling layers
        # self.up1 = self._Up(256, 64)
        # self.up2 = self._Up(128, 32)
        # self.up3 = self._Up(64, 16)
        # self.up4 = self._Up(32, 16)
        self.up1 = self._Up(128, 32)
        self.up2 = self._Up(64, 16)
        self.up3 = self._Up(32, 16)

        # Pixel-wise classification layer
        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Performs forward pass.

        Predicts raw pixel scores for the given batch of inputs. Scores 
        are unbounded - anything that's less than 0 means positive class
        belonging to the pixel is unlikely and anything that's above 0 
        indicates that positive class for a particular pixel is likely.

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note 
                that pixel values are normalized and squeezed between 
                0 and 1.

        Returns:
            dict[str, torch.Tensor]: A dictionary with a single "out" 
            entry (for compatibility). The value is an output tensor of 
            shape (N, 1, H, W) indicating which pixels in the image fall 
            under positive category. The scores are unbounded, thus, to 
            convert to probabilities, sigmoid function must be used.
        """
        # Extract primary features
        x1 = self.first(x)

        # Downsample features
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Updample features
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Predict one channel
        out = self.last(x)

        return {"out": out}