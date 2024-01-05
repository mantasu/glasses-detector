import torch
import torch.nn as nn


class TinyBinaryDetector(nn.Module):
    """Tiny binary detector.

    This is a custom detector created with the aim to contain very few
    parameters while maintaining a reasonable accuracy. It only has
    several sequential convolutional and pooling blocks (with
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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # FC for bounding box prediction
        self.fc = nn.Linear(80, 4)

    def _create_block(self, num_in, num_out, filter_size):
        return nn.Sequential(
            nn.Conv2d(num_in, num_out, filter_size, 1, "valid", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_out),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Predicts the bounding box for the given batch of inputs.

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note
                that pixel values are normalized and squeezed between
                0 and 1.

        Returns:
            torch.Tensor: An output tensor of shape (N, 4) indicating
            the bounding box coordinates for each nth image. The
            coordinates are in the format (x_min, y_min, x_max, y_max),
            where (x_min, y_min) is the top-left corner of the bounding
            box and (x_max, y_max) is the bottom-right corner.
        """
        return self.fc(self.features(x))
