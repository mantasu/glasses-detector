import torch
import torch.nn as nn


class TinyBinaryClassifier(nn.Module):
    """Tiny binary classifier.

    This is a custom classifier created with the aim to contain very few
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
        unbounded, anything that's less than 0, means positive class is 
        unlikely and anything that's above 0 indicates that the positive 
        class is likely

        Args:
            x (torch.Tensor): Image batch of shape (N, C, H, W). Note 
                that pixel values are normalized and squeezed between 
                0 and 1.

        Returns:
            torch.Tensor: An output tensor of shape (N,) indicating 
            whether each nth image falls under the positive class or 
            not. The scores are unbounded, thus, to convert to a 
            probability, sigmoid function must be used.
        """
        return self.fc(self.features(x))