import torch
import torch.nn as nn
from .base_classifier_mixin import BaseClassifierMixin

class EyeglassesClassifier(nn.Module, BaseClassifierMixin):
    def __init__(self, base_model: str = "medium", pretrained: bool = False):
        super().__init__()
        self.base_model = self.load_base_model(base_model)
        self.base_model = self.load_weights_from_url(self.base_model)
    
    def forward():
        # TODO
        pass

class SunglassesClassifier(nn.Module, BaseClassifierMixin):
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
        * - TinyBinNet (tiny)
          - TODO
          - TODO
          - TODO
          - TODO
          - TODO
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

            * "tinyclsnet" or "tiny" - The smallest model that is 
                uniquely part of this package. For more information, see
                :class:`.._models.TinyBinaryClassifier`.
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
    def __init__(self, base_model: str = "mobilenet", pretrained: bool = False):
        super().__init__()
        self.base_model = self.load_base_model(base_model)
    
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


class GlassesClassifier(nn.Module, BaseClassifierMixin):
    def __init__(
        self, 
        base_model: str | tuple[str, str] = "medium", pretrained: bool = False):
        super().__init__()

        if isinstance(base_model, str):
            base_model = (base_model, base_model)
        
        self.eyeglasses_classifier = self.load_base_model(base_model[0])
        self.sunglasses_classifier = self.load_base_model(base_model[1])

        if pretrained:
            self.load_weights_from_url(self.eyeglasses_classifier)
            self.load_weights_from_url(self.sunglasses_classifier)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat_eyeg = self.eyeglasses_classifier(x)
        y_hat_sung = self.sunglasses_classifier(x)

        if y_hat_eyeg.data > 0 or y_hat_sung.data > 0:
            y_hat = torch.max(y_hat_eyeg, y_hat_sung)
        else:
            y_hat = torch.mean(torch.stack([y_hat_eyeg, y_hat_sung]), dim=0)
        
        return y_hat