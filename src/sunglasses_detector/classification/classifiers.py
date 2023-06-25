import torch
import torch.nn as nn
from .base_classifier import BaseClassifier

class EyeglassesClassifier(BaseClassifier):
    def __init__(self, base_model: str = "medium", pretrained: bool = False):
        super().__init__(base_model, pretrained)

class SunglassesClassifier(BaseClassifier):
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
    def __init__(self, base_model: str = "medium", pretrained: bool = False):
        super().__init__(base_model, pretrained)

class GlassesClassifier(BaseClassifier):
    def __init__(
        self, 
        base_model: str | tuple[str, str] = "medium",
        pretrained: bool = False
    ):
        super().__init__()

        if isinstance(base_model, str):
            base_model = (base_model, base_model)
        
        self.eyeglasses_classifier = EyeglassesClassifier(base_model[0], pretrained)
        self.sunglasses_classifier = SunglassesClassifier(base_model[1], pretrained)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat_eyeg = self.eyeglasses_classifier(x)
        y_hat_sung = self.sunglasses_classifier(x)

        if y_hat_eyeg.data > 0 or y_hat_sung.data > 0:
            y_hat = torch.max(y_hat_eyeg, y_hat_sung)
        else:
            y_hat = torch.mean(torch.stack([y_hat_eyeg, y_hat_sung]), dim=0)
        
        return y_hat