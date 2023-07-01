import torch
from copy import deepcopy
from .bases import BaseClassifier

class EyeglassesClassifier(BaseClassifier):
    """Classifier to determine if the person is wearing eyeglasses.

    A binary classifier that tells whether a person in the image is
    wearing eyeglasses (sunglasses or any other occluded glasses don't 
    count!). It has 5 different base models ranging from tiny to huge - 
    check the table below for more details:

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
        * - TBA (huge)
          - TODO
          - TODO
          - TODO
          - TODO
          - TODO

    Args:
        base_model (str | torch.nn.Module, optional): The abbreviation 
            of the base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. Finally, it can 
            also be custom torch model, e.g., personally trained on some 
            other data. Defaults to "medium".
        pretrained (bool, optional): Whether to load the pretrained 
            weights for the chosen base model. Check the note inside the 
            documentation of :class:`.BaseModel` to see how the weights 
            are automatically downloaded and loaded. Defaults to False.
    """
    def __init__(
        self, 
        base_model: str | torch.nn.Module = "medium", 
        pretrained: bool = False,
    ):
        super().__init__(base_model, pretrained)

class SunglassesClassifier(BaseClassifier):
    """Classifier to determine if the person is wearing sunglasses.

    A binary classifier that tells whether a person in the image is
    wearing sunglasses. It has 5 different base models ranging from 
    tiny to huge - check the table below for more details:

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
        * - TBA (huge)
          - TODO
          - TODO
          - TODO
          - TODO
          - TODO

    Args:
        base_model (str | torch.nn.Module, optional): The abbreviation 
            of the base model to use for classification. One of "tiny", 
            "small", "medium", "large", "huge". It can also be the name 
            of the model architecture - for available classification 
            architecture names, check 
            :meth:`~.BaseClassifier.create_base_model`. Finally, it can 
            also be custom torch model, e.g., personally trained on some 
            other data. Defaults to "medium".
        pretrained (bool, optional): Whether to load the pretrained 
            weights for the chosen base model. Check the note inside the 
            documentation of :class:`.BaseModel` to see how the weights 
            are automatically downloaded and loaded. Defaults to False.
    """
    def __init__(
        self, 
        base_model: str | torch.nn.Module = "medium", 
        pretrained: bool = False,
    ):
        super().__init__(base_model, pretrained)

class GlassesClassifier(BaseClassifier):
    """Classifier to determine if the person is wearing glasses.

    A binary classifier that tells whether a person in the image is
    wearing any type of glasses. It has 5 different base models ranging 
    from tiny to huge for both eyeglasses and sunglasses sub-models. The 
    way this model works is by taking the eyeglasses classifier and 
    sunglasses classifier and produces a combined output. Therefore, 
    weights and accuracy will be a combination of those defined in 
    :class:`.EyeglassesClassifier` and :class:`.SunglassesClassifier`.

    Args:
        base_model (str | torch.nn.Module | tuple[str | torch.nn.Module, 
            str | torch.nn.Module]): The abbreviation of the base model 
            to use for classification. One of "tiny", "small", "medium", 
            "large", "huge". It can also be the name of the model 
            architecture - for available classification architecture 
            names, check :meth:`~.BaseClassifier.create_base_model`. 
            Finally, it can also be custom torch model, e.g., 
            personally trained on some other data. If provided as a 
            tuple of 2 base models, the first one will be used for 
            eyeglasses classifier and the second will be used to 
            construct a sunglasses classifier. So if it is not a tuple, 
            but a single value, the same value will be used for both 
            models. Defaults to "medium".
        pretrained (bool, optional): Whether to load the pretrained 
            weights for the chosen base model(-1). Check the note inside 
            the documentation of :class:`.BaseModel` to see how the 
            weights are automatically downloaded and loaded. If provided 
            as a tuple, then the first value will be used for eyeglasses 
            model and the second value for sunglasses model. Defaults to 
            False.
    """
    def __init__(
        self, 
        base_model: str | torch.nn.Module | tuple[str | torch.nn.Module, str | torch.nn.Module] = "medium",
        pretrained: bool | tuple[bool, bool] = False,
    ):
        super().__init__()

        if isinstance(base_model, str):
            # Same base for both classifiers
            base_model = (base_model, base_model)
        
        if isinstance(base_model, torch.nn.Module):
            # Same base architecture and weights for both classifiers
            base_model = (deepcopy(base_model), deepcopy(base_model))
        
        if isinstance(pretrained, bool):
            # Same bool value for both classifiers
            pretrained = (pretrained, pretrained)
        
        # Init inner classifiers (optionally, with pretrained weights)
        self.eyeg_classifier = EyeglassesClassifier(base_model[0], pretrained[0])
        self.sung_classifier = SunglassesClassifier(base_model[1], pretrained[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the classifier.

        This is an updated forward pass from the parent method, in a 
        way that it uses 2 inner models to perform a full forward pass.
        There are 2 things to note:

        1. If either the eyeglasses classifier or the sunglasses 
           classifier produces a logit with a score more than 0, the 
           logit is taken whichever is the highest.
        2. If both classifiers produce a logit that is less than 0, 
           an average is taken of both logits.

        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The output batch of scores (logits) of shape 
                (N,).
        """
        # Retrieve the scores from both classifiers
        y_hat_eyeg = self.eyeg_classifier(x)
        y_hat_sung = self.sung_classifier(x)

        if y_hat_eyeg.data > 0 or y_hat_sung.data > 0:
            # If either is >0, take the larger one
            y_hat = torch.max(y_hat_eyeg, y_hat_sung)
        else:
            # If both are <0, take an average of those scores
            y_hat = torch.mean(torch.stack([y_hat_eyeg, y_hat_sung]), dim=0)
        
        return y_hat