import re
import types
import torch
import inspect
import torch.nn as nn

from .tiny_binary_segmenter import TinyBinarySegmenter
from .tiny_binary_classifier import TinyBinaryClassifier
from torchvision.models.segmentation.lraspp import LRASPPHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)

from torchvision.models import (
    shufflenet_v2_x0_5,
    mobilenet_v3_small,
    efficientnet_b0,
)


class BaseModel(nn.Module):
    def __init__(
        self, 
        base_model: str | torch.nn.Module | None = None, 
        pretrained: bool = False, 
        kind: str | None = None, 
        name: str | None = None, 
        abbrev_map: dict[str, str] = {}, 
        version_map: dict[str, str] = {},
    ):
        """_summary_

        Args:

        It is also possible to specify ``model_type`` as some 
                abbreviation that can be mapped to a specific name. For 
                example, it could be one of "tiny", "small", "medium", 
                "large", "huge". For more details, see ``abbrev_map``
                argument specification.
            abbrev_map (dict[str, str], optional): abbrev_map (str | dict[str, str] | None, optional): Custom 
                dictionary that maps abbreviations specified via 
                ``model_type`` argument to actual model types supported 
                by this function. If not specified, ``model_type`` will 
                be assumed not to be an abbreviation but an actual model 
                type. Defaults to None.
        """
        super().__init__()

        self.kind = kind
        self.name = name
        self.abbrev_map = abbrev_map
        self.version_map = version_map

        if base_model is not None:
            self._init_from_base_model(base_model, pretrained)

    @staticmethod
    def split_at_capitals(string: str):
        return '_'.join(re.findall('[A-Z][^A-Z]*', string)).lower()

    @staticmethod
    def create_base_model(model_name: str) -> torch.nn.Module:
        """Creates a base model based on specified type.

        This method simply creates a new instance of torch model 
        according to the specified name. There are some pre-defined 
        architecture names, according to which those instances can be 
        created.

        Note:
            No weights are loaded, only the model itself.

        Args:
            model_name (str): The name of the model to use for 
                classification/segmentation. These are the available 
                options:

                * "tinyclsnet_v1" - The smallest model that is uniquely 
                  part of this package. For more information, see 
                  :class:`.._models.TinyBinaryClassifier`.
                * "shufflenet_v2_x0_5" - ShuffleNet V2 model taken from 
                  torchvision package. For more information, see 
                  :func:`~torchvision.models.shufflenet_v2_x0_5`.
                * "mobilenet_v3_small" - MobileNet V3 model taken from 
                  torchvision package. For more information, see 
                  :func:`~torchvision.models.mobilenet_v3_small`.
                * "efficientnet_b0" - EfficientNet B0 model taken from 
                  torchvision package. For more information, see 
                  :func:`~torchvision.models.efficientnet_b0`.
                * "tinysegnet_v1" - The smallest model that is uniquely 
                  part of this package. For more information, see 
                  :class:`.TinyBinarySegmenter`.
                * "lraspp_mobilenet_v3_large" - LR-ASPP model taken from 
                  torchvision package. For more information, see 
                  :func:`~torchvision.models.segmentation.lraspp_mobilenet_v3_large`.
                * "fcn_resnet50" - FCN model taken from torchvision 
                  package. For more information, see 
                  :func:`~torchvision.models.segmentation.fcn_resnet50`.
                * "deeplabv3_resnet101" - DeepLab V3 model taken from 
                  torchvision package. For more information, see 
                  :func:`~torchvision.models.segmentation.deeplabv3_resnet101`.

        Raises:
            ValueError: If the model type is not available.

        Returns:
            torch.nn.Module: A new instance of torch model based on the 
                specified model type.
        """
        # Match and init correct model type
        match model_name:
            case "tinyclsnet_v1":
                m = TinyBinaryClassifier()
            case "shufflenet_v2_x0_5":
                m = shufflenet_v2_x0_5()
                m.fc = nn.Linear(m.fc.in_features, 1)
            case "mobilenet_v3_small":
                m = mobilenet_v3_small()
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
            case "efficientnet_b0":
                m = efficientnet_b0()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
            case "tinysegnet_v1":
                m = TinyBinarySegmenter()
            case "lraspp_mobilenet_v3_large":
                m = lraspp_mobilenet_v3_large()
                m.classifier = LRASPPHead(40, 960, 1, 128)
            case "fcn_resnet50":
                m = fcn_resnet50()
                m.classifier[-1] = nn.Conv2d(512, 1, 1)
                m.aux_classifier = None
            case "deeplabv3_resnet101":
                m = deeplabv3_resnet101()
                m.classifier = DeepLabHead(2048, 1)
                m.aux_classifier = None
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m
    
    def _init_from_base_model(self, base_model, pretrained=False):
        if isinstance(base_model, str):
            # Get the base model, assign name and instantiate the model
            base_model = self.abbrev_map.get(base_model, base_model)
            self.name = base_model if self.name is None else self.name
            base_model = self.create_base_model(base_model)
            
        if self.name is None:
            # If base_model was torch.nn.Module, just take its name
            self.name = self.split_at_capitals(base_model.__class__.__name__)
        
        if self.kind is None:
            # If self kind is not known, take child's class name
            self.kind = self.split_at_capitals(self.__class__.__name__)

        # Update the current class' attributes
        self.__dict__.update(base_model.__dict__)

        for name, method in inspect.getmembers(base_model, inspect.ismethod):
            # Update the current class' methods (copy them)
            name = "base_forward" if name == "forward" else name
            setattr(self, name, types.MethodType(method.__func__, self))

        if pretrained:
            # Load version if known, then load weights
            version = self.version_map.get(self.kind)
            self._load_weights_from_url(self.kind, self.name, version)

    def _load_weights_from_url(
        self,
        model_kind: str,
        model_name: str,
        version: str | None = None,
        base_url: str = "https://github.com/mantasu/glasses-detector/releases/download/", 
        device: str | torch.device | None = None,
    ):
        """Reads weights file and loads them into the given model.

        This method takes a model which needs to be loaded with weights, 
        a URL that specifies the location of weights to download and, 
        optionally, a device to load the model on. 

        Note: 
            Although URL must be specified, according to 
            :func:`~torch.hub.load_state_dict_from_url`, first, the 
            corresponding weights will be checked if they are already 
            present in hub cache, which by default is 
            ``~/.cache/torch/hub/checkpoints``, and, if they are not, 
            the weight will be downloaded there and then loaded.

        Args:
            model (torch.nn.Module): The model to load the weights to.
            base_model_type (str): The type of base model or model size 
                the corresponding weights should be downloaded for. One 
                of "tinysegnet", "lraspp", "fcn", "deeplab" or one of 
                "tiny", "small", "medium", "large".
            segmenter_type (str): The type of segmenter to download 
                the weights for. One of "full-glasses", "glass-frames".
            version (str): The GitHub release version at which the 
                model weights were uploaded, e.g., "v1.0.0".
            base_url (str): The base GitHub releases URL (without 
                version) at which the weights are uploaded. Defaults to 
                "https://github.com/mantasu/glasses-detector/releases/download/".
            device (str | torch.device | None, optional): The device to
                load the weight and the model onto. If not specified, 
                *cpu* will be used. Defaults to None.

        Returns:
            torch.nn.Module: The same given model but with weights 
                loaded.
        """
        # Create a full URL from parts to download weights
        base_url += '' if version is None else version + '/'
        url = base_url + f"{model_kind}_{model_name}.pth"

        # Get weights from the download path and load them into model
        weights = torch.hub.load_state_dict_from_url(url, map_location=device)
        self.load_state_dict(weights)

        if device is not None:
            # Cast to device
            self.to(device)

    def base_forward(self, *args, **kwargs):
        return None
    
    def forward(self, *args, **kwargs):
        return self.base_forward(*args, **kwargs)