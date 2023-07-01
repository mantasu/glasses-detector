import re
import types
import torch
import inspect
import torch.nn as nn

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

from ..models import TinyBinarySegmenter
from ..models import TinyBinaryClassifier


class BaseModel(nn.Module):
    """Base Model used any task.

    This model by itself does not serve any real purpose, however it 
    helps to keep the code DRY as the parent class. Here are the main 
    purposes of it:

    1. This can be loaded as a pre-defined model, e.g., from 
       torchvision, by specifying ``base_model`` argument. Such 
       pre-defined models also have weights that can be downloaded, 
       even multiple sets of weights - one for each kind of task, e.g., 
       for different things to classify.
    2. This can be extended via child classes which inherit the same 
       instantiation principle but have some extra methods, e.g., those 
       that are unique to a specific model type, like classification or 
       segmentation.
    3. This model has some cool attributes, like ``kind``, ``name`` 
       which generalize the behavior of any model.

    Other than that, this model can wrap around any other torch model
    with the custom attributes as mentioned in the 3rd purpose. 
    However, note that this model will copy everything from that torch 
    model, i.e., it will not keep it as an attribute (such as 
    ``self.torch_model``) but rather be that model itself (so it will 
    have all the layers as, for instance, ``self.layer1``, 
    ``self.layer2``, and functions from that torch model) with some 
    custom arguments.

    Note: 
        When ``pretrained`` is ``True``, the URL of the weights to be
        downloaded from will be constructed in the background 
        (private method) when a pre-defined model is initialized. 
        According to :func:`~torch.hub.load_state_dict_from_url`, first, 
        the corresponding weights will be checked if they are already 
        present in hub cache, which by default is 
        ``~/.cache/torch/hub/checkpoints``, and, if they are not, 
        the weight will be downloaded there and then loaded.

    Args:
        base_model (str | torch.nn.Module | None, optional): The base 
            model to copy the layers and the methods from. If provided 
            as a string (model name), see the available model options in 
            :meth:`create_base_model`. It is also possible to specify 
            ``base_model`` as some abbreviation that can be mapped to a 
            specific name. For example, it could be one of "tiny", 
            "small", "medium", "large", "huge". For more details, see 
            ``abbrev_map`` argument specification. It can also be a 
            custom torch model, instead of a name of a predefined model. 
            If not specified, it will not copy any layers from any 
            model, however, note that any child class will have to 
            overwrite :meth:`forward` as it will simply not exist. 
            Defaults to None.
        pretrained (bool, optional): Whether to load weights from a 
            custom URL (or local file if they're already downloaded)
            which will be inferred based on model's kind and name. 
            Defaults to False.
        name (str | None, optional): The name of the model architecture. 
            If not provided, but ``base_model`` is provided, the name 
            will based on it:

            * If ``base_model`` is :class:`str`, then it will also be 
              ``name``. However, if it is an abbreviation then the name 
              will be taken from ``abbrev_map``
            * If ``base_model`` is :class:`torch.Module`, then the name 
              will be based on the class' name, subject to 
              :meth:`split_at_capitals`.

            Defaults to None.
        kind (str | None, optional): The kind of the model, i.e., the 
            name for the set of weights that will be trained for some 
            specific task on the model architecture specified by 
            ``name``. If not provided, then the name will be based on 
            the current class name, subject to 
            :meth:`split_at_capitals`. Thus, if any class extends it, 
            its name will be used to determine ``kind``. Defaults to 
            None.
        abbrev_map (dict[str, str], optional): Custom dictionary that 
            maps abbreviations specified via ``base_model`` argument to 
            actual model types supported by :meth:`create_base_model`. 
            If not specified, ``base_model`` (if it's a string) will be 
            assumed not to be an abbreviation but an actual model type, 
            i.e., name of the architecture. Defaults to {}.
        version_map (dict[str, str], optional): Custom dictionary 
            mapping ``kind`` to some version which specifies the 
            release at which the model weights for this ``kind`` should 
            be taken. This is irrelevant if ``pretrained`` is ``False`` 
            or if the model weights are already downloaded. Defaults to 
            {}.
    """
    def __init__(
        self, 
        base_model: str | torch.nn.Module | None = None,
        pretrained: bool = False,
        name: str | None = None, 
        kind: str | None = None,
        abbrev_map: dict[str, str] = {},
        version_map: dict[str, str] = {},
    ):
        super().__init__()

        # Init attributes
        self.kind = kind
        self.name = name
        self.abbrev_map = abbrev_map
        self.version_map = version_map

        if base_model is not None:
            # Copy over layers and methods from base model
            self._init_from_base_model(base_model, pretrained)

    @staticmethod
    def split_at_capitals(string: str) -> str:
        """Splits string at capital letters and joins with '_'.

        Splits some :class:`str` value, e.g., the name of the class, at 
        places where capital letters are present, joins the separated 
        sub-strings via underscores '_' and lowercases everything.

        Example:
            >>> BaseModel.split_string_at_capitals("MyCustomClass")
            >>> "my_custom_class"

        Args:
            string (str): The string to split at capitals.

        Returns:
            str: A new string split at capitals and joined with '_'.
        """
        return '_'.join(re.findall('[A-Z][^A-Z]*', string)).lower()

    @staticmethod
    def create_base_model(model_name: str) -> torch.nn.Module:
        """Creates a base model based on specified type.

        This method simply creates a new instance of torch model 
        according to the specified name. There are some pre-defined 
        architecture names, according to which those instances can be 
        created.

        Note:
            No weights are loaded, only the model itself is created.

        Args:
            model_name (str): The name of the model to use for 
                classification/segmentation. These are the available 
                options for classification:

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
            
                And these are the available options for segmentation:

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
        # Create a full URL from parts to download weights
        base_url += '' if version is None else version + '/'
        url = base_url + f"{model_kind}_{model_name}.pth"

        # Get weights from the download path and load them into model
        weights = torch.hub.load_state_dict_from_url(url, map_location=device)
        self.load_state_dict(weights)

        if device is not None:
            # Cast to device
            self.to(device)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward function of the model.

        This is the same function as the one copied from ``base_model``. 
        Note that, when the methods are being copied (during 
        initialization in a private method), the forward function is 
        copied as ``base_function``, therefore here it is called by 
        default. This allows more flexibility should forward be altered 
        for different types of models (classes that extend this one).

        Args:
            *args: Any input arguments accepted by original ``forward``.
            **kwargs: Additional keyword arguments accepted by original 
                ``forward``.

        Returns:
            torch.Tensor: A model output after inference.
        """
        return self.base_forward(*args, **kwargs)