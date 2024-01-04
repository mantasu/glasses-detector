import os
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, Self, overload, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .._data import ImageLoaderMixin
from ..utils import ImgPath, is_image_path, is_url
from .pred_interface import PredInterface
from .pred_type import *


@dataclass
class BaseGlassesModel(nn.Module, PredInterface):
    """Base class for all glasses models.

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
        task (str): The task the model is built for. Used when
            automatically constructing URL to download the weights from.
        kind (str): The kind of the model. Used to access
            :meth:`model_info`.
        size (str): The size of the model. Used to access
            :meth:`model_info`.
        pretrained (bool | str, optional): Whether to load weights from
            a custom URL (or local file if they're already downloaded)
            which will be inferred based on model's kind and name. If a
            string is provided, it will be used as a path or a URL
            (determined automatically) to the model weights. Defaults to
            ``False``.
        device (str | torch.device, optional): Device to cast the model
            (once it is loaded) to. Defaults to ``"cpu"``.
    """

    task: str
    kind: str
    size: str
    pretrained: bool | str = field(default=False, repr=False)
    device: str | torch.device = field(default="cpu", repr=False)
    model: nn.Module = field(default_factory=lambda: None, init=False, repr=False)

    BASE_WEIGHTS_URL: ClassVar[
        str
    ] = "https://github.com/mantasu/glasses-detector/releases/download"
    """
    typing.ClassVar[str]: The base URL to download the weights from.
    """

    DEFAULT_KIND_MAP: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "kind": {"size": {"name": "", "version": ""}},
    }
    """
    typing.ClassVar[dict[str, dict[str, dict[str, str]]]]: The template
    for the model info. The model info is used to construct the URL to
    download the weights from.
    """

    def __post_init__(self):
        super().__init__()

        try:
            # Get the model name and create it
            model_name = self.model_info["name"]
            self.model = self.create_model(model_name)
        except KeyError:
            # Raise model info warning
            self._model_info_warning()
        except ValueError:
            # Raise model init (structure construction) warning
            message = f"Model structure named {model_name} does not exist. "
            self._model_init_warning(message=message)

        if self.pretrained:
            # Load weights if pretrained is True or a path
            self.load_weights(path_or_url=self.pretrained)

        # Cast to device
        self.to(self.device)

    @property
    def model_info(self) -> dict[str, dict[str, dict[str, str]]]:
        return self.DEFAULT_KIND_MAP.get(self.kind, {}).get(self.size, {})

    @staticmethod
    @abstractmethod
    def create_model(self, model_name: str) -> nn.Module:
        ...

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        **kwargs,
    ) -> Self:
        # Get the specified device or check the one from the model
        device = kwargs.get("device", next(iter(model.parameters())).device)

        with warnings.catch_warnings():
            # Ignore the warnings from the model init
            glasses_model = cls(
                task=kwargs.get("task", "custom"),
                kind=kwargs.get("kind", "custom"),
                size=kwargs.get("size", "custom"),
                pretrained=False,
                device=device,
            )

        # Assign the actual model
        glasses_model.model = model

        if pretrained := kwargs.get("pretrained", False):
            # Load weights if `pretrained` is True or a path
            glasses_model.load_weights(path_or_url=pretrained)

        # Cast to device
        glasses_model.to(device)

        return glasses_model

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @override
    @overload
    def predict(
        self,
        image: ImgPath | Image.Image | np.ndarray,
        format: Callable[[torch.Tensor], Default] = lambda x: str(x),
    ) -> Default:
        ...

    @override
    @overload
    def predict(
        self,
        image: Collection[ImgPath | Image.Image | np.ndarray],
        format: Callable[[torch.Tensor], Default] = lambda x: str(x),
    ) -> list[Default]:
        ...

    @torch.inference_mode()
    @override
    def predict(
        self,
        image: ImgPath
        | Image.Image
        | np.ndarray
        | Collection[ImgPath | Image.Image | np.ndarray],
        format: Callable[[torch.Tensor], Default] = lambda x: str(x),
    ) -> Default | list[Default]:
        """Predicts based on the model specified by the child class.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction generated
        by the child class.

        Warning:
            If the image is provided as :class:`numpy.ndarray`, make
            sure the last dimension specifies the channels, i.e., last
            dimension should be of size ``1`` or ``3``. If it is
            anything else, e.g., if the shape is ``(3, H, W)``, where
            ``W`` is neither ``1`` nor ``3``, this would be interpreted
            as 3 grayscale images.

        Args:
            image (ImgPath | PIL.Image.Image | numpy.ndarray | typing.Collection[ImgPath | PIL.Image.Image | numpy.ndarray]):
                The path(-s) to the image to generate the prediction for
                or the image(-s) itself represented as
                :class:`Image.Image` or as a :class:`numpy.ndarray`.
                Note that the image should have values between 0 and 255
                and be of RGB format. Normalization is not needed as the
                channels will be automatically normalized before passing
                through the network.
            format (typing.Callable[[torch.Tensor], Default], optional):
                Format callback. This is a custom function that takes a
                predicted tensor as input and outputs a formatted
                prediction of type :attr:`Default`. Defaults
                to ``lambda x: str(x)``.

        Returns:
            Default | typing.List[Default]: The
            formatted prediction or a list of formatted predictions if
            multiple images were provided.
        """
        # Get the device from the model and init vars
        device = next(iter(self.parameters())).device
        xs, preds, is_multiple = [], [], True

        # Warning: if the image has shape (3, H, W), it will be interpreted as 3 grayscale images
        if (is_image_path(image) or isinstance(image, Image.Image)) or (
            isinstance(image, np.ndarray)
            and (image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in [1, 3]))
        ):
            # Single image
            image = [image]
            is_multiple = False

        for img in image:
            # Load the image and cast to device and append to batch
            xs.append(ImageLoaderMixin.load_image(img).to(device))

        for pred in self(torch.stack(xs)):
            # Append formatted prediction
            preds.append(format(pred))

        return preds if is_multiple else preds[0]

    def load_weights(self, path_or_url: str | bool = True):
        if isinstance(path_or_url, bool) and path_or_url:
            try:
                # Get model name and release version
                name = self.model_info["name"]
                version = self.model_info["version"]
            except KeyError:
                # Raise model info warning for not constructing URL
                message = "Path/URL to weights cannot be constructed. "
                self._model_info_warning(message)
                return

            # Construct weights URL from base URL and model info
            weights_name = f"{self.task}_{self.kind}_{name}.pth"
            path_or_url = f"{self.BASE_WEIGHTS_URL}/{version}/{weights_name}"
        elif isinstance(path_or_url, bool):
            return

        if self.model is None:
            # Raise model init warning for not loading weights
            message = "Cannot load weights for the unspecified model. "
            self._model_init_warning(message)
            return

        if is_url(path_or_url):
            # Get weights from download path (and download if needed)
            weights = torch.hub.load_state_dict_from_url(
                url=path_or_url,
                map_location=self.device,
            )
        else:
            # Load weights from local path
            weights = torch.load(path_or_url, map_location=self.device)

        # Actually load the weights
        self.model.load_state_dict(weights)

        if self.device is not None:
            # Cast self to device
            self.to(self.device)

    def _model_info_warning(self, message: str = ""):
        warnings.warn(
            f"{message}Model info (name and release version) not found for the "
            f"specified configuration: {self.task=} {self.kind=} {self.size=}."
        )

    def _model_init_warning(self, message: str = ""):
        warnings.warn(
            f"{message}Model is not initialized. Try assigning a custom model "
            f"via `self.model` attribute, for instance, create a custom model "
            f"using `GlassesModel.create_model` and assign it."
        )
