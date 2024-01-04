from dataclasses import dataclass
from typing import Callable, ClassVar, Collection, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .components.base_model import BaseGlassesModel
from .components.pred_type import PredType
from .models import TinyBinaryClassifier
from .utils import ImgPath


@dataclass
class GlassesClassifier(BaseGlassesModel):
    """Glasses classifier for specific glasses type."""

    task: str = "classification"
    kind: str = "anyglasses"
    size: str = "normal"
    pretrained: bool = True

    DEFAULT_SIZE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "small": {"name": "tinyclsnet_v1", "version": "v1.0.0"},
        "medium": {"name": "TBA", "version": "v1.0.0"},
        "large": {"name": "TBA", "version": "v1.0.0"},
    }

    DEFAULT_KIND_MAP: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "anyglasses": DEFAULT_SIZE_MAP,
        "eyeglasses": DEFAULT_SIZE_MAP,
        "sunglasses": DEFAULT_SIZE_MAP,
    }

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinyclsnet_v1":
                m = TinyBinaryClassifier()
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @override
    def predict(
        self,
        image: ImgPath
        | Image.Image
        | np.ndarray
        | Collection[ImgPath | Image.Image | np.ndarray],
        format: str
        | dict[bool, PredType.Custom]
        | Callable[[torch.Tensor], PredType.Default] = {
            True: "wears",
            False: "does_not_wear",
        },
    ) -> PredType.Default | list[PredType.Default]:
        """Predicts whether the positive class is present.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a boolean value for each image
        indicating whether the it belongs to a positive class, e.g.,
        *"anyglasses"*, or not. The prediction could be mapped to
        some :attr:`PredType.Default` type.

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
            format (str | dict[bool, PredType.Default] | typing.Callable[[torch.Tensor], PredType.Default], optional):
                The string specifying the way to map the predictions to
                labels. These are the following options:

                * "bool" - maps image to ``True`` (if predicted as
                  positive) and to ``False`` (if predicted as negative).
                * "int" - maps image to ``1`` (if predicted as positive)
                  and to ``0`` (if predicted as negative).
                * "str" - maps image to ``"present"`` (if predicted as
                  positive) and to ``"not_present"`` (if predicted as
                  negative).
                * "logit" - maps image to a raw score (real number) of
                  a positive class.
                * "proba" - maps image to a probability (a number
                  between 0 and 1) of a positive class.

                It is also possible to provide a dictionary with 2 keys:
                ``True`` and ``False``, each mapping to values
                corresponding to what to output if the predicted label
                is positive or negative. Further, a custom callback
                function is also possible that specifies how to map a
                raw :class:`torch.Tensor` score of type
                ``torch.float32`` of shape ``(1,)`` to a label. Defaults
                to ``{True: "wears", False: "does_not_wear"}``.

        Returns:
            PredType.Default | list[PredType.Default]: The formatted
            prediction or a list of formatted predictions if multiple
            images were provided.

        Raises:
            ValueError: If the specified ``format`` as a string is
                not recognized.
        """
        if isinstance(format, str):
            # Check format
            match format:
                case "bool":
                    format = {True: True, False: False}
                case "int":
                    format = {True: 1, False: 0}
                case "str":
                    format = {True: "present", False: "not_present"}
                case "logit":
                    format = lambda x: x.item()
                case "proba":
                    format = lambda x: x.sigmoid().item()
                case _:
                    raise ValueError(f"Invalid format: {format}")

        if isinstance(d := format, dict):
            # If the format was specified as dictionary
            format = lambda x: d[(x > 0).item()]

        return super().predict(image, format)
