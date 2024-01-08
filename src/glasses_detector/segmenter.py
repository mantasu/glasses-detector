from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import fcn_resnet101, lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPPHead

from .components.base_model import BaseGlassesModel
from .components.pred_type import Default
from .models import TinyBinarySegmenter
from .utils import FilePath


@dataclass
class GlassesSegmenter(BaseGlassesModel):
    """Glasses segmenter for glasses and their parts."""

    task: str = field(default="segmentation", init=False)
    kind: str = "smart"
    size: str = "medium"
    pretrained: bool | str | None = field(default=True, repr=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "small": {"name": "tinysegnet_v1", "version": "v1.0.0"},
        "medium": {"name": "lraspp_mobilenet_v3_large", "version": "v1.0.0"},
        "large": {"name": "fcn_resnet101", "version": "v1.0.0"},
    }

    DEFAULT_KIND_MAP: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "frames": DEFAULT_SIZE_MAP,
        "full": DEFAULT_SIZE_MAP,
        "legs": DEFAULT_SIZE_MAP,
        "lenses": DEFAULT_SIZE_MAP,
        "shadows": DEFAULT_SIZE_MAP,
        "smart": DEFAULT_SIZE_MAP,
    }

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinysegnet_v1":
                m = TinyBinarySegmenter()
            case "lraspp_mobilenet_v3_large":
                m = lraspp_mobilenet_v3_large()
                m.classifier = LRASPPHead(40, 960, 1, 128)
            case "fcn_resnet101":
                m = fcn_resnet101()
                m.classifier[-1] = nn.Conv2d(512, 1, 1)
                m.aux_classifier = None
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @override
    def predict(
        self,
        image: FilePath
        | Image.Image
        | np.ndarray
        | Collection[FilePath | Image.Image | np.ndarray],
        format: str | dict[bool, Default] | Callable[[torch.Tensor], Default] = "img",
    ) -> Default | list[Default]:
        """Predicts which pixels in the image are positive.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction (typically
        a 2D mask of type :class:`torch.tensor`, e.g., with values of
        either 255 (white) indicating pixels under positive category or
        0 (black) indicating the rest of the pixels). In general, the
        prediction could be converted to any :attr:`Default`
        type.

        Warning:
            If the image is provided as :class:`numpy.ndarray`, make
            sure the last dimension specifies the channels, i.e., last
            dimension should be of size ``1`` or ``3``. If it is
            anything else, e.g., if the shape is ``(3, H, W)``, where
            ``W`` is neither ``1`` nor ``3``, this would be interpreted
            as 3 grayscale images.

        Args:
            image (FilePath | PIL.Image.Image | numpy.ndarray | typing.Collection[FilePath | PIL.Image.Image | numpy.ndarray]):
                The path(-s) to the image to generate the prediction for
                or the image(-s) itself represented as
                :class:`Image.Image` or as a :class:`numpy.ndarray`.
                Note that the image should have values between 0 and 255
                and be of RGB format. Normalization is not needed as the
                channels will be automatically normalized before passing
                through the network.
            format (str | dict[bool, Default] | typing.Callable[[torch.Tensor], Default], optional):
                The string specifying the way to map the predictions
                (pixel scores) to masks. These are the following
                options:

                * "bool" - maps image pixels to ``True`` (those
                  predicted as positive) and to ``False`` (those
                  predicted as negative).
                * "int" - maps image pixels to ``1`` (those predicted
                  as positive) and to ``0`` (those predicted as
                  negative).
                * "img" - maps image pixels to ``255`` (those predicted
                  as positive) and to ``0`` (those predicted as
                  negative). The returned type is :class:`Image.Image`
                  of mode "L" (grayscale).
                * "logit" - maps image pixels to raw scores (real
                  numbers) of them being positive.
                * "proba" - maps image pixels to probabilities (numbers
                  between 0 and 1) of them being positive.

                It is also possible to provide a a dictionary with 2
                keys: ``True`` and ``False``, each mapping to values
                corresponding to what to output if the predicted pixel
                is positive or negative. Further, a custom callback
                function can be provided which specifies how to map a
                raw :class:`torch.Tensor` output of type
                ``torch.float32`` of shape ``(H, W)`` to a mask.
                Defaults to "img".

        Returns:
            Default | list[Default]: The formatted
            prediction(s) of type :attr:`Default`. In most
            cases the output is a mask of type :class:`torch.Tensor`
            and of shape (H, W)  with each pixel mapped to some ranged
            value or to a binary value. But it can also be a grayscale
            mask image of type :class:`Image.Image` or any other
            :attr:`Default` type, depending on the ``format``
            argument.

        Raises:
            ValueError: If the specified ``format`` as a string is
                not recognized.
        """
        if isinstance(format, str):
            # Update mask type
            match format:
                case "bool":
                    format = {True: True, False: False}
                case "int":
                    format = {True: 1, False: 0}
                case "img":
                    format = lambda x: Image.fromarray(
                        ((x > 0) * 255).to(torch.uint8),
                        mode="L",
                    )
                case "logit":
                    format = lambda x: x
                case "proba":
                    format = lambda x: x.sigmoid()
                case _:
                    raise ValueError(f"Invalid format: {format}")

        if isinstance(d := format, dict):
            # If mask type was specified as dictionary
            format = lambda x: torch.where((x > 0), d[True], d[False])

        return super().predict(image, format)
