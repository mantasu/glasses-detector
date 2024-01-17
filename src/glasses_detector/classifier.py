from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_v2_s, shufflenet_v2_x1_0

from .architectures import TinyBinaryClassifier
from .components.base_model import BaseGlassesModel
from .components.pred_type import Default
from .utils import FilePath


@dataclass
class GlassesClassifier(BaseGlassesModel):
    r"""Binary classifier to check if glasses are present.

    This class allows to perform binary classification for images with
    glasses, i.e., determines whether or not the glasses are present in
    the image (primarily focus is on whether or not eyeglasses are worn
    by a person). It is possible to specify only a particular kind of
    glasses to focus on, e.g., sunglasses.

    .. list-table:: Performance of the Pre-trained Classifiers
        :header-rows: 1

        * - Kind
          - Size
          - BCE :math:`\downarrow`
          - F1 :math:`\uparrow`
          - ROC-AUC :math:`\uparrow`
          - PR-AUC :math:`\uparrow`
        * - ``anyglasses``
          - ``small``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``anyglasses``
          - ``medium``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``anyglasses``
          - ``large``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``eyeglasses``
          - ``small``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``eyeglasses``
          - ``medium``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``eyeglasses``
          - ``large``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``sunglasses``
          - ``small``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``sunglasses``
          - ``medium``
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``sunglasses``
          - ``large``
          - TODO
          - TODO
          - TODO
          - TODO

    .. list-table:: Size Info of teh Pre-trained Classifiers
        :header-rows: 1

        * - Size
          - Architecture
          - Paper
          - Params :math:`\downarrow`
          - GFLOPs :math:`\downarrow`
          - Memory :math:`\downarrow`
          - Filesize :math:`\downarrow`
        * - ``small``
          - :class:`tinyclsnet_v1 <.architectures.TinyBinaryClassifier>`
          - N/A
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``medium``
          - :func:`~torchvision.models.shufflenet_v2_x1_0`
          - `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design <https://arxiv.org/abs/1807.11164>`_
          - TODO
          - TODO
          - TODO
          - TODO
        * - ``large``
          - :func:`~torchvision.models.efficientnet_v2_s`
          - `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_
          - TODO
          - TODO
          - TODO
          - TODO

    Args:
        kind (str, optional): The kind of glasses to perform binary
            classification for. Available options are:

                * ``"anyglasses"`` - any kind glasses/googles/spectacles
                * ``"eyeglasses"`` - transparent eyeglasses
                * ``"sunglasses"`` - opaque and semi-transparent glasses

            Each kind is only responsible for its category, e.g., if
            ``kind`` is set to ``"sunglasses"``, then images with
            transparent eyeglasses will not be identified as positive.
            Defaults to ``"anyglasses"``.
        size (str, optional): The size of the model to use. Available
            options are:

                * ``"small"`` - a tiny model with very few parameters
                  but a lower accuracy.
                * ``"medium"`` - a model with a balance between the
                  number of parameters and the accuracy.
                * ``"large"`` - a model with a large number of
                  parameters but a higher accuracy.

            Please check :attr:`DEFAULT_SIZE_MAP` to see which
            architecture each size maps to and the details about the
            number of parameters. Defaults to ``"medium"``.
        pretrained (bool | str | None, optional): Whether to load
            weights from a custom URL (or local file if they're already
            downloaded) which will be inferred based on model's
            :attr:`kind` and :attr:`size`. If a string is provided, it
            will be used as a path or a URL (determined automatically)
            to the model weights. Defaults to ``True``.
    """
    kind: str = "anyglasses"
    size: str = "medium"
    pretrained: bool | str | None = field(default=True, repr=False)
    task: str = field(default="classification", init=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "small": {"name": "tinyclsnet_v1", "version": "v1.0.0"},
        "medium": {"name": "shufflenet_v2_x1_0", "version": "v1.0.0"},
        "large": {"name": "efficientnet_v2_s", "version": "v1.0.0"},
    }
    """
    typing.ClassVar[dict[str, dict[str, str]]]: The model info
    dictionary mapping from the size of the model to the model info
    dictionary which contains the name of the architecture and the
    version of the release. This is just a helper component for
    :attr:`DEFAULT_KIND_MAP` because each default kind has the same set
    of default models.
    """

    DEFAULT_KIND_MAP: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "anyglasses": DEFAULT_SIZE_MAP,
        "eyeglasses": DEFAULT_SIZE_MAP,
        "sunglasses": DEFAULT_SIZE_MAP,
    }
    """
    typing.ClassVar[dict[str, dict[str, dict[str, str]]]]: The model
    info dictionary used to construct the URL to download the weights
    from. It has 3 nested levels:

        1. ``kind`` - the kind of the model, e.g., ``"sunglasses"``
        2. ``size`` - the size of the model, e.g., ``"medium"``
        3. ``info`` - the model info, i.e., ``"name"`` and ``"version"``
    
    For example, ``DEFAULT_KIND_MAP["sunglasses"]["medium"]`` would
    return ``{"name": <arch-name>, "version": <release-version>}``.
    """

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinyclsnet_v1":
                m = TinyBinaryClassifier()
            case "shufflenet_v2_x1_0":
                m = shufflenet_v2_x1_0()
                m.fc = nn.Linear(1024, 1)
            case "efficientnet_v2_s":
                m = efficientnet_v2_s()
                m.classifier = nn.Linear(1280, 1)
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
        format: str
        | dict[bool, Default]
        | Callable[[torch.Tensor], Default] = {
            True: "wears",
            False: "does_not_wear",
        },
    ) -> Default | list[Default]:
        """Predicts whether the positive class is present.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction for each
        image indicating whether the it belongs to a positive class,
        e.g., *"anyglasses"*, or not. The format of the prediction,
        i.e., the prediction type is :attr:`Default` type which
        corresponds to :attr:`~.PredType.DEFAULT`.

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
            Default | list[Default]: The formatted
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
