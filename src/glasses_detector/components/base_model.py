import inspect
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import Callable, ClassVar, Collection, Self, overload, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .._data import ImageLoaderMixin
from ..utils import FilePath, is_path_type, is_url
from .pred_interface import PredInterface
from .pred_type import *


@dataclass
class BaseGlassesModel(PredInterface):
    """Base class for all glasses models.

    Base class with common functionality, i.e., prediction and weight
    loading methods, that should be inherited by all glasses models.
    Child classes must implement :meth:`create_model` method which
    should return the model architecture based on :meth:`model_info`
    which is a dictionary containing the model name and release
    version. The dictionary depends on the model's :attr:`kind` and
    :attr:`size`, both of which are used when creating an instance.
    An instance can be created by providing a custom model instead of
    creating a predefined one, see :meth:`from_model`.

    Note:
        When ``pretrained`` is ``True``, the URL of the weights to be
        downloaded from will be constructed in the background
        (private method) when a pre-defined model is initialized.
        According to :func:`~torch.hub.load_state_dict_from_url`, first,
        the corresponding weights will be checked if they are already
        present in hub cache, which by default is
        ``~/.cache/torch/hub/checkpoints``, and, if they are not,
        the weight will be downloaded there and then loaded.

    Warning:
        Any instance of this class should not be used for model training
        directly - some properties are unhashable. Instead, retrieve
        the model from the :attr:`model` attribute and train it instead.

    Args:
        task (str): The task the model is built for. Used when
            automatically constructing URL to download the weights from.
        kind (str): The kind of the model. Used to access
            :meth:`model_info`.
        size (str): The size of the model. Used to access
            :meth:`model_info`.
        pretrained (bool | str | None, optional): Whether to load
            weights from a custom URL (or local file if they're already
            downloaded) which will be inferred based on model's
            :attr:`task`, :attr:`kind`, and :attr:`size`. If a string
            is provided, it will be used as a path or a URL (determined
            automatically) to the model weights. Defaults to ``False``.
        device (str | torch.device, optional): Device to cast the model
            (once it is loaded) to. Defaults to ``"cpu"``.
    """

    task: str
    kind: str
    size: str
    pretrained: bool | str | None = field(default=False, repr=False)
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
    download the weights from. This nested dictionary has 3 levels which
    are expected to be as follows:

        1. ``kind`` - the kind of the model, e.g., ``"sunglasses"``
        2. ``size`` - the size of the model, e.g., ``"medium"``
        3. ``info`` - the model info, i.e., ``"name"`` and ``"version"``

    For example, ``DEFAULT_KIND_MAP["sunglasses"]["medium"]`` would
    return ``{"name": <arch-name>, "version": <release-version>}`` which
    is the expected format for :attr:`model_info`.
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
    def model_info(self) -> dict[str, str]:
        """Model info property.

        This contains the information about the model used (e.g.,
        architecture and weights). By default, it should have 2 fields:
        `"name"` and `"version"`, both of which are used when
        initializing the architecture and looking for pretrained weights
        (see :meth:`load_weights`).

        Note:
            This is the default implementation which accesses
            :attr:`DEFAULT_KIND_MAP` based on :attr:`kind` and
            :attr:`size`. Child classes can override either
            :attr:`DEFAULT_KIND_MAP` or this property itself for a
            custom dictionary.

        Returns:
            dict[str,str]: The model info dictionary with 2 fields -
            `"name"` and `"version"` which allow to construct model
            architecture and download the pretrained model weights, if
            present.
        """
        return self.DEFAULT_KIND_MAP.get(self.kind, {}).get(self.size, {})

    @staticmethod
    @abstractmethod
    def create_model(self, model_name: str) -> nn.Module:
        """Creates the model architecture.

        Takes the name of the model architecture and returns the
        corresponding model instance.

        Args:
            model_name (str): The name of the model architecture to
                create, for example, ``"efficientnet_v2_s"``.

        Returns:
            torch.nn.Module: The model instance with the corresponding
            architecture.

        Raises:
            ValueError: If the architecture for the model name is not
                implemented or is not valid.
        """
        ...

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        **kwargs,
    ) -> Self:
        """Creates a glasses model from a custom model.

        Creates a glasses model wrapper for a custom provided
        :class:`~torch.nn.Module`, instead of creating a predefined
        one based on :attr:`kind` and :attr:`size`.

        Note:
            Make sure the provided model's ``forward`` method behaves as
            expected, i.e., returns the prediction in expected format
            for compatibility with :meth:`predict`.

        Warning:
            :meth:`model_info` property will not be useful as it would
            return an empty dictionary for custom specified :attr:`kind`
            and :attr:`size` (if specified at all).

        Args:
            model (torch.nn.Module): The custom model that will be
                assigned as :attr:`model`.
            **kwargs: Keyword arguments to pass to the constructor, see
                :class:`BaseGlassesModel` for more details. If
                ``task``, ``kind``, and ``size`` are not provided, they
                will be set to ``"custom"``. If the model architecture
                is custom, you may still specify the path to the
                pretrained wights via ``pretrained`` argument. Finally,
                if ``device`` is not provided, the wrapper will be cast
                to the the same one as the provided model is currently
                on.

        Returns:
            The glasses model wrapper of the same class type from which
            this method was called for the provided custom model.
        """
        # Set default values for class args
        kwargs.setdefault("task", "custom")
        kwargs.setdefault("kind", "custom")
        kwargs.setdefault("size", "custom")
        kwargs.setdefault("device", device := next(iter(model.parameters())).device)

        # Weights will be handled after instantiation
        pretrained = kwargs.get("pretrained", False)
        kwargs["pretrained"] = False

        # Filter out the arguments that are not for the model init
        is_init = {f.name: f.init for f in fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if is_init[k]}

        with warnings.catch_warnings():
            # Ignore warnings from model init
            warnings.simplefilter("ignore")
            glasses_model = cls(**kwargs)

        # Assign the actual model
        glasses_model.model = model

        if pretrained := kwargs.get("pretrained", False):
            # Load weights if `pretrained` is True or a path
            glasses_model.load_weights(path_or_url=pretrained)

        # Cast to device
        glasses_model.to(device)

        return glasses_model

    @override
    def forward(self, *args) -> Iterable[Any]:
        """Performs forward pass.

        Calls the forward method of the inner :attr:`model`, by passing
        any inputs it can process (first argument is typically a batch
        of images, i.e., a :class:`~torch.Tensor` of shape
        ``(N, C, H, W)``.

        Note:
            The default :meth:`predict` that uses this method assumes an
            input is a batch of images of type :class:`~torch.Tensor`
            and the output can be anything that is
            :class:`~typing.Iterable`, e.g., a :class:`~torch.Tensor`.

        Args:
            *args: any inputs that can be passed to :attr:`model`.
                Usually, it is just a single input, i.e., a batch of
                images: a :class:`~torch.Tensor` of shape
                ``(N, C, H, W)``. with normalized pixel values between
                0 and 1.

        Returns:
            An iterable of predictions (one for each input). Usually,
            it is a :class:`~torch.Tensor` with the first dimension of
            size ``N``.
        """
        return self.model(*args)

    @override
    @overload
    def predict(
        self,
        image: FilePath | Image.Image | np.ndarray,
        format: Callable[[torch.Tensor], Default] = lambda x: str(x),
    ) -> Default:
        ...

    @override
    @overload
    def predict(
        self,
        image: Collection[FilePath | Image.Image | np.ndarray],
        format: Callable[[torch.Tensor], Default] = lambda x: str(x),
    ) -> list[Default]:
        ...

    @torch.inference_mode()
    @override
    def predict(
        self,
        image: FilePath
        | Image.Image
        | np.ndarray
        | Collection[FilePath | Image.Image | np.ndarray],
        format: Callable[[Any], Default]
        | Callable[[Image.Image, Any], Default] = lambda x: str(x),
        resize: tuple[int, int] | None = (256, 256),
    ) -> Default | list[Default]:
        """Predicts based on the model specified by the child class.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction generated
        by the child class.

        Note:
            This method expects that :meth:`forward` always returns an
            :class:`typing.Iterable` of any type of predictions
            (typically, they would be of type :class:`~torch.Tensor`),
            even if there is only one prediction.

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
            format (typing.Callable[[typing.Any], Default] | (typing.Callable[[Image.Image, typing.Any], Default], optional):
                Format callback. This is a custom function that takes
                a predicted elements from the iterable output of
                :meth:`forward` (elements are usually of type
                :class:`~torch.Tensor`) as input or the original image
                and its prediction as inputs (it will be determined
                automatically which function it is) and outputs a
                formatted prediction of type :attr:`Default`. Defaults
                to ``lambda x: str(x)``.
            resize (tuple[int, int] | None, optional): The size (width,
                height) to resize the image to before passing it through
                the network. If ``None``, the image will not be resized.
                It is recommended to resize it to the size the model was
                trained on, which by default is ``(256, 256)``. Defaults
                to ``None``.

        Returns:
            Default | typing.List[Default]: The
            formatted prediction or a list of formatted predictions if
            multiple images were provided.
        """
        # Get the device from the model and init vars
        device = next(iter(self.parameters())).device
        xs, preds, is_multiple = [], [], True

        # Warning: if the image has shape (3, H, W), it will be interpreted as 3 grayscale images
        if (is_path_type(image) or isinstance(image, Image.Image)) or (
            isinstance(image, np.ndarray)
            and (image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in [1, 3]))
        ):
            # Single image
            image = [image]
            is_multiple = False

        if require_original := (len(inspect.signature(format).parameters) == 2):
            # Init original images
            original_images = []

        for img in image:
            if require_original:
                # Keep track of original
                original_images.append(
                    Image.open(img)
                    if isinstance(img, str)
                    else Image.fromarray(img)
                    if isinstance(img, np.ndarray)
                    else img
                )
            # Load the image and cast to device and append to batch
            xs.append(ImageLoaderMixin.load_image(img, resize=resize).to(device))

        for i, pred in enumerate(self(torch.stack(xs))):
            if require_original:
                # Format prediction with original image
                preds.append(format(original_images[i], pred))
            else:
                # Append formatted prediction
                preds.append(format(pred))

        return preds if is_multiple else preds[0]

    def load_weights(self, path_or_url: str | bool = True):
        """Loads inner :attr:`model` weights.

        Takes a path of a URL to the weights file, or ``True`` to
        construct the URL automatically based on :attr:`model_info` and
        loads the weights into :attr:`model`.

        Note:
            If the weights are already downloaded, they will be loaded
            from the hub cache, which by default is
            ``~/.cache/torch/hub/checkpoints``.

        Warning:
            If the fields in :attr:`model_info` are not recognized,
            e.g., by providing an unrecognized :attr:`kind` or
            :attr:`size` or by initializing with :meth:`from_model`,
            this method will not be able to construct the URL (if
            ``path_or_url`` is ``True``) and will raise a warning.

        Args:
            path_or_url (str | bool, optional): The path or the URL (it
                will be inferred automatically) to the model weights
                (``.pth`` file). It can also be :class:`bool`, in which
                case ``True`` indicates to construct ``URL`` for the
                pre-trained weights and ``False`` does nothing. Defaults
                to ``True``.
        """
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
