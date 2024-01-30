from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, overload, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)

from .architectures import TinyBinarySegmenter
from .components.base_model import BaseGlassesModel
from .components.pred_type import Default
from .utils import FilePath, copy_signature


@dataclass
class GlassesSegmenter(BaseGlassesModel):
    r"""**Binary** segmenter for glasses and their parts.

    This class allows to perform binary segmentation of glasses or their
    particular parts, e.g., frames, lenses, legs, shadows, etc.
    Specifically, it allows to generate a mask of the same size as the
    input image where each pixel is mapped to a value indicating whether
    it belongs to the positive category (e.g., glasses) or not.

    ----

    .. dropdown:: Performance of the Pre-trained Segmenters
        :icon: graph
        :color: info
        :animate: fade-in-slide-down
        :name: Performance of the Pre-trained Segmenters

        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        | Kind           | Size       | BCE :math:`\downarrow`  | F1 :math:`\uparrow` | Dice :math:`\uparrow`    | IoU :math:`\uparrow`    |
        +================+============+=========================+=====================+==========================+=========================+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``frames``     | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``full``       | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``legs``       | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``lenses``     | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``shadows``    | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``smart``      | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+

    .. dropdown:: Size Information of the Pre-trained Segmenters
        :icon: info
        :color: info
        :animate: fade-in-slide-down
        :name: Size Information of the Pre-trained Segmenters

        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | Size           | Architecture                                                                                                 | Params :math:`\downarrow` | GFLOPs :math:`\downarrow` | Memory :math:`\downarrow` | Filesize :math:`\downarrow` |
        +================+==============================================================================================================+===========================+===========================+===========================+=============================+
        | ``small``      | :class:`tinysegnet_v1<.architectures.tiny_binary_segmenter.TinyBinarySegmenter>`                             | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | ``medium``     | :func:`~torchvision.models.segmentation.lraspp_mobilenet_v3_large` :cite:p:`howard2019searching`             | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | ``large``      | :func:`~torchvision.models.segmentation.fcn_resnet101` :cite:p:`long2015fully,he2016deep`                    | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+

    Examples
    --------

    Let's instantiate the segmenter with default parameters:

          >>> from glasses_detector import GlassesSegmenter
          >>> seg = GlassesSegmenter()

    First, we can perform a raw prediction on an image expressed as
    either a path, a :class:`PIL Image<PIL.Image.Image>` or a
    :class:`numpy array<numpy.ndarray>`. See :meth:`predict` for more
    details.

        >>> seg(np.random.randint(0, 256, size=(2, 2, 3), dtype=np.uint8), format="bool")
        tensor([[False, False],
                [False, False]])
        >>> type(seg(["path/to/image1.jpg", "path/to/image2.jpg"], format="img")[0])
        <class 'PIL.Image.Image'>

    We can also use a more specific method :meth:`process_file` which
    allows to save the results to a file:

        >>> seg.process_file("path/to/img.jpg", "path/to/pred.jpg", show=True)
        ... # opens a new image window with overlaid mask
        >>> seg.process_file(["img1.jpg", "img2.jpg"], "preds.npy", format="proba")
        >>> np.load("preds.npy").shape
        (2, 256, 256)

    Finally, we can also use :meth:`process_dir` to process all images
    in a directory and save the predictions to a file or a directory:

        >>> seg.process_dir("path/to/dir", "path/to/preds.csv", format="logit")
        >>> subprocess.run(["cat", "path/to/preds.csv"])
        path/to/dir/img1.jpg,-0.234,-1.23,0.123,0.123,1.435,...
        path/to/dir/img2.jpg,0.034,-0.23,2.123,-1.123,0.435,...
        ...
        >>> seg.process_dir("path/to/dir", "path/to/pred_dir", ext=".jpg", format="mask")
        >>> subprocess.run(["ls", "path/to/pred_dir"])
        img1.jpg img2.jpg ...

    Args:
        kind (str, optional): The kind of glasses/parts to perform
            binary segmentation for. Available options are:

            +-------------------+---------------------------------------------------------------------+
            |                   |                                                                     |
            +-------------------+---------------------------------------------------------------------+
            | ``"frames"``      | Frames (including legs) of any glasses                              |
            +-------------------+---------------------------------------------------------------------+
            | ``"full"``        | Frames (including legs) and lenses of any glasses                   |
            +-------------------+---------------------------------------------------------------------+
            | ``"legs"``        | Legs of any glasses                                                 |
            +-------------------+---------------------------------------------------------------------+
            | ``"lenses"``      | Lenses of any glasses                                               |
            +-------------------+---------------------------------------------------------------------+
            | ``"shadows"``     | Cast shadows on the skin of glasses frames only                     |
            +-------------------+---------------------------------------------------------------------+
            | ``"smart"``       | Like ``"full"`` but does not segment lenses if they are transparent |
            +-------------------+---------------------------------------------------------------------+

            Defaults to ``"smart"``.
        size (str, optional): The size of the model to use. Available
            options are:

            +--------------+-------------------------------------------------------------+
            |              |                                                             |
            +--------------+-------------------------------------------------------------+
            | ``"small"``  | Very few parameters but lower accuracy                      |
            +--------------+-------------------------------------------------------------+
            | ``"medium"`` | A balance between the number of parameters and the accuracy |
            +--------------+-------------------------------------------------------------+
            | ``"large"``  | Large number of parameters but higher accuracy              |
            +--------------+-------------------------------------------------------------+

            Please check:

            * `Performance of the Pre-trained Segmenters`_: to see the
              results of the pre-trained models for each size depending
              on :attr:`kind`.
            * `Size Information of the Pre-trained Segmenters`_: to see
              which architecture each size maps to and the details
              about the number of parameters.

            Defaults to ``"medium"``.
        pretrained (bool | str | None, optional): Whether to load
            weights from a custom URL (or a local file if they're
            already downloaded) which will be inferred based on model's
            :attr:`kind` and :attr:`size`. If a string is provided, it
            will be used as a custom path or a URL (determined
            automatically) to the model weights. Defaults to
            :data:`True`.
        device (str | torch.device, optional): Device to cast the model
            (once it is loaded) to. Defaults to ``"cpu"``.
    """

    task: str = field(default="segmentation", init=False)
    kind: str = "smart"
    size: str = "medium"
    pretrained: bool | str | None = field(default=True, repr=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "small": {"name": "tinysegnet_v1", "version": "v1.0.0"},
        "medium": {"name": "lraspp_mobilenet_v3_large", "version": "v1.0.0"},
        "large": {"name": "deeplabv3_resnet50", "version": "v1.0.0"},
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
                m = lraspp_mobilenet_v3_large(num_classes=1)
                # m.classifier = LRASPPHead(40, 960, 1, 128)
            case "fcn_resnet101":
                m = fcn_resnet101()
                m.classifier[-1] = nn.Conv2d(512, 1, 1)
                m.aux_classifier = None
            case "deeplabv3_resnet50":
                m = deeplabv3_resnet50(num_classes=1)
                # m.classifier[-1] = nn.Conv2d(256, 1, 1)
                m.aux_classifier = None
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @staticmethod
    def draw_mask(
        img: Image.Image,
        mask: torch.Tensor | np.ndarray | Image.Image,
        color: str | tuple[int, int, int] = "red",
        alpha: float = 0.5,
    ) -> Image.Image:
        """Draws a mask over an image.

        Takes the original image and a mask and overlays the mask over
        the image with a specified color and transparency.

        Args:
            img (PIL.Image.Image): The image to draw the mask over.
            mask (torch.Tensor | numpy.ndarray | PIL.Image.Image): The
                mask to draw over the image. It should be of the same
                size as the image. If it is a :class:`torch.Tensor` or
                a :class:`numpy.ndarray`, it should be of shape
                ``(H, W)``, contain values between ``0`` and ``255``,
                and be of type :data:`torch.uint8` or
                :data:`numpy.uint8` respectively.
            color (str | tuple[int, int, int], optional): The color of
                the overlaid mask. Defaults to ``"red"``.
            alpha (float, optional): The transparency of the mask.
                Defaults to ``0.5``.

        Returns:
            PIL.Image.Image: The RGB image with the mask drawn over it.
        """
        if isinstance(mask, torch.Tensor):
            # Convert to numpy array
            mask = mask.numpy(force=True)

        if isinstance(mask, np.ndarray):
            # Convert to PIL Image
            mask = Image.fromarray(mask)

        # Overlay the mask on the image
        mask = mask.convert("L").point(lambda p: p * alpha)
        colored_mask = Image.new("RGB", mask.size, color)
        img.paste(colored_mask, mask=mask)

        return img

    @overload
    def predict(
        self,
        image: FilePath | Image.Image | np.ndarray,
        format: str
        | dict[bool, Default]
        | Callable[[torch.Tensor], Default]
        | Callable[[Image.Image, torch.Tensor], Default] = "img",
        output_size: tuple[int, int] | None = None,
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default:
        ...

    @overload
    def predict(
        self,
        image: Collection[FilePath | Image.Image | np.ndarray],
        format: str
        | dict[bool, Default]
        | Callable[[torch.Tensor], Default]
        | Callable[[Image.Image, torch.Tensor], Default] = "img",
        output_size: tuple[int, int] | None = None,
        input_size: tuple[int, int] | None = (256, 256),
    ) -> list[Default]:
        ...

    @override
    def predict(
        self,
        image: FilePath
        | Image.Image
        | np.ndarray
        | Collection[FilePath | Image.Image | np.ndarray],
        format: str
        | dict[bool, Default]
        | Callable[[torch.Tensor], Default]
        | Callable[[Image.Image, torch.Tensor], Default] = "img",
        output_size: tuple[int, int] | None = None,
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default | list[Default]:
        """Predicts which pixels in the image are positive.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction indicating
        the semantic mask of the present glasses or their specific
        part(-s). The format of the prediction, i.e., the prediction
        type is :data:`~glasses_detector.components.pred_type.Default`
        type which corresponds to :attr:`~.PredType.DEFAULT`.

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
                :class:`~PIL.Image.Image` or as :class:`~numpy.ndarray`.
                Note that the image should have values between 0 and 255
                and be of RGB format. Normalization is not needed as the
                channels will be automatically normalized before passing
                through the network.
            format (str | dict[bool, Default] | typing.Callable[[torch.Tensor], Default], optional):
                The string specifying the way to map the predictions
                (pixel scores) to outputs (masks) of specific format.
                These are the following options (if ``image`` is a
                :class:`~typing.Collection`, then the output will be a
                :class:`list` of corresponding items of **output type**):

                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
                | **format**    | **output type**                                                         | **prediction mapping**                                                                    |
                +===============+=========================================================================+===========================================================================================+
                | ``"bool"``    | :class:`torch.Tensor` of type :data:`torch.bool` of shape ``(H, W)``    | :data:`True` for positive pixels, :data:`False` for negative pixels                       |
                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
                | ``"int"``     | :class:`torch.Tensor` of type :data:`torch.int64` of shape ``(H, W)``   | ``1`` for positive pixels, ``0`` for negative pixels                                      |
                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
                | ``"logit"``   | :class:`torch.Tensor` of type :data:`torch.float32` of shape ``(H, W)`` | Raw score (real number) of being a positive pixel                                         |
                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
                | ``"proba"``   | :class:`torch.Tensor` of type :data:`torch.float32` of shape ``(H, W)`` | Probability (a number between 0 and 1) of being a positive pixel                          |
                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
                | ``"mask"``    | :class:`PIL.Image.Image` of mode ``"L"`` (grayscale)                    | *White* for positive pixels, *black* for negative pixels                                  |
                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
                | ``"img"``     | :class:`PIL.Image.Image` of mode ``"RGB"`` (RGB)                        | The original image with the mask overlaid on it using default values in :meth:`draw_mask` |
                +---------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+

                It is also possible to provide a a dictionary with 2
                keys: :data:`True` and :data:`False`, each mapping to
                values corresponding to what to output if the predicted
                pixel is positive or negative. Further, a custom
                callback function is also possible that specifies how to
                map the original image (:class:`~PIL.Image.Image`) and
                the mask prediction (:class:`~torch.Tensor` of type
                :data:`torch.float32` of shape ``(H, W)``), or just the
                predictions to a formatted
                :data:`~glasses_detector.components.pred_type.Default`
                output. Defaults to "img".
            output_size (tuple[int, int] | None, optional): The size
                (width, height), or ``(W, H)``, to resize the prediction
                (output mask) to. If :data:`None`, the prediction will
                have the same size as the input image. Defaults to
                :data:`None`.
            input_size (tuple[int, int] | None, optional): The size
                (width, height), or ``(W, H)``, to resize the image to
                before passing it through the network. If :data:`None`,
                the image will not be resized. It is recommended to
                resize it to the size the model was trained on, which by
                default is ``(256, 256)``. Defaults to ``(256, 256)``.


        Returns:
            Default | list[Default]: The formatted prediction or a list
            of formatted predictions if multiple images were provided.

        Raises:
            ValueError: If the specified ``format`` as a string is
                not recognized.
        """
        if isinstance(f := format, str):
            # Update mask type
            match format:
                case "bool":
                    format = {True: True, False: False}
                case "int":
                    format = {True: 1, False: 0}
                case "logit":
                    format = lambda x: x
                case "proba":
                    format = lambda x: x.sigmoid()
                case "mask":
                    format = (
                        lambda img, x: Image.fromarray(
                            ((x > 0) * 255).to(torch.uint8),
                            mode="L",
                        ).resize(output_size if output_size else img.size),
                    )
                case "img":
                    format = lambda img, x: self.draw_mask(
                        img.resize(output_size) if output_size else img,
                        Image.fromarray(
                            ((x > 0) * 255).to(torch.uint8),
                            mode="L",
                        ).resize(output_size if output_size else img.size),
                    )
                case _:
                    raise ValueError(f"Invalid format: {format}")

        if isinstance(d := format, dict):
            # If mask type was specified as dictionary
            format = lambda x: torch.where((x > 0), d[True], d[False])

        if isinstance(f, str) and f not in {"mask", "img"}:
            # Apply torch transform if not mask or img
            format = lambda img, x: transforms.Resize(
                output_size if output_size else img.size
            )(format(x))

        return super().predict(image, format, input_size)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]

    @override
    @copy_signature(predict)
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
