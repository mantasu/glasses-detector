from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, overload, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import regnet_x_3_2gf, shufflenet_v2_x1_0
from torchvision.transforms.v2.functional import to_pil_image

from .architectures import TinyBinaryClassifier
from .components.base_model import BaseGlassesModel
from .components.pred_type import Default
from .utils import FilePath, copy_signature


@dataclass
class GlassesClassifier(BaseGlassesModel):
    r"""**Binary** classifier to check if glasses are present.

    This class allows to perform binary classification for images with
    glasses, i.e., determines whether or not the glasses are present in
    the image (primarily focus is on whether or not eyeglasses are worn
    by a person). It is possible to specify a particular kind of glasses
    to focus on, e.g., sunglasses.

    ----

    .. dropdown:: Performance of the Pre-trained Classifiers
        :icon: graph
        :color: info
        :animate: fade-in-slide-down
        :name: Performance of the Pre-trained Classifiers

        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        | Kind           | Size       | BCE :math:`\downarrow`  | F1 :math:`\uparrow` | ROC-AUC :math:`\uparrow` | PR-AUC :math:`\uparrow` |
        +================+============+=========================+=====================+==========================+=========================+
        |                | ``small``  | 0.2160                  | 0.9431              | 0.9866                   | 0.9757                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``anyglasses`` | ``medium`` | 0.1539                  | 0.9693              | 0.9933                   | 0.9895                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TBA                     | TBA                 | TBA                      | TBA                     |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | 0.2210                  | 0.9082              | 0.9808                   | 0.9590                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``eyeglasses`` | ``medium`` | 0.1342                  | 0.9502              | 0.9922                   | 0.9810                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TBA                     | 0.9490              | TBA                      | TBA                     |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | 0.2331                  | 0.8827              | 0.9852                   | 0.9551                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``sunglasses`` | ``medium`` | 0.1794                  | 0.9311              | 0.9912                   | 0.9739                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TBA                     | TBA                 | TBA                      | TBA                     |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | 0.3956                  | 0.8158              | 0.9326                   | 0.9075                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``shadows``    | ``medium`` | 0.3314                  | 0.8468              | 0.9537                   | 0.9354                  |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TBA                     | TBA                 | TBA                      | TBA                     |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+

    .. dropdown:: Size Information of the Pre-trained Classifiers
        :icon: info
        :color: info
        :animate: fade-in-slide-down
        :name: Size Information of the Pre-trained Classifiers

        +----------------+-----------------------------------------------------------------------------------------+---------------------------+---------------------------+--------------------------------+----------------------------------+
        | Size           | Architecture                                                                            | Params                    | GFLOPs                    | Memory (MB)                    | Filesize (MB)                    |
        +================+=========================================================================================+===========================+===========================+================================+==================================+
        | ``small``      | :class:`Tiny Classifier <.architectures.tiny_binary_classifier.TinyBinaryClassifier>`   | 0.03M                     | 0.001                     | 23.38                          | 0.12                             |
        +----------------+-----------------------------------------------------------------------------------------+---------------------------+---------------------------+--------------------------------+----------------------------------+
        | ``medium``     | :func:`ShuffleNet <torchvision.models.shufflenet_v2_x1_0>` :cite:p:`ma2018shufflenet`   | 1.25M                     | 0.19                      | 84.03                          | 4.95                             |
        +----------------+-----------------------------------------------------------------------------------------+---------------------------+---------------------------+--------------------------------+----------------------------------+
        | ``large``      | TBA                                                                                     | TBA                       | TBA                       | TBA                            | TBA                              |
        +----------------+-----------------------------------------------------------------------------------------+---------------------------+---------------------------+--------------------------------+----------------------------------+

    Examples
    --------

    Let's instantiate the classifier with default parameters:

    .. code-block:: python

        >>> from glasses_detector import GlassesClassifier
        >>> clf = GlassesClassifier()

    First, we can perform a raw prediction on an image expressed as
    either a path, a :class:`PIL Image<PIL.Image.Image>` or a
    :class:`numpy array<numpy.ndarray>`. See :meth:`predict` for more
    details.

    .. code-block:: python

        >>> clf(np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8))
        'absent'
        >>> clf(["path/to/image1.jpg", "path/to/image2.jpg"], format="bool")
        [True, False]

    We can also use a more specific method :meth:`process_file` which
    allows to save the results to a file:

    .. code-block:: python

        >>> clf.process_file("path/to/img.jpg", "path/to/pred.txt", show=True)
        'present'
        >>> clf.process_file(["img1.jpg", "img2.jpg"], "preds.npy", format="proba")
        >>> np.load("preds.npy")
        array([0.96, 0.81562], dtype=float32)

    Finally, we can also use :meth:`process_dir` to process all images
    in a directory and save the predictions to a file or a directory:

    .. code-block:: python

        >>> clf.process_dir("path/to/dir", "path/to/preds.csv", format="str")
        >>> subprocess.run(["cat", "path/to/preds.csv"])
        img1.jpg,present
        img2.jpg,absent
        ...
        >>> clf.process_dir("path/to/dir", "path/to/pred_dir", ext=".txt")
        >>> subprocess.run(["ls", "path/to/pred_dir"])
        img1.txt img2.txt ...

    Args:
        kind (str, optional): The kind of glasses to perform binary
            classification for. Available options are:

            +-------------------+----------------------------------------+
            |                   |                                        |
            +-------------------+----------------------------------------+
            | ``"anyglasses"``  | Any kind glasses/googles/spectacles    |
            +-------------------+----------------------------------------+
            | ``"eyeglasses"``  | Transparent eyeglasses                 |
            +-------------------+----------------------------------------+
            | ``"sunglasses"``  | Opaque and semi-transparent glasses    |
            +-------------------+----------------------------------------+
            | ``"shadows"``     | Visible cast shadows of glasses frames |
            +-------------------+----------------------------------------+

            Each kind is only responsible for its category, e.g., if
            ``kind`` is set to ``"sunglasses"``, then images with
            transparent eyeglasses will not be identified as positive.
            Defaults to ``"anyglasses"``.
        size (str, optional): The size of the model to use (check
            :attr:`.ALLOWED_SIZE_ALIASES` for size aliases). Available
            options are:

            +-------------------------+-------------------------------------------------------------+
            |                         |                                                             |
            +-------------------------+-------------------------------------------------------------+
            | ``"small"`` or ``"s"``  | Very few parameters but lower accuracy                      |
            +-------------------------+-------------------------------------------------------------+
            | ``"medium"`` or ``"m"`` | A balance between the number of parameters and the accuracy |
            +-------------------------+-------------------------------------------------------------+
            | ``"large"`` or ``"l"``  | Large number of parameters but higher accuracy              |
            +-------------------------+-------------------------------------------------------------+

            Please check:

            * `Performance of the Pre-trained Classifiers`_: to see the
              results of the pre-trained models for each size depending
              on :attr:`kind`.
            * `Size Information of the Pre-trained Classifiers`_: to see
              which architecture each size maps to and the details
              about the number of parameters.

            Defaults to ``"medium"``.
        weights (bool | str | None, optional): Whether to load weights
            from a custom URL (or a local file if they're already
            downloaded) which will be inferred based on model's
            :attr:`kind` and :attr:`size`. If a string is provided, it
            will be used as a custom path or a URL (determined
            automatically) to the model weights. Defaults to
            :data:`True`.
        device (str | torch.device | None, optional): Device to cast the
            model to (once it is loaded). If specified as :data:`None`,
            it will be automatically checked if
            `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ or
            `MPS <https://developer.apple.com/documentation/metalperformanceshaders>`_
            is supported. Defaults to :data:`None`.
    """

    kind: str = "anyglasses"
    size: str = "medium"
    weights: bool | str | None = True
    task: str = field(default="classification", init=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "small": {"name": "tinyclsnet_v1", "version": "v1.0.0"},
        "medium": {"name": "shufflenet_v2_x1_0", "version": "v1.0.0"},
        "large": {"name": "regnet_x_3_2gf", "version": "v1.1.0"},
    }

    DEFAULT_KIND_MAP: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "anyglasses": DEFAULT_SIZE_MAP,
        "eyeglasses": DEFAULT_SIZE_MAP,
        "sunglasses": DEFAULT_SIZE_MAP,
        "shadows": DEFAULT_SIZE_MAP,
    }

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinyclsnet_v1":
                m = TinyBinaryClassifier()
            case "shufflenet_v2_x1_0":
                m = shufflenet_v2_x1_0()
                m.fc = nn.Linear(1024, 1)
            case "regnet_x_3_2gf":
                m = regnet_x_3_2gf(num_classes=1)
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @staticmethod
    def draw_label(
        image: Image.Image | np.ndarray | torch.Tensor,
        label: str,
        font: str | None = None,
        font_size: int = 15,
    ) -> Image.Image:
        """Draws a label on the image.

        This method takes an image and a label and draws a caption box
        with the given text which is appended to the bottom of the
        image.

        Args:
            image (PIL.Image.Image | numpy.ndarray | torch.Tensor): The
                original image. It can be either a *PIL*
                :class:`~PIL.Image.Image`, a *numpy*
                :class:`~numpy.ndarray` of shape ``(H, W, 3)`` or
                ``(H, W)`` and type :attr:`~numpy.uint8` or a *torch*
                :class:`~torch.Tensor` of shape ``(3, H, W)`` or
                ``(H, W)`` and type :attr:`~torch.uint8`.
            label (str): The label to write in the caption box that is
                appended to the bottom of the image.
            font (str | None, optional): A filename containing a
                *TrueType* font. If the file is not found in this
                filename, the loader may also search in other
                directories, such as the ``fonts/`` directory on Windows
                or ``/Library/Fonts/``, ``/System/Library/Fonts/`` and
                ``~/Library/Fonts/`` on macOS. Defaults to :data:`None`.
            font_size (int, optional): The requested font size in
                points used when calling
                :meth:`~PIL.ImageFont.truetype`. Defaults to ``15``.

        Returns:
            PIL.Image.Image: The extended original image in height with
            the caption box appended to the bottom.
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL image
            image = to_pil_image(image)

        if isinstance(image, np.ndarray):
            # Convert ndarray to PIL image
            image = Image.fromarray(image)

        if font is None:
            # Use the default system font
            font = ImageFont.load_default()
        else:
            # Use the specified font with the specified font size
            font = ImageFont.truetype(font=font, size=font_size)

        # Create a new image with extra space for the title
        new_image = Image.new("RGB", (image.width, image.height + 2 * font_size))
        new_image.paste(image)

        # Draw the title
        draw = ImageDraw.Draw(new_image)
        _, _, text_width, text_height = font.getbbox(label)

        x = (new_image.width - text_width) / 2
        y = image.height + (font_size * 2 - text_height) / 2
        draw.text((x, y), label, font=font, fill="white")

        return new_image

    @overload
    def predict(
        self,
        image: FilePath | Image.Image | np.ndarray,
        format: str | dict[bool, Default] | Callable[[torch.Tensor], Default] = "str",
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default: ...

    @overload
    def predict(
        self,
        image: Collection[FilePath | Image.Image | np.ndarray],
        format: str | dict[bool, Default] | Callable[[torch.Tensor], Default] = "str",
        input_size: tuple[int, int] | None = (256, 256),
    ) -> list[Default]: ...

    @override
    def predict(
        self,
        image: (
            FilePath
            | Image.Image
            | np.ndarray
            | Collection[FilePath | Image.Image | np.ndarray]
        ),
        format: str | dict[bool, Default] | Callable[[torch.Tensor], Default] = "str",
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default | list[Default]:
        """Predicts whether the positive class is present.

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction for each
        image indicating whether it belongs to a positive class, e.g.,
        *"anyglasses"*, or not. The format of the prediction,
        i.e., the prediction type is
        :data:`~glasses_detector.components.pred_type.Default` type
        which corresponds to :attr:`~.PredType.DEFAULT`.

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
                The string specifying the way to map the predictions to
                labels. These are the following options (if ``image`` is
                a :class:`~typing.Collection`, then the output will be a
                :class:`list` of corresponding items of **output
                type**):

                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
                | **format**  | **output type**          | **prediction mapping**                                                                                                               |
                +=============+==========================+======================================================================================================================================+
                | ``"bool"``  | :class:`bool`            | :data:`True` if positive, :data:`False` if negative                                                                                  |
                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
                | ``"int"``   | :class:`int`             | ``1`` if positive, ``0`` if negative                                                                                                 |
                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
                | ``"str"``   | :class:`str`             | ``"present"`` if positive, ``"absent"`` if negative                                                                                  |
                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
                | ``"logit"`` | :class:`float`           | Raw score (real number) of a positive class                                                                                          |
                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
                | ``"proba"`` | :class:`float`           | Probability (a number between 0 and 1) of a positive class                                                                           |
                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
                | ``"img"``   | :class:`PIL.Image.Image` | The original image with an inserted title using default values in :meth:`.draw_label` (caption text corresponds to ``"str"`` format) |
                +-------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------+

                It is also possible to provide a dictionary with 2 keys:
                :data:`True` and :data:`False`, each mapping to values
                corresponding to what to output if the predicted label
                is positive or negative. Further, a custom callback
                function is also possible that specifies how to map a
                raw :class:`torch.Tensor` score of type
                ``torch.float32`` of shape ``(1,)`` to a label. Defaults
                to ``"str"``.
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
        if isinstance(format, str):
            # Check format
            match format:
                case "bool":
                    format = {True: True, False: False}
                case "int":
                    format = {True: 1, False: 0}
                case "str":
                    format = {True: "present", False: "absent"}
                case "logit":
                    format = lambda x: x.item()
                case "proba":
                    format = lambda x: x.sigmoid().item()
                case "img":
                    format = lambda img, x: self.draw_label(
                        img, "present" if (x > 0).item() else "absent"
                    )
                case _:
                    raise ValueError(f"Invalid format: {format}")

        if isinstance(d := format, dict):
            # If the format was specified as dictionary
            format = lambda x: d[(x > 0).item()]

        return super().predict(image, format, input_size)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    @override
    @copy_signature(predict)
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
