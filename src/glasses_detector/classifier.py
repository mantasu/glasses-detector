from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, overload, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_v2_s, shufflenet_v2_x1_0

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
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``anyglasses`` | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``eyeglasses`` | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        | ``sunglasses`` | ``medium`` | TODO                    | TODO                | TODO                     | TODO                    |
        |                +------------+-------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                    | TODO                | TODO                     | TODO                    |
        +----------------+------------+-------------------------+---------------------+--------------------------+-------------------------+

    .. dropdown:: Size Information of the Pre-trained Classifiers
        :icon: info
        :color: info
        :animate: fade-in-slide-down
        :name: Size Information of the Pre-trained Classifiers

        +----------------+--------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | Size           | Architecture                                                                         | Params :math:`\downarrow` | GFLOPs :math:`\downarrow` | Memory :math:`\downarrow` | Filesize :math:`\downarrow` |
        +================+======================================================================================+===========================+===========================+===========================+=============================+
        | ``small``      | :class:`tinyclsnet_v1<.architectures.tiny_binary_classifier.TinyBinaryClassifier>`   | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | ``medium``     | :func:`~torchvision.models.shufflenet_v2_x1_0` :cite:p:`ma2018shufflenet`            | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | ``large``      | :func:`~torchvision.models.efficientnet_v2_s` :cite:p:`tan2021efficientnetv2`        | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+

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
        'not_present'
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
        path/to/dir/img1.jpg,present
        path/to/dir/img2.jpg,not_present
        ...
        >>> clf.process_dir("path/to/dir", "path/to/pred_dir", ext=".txt")
        >>> subprocess.run(["ls", "path/to/pred_dir"])
        img1.txt img2.txt ...

    Args:
        kind (str, optional): The kind of glasses to perform binary
            classification for. Available options are:

            +-------------------+-------------------------------------+
            |                   |                                     |
            +-------------------+-------------------------------------+
            | ``"anyglasses"``  | Any kind glasses/googles/spectacles |
            +-------------------+-------------------------------------+
            | ``"eyeglasses"``  | Transparent eyeglasses              |
            +-------------------+-------------------------------------+
            | ``"sunglasses"``  | Opaque and semi-transparent glasses |
            +-------------------+-------------------------------------+

            Each kind is only responsible for its category, e.g., if
            ``kind`` is set to ``"sunglasses"``, then images with
            transparent eyeglasses will not be identified as positive.
            Defaults to ``"anyglasses"``.
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

            * `Performance of the Pre-trained Classifiers`_: to see the
              results of the pre-trained models for each size depending
              on :attr:`kind`.
            * `Size Information of the Pre-trained Classifiers`_: to see
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
    kind: str = "anyglasses"
    size: str = "medium"
    pretrained: bool | str | None = field(default=True, repr=False)
    task: str = field(default="classification", init=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "small": {"name": "tinyclsnet_v1", "version": "v1.0.0"},
        "medium": {"name": "shufflenet_v2_x1_0", "version": "v1.0.0"},
        "large": {"name": "efficientnet_v2_s", "version": "v1.0.0"},
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
            case "shufflenet_v2_x1_0":
                m = shufflenet_v2_x1_0()
                m.fc = nn.Linear(1024, 1)
            case "efficientnet_v2_s":
                m = efficientnet_v2_s()
                m.classifier = nn.Linear(1280, 1)
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @overload
    def predict(
        self,
        image: FilePath | Image.Image | np.ndarray,
        format: str | dict[bool, Default] | Callable[[torch.Tensor], Default] = "str",
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default:
        ...

    @overload
    def predict(
        self,
        image: Collection[FilePath | Image.Image | np.ndarray],
        format: str | dict[bool, Default] | Callable[[torch.Tensor], Default] = "str",
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
                :class:`lis` of corresponding items of **output type**):

                +------------+---------------------+------------------------------------------------------------+
                | **format** | **output type**     | **prediction mapping**                                     |
                +============+=====================+============================================================+
                | ``"bool"`` | :class:`bool`       | :data:`True` if positive, :data:`False` if negative        |
                +------------+---------------------+------------------------------------------------------------+
                | ``"int"``  | :class:`int`        | ``1`` if positive, ``0`` if negative                       |
                +------------+---------------------+------------------------------------------------------------+
                | ``"str"``  | :class:`str`        | ``"present"`` if positive, ``"not_present"`` if negative   |
                +------------+---------------------+------------------------------------------------------------+
                | ``"logit"``| :class:`float`      | Raw score (real number) of a positive class                |
                +------------+---------------------+------------------------------------------------------------+
                | ``"proba"``| :class:`float`      | Probability (a number between 0 and 1) of a positive class |
                +------------+---------------------+------------------------------------------------------------+

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

        return super().predict(image, format, input_size)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    @override
    @copy_signature(predict)
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
