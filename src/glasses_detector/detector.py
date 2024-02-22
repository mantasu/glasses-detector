from dataclasses import dataclass, field
from typing import Callable, ClassVar, Collection, overload, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.v2.functional import to_image, to_pil_image
from torchvision.utils import draw_bounding_boxes

from .architectures import TinyBinaryDetector
from .components import BaseGlassesModel
from .components.pred_type import Default
from .utils import FilePath, copy_signature


@dataclass
class GlassesDetector(BaseGlassesModel):
    r"""**Binary** detector to check where the glasses are in the image.

    This class allows to perform binary glasses and eye-area detection.
    By *binary*, it means only a single class is detected. It is
    possible to specify a particular kind of detection to perform, e.g.,
    standalone glasses, worn glasses, or just the eye area.

    Important:
        The detector cannot determine whether or not the glasses are
        present in the image, i.e., it will always try to predict a
        bounding box. If you are not sure whether the glasses may be
        present, please additionally use
        :class:`~glasses_detector.classifier.GlassesClassifier`.

    Warning:
        The pre-trained models are trained on datasets that contain just
        a single bounding box per image. For this reason, the number of
        predicted bounding boxes will always be 1. If you want to detect
        multiple objects in the image, please train custom models on
        custom datasets or share those datasets with me :).

    Note:
        If you want to use a custom inner :attr:`model`, e.g., by
        instantiating through :meth:`from_model`, please ensure that
        during inference in evaluation mode it outputs a list of
        dictionaries (one for each image in the batch) with at least
        one key being ``"boxes"`` which corresponds to the bounding
        boxes of the detected objects.

    ----

    .. dropdown:: Performance of the Pre-trained Detectors
        :icon: graph
        :color: info
        :animate: fade-in-slide-down
        :name: Performance of the Pre-trained Detectors

        +----------------+------------+--------------------------+---------------------+--------------------------+-------------------------+
        | Kind           | Size       | MSLE :math:`\downarrow`  | F1 :math:`\uparrow` | R2 :math:`\uparrow`      | IoU :math:`\uparrow`    |
        +================+============+==========================+=====================+==========================+=========================+
        |                | ``small``  | TODO                     | TODO                | TODO                     | TODO                    |
        |                +------------+--------------------------+---------------------+--------------------------+-------------------------+
        | ``eyes``       | ``medium`` | TODO                     | TODO                | TODO                     | TODO                    |
        |                +------------+--------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                     | TODO                | TODO                     | TODO                    |
        +----------------+------------+--------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                     | TODO                | TODO                     | TODO                    |
        |                +------------+--------------------------+---------------------+--------------------------+-------------------------+
        | ``solo``       | ``medium`` | TODO                     | TODO                | TODO                     | TODO                    |
        |                +------------+--------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                     | TODO                | TODO                     | TODO                    |
        +----------------+------------+--------------------------+---------------------+--------------------------+-------------------------+
        |                | ``small``  | TODO                     | TODO                | TODO                     | TODO                    |
        |                +------------+--------------------------+---------------------+--------------------------+-------------------------+
        | ``worn``       | ``medium`` | TODO                     | TODO                | TODO                     | TODO                    |
        |                +------------+--------------------------+---------------------+--------------------------+-------------------------+
        |                | ``large``  | TODO                     | TODO                | TODO                     | TODO                    |
        +----------------+------------+--------------------------+---------------------+--------------------------+-------------------------+

        **NB**: **F1 score** is useless because there is only one class,
        but is still here to emphasize this fact. Not even background is
        considered as a class - bbox prediction will always happen.

    .. dropdown:: Size Information of the Pre-trained Detectors
        :icon: info
        :color: info
        :animate: fade-in-slide-down
        :name: Size Information of the Pre-trained Detectors

        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | Size           | Architecture                                                                                                 | Params :math:`\downarrow` | GFLOPs :math:`\downarrow` | Memory :math:`\downarrow` | Filesize :math:`\downarrow` |
        +================+==============================================================================================================+===========================+===========================+===========================+=============================+
        | ``small``      | :class:`tinydetnet_v1<.architectures.tiny_binary_detector.TinyBinaryDetector>`                               | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | ``medium``     | :func:`~torchvision.models.detection.ssdlite320_mobilenet_v3_large` :cite:p:`liu2016ssd,howard2019searching` | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+
        | ``large``      | :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` :cite:p:`ren2015faster,li2021benchmarking`  | TODO                      | TODO                      | TODO                      | TODO                        |
        +----------------+--------------------------------------------------------------------------------------------------------------+---------------------------+---------------------------+---------------------------+-----------------------------+

    Examples
    --------

    Let's instantiate the detector with default parameters:

    .. code-block:: python

          >>> from glasses_detector import GlassesDetector
          >>> det = GlassesDetector()

    First, we can perform a raw prediction on an image expressed as
    either a path, a :class:`PIL Image<PIL.Image.Image>` or a
    :class:`numpy array<numpy.ndarray>`. See :meth:`predict` for more
    details.

    .. code-block:: python

        >>> det(np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8), format="int")
        [[0, 0, 1, 1]]
        >>> det(["path/to/image1.jpg", "path/to/image2.jpg"], format="str")
        'BBoxes: 12 34 56 78; 90 12 34 56'

    We can also use a more specific method :meth:`process_file` which
    allows to save the results to a file:

    .. code-block:: python

        >>> det.process_file("path/to/img.jpg", "path/to/pred.jpg", show=True)
        ... # opens a new image window with drawn bboxes
        >>> det.process_file(["img1.jpg", "img2.jpg"], "preds.npy", format="bool")
        >>> np.load("preds.npy").shape
        (2, 256, 256)

    Finally, we can also use :meth:`process_dir` to process all images
    in a directory and save the predictions to a file or a directory:

    .. code-block:: python

        >>> det.process_dir("path/to/dir", "path/to/preds.json", format="float")
        >>> subprocess.run(["cat", "path/to/preds.json"])
        {
            "path/to/dir/img1.jpg": [[0.1, 0.2, 0.3, 0.4]],
            "path/to/dir/img2.jpg": [[0.5, 0.6, 0.7, 0.8], [0.2, 0.8, 0.4, 0.9]],
            ...
        }
        >>> det.process_dir("path/to/dir", "path/to/pred_dir", ext=".jpg")
        >>> subprocess.run(["ls", "path/to/pred_dir"])
        img1.jpg img2.jpg ...

    Args:
        kind (str, optional): The kind of objects to perform the
            detection for. Available options are:

            +-------------------+-------------------------------------+
            |                   |                                     |
            +-------------------+-------------------------------------+
            | ``"eyes"``        | No glasses, just the eye area       |
            +-------------------+-------------------------------------+
            | ``"solo"``        | Any standalone glasses in the wild  |
            +-------------------+-------------------------------------+
            | ``"worn"``        | Any glasses that are worn by people |
            +-------------------+-------------------------------------+

            Categories are not very strict, for example, ``"worn"`` may
            also detect glasses on the table. Defaults to ``"worn"``.
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

            * `Performance of the Pre-trained Detectors`_: to see the
              results of the pre-trained models for each size depending
              on :attr:`kind`.
            * `Size Information of the Pre-trained Detectors`_: to see
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
        device (str | torch.device, optional): Device to cast the model
            (once it is loaded) to. Defaults to ``"cpu"``.
    """

    kind: str = "worn"
    size: str = "medium"
    weights: bool | str | None = field(default=True, repr=False)
    task: str = field(default="detection", init=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, int]] = {
        "small": {"name": "tinydetnet_v1", "version": "v1.0.0"},
        "medium": {"name": "ssdlite320_mobilenet_v3_large", "version": "v1.0.0"},
        "large": {"name": "fasterrcnn_resnet50_fpn_v2", "version": "v1.0.0"},
    }

    DEFAULT_KIND_MAP: ClassVar[dict[str, str]] = {
        "eyes": DEFAULT_SIZE_MAP,
        "solo": DEFAULT_SIZE_MAP,
        "worn": DEFAULT_SIZE_MAP,
    }

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinydetnet_v1":
                m = TinyBinaryDetector()
            case "ssdlite320_mobilenet_v3_large":
                m = ssdlite320_mobilenet_v3_large(
                    num_classes=2,
                    detections_per_img=1,
                    topk_candidates=10,
                )
            case "fasterrcnn_resnet50_fpn_v2":
                m = fasterrcnn_resnet50_fpn_v2(
                    num_classes=2,
                    box_detections_per_img=1,
                    box_batch_size_per_image=10,
                )

            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @staticmethod
    def draw_boxes(
        image: Image.Image | np.ndarray | torch.Tensor,
        boxes: list[list[int | float]] | np.ndarray | torch.Tensor,
        labels: list[str] | None = None,
        colors: (
            str | tuple[int, int, int] | list[str | tuple[int, int, int]] | None
        ) = "red",
        fill: bool = False,
        width: int = 3,
        font: str | None = None,
        font_size: int | None = None,
    ) -> Image.Image:
        """Draws bounding boxes on the image.

        Takes the original image and the bounding boxes and draws the
        them on the image. Optionally, the labels can be provided to
        write the label next to the bounding box.

        See Also:

            * :func:`~torchvision.utils.draw_bounding_boxes` for more
              details about how the bounding boxes are drawn.
            * :func:`~torchvision.transforms.v2.functional.to_image` for
              more details about the expected formats if the input
              image is of type :class:`PIL.Image.Image` or
              :class:`numpy.ndarray`.

        Args:
            image (PIL.Image.Image | numpy.ndarray | torch.Tensor): The
                original image. It can be either a *PIL*
                :class:`~PIL.Image.Image`, a *numpy*
                :class:`~numpy.ndarray` of shape ``(H, W, 3)`` or
                ``(H, W)`` and type :attr:`~numpy.uint8` or a *torch*
                :class:`~torch.Tensor` of shape ``(3, H, W)`` or
                ``(H, W)`` and type :attr:`~torch.uint8`.
            boxes (list[list[int | float]] | numpy.ndarray | torch.Tensor):
                The bounding boxes to draw. The expected shape is
                ``(N, 4)`` where ``N`` is the number of bounding boxes
                and the last dimension corresponds to the coordinates of
                the bounding box in the following order: ``x_min``,
                ``y_min``, ``x_max``, ``y_max``.
            labels (list[str] | None, optional): The labels
                corresponding to ``N`` bounding boxes. If :data:`None`,
                no labels will be written next to the drawn bounding
                boxes. Defaults to :data:`None`.
            colors (list[str | tuple[int, int, int]] | str | tuple[int, int, int] | None, optional):
                List containing the colors of the boxes or single color
                for all boxes. The color can be represented as PIL
                strings e.g. "red" or "#FF00FF", or as RGB tuples e.g.
                ``(240, 10, 157)``. If :data:`None`, random colors are
                generated for boxes. Defaults to ``"red"``.
            fill (bool, optional): If :data:`True`, fills the bounding
                box with the specified color. Defaults to :data:`False`.
            width (int, optional): Width of bounding box used when
                calling :meth:`~PIL.ImageDraw.rectangle`. Defaults to
                ``3``.
            font (str | None, optional): A filename containing a
                *TrueType* font. If the file is not found in this
                filename, the loader may also search in other
                directories, such as the ``fonts/`` directory on Windows
                or ``/Library/Fonts/``, ``/System/Library/Fonts/`` and
                ``~/Library/Fonts/`` on macOS. Defaults to :data:`None`.
            font_size (int | None, optional): The requested font size in
                points used when calling
                :meth:`~PIL.ImageFont.truetype`. Defaults to
                :data:`None`.

        Returns:
            PIL.Image.Image: The image with bounding boxes drawn on it.
        """
        if isinstance(image, np.ndarray):
            # TODO: https://github.com/pytorch/vision/issues/8255
            image = np.atleast_3d(image)

        if (image := to_image(image)).ndim == 2:
            # Add a channel dimension
            image = image.unsqueeze(0)

        if not isinstance(boxes, torch.Tensor):
            # Convert bboxes to torch Tensor
            boxes = torch.tensor(boxes, dtype=torch.float32)

        # Draw the bounding boxes on the image
        new_image = draw_bounding_boxes(
            image=image,
            boxes=boxes,
            labels=labels,
            colors=colors,
            fill=fill,
            width=width,
            font=font,
            font_size=font_size,
        )

        return to_pil_image(new_image)

    @overload
    def predict(
        self,
        image: FilePath | Image.Image | np.ndarray,
        format: (
            str
            | Callable[[torch.Tensor], Default]
            | Callable[[Image.Image, torch.Tensor], Default]
        ) = "img",
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default: ...

    @overload
    def predict(
        self,
        image: Collection[FilePath | Image.Image | np.ndarray],
        format: (
            str
            | Callable[[torch.Tensor], Default]
            | Callable[[Image.Image, torch.Tensor], Default]
        ) = "img",
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
        format: (
            Callable[[torch.Tensor], Default]
            | Callable[[Image.Image, torch.Tensor], Default]
        ) = "img",
        output_size: tuple[int, int] | None = None,
        input_size: tuple[int, int] | None = (256, 256),
    ) -> Default | list[Default]:
        """Predicts the bounding box(-es).

        Takes a path or multiple paths to image files or the loaded
        images themselves and outputs a formatted prediction for each
        image indicating the location of the object (typically,
        glasses). The format of the prediction, i.e., the prediction
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
            format (str | dict[bool, Default] | typing.Callable[[torch.Tensor], Default] | typing.Callable[[PIL.Image.Image, torch.Tensor], Default], optional):
                The string specifying the way to map the predictions to
                outputs of specific format. These are the following
                options (if ``image`` is a :class:`~typing.Collection`,
                then the output will be a :class:`list` of corresponding
                items of **output type**):

                +------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
                | **format** | **output type**                                                        | **prediction mapping**                                                                                                                                     |
                +============+========================================================================+============================================================================================================================================================+
                | ``"bool"`` | :class:`numpy.ndarray` of type :class:`numpy.bool` of shape ``(H, W)`` | A :class:`numpy array<numpy.ndarray>` of shape ``(H, W)`` (original image size) with :data:`True` values for pixels that fall in any of the bounding boxes |
                +------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
                | ``"int"``  | :class:`list` of :class:`list` of :class:`int`                         | Bounding boxes with integer coordinates w.r.t. the original ``image`` size: ``[[x_min, y_min, x_max, y_max], ...]``                                        |
                +------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
                | ``"float"``| :class:`list` of :class:`list` of :class:`float`                       | Bounding boxes with float coordinates normalized between 0 and 1: ``[[x_min, y_min, x_max, y_max], ...]``                                                  |
                +------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
                | ``"str"``  | :class:`str`                                                           | A string of the form ``"BBoxes: x_min y_min x_max y_max; ..."``                                                                                            |
                +------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
                | ``"img"``  | :class:`PIL.Image.Image`                                               | The original image with bounding boxes drawn on it using default values in :meth:`draw_rects`                                                              |
                +------------+------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+

                A custom callback function is also possible that
                specifies how to map the original image
                (:class:`~PIL.Image.Image`) and the bounding box
                predictions (:class:`~torch.Tensor` of type
                :data:`torch.float32` of shape ``(K, 4)`` with ``K``
                being the number of detected bboxes), or just the
                predictions to a formatted
                :data:`~glasses_detector.components.pred_type.Default`
                output. Defaults to ``"img"``.
            output_size (tuple[int, int] | None, optional): The size
                (width, height), or ``(W, H)``, the prediction (either
                the bbox coordinates or the images itself) should
                correspond to. If :data:`None`, the prediction will
                correspond to the same size as the input image. Defaults
                to :data:`None`.
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

        def verify_bboxes(ori: Image.Image, boxes: torch.Tensor):
            # Set output size to original size if not specified
            w, h = output_size if output_size else ori.size

            if input_size is not None and input_size != (w, h):
                # Convert bboxes to output size
                boxes[:, 0] = boxes[:, 0] * w / input_size[0]
                boxes[:, 1] = boxes[:, 1] * h / input_size[1]
                boxes[:, 2] = boxes[:, 2] * w / input_size[0]
                boxes[:, 3] = boxes[:, 3] * h / input_size[1]

            return boxes

        if isinstance(format, str):
            match format:
                case "bool":
                    # NumPy array of shape (H, W) with True values for
                    # pixels that fall in any of the bounding boxes

                    def format_fn(ori, pred):
                        pred = verify_bboxes(ori, pred).numpy(force=True)
                        w, h = output_size if output_size else ori.size
                        fn = lambda b, x, y: b[0] <= x <= b[2] and b[1] <= y <= b[3]
                        pred = np.any(
                            [
                                [[fn(b, x, y) for x in range(w)] for y in range(h)]
                                for b in pred
                            ],
                            axis=0,
                        )

                        return pred

                case "int":
                    # Bounding boxes with integer coordinates w.r.t. the
                    # original image size: [[x_min, y_min, x_max, y_max], ...]

                    def format_fn(ori, pred):
                        pred = verify_bboxes(ori, pred)
                        return [[int(p.item()) for p in b] for b in pred]

                case "float":
                    # Bounding boxes with float coordinates normalized
                    # between 0 and 1: [[x_min, y_min, x_max, y_max], ...]

                    def format_fn(ori, pred):
                        w, h = ori.size if input_size is None else input_size
                        pred = pred / torch.tensor([w, h, w, h], device=pred.device)

                        return [[float(p.item()) for p in b] for b in pred]

                case "str":
                    # A string of the form
                    # "BBoxes: x_min y_min x_max y_max; ..."

                    def format_fn(ori, pred):
                        pred = verify_bboxes(ori, pred)
                        return "BBoxes: " + "; ".join(
                            [" ".join(map(str, map(int, b))) for b in pred]
                        )

                case "img":
                    # The original image with bounding boxes drawn on it
                    # using default values in draw_boxes

                    def format_fn(ori, pred):
                        pred = verify_bboxes(ori, pred)
                        ori = ori.resize(output_size) if output_size else ori
                        img = self.draw_boxes(ori, pred)

                        return img

                case _:
                    raise ValueError(f"{format} is not a valid choice!")

            # Convert to function
            format = format_fn

        return super().predict(image, format, input_size)

    @override
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [pred["boxes"] for pred in self.model([*x])]

    @override
    @copy_signature(predict)
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
