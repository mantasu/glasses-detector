import os
from collections.abc import Callable
from typing import Any

import numpy
import torch
import tqdm
from PIL import Image

from .._data import ImageLoaderMixin
from .base_model_old import BaseModel


class BaseClassifier(BaseModel, ImageLoaderMixin):
    """Base model class for classification.

    This is a base classifier class that should be used to instantiate
    any type of classifier. The working principle is the same as for
    the parent class :class:`.BaseModel`, however this one provides a
    couple of methods for processing actual input files and predicting
    outputs. If used with ``base_model`` parameter as a string for
    pre-defined architectures, this class provides some constants to
    specifically refer to those model architectures, e.g., if
    ``base_model`` is specified as some abbreviation, like "tiny". You
    do not need to worry about those constants.

    Note:
        As noted in the parent model, if ``kind`` argument is not
        provided, it will be automatically inferred from the class name,
        in this case, it would be "base_classifier", thus if some class
        extends this, the *kind* attribute would be inferred
        correspondingly.

    Args:
        *args: Same arguments as for :class:`.BaseModel`.
        **kwargs: Same keyword arguments as for :class:`.BaseModel`.
    """

    ABBREV_MAP = {
        "tiny": "tinyclsnet_v1",
        "small": "shufflenet_v2_x0_5",
        "medium": "mnasnet0_5",
        "large": "mobilenet_v3_large",
        "huge": "efficientnet_v2_s",
    }
    """
    dict[str, str]: A dictionary mapping the abbreviation, i.e., names 
        "tiny", "small", "medium", "large", "huge", to corresponding 
        base model names, e.g., "small" is mapped to 
        "shufflenet_v2_x0_5". This is the default abbreviation map, 
        based on which pretrained weights for some kinds of 
        classification models can be downloaded. To specify a custom 
        abbreviation map, use ``abbrev_map`` argument when instantiating 
        an object and either provide an empty dictionary or a custom one.
    """

    VERSION_MAP = {"eyeglasses_classifier": None, "sunglasses_classifier": "v0.1.0"}
    """
    dict[str, str]: A dictionary mapping from the possible pretrained 
        classification model kinds, e.g., "eyeglasses", "sunglasses", 
        to their corresponding GitHub release versions, e.g., "v1.0.0" 
        (where the newest weights are stored). This can be customized 
        via ``version_map`` argument as well, e.g., if other versions 
        can be chosen.
    """

    def __init__(self, *args, **kwargs):
        kwargs["abbrev_map"] = kwargs.get("abbrev_map", self.ABBREV_MAP)
        kwargs["version_map"] = kwargs.get("version_map", self.VERSION_MAP)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def predict(
        self,
        image: str | Image.Image | numpy.ndarray,
        label_type: str | Callable[[torch.Tensor], Any] | dict[bool, Any] = "int",
    ) -> Any:
        """Predicts whether the positive class is present.

        Takes an image or a path to the image and outputs a boolean
        value indicating whether the image belongs to a positive class
        or not.

        Args:
            image (str | PIL.Image.Image | numpy.ndarray): The path to
                the image to generate the prediction for or the image
                itself represented as :class:`Image.Image` or as a
                :class:`numpy.ndarray`. Note that the image should have
                values between 0 and 255 and be of RGB format.
                Normalization is not needed as the channels will be
                automatically normalized before passing through the
                network.
            label_type (str | collections.abc.Callable[[torch.Tensor], typing.Any] | dict[bool, typing.Any], optional):
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

                It is also possible to provide a callable function which
                specifies how to map a raw :class:`torch.Tensor` score
                of type ``torch.float32`` of shape ``(1,)`` to a label,
                or a dictionary with 2 keys: ``True`` and ``False``,
                each mapping to values corresponding to what to output
                if the predicted label is positive or negative. Defaults
                to "int".

        Returns:
            bool: ``True`` if the label is likely to be positive and
            ``False`` otherwise.

        Raises:
            ValueError: If the specified ``label_type`` as a string is
                not recognized.
        """
        if isinstance(label_type, str):
            # Update label type
            match label_type:
                case "bool":
                    label_type = {True: True, False: False}
                case "int":
                    label_type = {True: 1, False: 0}
                case "str":
                    label_type = {True: "present", False: "not_present"}
                case "logit":
                    label_type = lambda x: x.item()
                case "proba":
                    label_type = lambda x: x.sigmoid().item()
                case _:
                    raise ValueError(f"Invalid label map type: {label_type}")

        if isinstance(map := label_type, dict):
            # If the label type was specified as dict
            label_type = lambda x: map[(x > 0).item()]

        # Load the image properly and predict
        device = next(iter(self.parameters())).device
        x = self.load_image(image)[None, ...].to(device)
        prediction = label_type(self(x))

        return prediction

    def process(
        self,
        input_path: str,
        output_path: str | None = None,
        label_type: str | Callable[[torch.Tensor], Any] | dict[bool, Any] = "int",
        sep=",",
        desc: str | None = "Processing",
    ):
        """Generates a prediction for each image in the directory.

        Goes through all images in the directory and generates a
        prediction of whether each image falls under the positive class
        or not. Each prediction is then written to an output file line
        by line, i.e., Each line is of the form::

            <image_name.ext><separator><prediction>

        For example::

            my_image.jpg,1

        .. warning::
            Please ensure the directory contains valid images and only
            image files, otherwise errors may occur.

        Args:
            input_dir (str): The path to the input directory with image
                files.
            output_file (str | None, optional): The output file path to
                which to write the predictions. If not specified, the
                same directory where the ``input_dir`` folder is located
                will be used and the file in that directory will have
                the same name as ``input_dir`` basename, just with a
                suffix of *_label_preds.csv*. Defaults to None.
            label_type (str | collections.abc.Callable[[torch.Tensor], typing.Any] | dict[bool, typing.Any], optional):
                The kind of label to use, for instance, "int" would
                mean to use ``1`` for positive predictions and ``0`` for
                negative ones. It could be another type of mapping, see
                :meth:`predict` for a more detailed explanation. The
                mapped values will be used as predictions that will be
                written next to image file names. Defaults to "int".
            sep (str, optional): The separator to use to separate image
                file names and the predictions. Defaults to ','.
            desc (str | None, optional): Only used if input path leads
                to a directory of images. It is the description that is
                used for the progress bar. If specified as ``None``,
                no progress bar is shown. Defaults to "Processing".
        """
        if os.path.isfile(input_path):
            # Generate the sole prediction for given image
            pred = self.predict(input_path, label_type)

            if output_path is None:
                # Print if no path
                print(pred)
                return

            if (dir := os.path.dirname(output_path)) != "":
                # Create dir if doesn't exist
                os.makedirs(dir, exist_ok=True)

            with open(output_path, "w") as f:
                # Write to file
                f.write(str(pred))
        else:
            if output_path is None:
                # Create a default file at the same root as input dir
                ext = ".csv" if sep == "," else ".txt"
                output_path = input_path + "_label_preds" + ext

            if (dir := os.path.dirname(output_path)) != "":
                # Create dir if doesn't exist
                os.makedirs(dir, exist_ok=True)

            with open(output_path, "w") as f:
                # Read the directory of images
                imgs = list(os.scandir(input_path))

                if desc is not None:
                    # If desc is provided, wrap pbar
                    imgs = tqdm.tqdm(imgs, desc=desc)

                for file in imgs:
                    # Predict and write the prediction
                    pred = self.predict(file.path, label_type)
                    f.write(f"{file.name}{sep}{pred}\n")
