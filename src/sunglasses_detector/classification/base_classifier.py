import os
import numpy
import torch

from PIL import Image
from typing import Any
from .._models import BaseModel
from .._data import ImageLoaderMixin
from collections.abc import Callable


class BaseClassifier(BaseModel, ImageLoaderMixin):
    ABBREV_MAP = {
        "tiny": "tinyclsnet_v1",
        "small": "shufflenet_v2_x0_5",
        "medium": "mobilenet_v3_small",
        "large": "efficientnet_b0",
        "huge": None,
    }

    VERSION_MAP = {
        "eyeglasses": None,
        "sunglasses": None,
    }

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
        """Predicts whether some type of glasses are present.

        Takes an image or a path to the image and outputs a boolean 
        value indicating whether the person in the image is wearing 
        that specific type of glasses or not.

        Args:
            image (str | Image.Image | numpy.ndarray): The path to the 
                image to generate the prediction for or the image itself
                represented as :class:`Image.Image` or as a 
                :class:`numpy.ndarray`. Note that the image should have 
                values between 0 and 255 and be of RGB format. 
                Normalization is not needed as the channels will be 
                automatically normalized before passing through the 
                network.
            label_type (str | callable | dict[bool, Any], optional): 
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
            bool: ``True`` if the person in the image is likely to wear 
                that specific type of glasses and ``False`` otherwise.

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
                case "logits":
                    label_type = lambda x: x.data
                case "probas":
                    label_type = lambda x: x.sigmoid().data
                case _:
                    raise ValueError(f"Invalid label map type: {label_type}")
            
        if isinstance(map := label_type, dict):
            # If the label type was specified as dict
            label_type = lambda x: map[(x > 0).data]
        
        # Loads the image properly and predict
        x = self.load_image(image)[None, ...]
        prediction = label_type(self(x))

        return prediction

    def process(
        self,
        input_path: str,
        output_path: str | None = None,
        label_type: str | Callable[[torch.Tensor], Any] | dict[bool, Any] = "int",
        sep=','
    ):
        """Generates a prediction for each image in the directory.

        Goes though all images in the directory and generates a 
        prediction of whether the person in the image is wearing some
        specific type of glasses or not. Each prediction is then written 
        to an output file line by line, i.e., Each line is of the form::

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
            label_type (str | callable | dict[bool, Any], optional): 
                The kind of label to use, for instance, "int" would 
                mean to use ``1`` for positive predictions and ``0`` for 
                negative ones. It could be another type of mapping, see 
                :meth:`predict` for a more detailed explanation. The 
                mapped values will be used as predictions that will be 
                written next to image file names. Defaults to "int".
            sep (str, optional): The separator to use to separate image 
                file names and the predictions. Defaults to ','.
        """
        if os.path.isfile(input_path):
            # Generate the sole prediction for given image
            pred = self.predict(input_path, label_type)

            if output_path is None:
                # Print if no path
                print(pred)
                return
            
            with open(output_path, 'w') as f:
               # Write to file
               f.write(pred)
        else:
            if output_path is None:
                # Create a default file at the same root as input dir
                output_path = input_path + "_label_preds.csv"
            
            with open(output_path, 'w') as f:
                for file in os.scandir(input_path):
                    # Predict and write the prediction
                    pred = self.predict(file.path, label_type)
                    f.write(f"{file.name}{sep}{pred}\n")
