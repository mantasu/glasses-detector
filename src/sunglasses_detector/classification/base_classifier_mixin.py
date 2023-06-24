import os
import numpy
import torch
import torch.nn as nn

from PIL import Image
from typing import Any
from .._data import ImageLoaderMixin
from .._models import TinyBinaryClassifier
from torchvision.models import (
    shufflenet_v2_x0_5,
    mobilenet_v3_small,
    efficientnet_b0,
)


class BaseClassifierMixin(ImageLoaderMixin):
    def load_base_model(self, model_type: str) -> torch.nn.Module:
        """Creates the base model for binary classification.

        This method simply creates a new instance of torch model 
        according to the specified type. There are some pre-defined 
        architecture names, according to which those instances can be 
        created.

        Note:
            No weights are loaded, only the model itself.

        Args:
            model_type (str): The type of the model to use for 
                classification. These are the available options:

                * "tinyclsnet" or "tiny" - The smallest model that is 
                  uniquely part of this package. For more information, 
                  see :class:`.._models.TinyBinaryClassifier`.
                * "shufflenet" or "small" - ShuffleNet V2 model taken 
                  from torchvision package. For more information, see 
                  :func:`~torchvision.models.shufflenet_v2_x0_5`.
                * "mobilenet" or "medium" - MobileNet V3 model taken 
                  from torchvision package. For more information, see 
                  :func:`~torchvision.models.mobilenet_v3_small`.
                * "efficientnet" or "large" - EfficientNet B0 model 
                  taken from torchvision package. For more information, 
                  see :func:`~torchvision.models.efficientnet_b0`.

        Raises:
            ValueError: If the model type is not available.

        Returns:
            torch.nn.Module: A new instance of torch model based on the 
                specified model type.
        """
        match model_type:
            case "tinyclsnet" | "tiny":
                m = TinyBinaryClassifier()
            case "shufflenet" | "small":
                m = shufflenet_v2_x0_5()
                m.fc = nn.Linear(m.fc.in_features, 1)
            case "mobilenet" | "medium":
                m = mobilenet_v3_small()
                m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
            case "efficientnet" | "large":
                m = efficientnet_b0()
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
            case _:
                raise ValueError(f"{model_type} is not a valid choice!")

        return m
    
    def load_weights_from_url(
        self, 
        model: torch.nn.Module,
        base_model_type: str,
        classifier_type: str,
        version: str = "",
        base_url: str = "https://github.com/mantasu/glasses-detector/releases/download", 
        device: str | torch.device | None = None,
    ) -> torch.nn.Module:
        """Reads weights file and loads them into the given model.

        This method takes a model which needs to be loaded with weights, 
        a URL that specifies the location of weights to download and, 
        optionally, a device to load the model on. 

        Note: 
            Although URL must be specified, according to 
            :func:`~torch.hub.load_state_dict_from_url`, first, the 
            corresponding weights will be checked if they are already 
            present in hub cache, which by default is 
            ``~/.cache/torch/hub/checkpoints``, and, if they are not, 
            the weight will be downloaded there and then loaded.

        Args:
            model (torch.nn.Module): The model to load the weights to.
            base_model_type (str): The type of base model or model size 
                the corresponding weights should be downloaded for. One 
                of "tinyclsnet", "shufflenet", "mobilenet", 
                "efficientnet" or one of "tiny", "small", "medium", 
                "large".
            classifier_type (str): The type of classifier to download 
                the weights for. One of "eyeglasses", "sunglasses", 
                "glasses".
            version (str): The GitHub release version at which the 
                model weights were uploaded, e.g., "v1.0.0".
            base_url (str): The base GitHub releases URL (without 
                version) at which the weights are uploaded. Defaults to 
                "https://github.com/mantasu/glasses-detector/releases/download".
            device (str | torch.device | None, optional): The device to
                load the weight and the model onto. If not specified, 
                *cpu* will be used. Defaults to None.

        Returns:
            torch.nn.Module: The same given model but with weights 
                loaded.
        """
        # Generate correct name
        match base_model_type:
            case "tinyclsnet" | "tiny":
                base_model_type = "tinyclsnet_v1"
            case "shufflenet" | "small":
                base_model_type = "shufflenet_v2_x0_5"
            case "mobilenet" | "medium":
                base_model_type = "mobilenet_v3_small"
            case "efficientnet" | "large":
                base_model_type = "efficientnet_b0"
        
        # Create a full URL from parts to download the weights from it
        url = f"{base_url}/{version}/{classifier_type}-classifier-{base_model_type}.pth"

        # Get weights from the download path and load them into model
        weights = torch.hub.load_state_dict_from_url(url, map_location=device)
        model.load_state_dict(weights)

        if device is not None:
            # Cast to device
            model.to(device)
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image: str | Image.Image | numpy.ndarray,
        label_type: str | callable | dict[bool, Any] = "int",
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
                if the predicted label is positive or negative.
        Returns:
            bool: ``True`` if the person in the image is likely to wear 
                that specific type of glasses and ``False`` otherwise.
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
            
        if isinstance(label_map := label_type, dict):
            # If the label type was specified as dict
            label_type = lambda x: label_map[(x > 0).data]
        
        # Loads the image properly and predict
        x = self.load_image(image)[None, ...]
        prediction = label_type(self(x))

        return prediction
    
    @torch.no_grad()
    def process_dir(
        self,
        input_dir: str,
        output_file: str | None = None,
        label_type: str | callable | dict[bool, Any] = "int",
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
        
        Raises:
            ValueError: If the specified ``label_map`` as a string is 
                not recognized.
        """
        if output_path is None:
            # Create a default CSV output file
            output_path = input_dir + "_label_preds.csv"

        with open(output_file, 'w') as f:
            for file in os.scandir(input_dir):
                # Predict and write the prediction
                pred = self.predict(file.path, label_type)
                f.write(f"{file.name}{sep}{pred}\n")
