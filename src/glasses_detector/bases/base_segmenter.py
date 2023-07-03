import os
import numpy
import torch
import pickle
import warnings


from PIL import Image
from typing import Any
from copy import deepcopy
from collections import defaultdict
from collections.abc import Callable

from .base_model import BaseModel
from .._data import ImageLoaderMixin

VALID_EXTENSIONS = {
    ".rgb", ".gif", ".pbm", ".pgm", ".ppm", ".tiff", ".rast", 
    ".xbm", ".jpeg", ".jpg", ".bmp", ".png", ".webp", ".exr",
}


class BaseSegmenter(BaseModel, ImageLoaderMixin):
    """Base model class for segmentation.

    This is a base segmenter class that should be used to instantiate 
    any type of segmenter. The working principle is the same as for 
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
        in this case, it would be "base_segmenter", thus if some class
        extends this, the *kind* attribute would be inferred 
        correspondingly.
    
    .. warn::
        A base model is expected to produce a dictionary of outputs from 
        its ``forward`` method and this class will automatically select 
        *"out"* entry. If you pass a custom model as ``base_model``, and 
        it simply returns :class:`torch.Tensor`, please ensure to 
        modify it so it returns a dummy dictionary 
        ``{"out": actual_output}``.

    Attributes:
        ABBREV_MAP (dict[str, str]): A dictionary mapping the 
            abbreviation, i.e., names "tiny", "small", "medium", 
            "large", "huge", to corresponding base model names, e.g., 
            "small" is mapped to "lraspp_mobilenet_v3_large". This is 
            the default abbreviation map, based on which pretrained 
            weights for some kinds of segmentation models can be 
            downloaded. To specify a custom abbreviation map, use 
            ``abbrev_map`` argument when instantiating an object and 
            either provide an empty dictionary or a custom one.
        VERSION_MAP (dict[str, str]): A dictionary mapping from the 
            possible pretrained segmentation model kinds, e.g., 
            "full glasses", "glass frames", to their corresponding 
            GitHub release versions, e.g., "v1.0.0" (where the newest 
            weights are stored). This can be customized via 
            ``version_map`` argument as well, e.g., if other versions 
            can be chosen.

    Args:
        *args: Same arguments as for :class:`.BaseModel`.
        **kwargs: Same keyword arguments as for :class:`.BaseModel`.
    """
    ABBREV_MAP = {
        "tiny": "tinysegnet_v1",
        "small": "lraspp_mobilenet_v3_large",
        "medium": "fcn_resnet50",
        "large": "deeplabv3_resnet101",
        "huge": None,
    }

    VERSION_MAP = {
        "full_glasses_segmenter": None,
        "glass_frames_segmenter": None,
    }

    def __init__(self, *args, **kwargs):
        kwargs["abbrev_map"] = kwargs.get("abbrev_map", self.ABBREV_MAP)
        kwargs["version_map"] = kwargs.get("version_map", self.VERSION_MAP)
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward function of the model.

        This is an updated function of the parent class. By default, 
        when base models are created (from which the layers and 
        functions are copied), their forward functions return a 
        dictionary of outputs - one actual and one auxiliary. We only 
        care about the actual output *"out"*, thus we take ``["out"]``
        entry from an output produced by ``base_forward``, which is the 
        original ``forward`` function of the base model.

        Args:
            *args: Any input arguments accepted by original ``forward`` 
                that belongs to base model.
            **kwargs: Additional keyword arguments accepted by original 
                ``forward`` that belongs to base model.

        Returns:
            torch.Tensor: A model output after inference.
        """
        return self.base_forward(*args, **kwargs)["out"]

    @torch.no_grad()
    def predict(
        self,
        image: str | Image.Image | numpy.ndarray,
        mask_type: str | Callable[[torch.Tensor], Any] | dict[bool, Any] = "img"
    ) -> numpy.ndarray:
        """Predicts which pixels in the image are positive.

        Takes an image or a path to the image and outputs a mask as 
        :class:`numpy.ndarray` of shape (H, W) and of some custom type, 
        e.g., :attr:`numpy.uint8`. For example, a mask could be black 
        and white with values of either 255 (white) indicating pixels 
        under positive category or 0 (black) indicating the rest of the 
        pixels. Mask type can be customized with ``mask_type`` argument.

        Args:
            image (str | Image.Image | numpy.ndarray): The path to the 
                image to generate the mask for or the image itself
                represented as :class:`Image.Image` or as a 
                :class:`numpy.ndarray`. Note that the image should have 
                values between 0 and 255 and be of RGB format. 
                Normalization is not needed as the channels will be 
                automatically normalized before passing through the 
                network.
            mask_type (str | callable | dict[bool, Any], optional):
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
                  negative). Additionally, this ensures the returned 
                  dtype is UINT8, i.e., :attr:`numpy.uint8`.
                * "logit" - maps image pixels to raw scores (real 
                  numbers) of them being positive.
                * "proba" - maps image pixels to probabilities (numbers 
                  between 0 and 1) of them being positive.
                
                It is also possible to provide a callable function which 
                specifies how to map a raw :class:`torch.Tensor` output 
                of type ``torch.float32`` of shape ``(H, W)`` to a mask,
                or a dictionary with 2 keys: ``True`` and ``False``, 
                each mapping to values corresponding to what to output 
                if the predicted pixel is positive or negative. Defaults 
                to "img".

        Returns:
            numpy.ndarray: Output mask of shape (H, W) with each pixel 
                mapped to some ranged value or to a binary value, based 
                on whether which are positive and which are negative.

        Raises:
            ValueError: If the specified ``mask_type`` as a string is 
                not recognized.
        """
        if isinstance(mask_type, str):
            # Update mask type
            match mask_type:
                case "bool":
                    mask_type = {True: True, False: False}
                case "int":
                    mask_type = {True: 1, False: 0}
                case "img":
                    mask_type = lambda x: ((x > 0) * 255).to(torch.uint8)
                case "logit":
                    mask_type = lambda x: x
                case "proba":
                    mask_type = lambda x: x.sigmoid()
                case _:
                    raise ValueError(f"Invalid mask map type: {mask_type}")
        
        if isinstance(map := mask_type, dict):
            # If mask type was specified as dict
            mask_type = lambda x: torch.where((x > 0), map[True], map[False])
        
        # Loads the image properly and predict
        device = next(iter(self.parameters())).device
        x = self.load_image(image)[None, ...].to(device)
        mask = mask_type(self(x)[0, 0]).numpy(force=True)

        return mask
    
    def process(
        self,
        input_path: str,
        output_path: str | None = None,
        mask_type: str | Callable[[torch.Tensor], Any] | dict[bool, Any] = "img",
        ext: str | None = None,
    ):
        """Process a file or a dir of images by predicting mask(-s).

        If ``input_path`` is an image file, then a mask is generated for 
        it, otherwise, if it is a directory, a separate directory of 
        corresponding masks is created.

        Args:
            input_path (str): The path to the input image or a directory 
                with images. If it is a directory, please ensure it only 
                contains images.
            output_path (str | None, optional): The path to the output 
                mask or a directory where the masks should be saved - 
                depends on whether ``input_path`` is a path to an image 
                or to a directory with images. If not specified, for an 
                image, a mask will be generated at the same location as 
                the original image and with the same name but with an 
                added "_mask" suffix, otherwise, for a directory, 
                another directory will be created at the same root path 
                as the input directory and with the same name, but with 
                "_masks" suffix added. Defaults to None.
            mask_type (str | callable | dict[bool, Any], optional): 
                The type of mask to generate. For example, a mask could 
                be a black and white image, in which case "img" should 
                be specified. For more details, check :meth:`predict`. 
                Defaults to "img".
            ext (str | None, optional): The extension to use for the 
                masks to save. ``ext`` (if it is not ``None``) will
                overwrite the extension of ``output_path`` (if it is not 
                ``None`` and is a path to a file). These are the 
                possible options:

                * "npy" - saves mask using :func:`numpy.save`
                * "npz" - saves mask using :func:`numpy.savez_compressed`
                * "txt" - saves mask using :func:`numpy.savetxt` with 
                  default delimiter (space)
                * "csv" - saves mask using :func:`numpy.savetxt` with
                  ',' as delimiter
                * "pkl" - saves mask using :func:`pickle.dump`
                * "dat" - saves mask using :meth:`numpy.ndarray.tofile`
                * Any other extension is assumed to be an image 
                  extension, e.g., "jpg", "png", "bmp" and the mask is 
                  saved as a grayscale image, i.e., with a single 
                  channel.

                If not specified, then extension will be taken from 
                ``output_file``, or if it is not specified or is a path 
                to directory, ``ext`` will be based on ``mask_type``, 
                i.e., if mask type is "img", then the extension will 
                be "jpg" by default, otherwise it will be "npy". 
                Defaults to None.
        """
        # Check if the input is file or dir
        is_file = os.path.isfile(input_path)

        if ext is None and is_file and output_path is not None:
            # If None, ext is taken from output file
            ext = os.path.splitext(output_path)[1]
        elif ext is None:
            # If output file not given, we use default
            ext = "jpg" if mask_type == "img" else "npy"

        if mask_type == "img" and '.' + ext not in VALID_EXTENSIONS:
            # Raise a warning if ext doesn't support mask type
            warnings.warn(
                f"Mask type is not 'img', therefore, it is not possible to "
                f"save mask(-s) to a specified file type '{ext}'. "
                f"Switching extension (file type) to 'npy'."
            )
            ext = "npy"
        
        # Create on_file function
        on_file = defaultdict(
            lambda: lambda msk, pth: Image.fromarray(msk, mode='L').save(pth),
            {
                "npy": lambda msk, pth: numpy.save(pth, msk),
                "npz": lambda msk, pth: numpy.savez_compressed(pth, msk),
                "txt": lambda msk, pth: numpy.savetxt(pth, msk),
                "csv": lambda msk, pth: numpy.savetxt(pth, msk, delimiter=','),
                "pkl": lambda msk, pth: pickle.dump(msk, open(pth, "wb")),
                "dat": lambda msk, pth: msk.tofile(pth)
            }
        )[ext]

        if is_file:
            if output_path is None:
                # Define default mask output file path
                name = os.path.splitext(input_path)[0]
                output_path = name + "_mask." + ext
            
            # Predict and save single mask
            mask = self.predict(input_path)
            on_file(mask, output_path)
        else:
            if output_path is None:
                # Define default output directory
                output_path = input_path + "_masks"
            
            # Create the possibly non-existing dirs
            os.makedirs(output_path, exist_ok=True)

            for img in os.scandir(input_path):
                # Gen mask path, predict and save a single mask
                mask_name = os.path.splitext(img.name)[0] + '.' + ext
                mask_path = os.path.join(output_path, mask_name)
                mask = self.predict(img.path)
                on_file(mask, mask_path)


class _BaseConditionalSegmenter(BaseSegmenter):
    def __init__(self, classifier_cls, segmenter_cls, base_model, pretrained):
        super().__init__()

        if isinstance(base_model, str):
            # Both base model abbrevs are the same
            base_model = (base_model, base_model)
        
        if isinstance(pretrained, bool):
            # Same bool value for both models
            pretrained = (pretrained, pretrained)
        
        # Create a classifier and a segmenter inner models
        self.classifier = classifier_cls(base_model[0], pretrained[0])
        self.segmenter = segmenter_cls(base_model[1], pretrained[1])
    
    def forward(self, x):
        # Check which inputs to segment
        is_positive = self.classifier(x)
        masks = torch.zeros_like(x)

        if is_positive.any():
            # Segment only the inputs which are identified pos
            masks[is_positive] = self.segmenter(x[is_positive])
        
        return masks
