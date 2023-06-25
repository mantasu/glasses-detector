import os
import numpy
import torch
import pickle
import warnings
import PIL.Image as Image

from typing import Any
from .._models import BaseModel
from collections import defaultdict
from collections.abc import Callable
from .._data import ImageLoaderMixin
from ..classification import BaseClassifier

VALID_EXTENSIONS = {
    ".rgb", ".gif", ".pbm", ".pgm", ".ppm", ".tiff", ".rast", 
    ".xbm", ".jpeg", ".jpg", ".bmp", ".png", ".webp", ".exr",
}

class BaseSegmenter(BaseModel, ImageLoaderMixin):
    ABBREV_MAP = {
        "tiny": "tinysegnet_v1",
        "small": "lraspp_mobilenet_v3_large",
        "medium": "fcn_resnet50",
        "large": "deeplabv3_resnet101",
        "huge": None,
    }

    VERSION_MAP = {
        "full_glasses": None,
        "glass_frames": None,
    }

    def __init__(self, *args, **kwargs):
        kwargs["abbrev_map"] = kwargs.get("abbrev_map", self.ABBREV_MAP)
        kwargs["version_map"] = kwargs.get("version_map", self.VERSION_MAP)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return self.base_forward(x)["out"]
    
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
                case "logits":
                    mask_type = lambda x: x
                case "probas":
                    mask_type = lambda x: x.sigmoid()
                case _:
                    raise ValueError(f"Invalid mask map type: {mask_type}")
        
        if isinstance(map := mask_type, dict):
            # If mask type was specified as dict
            mask_type = lambda x: torch.where((x > 0), map[True], map[False])
        

        # Loads the image properly and predict
        device = next(iter(self.parameters())).device
        x = self.load_image(image)[None, ...].to(device)
        print(self(x))
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


class BaseConditionalSegmenter(BaseSegmenter):
    def __init__(
        self, 
        classifier_cls: BaseClassifier, 
        segmenter_cls: BaseSegmenter, 
        base_model: str | tuple[str, str] = "medium", 
        pretrained: bool = False
    ):
        super().__init__(base_model, pretrained)

        if isinstance(base_model, str):
            base_model = (base_model, base_model)
        
        self.classifier = classifier_cls(base_model[0], pretrained)
        self.segmenter = segmenter_cls(base_model[1], pretrained)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selection_mask = self.classifier(x)
        masks = torch.zeros_like(x)

        if selection_mask.any():
            masks[selection_mask] = self.segmenter(x)
        
        return masks