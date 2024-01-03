import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Self, TypedDict, Unpack, override

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ._data import ImageLoaderMixin
from .utils import PredType, is_url


@dataclass
class BaseGlassesModel(nn.Module, ABC):
    task: str
    kind: str
    size: str
    pretrained: bool | str = field(default=False, repr=False)
    device: str | torch.device = field(default="cpu", repr=False)
    model: nn.Module = field(default=None, init=False, repr=False)

    BASE_WEIGHTS_URL: ClassVar[
        str
    ] = "https://github.com/mantasu/glasses-detector/releases/download"

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

    def _format_pred(self, preds: list[tuple[str, PredType.ANY]], ext: str):
        if not isinstance(pred, PredType.DEFAULT):
            raise ValueError("Prediction type not recognized.")

        match ext:
            case ".txt":
                init, agg_fn = "", lambda x, y, z: z + f"{x} {y}\n"
            case ".csv":
                init, agg_fn = "", lambda x, y, z: z + f"{x},{y}\n"
            case ".json":
                init, agg_fn = {}, lambda x, y, z: z.update({x: y})
            case ".yml" | ".yaml":
                init, agg_fn = {}, lambda x, y, z: z.update({x: y})
            case ".pkl":
                init, agg_fn = {}, lambda x, y, z: z.update({x: y})
            case ".npy":
                init, agg_fn = {}, lambda x, y, z: z.update({x: y})
            case ".npz":
                init, agg_fn = {}, lambda x, y, z: z.update({x: y})
            case ".dat":
                init, agg_fn = "", lambda x, y, z: z + f"{x} {y}\n"
            case ".jpg" | ".jpeg" | ".png" | ".bmp" | ".pgm" | ".webp":
                pass
            case _:
                pass

        for name, pred in preds:
            if not PredType.is_default(pred):
                raise ValueError("Prediction type not recognized.")

            if PredType.is_scalar(pred):
                pred = [pred]

    def _process_file(
        self,
        input_path: os.PathLike,
        output_path: os.PathLike | None,
        pred_type: str
        | dict[bool, PredType.ANY]
        | Callable[[torch.Tensor], PredType.ANY],
        show: bool = False,
    ) -> tuple[str, PredType.ANY] | None:
        if output_path is not None:
            ext = os.path.splitext(output_path)[1]

        ext_map = {
            ".txt",
            ".csv",
            ".json",
            ".xml",
            ".yaml",
        }

        pred = self.predict(input_path, pred_type)

        match pred:
            case PredType.SCALAR():
                if show:
                    # Print pred
                    print(pred)

                if output_path is None:
                    return os.path.basename(input_path), pred
                elif isinstance(output_path, os.PathLike):
                    ext = os.path.splitext(output_path)[1]

                    with open(output_path, "w") as f:
                        # Write to file
                        f.write(pred)
                elif isinstance(output_path, Callable):
                    return output_path(input_path, pred)
            case PredType.ARRAY():
                pass
            case PredType.IMAGE():
                if output_path is None:
                    # Show the image
                    plt.imshow(pred)
                elif isinstance(output_path, os.PathLike):
                    # Save to output path
                    pred.save(output_path)
                elif aggregate_fn is not None:
                    # Aggregate to dict
                    aggregate_fn(input_path, pred)
                else:
                    pass
            case _:
                raise ValueError("Prediction type not recognized")

        if output_path is None and isinstance(pred, Image.Image):
            plt.imshow(pred)
        elif output_path is None:
            print(pred)
        elif isinstance(pred, Image.Image):
            pred.save(output_path)
        else:
            with open(output_path, "w") as f:
                f.write(pred)

    def _process_dir(
        self,
        input_path: str,
        output_path: str | None,
        format: str | dict[bool, T] | Callable[[torch.Tensor], T],
    ):
        pass

    def _process_dir_to_file():
        pass

    @property
    @abstractmethod
    def model_info(self) -> dict[str, str]:
        ...

    @staticmethod
    @abstractmethod
    def create_model(self, model_name: str) -> nn.Module:
        ...

    @abstractmethod
    def _get_format_fn(self, format: str | dict[bool, T]) -> T:
        ...

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        **kwargs: Unpack[BaseGlassesModelParams],
    ) -> Self:
        # Get the specified device or check the one from the model
        device = kwargs.get("device", next(iter(model.parameters())).device)

        with warnings.catch_warnings():
            # Ignore the warnings from the model init
            glasses_model = cls(
                task=kwargs.get("task", "custom"),
                kind=kwargs.get("kind", "custom"),
                size=kwargs.get("size", "custom"),
                pretrained=False,
                device=device,
            )

        # Assign the actual model
        glasses_model.model = model

        if pretrained := kwargs.get("pretrained", False):
            # Load weights if `pretrained` is True or a path
            glasses_model.load_weights(path_or_url=pretrained)

        # Cast to device
        glasses_model.to(device)

        return glasses_model

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_weights(self, path_or_url: str | bool = True):
        if isinstance(path_or_url, bool) and path_or_url:
            try:
                # Get model name and release version
                name = self.model_info["name"]
                version = self.model_info["version"]
            except KeyError:
                # Raise model info warning for not constructing URL
                message = "Path/URL to weights cannot be constructed"
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

    @torch.inference_mode()
    # @abstractmethod
    def predict(
        self,
        image: str | Image.Image | np.ndarray,
        format: str | dict[bool, T] | Callable[[torch.Tensor], T],
    ) -> T:
        if isinstance(format, (str, dict)):
            # Get the format function from child
            format = self._get_format_fn(format)

        # Load the image properly and predict
        device = next(iter(self.parameters())).device
        x = ImageLoaderMixin.load_image(image)[None, ...].to(device)
        prediction = format(self(x))

        return prediction

    @abstractmethod
    def process(
        self,
        input_path: str,
        output_path: str | None,
        format: str | dict[bool, T] | Callable[[torch.Tensor], T],
        ext: str | None = None,
        desc: str | None = "Processing",
    ):
        ...
