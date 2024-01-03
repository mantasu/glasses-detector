import os
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Callable, ClassVar, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.types import Number

from .components.base_model import BaseGlassesModel, T
from .models import TinyBinaryClassifier


@dataclass
class GlassesClassifier(BaseGlassesModel):
    task: str = "classification"
    kind: str = "anyglasses"
    size: str = "normal"
    pretrained: bool = True

    DEFAULT_SIZE_MAP: ClassVar[dict[str, int]] = {
        "small": {"name": "tinyclsnet_v1", "version": "v1.0.0"},
        "medium": {"name": "TBA", "version": "v1.0.0"},
        "large": {"name": "TBA", "version": "v1.0.0"},
    }

    DEFAULT_KIND_MAP: ClassVar[dict[str, str]] = {
        "anyglasses": DEFAULT_SIZE_MAP,
        "eyeglasses": DEFAULT_SIZE_MAP,
        "sunglasses": DEFAULT_SIZE_MAP,
    }

    @property
    @override
    def model_info(self) -> dict[str, str]:
        return self.DEFAULT_KIND_MAP.get(self.kind, {}).get(self.size, {})

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinyclsnet_v1":
                m = TinyBinaryClassifier()
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m

    @override
    def _get_format_fn(
        self,
        format: str | dict[bool, T],
    ) -> Callable[[torch.Tensor], T]:
        # Specify format_fn type
        format_fn: Callable[[torch.Tensor], T]

        if isinstance(format, str):
            # Check format
            match format:
                case "bool":
                    format_fn = {True: True, False: False}
                case "int":
                    format_fn = {True: 1, False: 0}
                case "str":
                    format_fn = {True: "present", False: "not_present"}
                case "logit":
                    format_fn = lambda x: x.item()
                case "proba":
                    format_fn = lambda x: x.sigmoid().item()
                case _:
                    raise ValueError(f"Invalid format: {format}")

        if isinstance(format, dict):
            # If the label type was specified as dict
            format_fn = lambda x: format[(x > 0).item()]

        return format_fn

    @override
    def predict(
        self,
        image: str | Image.Image | np.ndarray,
        format: str | dict[bool, T] | Callable[[torch.Tensor], T],
    ) -> T:
        return super().predict(image, format)

    @override
    def process(
        self,
        input_path: str,
        output_path: str | None,
        format: str | dict[bool, T] | Callable[[torch.Tensor], T],
        desc: str | None = "Processing",
    ):
        if os.path.isfile(input_path):
            # Generate prediction for the given image
            pred = self.predict(input_path, format)

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
        elif os.path.isdir(input_path):
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
