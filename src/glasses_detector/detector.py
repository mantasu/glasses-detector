from dataclasses import dataclass
from typing import ClassVar, override

import torch.nn as nn

from .components.base_model import BaseGlassesModel
from .models import TinyBinaryDetector


@dataclass
class GlassesDetector(BaseGlassesModel):
    task: str = "detection"
    kind: str = "worn"
    size: str = "normal"
    pretrained: bool = True

    DEFAULT_SIZE_MAP: ClassVar[dict[str, int]] = {
        "small": {"name": "tinydetnet_v1", "version": "v1.0.0"},
        "medium": {"name": "TBA", "version": "v1.0.0"},
        "large": {"name": "TBA", "version": "v1.0.0"},
    }

    DEFAULT_KIND_MAP: ClassVar[dict[str, str]] = {
        "eyes": DEFAULT_SIZE_MAP,
        "standalone": DEFAULT_SIZE_MAP,
        "worn": DEFAULT_SIZE_MAP,
    }

    @staticmethod
    @override
    def create_model(model_name: str) -> nn.Module:
        match model_name:
            case "tinydetnet_v1":
                m = TinyBinaryDetector()
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m
