from dataclasses import dataclass, field
from typing import ClassVar, override

import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDHead

from .components.base_model import BaseGlassesModel
from .models import TinyBinaryDetector


@dataclass
class GlassesDetector(BaseGlassesModel):
    task: str = field(default="detection", init=False)
    kind: str = "worn"
    size: str = "medium"
    pretrained: bool | str | None = field(default=True, repr=False)

    DEFAULT_SIZE_MAP: ClassVar[dict[str, int]] = {
        "small": {"name": "tinydetnet_v1", "version": "v1.0.0"},
        "medium": {"name": "ssdlite320_mobilenet_v3_large", "version": "v1.0.0"},
        "large": {"name": "fasterrcnn_resnet50_fpn_v2", "version": "v1.0.0"},
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
            case "ssdlite320_mobilenet_v3_large":
                m = ssdlite320_mobilenet_v3_large(
                    num_classes=2,
                    detections_per_img=1,
                    topk_candidates=10,
                )
                # num_in = m.backbone.out_channels
                # m.head = SSDHead(num_in, m.head.num_anchors, 2)
            case "fasterrcnn_resnet50_fpn_v2":
                m = fasterrcnn_resnet50_fpn_v2(num_classes=2)
                # num_in = m.roi_heads.box_predictor.cls_score.in_features
                # m.roi_heads.box_predictor = FastRCNNPredictor(num_in, 2)
            case _:
                raise ValueError(f"{model_name} is not a valid choice!")

        return m
