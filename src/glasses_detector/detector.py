from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Collection, override

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)

from .components import BaseGlassesModel
from .components.pred_type import Default
from .models import TinyBinaryDetector
from .utils import FilePath


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
    def draw_rects(
        img: Image.Image,
        bboxes: torch.Tensor | np.ndarray | list[list[int | float]],
        labels: torch.Tensor | np.ndarray | list[int] | None = None,
        label2color: dict[int, str] = {0: "red", 1: "green", 2: "blue"},
        label2name: dict[int, str] = {},
    ) -> Image.Image:
        """Draw bounding boxes on an image."""
        # Convert to numpy
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Convert to list
        if isinstance(bboxes, np.ndarray):
            bboxes = bboxes.tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        if labels is None:
            # Match len for zipping
            labels = [None] * len(bboxes)

        if label2name == {}:
            # Construct default label to name mapping
            label2name = {i: str(i) for i in np.unique(labels)}

        for bbox, label in zip(bboxes, labels):
            # Draw the rectangle around the object
            img = ImageDraw.Draw(img).rectangle(
                bbox,
                outline="red"
                if label is None
                else label2color[label % len(label2color)],
                width=3,
            )

            if label is None:
                continue

            # Write down the label next to bbox
            img = ImageDraw.Draw(img).text(
                bbox[:2],
                label2name[label],
                fill=label2color[label % len(label2color)],
                font=ImageFont.truetype("arial.ttf", 20),
            )

        return img

    @override
    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return self.model([*x])

    @override
    def predict(
        self,
        image: FilePath
        | Image.Image
        | np.ndarray
        | Collection[FilePath | Image.Image | np.ndarray],
        format: Callable[[Any], Default]
        | Callable[[Image.Image, Any], Default] = "img",
        resize: tuple[int, int] | None = (256, 256),
    ) -> Default | list[Default]:
        def verify_bboxes(ori: Image.Image, boxes: torch.Tensor):
            w, h = ori.size

            if (w, h) != resize:
                # Convert bboxes back to original size
                boxes[:, 0] = boxes[:, 0] * w / resize[0]
                boxes[:, 1] = boxes[:, 1] * h / resize[1]
                boxes[:, 2] = boxes[:, 2] * w / resize[0]
                boxes[:, 3] = boxes[:, 3] * h / resize[1]

            return boxes

        if isinstance(format, str):
            match format:
                case "int":

                    def format_fn(ori, pred):
                        pred["boxes"] = verify_bboxes(ori, pred["boxes"])
                        return [list(map(int, b)) for b in pred["boxes"]]

                case "float":

                    def format_fn(ori, pred):
                        pred["boxes"] = verify_bboxes(ori, pred["boxes"])
                        return pred["boxes"]

                case "str":

                    def format_fn(ori, pred):
                        pred["boxes"] = verify_bboxes(ori, pred["boxes"])
                        return "BBoxes: " + "; ".join(
                            [" ".join(map(int, b)) for b in pred["boxes"]]
                        )

                case "img":

                    def format_fn(ori, pred):
                        pred["boxes"] = verify_bboxes(ori, pred["boxes"])
                        img = self.draw_rects(ori, pred["boxes"])

                        return img

                case _:
                    raise ValueError(f"{format} is not a valid choice!")

            # Convert to function
            format = format_fn

        return super().predict(image, format)
