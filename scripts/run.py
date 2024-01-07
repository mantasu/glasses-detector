import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KINDS = {
    "classification": "anyglasses",
    "detection": "worn",
    "segmentation": "smart",
}

sys.path.append(os.path.join(PROJECT_DIR, "src"))
torch.set_float32_matmul_precision("medium")

from glasses_detector import GlassesClassifier, GlassesDetector, GlassesSegmenter
from glasses_detector._data import (
    ImageClassificationDataset,
    ImageDetectionDataset,
    ImageSegmentationDataset,
)
from glasses_detector._wrappers import BinaryClassifier, BinaryDetector, BinarySegmenter


class RunCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add args for wrapper creation
        parser.add_argument(
            "-r",
            "--root",
            metavar="path/to/data/root",
            type=str,
            default=os.path.join(PROJECT_DIR, "data"),
            help="Path to the data directory with classification and segmentation subdirectories that contain datasets for different kinds of tasks. Defaults to 'data' under project root.",
        )
        parser.add_argument(
            "-t",
            "--task",
            metavar="<task-name>",
            type=str,
            default="classification-anyglasses",
            choices=(ch := [
                "classification",
                "classification-anyglasses",
                "classification-sunglasses",
                "classification-eyeglasses",
                "detection",
                "detection-eyes",
                "detection-standalone",
                "detection-worn",
                "segmentation",
                "segmentation-frames",
                "segmentation-full",
                "segmentation-legs",
                "segmentation-lenses",
                "segmentation-shadows",
                "segmentation-smart",
            ]),
            help=f"The kind of task to train/test the model for. One of {[f"'{c}'" for c in ch]}. If specified only as 'classification', 'detection', or 'segmentation', the subcategories 'anyglasses', 'worn', and 'smart' will be chosen, respectively. Defaults to 'classification-anyglasses'.",
        )
        parser.add_argument(
            "-s",
            "--size",
            metavar="<model-size>",
            type=str,
            default="medium",
            choices=["small", "medium", "large"],
            help="The model size which determines architecture type. One of 'small', 'medium', 'large'. Defaults to 'medium'.",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            metavar="<int>",
            type=int,
            default=64,
            help="The batch size used for training. Defaults to 64.",
        )
        parser.add_argument(
            "-n",
            "--num-workers",
            metavar="<int>",
            type=int,
            default=8,
            help="The number of workers for the data loader. Defaults to 8.",
        )
        parser.add_argument(
            "-w",
            "--weights-path",
            metavar="path/to/weights",
            type=str | None,
            default=None,
            help="Path to weights to load into the model. Defaults to None.",
        )
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

        # Checkpoint and trainer defaults
        parser.set_defaults(
            {
                "checkpoint.dirpath": "checkpoints",
                "checkpoint.save_last": False,
                "checkpoint.monitor": "val_loss",
                "checkpoint.mode": "min",
                "trainer.max_epochs": 300,
            }
        )

        # Link argument with wrapper creation callback arguments
        parser.link_arguments("root", "model.root", apply_on="parse")
        parser.link_arguments("task", "model.task", apply_on="parse")
        parser.link_arguments("size", "model.size", apply_on="parse")
        parser.link_arguments("batch_size", "model.batch_size", apply_on="parse")
        parser.link_arguments("num_workers", "model.num_workers", apply_on="parse")
        parser.link_arguments("weights_path", "model.weights_path", apply_on="parse")

    def before_fit(self):
        if self.config.fit.checkpoint.filename is None:
            # Update default filename for checkpoint saver callback
            self.model_name = (
                self.config.fit.model.task + "-" + self.config.fit.model.size
            )
            self.trainer.callbacks[-1].filename = (
                self.model_name + "-{epoch:02d}-{val_loss:.3f}"
            )

    def after_fit(self):
        # Get the best checkpoint path and load it
        ckpt_path = self.trainer.callbacks[-1].best_model_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load wights and save the inner model as pth
        self.model.load_state_dict(ckpt["state_dict"])
        torch.save(
            self.model.model.state_dict(),
            os.path.join(ckpt_dir, self.model_name + ".pth"),
        )

def create_wrapper_callback(
    root: str = "data",
    task: str = "classification",
    size: str = "medium",
    batch_size: int = 64,
    num_workers: int = 8,
    weights_path: str | None = None,
) -> pl.LightningModule:
    
    # Get task and kind
    
    task_and_kind = task.split("-", maxsplit=1)
    task = task_and_kind[0]
    kind = DEFAULT_KINDS[task] if len(task_and_kind) == 1 else task_and_kind[1]

    # Get model and dataset classes
    model_cls, data_cls = {
        "classification": (GlassesClassifier, ImageClassificationDataset),
        "detection": (GlassesDetector, ImageDetectionDataset),
        "segmentation": (GlassesSegmenter, ImageSegmentationDataset),
    }[task]

    # Set-up wrapper initialization kwargs
    kwargs = {
        "root": os.path.join(root, task, kind),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    # Update wrapper initialization kwargs and set the initializer class
    if task == "classification":
        kwargs["label_type"] = {kind: 1, "no_" + kind: 0}
        wrapper_cls = BinaryClassifier
    elif task == "detection":
        wrapper_cls = BinaryDetector
    elif task == "segmentation":
        wrapper_cls = BinarySegmenter

    # Initialize model architecture and load weights if needed
    model = model_cls(kind=kind, size=size, pretrained=weights_path).model

    # if weights_path is not None:
    #     # Load weights if the path is specified to them
    #     model.load_state_dict(torch.load(weights_path))

    return wrapper_cls(model, *data_cls.create_loaders(**kwargs))


def cli_main():
    cli = RunCLI(create_wrapper_callback, seed_everything_default=0)


if __name__ == "__main__":
    cli_main()
