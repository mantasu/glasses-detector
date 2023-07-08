import os
import sys
import torch
import pytorch_lightning as pl

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_DIR, "src"))
torch.set_float32_matmul_precision("medium")

from glasses_detector._data import (
    ImageClassificationDataset, 
    ImageSegmentationDataset,
)

from glasses_detector._wrappers import (
    BinaryClassifier,
    BinarySegmenter,
)

from glasses_detector import (
    EyeglassesClassifier, 
    SunglassesClassifier,
    FullGlassesSegmenter,
    GlassFramesSegmenter,
)


class RunCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add args for wrapper creation
        parser.add_argument(
            "-r", "--root", 
            metavar="path/to/data/root", 
            type=str, 
            default=os.path.join(PROJECT_DIR, "data"),
            help="Path to the data directory with classification and segmentation subdirectories that contain datasets for different kinds of tasks. Defaults to 'data' under project root."
        )
        parser.add_argument(
            "-t", "--task",
            metavar="<task-name>",
            type=str,
            default="sunglasses-classification",
            choices=["eyeglasses-classification", "sunglasses-classification", "full-glasses-segmentation", "glass-frames-segmentation"],
            help="The kind of task to train/test the model for. One of 'eyeglasses-classification', 'sunglasses-classification', 'full-glasses-segmentation', 'glass-frames-segmentation'. Defaults to 'sunglasses-classification'."
        )
        parser.add_argument(
            "-s", "--size",
            metavar="<arch-name>",
            type=str,
            default="medium",
            choices=["tiny", "small", "medium", "large", "huge"],
            help="The model architecture name (model size). One of 'tiny', 'small', 'medium', 'large', 'huge'. Defaults to 'medium'."
        )
        parser.add_argument(
            "-b", "--batch-size",
            metavar="<int>",
            type=int,
            default=64,
            help="The batch size used for training. Defaults to 64."
        )
        parser.add_argument(
            "-n", "--num-workers",
            metavar="<int>",
            type=int,
            default=8,
            help="The number of workers for the data loader. Defaults to 8."
        )
        parser.add_argument(
            "-w", "--weights-path",
            metavar="path/to/weights",
            type=str | None,
            default=None,
            help="Path to weights to load into the model. Defaults to None."
        )
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")

        # Checkpoint and trainer defaults
        parser.set_defaults({
            "checkpoint.dirpath": "checkpoints", 
            "checkpoint.save_last": False,
            "checkpoint.monitor": "val_loss",
            "checkpoint.mode": "min",
            "trainer.max_epochs": 300,
        })

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
            self.model_name = self.config.fit.model.task + '-' + self.config.fit.model.size
            self.trainer.callbacks[-1].filename = self.model_name + "-{epoch:02d}-{val_loss:.3f}"
    
    def after_fit(self):
        # Get the best checkpoint path and load it
        ckpt_path = self.trainer.callbacks[-1].best_model_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load wights and save the inner model as pth
        self.model.load_state_dict(ckpt["state_dict"])
        torch.save(self.model.model.state_dict(), os.path.join(ckpt_dir, self.model_name + ".pth"))


def create_wrapper_callback(
    root: str = "data", 
    task: str = "sunglasses-classification",
    size: str = "medium",
    batch_size: int = 64,
    num_workers: int = 8,
    weights_path: str | None = None,
) -> pl.LightningModule:
    
    # Get model and dataset classes
    model_cls, data_cls = {
        "eyeglasses-classification": (EyeglassesClassifier, ImageClassificationDataset),
        "sunglasses-classification": (SunglassesClassifier, ImageClassificationDataset),
        "full-glasses-segmentation": (FullGlassesSegmenter, ImageSegmentationDataset),
        "glass-frames-segmentation": (GlassFramesSegmenter, ImageSegmentationDataset),
    }[task]

    # Set-up wrapper initialization kwargs
    kwargs = {
        "root": os.path.join(root, task.split('-')[-1], '-'.join(task.split('-')[:-1])),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    # Update wrapper initialization kwargs and set the initializer class
    if task.split('-')[-1] == "classification":
        kwargs["label_type"] = {"eyeglasses": 1, "no_eyeglasses": 0} \
                               if task.split('-')[0] == "eyeglasses" \
                               else {"sunglasses": 1, "no_sunglasses": 0}
        wrapper_cls = BinaryClassifier
    else:
        kwargs["img_dirname"] = "images"
        kwargs["name_map_fn"] = {"masks": lambda x: f"{int(x[:5])}.jpg"}
        wrapper_cls = BinarySegmenter
    
    # Initialize model arch
    model = model_cls(size)
    
    if weights_path is not None:
        # Load weights if the path is specified to them
        model.load_state_dict(torch.load(weights_path))

    return wrapper_cls(model, *data_cls.create_loaders(**kwargs))

def cli_main():
    cli = RunCLI(create_wrapper_callback, seed_everything_default=0)

if __name__ == "__main__":
    cli_main()