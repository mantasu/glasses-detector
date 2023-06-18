import sys
import torch
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(0)
sys.path.append("src")
torch.set_float32_matmul_precision("medium")

from sunglasses_detector import SunglassesClassifier, GlassesSegmenter
from sunglasses_detector._wrappers import BinaryClassifier, BinarySegmenter
from sunglasses_detector._data import ImageClassificationDataset, ImageSegmentationDataset

CLASSIFICATION_DATA_DIRS = [
    "specs-on-faces",
    "cmu-face-images",
    "glasses-and-coverings",
    "face-attributes-grouped",
    "sunglasses-no-sunglasses",
]

SEGMENTATION_DATA_DIRS = [
    "celeba-mask-hq",
]

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data-dir", type=str, default="data",
        help="Path to the dataset directory. Defaults to data.")
    parser.add_argument("-n", "--num-epochs", type=int, default=300,
        help="The number of training epochs. Defaults to 300.")
    parser.add_argument("-t", "--task", type=str, default="classification", 
        choices=["classification", "segmentation"],
        help="The type of the model to train. Can be either 'classification'" +\
             "or 'segmentation'. Defaults to 'classification'.")
    parser.add_argument("-m", "--model-size", type=str, default="medium",
        choices=["tiny", "small", "medium", "large"],
        help="The size of the model to train. One of 'tiny', 'small', " +\
             "'medium', 'large'. Defaults to 'medium'.")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
        help="The batch size used for training. Defaults to 32.")
    parser.add_argument("-w", "--num-workers", type=int, default=8,
        help="The number of workers for the data loader. Defaults to 8.")
    parser.add_argument("-a", "--accelerator", type=str, default="gpu",
        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"],
        help="The choice of accelerator used for training. One of 'cpu', " +\
             "'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto'. Defaults to 'gpu'.")

    return parser.parse_args()

def get_checkpoint_callback(name):
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename=name + "-{epoch:02d}-{val_loss:.3f}",
        save_last=False,
        monitor="val_loss",
        mode="min"
    )

def train(wrapper, model_name, num_epochs=300, accelerator="gpu"):
    # Initialize the callback to save best checkpoints
    callback = get_checkpoint_callback(model_name)
    
    # Trainer kwargs
    kwargs = {
        "accelerator": accelerator,
        "callbacks": [callback],
        "max_epochs": num_epochs,
    }

    # Init trainer object, fit the data and test it
    trainer = Trainer(**kwargs)
    trainer.fit(wrapper)
    trainer.test(wrapper)

    # Load the best checkpoint and save raw model
    checkpoint = torch.load(callback.best_model_path)
    wrapper.load_state_dict(checkpoint["state_dict"])
    torch.save(wrapper.model.state_dict(), f"checkpoints/{model_name}.pth")

def train_sunglasses_classifier(**kwargs):
    # Create base model and the data loaders
    model = SunglassesClassifier(kwargs["model_size"])
    loaders = ImageClassificationDataset.create_loaders(
        root=kwargs["data_dir"], 
        dirs=kwargs["dirs"],
        label_type={"sunglasses": 1, "no_sunglasses": 0},
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
    )
    
    # Create trainable classifier and callback
    model_name = f"sunglasses-classifier-{kwargs['model_size']}"
    classifier = BinaryClassifier(model, *loaders)
    train(classifier, model_name, kwargs["num_epochs"], kwargs["accelerator"])

def train_glasses_segmenter(**kwargs):
    # Create base model and the data loaders
    model = GlassesSegmenter(kwargs["model_size"])
    loaders = ImageSegmentationDataset.create_loaders(
        root=kwargs["data_dir"],
        dirs=kwargs["dirs"],
        img_dirname="images",
        name_map_fn={"masks": lambda x: f"{int(x[:5])}.jpg"},
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
    )
    
    # Create trainable segmenter and callback
    model_name = f"glasses-segmenter-{kwargs['model_size']}"
    segmenter = BinarySegmenter(model, *loaders)
    train(segmenter, model_name, kwargs["num_epochs"], kwargs["accelerator"])

if __name__ == "__main__":
    # Get arguments, convert to kwargs
    kwargs = vars(parse_arguments())

    match kwargs.pop("task"):
        case "classification":
            kwargs["dirs"] = CLASSIFICATION_DATA_DIRS
            train_sunglasses_classifier(**kwargs)
        case "segmentation":
            kwargs["dirs"] = SEGMENTATION_DATA_DIRS
            train_glasses_segmenter(**kwargs)
