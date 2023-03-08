import sys
import torch
import argparse
import pytorch_lightning as pl

sys.path.append("src")
torch.set_float32_matmul_precision("medium")

from utils import get_checkpoint_callback, seed
from model import SunglassesClssifier
from data import SunglassesOrNotModule

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data-dir", type=str, default="data",
        help="Path to the dataset directory. Defaults to data.")
    parser.add_argument("-n", "--num-epochs", type=int, default=30,
        help="The number of training epochs. Defaults to 30.")

    return parser.parse_args()

def main():
    # Parse given arguments
    args = parse_args()
    seed(0)

    # Setup model, datamodule and trainer params
    model = SunglassesClssifier(args.num_epochs)
    datamodule = SunglassesOrNotModule(args.data_dir)
    checkpoint_callback = get_checkpoint_callback()
    
    # Initialize the trainer, train it using datamodule and finally test
    trainer = pl.Trainer(max_epochs=args.num_epochs, accelerator="gpu",
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)

    # Load the best model from saved checkpoints and save its weights
    best_model = SunglassesClssifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), "checkpoints/sunglasses-classsifier-best.pth")

if __name__ == "__main__":
    main()