import sys
import torch
import argparse

sys.path.append("src")
from model import SunglassesClssifier

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image-path", type=str, required=True,
        help="The path to the image to infer.")
    parser.add_argument("-m", "--model-path", type=str,
        default="checkpoints/sunglasses-classsifier-best.pth",
        help=f"The path to `.pth` weights file to load. Defaults to "
             f"'checkpoints/sunglasses-classsifier-best.pth'.")

    return parser.parse_args()

def predict():
    args = parse_args()

    classifier = SunglassesClssifier()
    classifier.load_state_dict(torch.load(args.model_path))
    prediction, confidence = classifier.predict(args.image_path)

    print(f"Prediction: {prediction} [{confidence * 100:.2f}%]")

if __name__ == "__main__":
    predict()