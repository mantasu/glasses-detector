import argparse
import os
import typing

import torch

from . import GlassesClassifier, GlassesDetector, GlassesSegmenter


def parse_kwargs():
    parser = argparse.ArgumentParser(description="Processing files with models.")

    parser.add_argument(
        "-i",
        "--input-path",
        metavar="path/to/dir/or/file",
        type=str,
        required=True,
        help="Path to the input image or the directory with images.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        metavar="path/to/dir/or/file",
        type=typing.Union[str, None],
        default=None,
        help=f"Path to the output file or the directory. If not provided, "
        f"then, if input is a file, the prediction will be printed (or shown "
        f"if it is an image), otherwise, if input is a directory, the "
        f"predictions will be written to a directory with the same name with "
        f"an added suffix '_preds'. If provided as a file, then the "
        f"prediction(-s) will be saved to this file (supported extensions "
        f"include: .txt, .csv, .json, .npy, .pkl, .jpg, .png). If provided as "
        f"a directory, then the predictions will be saved to this directory "
        f"use `--extension` flag to specify the file extensions in that "
        f"directory. Defaults to None.",
    )
    parser.add_argument(
        "-e",
        "--extension",
        metavar="<ext>",
        type=typing.Union[str, None],
        default=None,
        choices=(ch := [".txt", ".csv", ".json", ".npy", ".pkl", ".jpg", ".png"]),
        help=f"Only used if `--output-path` is a directory. The extension to "
        f"use to save the predictions as files. Common extensions include: "
        f"{", ".join([f"{c}" for c in ch])}. If not specified, it will be set "
        f"automatically to .jpg for image predictions and to .txt for all "
        f"other formats. Defaults to None.",
    )
    parser.add_argument(
        "-f",
        "--format",
        metavar="<format>",
        type=typing.Union[str, None],
        default=None,
        help=f"The format to use to map the raw prediction to. For "
        f"classification, common formats are bool, proba, str, for detection, "
        f"common formats are bool, int, img, for segmentation, common formats "
        f"are proba, img, mask. If not specified, it will be set "
        f"automatically to str, img, mask for classification, detection, "
        f"segmentation respectively. Check API documentation for more "
        f"details. Defaults to None.",
    )
    parser.add_argument(
        "-t",
        "--task",
        metavar="<task-name>",
        type=str,
        default="classification:anyglasses",
        choices=(ch := [
            "classification",
            "classification:anyglasses",
            "classification:sunglasses",
            "classification:eyeglasses",
            "detection",
            "detection:eyes",
            "detection:standalone",
            "detection:worn",
            "segmentation",
            "segmentation:frames",
            "segmentation:full",
            "segmentation:legs",
            "segmentation:lenses",
            "segmentation:shadows",
            "segmentation:smart",
        ]),
        help=f"The kind of task the model should perform. One of "
        f"{", ".join([f"{c}" for c in ch])}. If specified only as "
        f"classification, detection, or segmentation, the subcategories "
        f"anyglasses, worn, and smart will be chosen, respectively. Defaults "
        f"to classification:anyglasses.",
    )
    parser.add_argument(
        "-s",
        "--size",
        metavar="<model-size>",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help=f"The model size which determines architecture type. One of "
        f"'small', 'medium', 'large'. Defaults to 'medium'.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="<batch-size>",
        type=int,
        default=1,
        help=f"Only used if `--input-path` is a directory. The batch size to "
        f"use when processing the images. This groups the files in the input "
        f"directory to batches of size `batch_size` before processing them. "
        f"In some cases, larger batch sizes can speed up the processing at "
        f"the cost of more memory usage. Defaults to 1."
    )
    parser.add_argument(
        "-p",
        "--pbar",
        type=str,
        metavar="<pbar-desc>",
        default="Processing",
        help=f"Only used if `--input-path` is a directory. It is the "
        f"description that is used for the progress bar. If specified "
        f"as '' (empty string), no progress bar is shown. Defaults to "
        f"'Processing'.",
    )
    parser.add_argument(
        "-w",
        "--weights-path",
        metavar="path/to/weights.pth",
        type=typing.Union[str, None],
        default=None,
        help=f"Path to custom weights to load into the model. If not "
        f"specified, weights will be loaded from the default location (and "
        f"automatically downloaded there if needed). Defaults to None.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=typing.Union[str, None],
        metavar="<device>",
        default=None,
        help=f"The device on which to perform inference. If not specified, it "
        f"will be automatically checked if CUDA or MPS is supported. "
        f"Defaults to None.",
    )

    return vars(parser.parse_args())

def prepare_kwargs(kwargs):
    # Define the keys to use when calling process and init methods
    process_keys = ["is_file", "input_path", "output_path", "ext", "batch_size", "show", "pbar"]
    model_keys = ["task", "kind", "size", "pretrained", "device"]

    # Add "is_file" key to check which process method to call
    kwargs["is_file"] = os.path.splitext(kwargs["input_path"])[-1] != ""

    if not kwargs["is_file"] and kwargs["output_path"] is None:
        # Input is a directory but no output path is specified
        kwargs["output_path"] = os.path.splitext(kwargs["input_path"])[0] + "_preds"
    
    if kwargs["is_file"] and kwargs["output_path"] is None:
        # Input is a file and no output path is specified
        kwargs["show"] = True
    
    if kwargs["pbar"] == "":
        # No progress bar
        kwargs["pbar"] = None
    
    if len(splits := kwargs["task"].split(":")) == 2:
        # Task is specified as "task:kind"
        kwargs["task"] = splits[0]
        kwargs["kind"] = splits[1]
    
    if kwargs["weights_path"] is not None:
        # Custom weights path is specified
        kwargs["pretrained"] = kwargs["weights_path"]
    
    if kwargs["device"] is None and torch.cuda.is_available():
        # CUDA device is available
        kwargs["device"] = torch.device("cuda")
    elif kwargs["device"] is None and torch.backends.mps.is_available():
        # MPS device is available
        kwargs["device"] = torch.device("mps")
    elif kwargs["device"] is None:
        # CPU device is used by default
        kwargs["device"] = torch.device("cpu")

    # Get the kwargs for the process and init methods
    process_kwargs = {k: kwargs[k] for k in process_keys if k in kwargs}
    model_kwargs = {k: kwargs[k] for k in model_keys if k in kwargs}
    
    return process_kwargs, model_kwargs


def main():
    # Parse CLI args; prepare to create model and process images
    process_kwargs, model_kwargs = prepare_kwargs(parse_kwargs())
    is_file = process_kwargs.pop("is_file")
    task = model_kwargs.pop("task")

    # Create model
    match task:
        case "classification":
            model = GlassesClassifier(**model_kwargs)
        case "detection":
            model = GlassesDetector(**model_kwargs)
        case "segmentation":
            model = GlassesSegmenter(**model_kwargs)
        case _:
            raise ValueError(f"Unknown task '{task}'.")

    if is_file:
        # Process a single image file
        model.process_file(**process_kwargs)
    else:
        # Process a directory of images
        model.process_dir(**process_kwargs)

if __name__ == "__main__":
    main()
