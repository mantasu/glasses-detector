import torch
import argparse
from . import segmenters
from . import classifiers


def parse_kwargs():
    parser = argparse.ArgumentParser(description="Processing files with models.")

    parser.add_argument(
        "-i", "--input-path",
        metavar="path/to/dir/or/file",
        type=str,
        required=True,
        help="Path to the input image or the directory with images."
    )
    parser.add_argument(
        "-o", "--output-path",
        metavar="path/to/dir/or/file",
        type=str,
        default=None,
        help=f"For classification, it is the path to a file, e.g., txt or "
             f"csv, to which to write the predictions. If not provided, the "
             f"prediction will be either printed (if input is a file) or "
             f"written to a default file (if input is a dir). For segmentation, "
             f"it is a path to a mask file, e.g., jpg or png, (if input is a "
             f"file) or a path to a directory where the masks should be saved "
             f"(if input is a dir). If not provided, default output paths will "
             f"be generated. Defaults to None."
    )
    parser.add_argument(
        "-t", "--task",
        metavar="<task-name>",
        type=str,
        required=True,
        choices=[
            "eyeglasses-classification", 
            "sunglasses-classification", 
            "glasses-classification",
            "full-glasses-segmentation", 
            "full-glasses-eyeglasses-segmentation",
            "full-glasses-sunglasses-segmentation",
            "full-glasses-glasses-segmentation",
            "glass-frames-segmentation", 
            "glass-frames-eyeglasses-segmentation",
            "glass-frames-sunglasses-segmentation",
            "glass-frames-glasses-segmentation",
        ],
        help=f"The kind of task to process the files for. One of "
             f"'eyeglasses-classification', 'sunglasses-classification', "
             f"'glasses-classification', 'full-glasses-segmentation', "
             f"'full-glasses-eyeglasses-segmentation', "
             f"'full-glasses-sunglasses-segmentation', "
             f"'full-glasses-glasses-segmentation', "
             f"'glass-frames-segmentation', "
             f"'glass-frames-eyeglasses-segmentation', "
             f"'glass-frames-sunglasses-segmentation', "
             f"'glass-frames-glasses-segmentation'."
    )
    parser.add_argument(
        "-s", "--size",
        metavar="<arch-name>",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large", "huge"],
        help=f"The model architecture name (model size). One of 'tiny', "
             f"'small', 'medium', 'large', 'huge'. Defaults to 'medium'."
    )
    parser.add_argument(
        "-l", "--label-type",
        type=str,
        metavar="<label-type>",
        default="int",
        choices=["bool", "int", "str", "logit", "proba"],
        help=f"Only used if the task is classification. It is the string "
             f"specifying the way to map the predictions to labels. For "
             f"instance, if specified as 'int', positive labels will be 1 and "
             f"negative will be 0. If specified as 'proba', probabilities of "
             f"being positive will be shown. One of 'bool', 'int', 'str', "
             f"'logit', 'proba'. Defaults to 'int'."
    )
    parser.add_argument(
        "-sep", "--sep",
        type=str,
        metavar="<sep>",
        default=",",
        help=f"Only used if the task is classification. It is the separator "
             f"to use to separate image file names and the predictions. "
             f"Defaults to ','."
    )
    parser.add_argument(
        "-m", "--mask-type",
        type=str,
        metavar="<mask-type>",
        default="img",
        choices=["bool", "int", "img", "logit", "proba"],
        help=f"Only used if the task is segmentation. The type of mask to "
             f"generate. For example, a mask could be a black and white image, "
             f"in which case 'img' should be specified. A mask could be a "
             f"matrix of raw scores in npy format, in which case 'logit' "
             f"should be specified. One of 'bool', 'int', 'img', 'logit', "
             f"'proba'. Defaults to 'img'."
    )
    parser.add_argument(
        "-ext", "--ext",
        type=str,
        metavar="<ext>",
        default=None,
        help=f"The extension to use to save masks. Specifying it will "
             f"overwrite the extension existing as part of ``output_path`` "
             f"(if it is specified as a path to file). If ``mask-type`` is "
             f"'img', then possible extensions are 'jpg', 'png', 'bmp' etc. "
             f"If ``mask-type`` is some value, e.g., 'bool' or 'probas', then "
             f"possible extensions are 'npy', 'pkl', 'dat' etc. If not "
             f"specified, it will be inferred form ``output-path`` (if it is "
             f"given and is a path to a file), otherwise 'jpg' or 'npy' will "
             f"be used, depending on ``mask-type``. Defaults to None."
    )

    # Parse kwargs, create class name
    kwargs = vars(parser.parse_args())
    model_cls_name = ''.join(map(str.capitalize, kwargs["task"].split('-')[:-1]))

    match kwargs["task"].split('-')[-1]:
        # Delete unnecessary keys
        case "classification":
            del kwargs["ext"]
            del kwargs["mask_type"]
            kwargs["model_cls"] = getattr(classifiers, model_cls_name + "Classifier")
        case "segmentation":
            del kwargs["sep"]
            del kwargs["label_type"]
            kwargs["model_cls"] = getattr(segmenters, model_cls_name + "Segmenter")
    
    # Delete task key
    del kwargs["task"]

    return kwargs

def main():
    # Get the cli kwargs
    kwargs = parse_kwargs()
    
    # Pop out model-based kwargs
    model_cls = kwargs.pop("model_cls")
    base_model = kwargs.pop("size")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model and process the input path
    model = model_cls(base_model=base_model, pretrained=True).to(device).eval()
    model.process(**kwargs)

if __name__ == "__main__":
    main()
