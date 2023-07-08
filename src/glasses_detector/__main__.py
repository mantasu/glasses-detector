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
        "-k", "--kind",
        metavar="<kind-of-model>",
        type=str,
        required=True,
        choices=[
            "eyeglasses-classifier", 
            "sunglasses-classifier", 
            "anyglasses-classifier",
            "full-glasses-segmenter", 
            "full-eyeglasses-segmenter",
            "full-sunglasses-segmenter",
            "full-anyglasses-segmenter",
            "glass-frames-segmenter", 
            "eyeglasses-frames-segmenter",
            "sunglasses-frames-segmenter",
            "anyglasses-frames-segmenter",
        ],
        help=f"The kind of model to use to process the files. One of "
             f"'eyeglasses-classifier', 'sunglasses-classifier', "
             f"'anyglasses-classifier', 'full-glasses-segmenter', "
             f"'full-eyeglasses-segmenter', 'full-sunglasses-segmenter', "
             f"'full-anyglasses-segmenter', 'glass-frames-segmenter', "
             f"'eyeglasses-frames-segmenter', 'sunglasses-frames-segmenter', "
             f"'anyglasses-frames-segmenter'."
    )
    parser.add_argument(
        "-s", "--size",
        metavar="<arch-name>",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large", "huge"],
        help=f"The model architecture name (model size). One of 'tiny', "
             f"'small', 'medium', 'large', 'huge'. Defaults to 'small'."
    )
    parser.add_argument(
        "-l", "--label-type",
        type=str,
        metavar="<label-type>",
        default="int",
        choices=["bool", "int", "str", "logit", "proba"],
        help=f"Only used if `kind` is classifier. It is the string "
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
        help=f"Only used if `kind` is classifier. It is the separator "
             f"to use to separate image file names and the predictions. "
             f"Defaults to ','."
    )
    parser.add_argument(
        "-m", "--mask-type",
        type=str,
        metavar="<mask-type>",
        default="img",
        choices=["bool", "int", "img", "logit", "proba"],
        help=f"Only used if `kind` is segmenter. The type of mask to "
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
        help=f"Only used if `kind` is segmenter. The extension to use to save " 
             f"masks. Specifying it will overwrite the extension existing as "
             f"part of ``output_path`` (if it is specified as a path to file). "
             f"If ``mask-type`` is 'img', then possible extensions are 'jpg', "
             f"'png', 'bmp' etc. If ``mask-type`` is some value, e.g., 'bool' "
             f"or 'proba', then possible extensions are 'npy', 'pkl', 'dat' "
             f"etc. If not specified, it will be inferred form ``output-path`` "
             f"(if it is given and is a path to a file), otherwise 'jpg' or "
             f"'npy' will be used, depending on ``mask-type``. Defaults to "
             f"None."
    )
    parser.add_argument(
        "-d", "--desc",
        type=str,
        metavar="<pbar-desc>",
        default="Processing",
        help=f"Only used if input path leads to a directory of images. It is "
             f"the description that is used for the progress bar. If specified "
             f"as '' (empty string), no progress bar is shown. Defaults to "
             f"'Processing'."
    )
    parser.add_argument(
        "-dev", "--device",
        type=str,
        metavar="<device>",
        default="",
        help=f"The device on which to perform inference. If not specified, it "
             f"will be automatically checked if CUDA or MPS is supported. "
             f"Defaults to ''."
    )

    # Parse kwargs, create class name
    kwargs = vars(parser.parse_args())
    model_cls_name = ''.join(map(str.capitalize, kwargs["kind"].split('-')))

    match kwargs["kind"].split('-')[-1]:
        # Delete unnecessary keys
        case "classifier":
            del kwargs["ext"]
            del kwargs["mask_type"]
            kwargs["model_cls"] = getattr(classifiers, model_cls_name)
        case "segmenter":
            del kwargs["sep"]
            del kwargs["label_type"]
            kwargs["model_cls"] = getattr(segmenters, model_cls_name)
    
    # Delete kind key
    del kwargs["kind"]

    return kwargs

def main():
    # Get the cli kwargs
    kwargs = parse_kwargs()
    
    # Pop out model-based kwargs
    model_cls = kwargs.pop("model_cls")
    base_model = kwargs.pop("size")
    device = kwargs.pop("device")

    # Automatically determine the device
    if device == '' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == '' and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device == '':
        device = torch.device("cpu")

    # Instantiate model and process the input path
    model = model_cls(base_model=base_model, pretrained=True).to(device).eval()
    model.process(**kwargs)

if __name__ == "__main__":
    main()
