import argparse
import os
import random
import shutil
import tarfile
import warnings
import zipfile
from multiprocessing import cpu_count

import numpy as np
import rarfile
import torch
from face_crop_plus import Cropper
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

random.seed(0)

VALID_EXTENSIONS = {
    ".rgb",
    ".gif",
    ".pbm",
    ".pgm",
    ".ppm",
    ".tiff",
    ".rast",
    ".xbm",
    ".jpeg",
    ".jpg",
    ".bmp",
    ".png",
    ".webp",
    ".exr",
}


def generate_title(title, pad=5):
    # Generate title with borders and print it
    midline = "#" * pad + " " + title + " " + "#" * pad
    top_bot = "#" * len(midline)
    print("\n".join(["\n" + top_bot, midline, top_bot]))


def folder_name(filepath):
    return os.path.basename(os.path.dirname(filepath))


def unpack(filename, root=".", members=set()):
    if not os.path.exists(file_path := os.path.join(root, filename)):
        warnings.warn(f"Please ensure {file_path} is a valid path.")
        return []

    # Check the contents before extracting
    contents = set(os.listdir(root))

    # Choose a correct unpacking interface
    if file_path.endswith(".zip"):
        open_file = zipfile.ZipFile
    elif file_path.endswith(".rar"):
        open_file = rarfile.RarFile
    elif file_path.endswith(".tar.gz"):
        open_file = tarfile.open

    with open_file(file_path) as file:
        # Choose a correct method to list files inside pack
        if isinstance(file, (zipfile.ZipFile, rarfile.RarFile)):
            iterable = file.namelist()
        elif isinstance(file, tarfile.TarFile):
            iterable = file.getnames()

        for member in tqdm(iterable, desc=f"    * Extracting '{filename}'"):
            if len(members) > 0 and not any(member.startswith(m) for m in members):
                # Skip if not needed to extract
                continue

            if os.path.exists(os.path.join(root, member)):
                # Not extracting if it is already extracted
                continue

            try:
                # Extract and print error is failed
                file.extract(member=member, path=root)
            except zipfile.error | rarfile.Error | tarfile.TarError as e:
                print(e)

    return list(set(os.listdir(root)) - contents) + [filename]


def categorize(**kwargs):
    # Retreive items from kwargs
    data_dir = kwargs["inp_dir"]
    criteria_fn = kwargs["criteria_fn"]
    categories = kwargs["categories"]

    # Create positive and negative dirs (for sunglasses/no sunglasses)
    pos_dir = os.path.join(os.path.dirname(data_dir), categories[0] + "/")
    neg_dir = os.path.join(os.path.dirname(data_dir), categories[1] + "/")

    # Make the actual directories
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # Count the total number of files in the directory tree, init tqdm
    basedir = os.path.basename(data_dir)
    total = sum(len(files) for _, _, files in os.walk(data_dir))
    pbar = tqdm(desc=f"    * Grouping images in '{basedir}'", total=total)

    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] not in VALID_EXTENSIONS:
                # Skip non-image files
                pbar.update(1)
                continue

            # Choose correct target directory
            filepath = os.path.join(root, filename)
            target_dir = pos_dir if criteria_fn(filepath) else neg_dir

            try:
                # Move to target dir, ignore duplicates
                shutil.move(filepath, target_dir)
            except shutil.Error:
                pass

            # Update pbar
            pbar.update(1)


def gen_splits(**kwargs):
    # Retrieve kwarg vals
    root = kwargs["root"]
    dirs = kwargs["categories"]
    out_dir = kwargs["out_dir"]

    # Calculate total number of file and initialize progress bar
    total = sum(len(os.listdir(os.path.join(root, x))) for x in dirs)
    pbar = tqdm(desc=f"    * Creating data splits", total=total)

    for dir in dirs:
        # List the files in the directory that has to be split and shuffle
        filenames = list(os.listdir(os.path.join(root, dir)))
        random.shuffle(filenames)

        # Compute the number of val and test files
        num_val = int(len(filenames) * kwargs["val_size"])
        num_test = int(len(filenames) * kwargs["test_size"])

        # Split filenames to 3 group types
        splits = {
            "train/": filenames[num_test + num_val :],
            "val/": filenames[num_test : num_test + num_val],
            "test/": filenames[:num_test],
        }

        for splitname, files in splits.items():
            # Create a split directory for the dir to split
            split_dir = os.path.join(out_dir, splitname, dir)
            os.makedirs(split_dir, exist_ok=True)

            for file in files:
                # Move all the files in this split to the created directory
                shutil.move(os.path.join(root, dir, file), split_dir)
                pbar.update(1)


def crop_align(**kwargs):
    # Initialize cropper
    cropper = Cropper(
        output_size=kwargs.get("size", (256, 256)),
        landmarks=kwargs.get("landmarks", None),
        output_format="jpg",
        padding="replicate",
        enh_threshold=None,
        device=kwargs.get("device", "cpu"),
        num_processes=kwargs.get("num_processes", 1),
    )

    for category in kwargs["categories"]:
        # Process directory (crop and align faces inside it)
        input_dir = os.path.join(kwargs["root"], category)
        pbar_desc = f"    * Cropping faces for {category}"
        cropper.process_dir(input_dir, desc=pbar_desc)

        # Remove the original dir
        shutil.rmtree(input_dir)
        os.rename(input_dir + "_faces", input_dir)


def resize_all(**kwargs):
    # Retrieve kwarg vals
    size = kwargs["size"]
    root = kwargs["root"]
    dirs = kwargs["categories"]

    # Calculate total number of file and initialize progress bar
    total = sum(len(os.listdir(os.path.join(root, dir))) for dir in dirs)
    pbar = tqdm(desc=f"    * Resizing images", total=total)

    for dir in dirs:
        for filename in os.listdir(os.path.join(root, dir)):
            # Generate filepath and open the image
            filepath = os.path.join(root, dir, filename)
            image = Image.open(filepath)

            if image.size != size:
                # Resize if needed
                image = image.resize(size)

            # Save under same path
            image.save(filepath)
            pbar.update(1)


def clean(contents, root="."):
    for file_or_dir in contents:
        # Create full path to the file or dir
        path = os.path.join(root, file_or_dir)

        if not os.path.exists(path):
            # Skip if not exists
            continue

        # Remove either file or dir
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def clear_keys(**kwargs):
    # Get key set and deletables
    key_set = set(kwargs.keys())
    to_del = ["inp_dir", "outp_dir", "criteria_fn", "landmarks", "num_processes"]

    for key in to_del:
        if key in key_set:
            # Del added key
            del kwargs[key]


def prepare_specs_on_faces(**kwargs):
    # Generate title to show in terminal
    generate_title("Specs on Faces")

    # Get root, update kwargs
    root = kwargs["root"]
    kwargs["inp_dir"] = os.path.join(root, "original images")
    kwargs["out_dir"] = os.path.join(root, "specs-on-faces")

    # Unpack contents that later will be removed
    contents = unpack("original images.rar", root)
    contents += unpack("metadata.rar", root)
    contents += kwargs["categories"]

    # Init landmarks, is_sunglasses set, metadata path and get_name fn
    names, coords, landmarks, is_sunglasses_set = [], [], {}, set()
    get_name = lambda path: "_".join(os.path.basename(path).split("_")[:4])
    mat_path = os.path.join(root, "metadata", "metadata.mat")

    for sample in loadmat(mat_path)["metadata"][0]:
        # Add landmarks to the dictionary
        name = sample[-1][0][0][0][:-2]
        landmarks[name] = np.array(sample[12][0]).reshape(-1, 2)

        if sample[10][0][0] in [2, 3]:
            # If sunglasses exist, add
            is_sunglasses_set.add(name)

    for filename in os.listdir(kwargs["inp_dir"]):
        # Append filenames and landms
        names.append(filename)
        coords.append(landmarks[get_name(filename)])

    # Create landmarks required to align and center-crop images
    kwargs["landmarks"] = np.stack(coords)[:, [3, 0, 14, 7, 6]], np.array(names)
    kwargs["criteria_fn"] = lambda path: get_name(path) in is_sunglasses_set
    kwargs["num_processes"] = cpu_count()

    # Sequential operations
    categorize(**kwargs)
    crop_align(**kwargs)
    gen_splits(**kwargs)
    clear_keys(**kwargs)
    clean(contents, root)


def prepare_cmu_face_images(**kwargs):
    # Generate title to show in terminal
    generate_title("CMU Face Images")

    # Get root, update kwargs
    root = kwargs["root"]
    kwargs["inp_dir"] = os.path.join(root, "faces")
    kwargs["out_dir"] = os.path.join(root, "cmu-face-images")
    kwargs["criteria_fn"] = lambda path: "sunglasses" in os.path.basename(path)

    # Unpack the contents from faces.tar.gz that's insde a zip file
    contents = unpack("cmu+face+images.zip", root, {"faces.tar.gz"})
    contents += unpack("faces.tar.gz", root)
    contents += kwargs["categories"]

    # Sequential operations
    categorize(**kwargs)
    crop_align(**kwargs)
    gen_splits(**kwargs)
    clear_keys(**kwargs)
    clean(contents, root)


def prepare_sunglasses_no_sunglasses(**kwargs):
    # Generate title to show in terminal
    generate_title("Sunglasses/No Sunglasses")

    # Get root, update kwargs
    root = kwargs["root"]
    kwargs["inp_dir"] = os.path.join(root, "glasses_noGlasses")
    kwargs["out_dir"] = os.path.join(root, "sunglasses-no-sunglasses")
    kwargs["criteria_fn"] = lambda filepath: "with_glasses" in filepath

    # Unpack the contents from the zip directory
    contents = unpack("sunglasses-no-sunglasses.zip", root)
    contents += kwargs["categories"]

    # Sequential operations
    categorize(**kwargs)
    resize_all(**kwargs)
    gen_splits(**kwargs)
    clear_keys(**kwargs)
    clean(contents, root)


def prepare_glasses_and_coverings(**kwargs):
    # Generate title to show in terminal
    generate_title("Glasses and Coverings")

    # Get root, update kwargs
    root = kwargs["root"]
    kwargs["inp_dir"] = os.path.join(root, "glasses-and-coverings")
    kwargs["out_dir"] = os.path.join(root, "glasses-and-coverings-done")

    target_dir = kwargs["categories"][0]  # eyeglasses or sunglasses
    target_dir = "glasses" if target_dir == "eyeglasses" else target_dir
    kwargs["criteria_fn"] = lambda path: target_dir == folder_name(path)

    # Unpack the contents from the zip directory
    contents = unpack("glasses-and-coverings.zip", root)
    contents += kwargs["categories"]

    # Sequential operations
    categorize(**kwargs)
    resize_all(**kwargs)
    gen_splits(**kwargs)
    clear_keys(**kwargs)
    clean(contents, root)

    # Rename the output directory to more exact name
    os.rename(kwargs["out_dir"], kwargs["inp_dir"])


def prepare_face_attributes_grouped(**kwargs):
    # Generate title to show in terminal, define 2 dirs
    generate_title("Face Attributes Grouped")
    inp1 = "face-attributes-grouped"
    inp2 = "face-attributes-extra"

    # Get root, update kwargs
    root = kwargs["root"]
    kwargs["inp_dir"] = os.path.join(root, inp1)
    kwargs["out_dir"] = os.path.join(root, inp1 + "-done")

    target_dir = kwargs["categories"][0]  # eyeglasses or sunglasses
    kwargs["criteria_fn"] = lambda path: target_dir in folder_name(path)

    # Define members to extract from both of the zip files
    mem1 = [f"{inp1}/{x}/eyewear" for x in ["train", "val", "test"]]
    mem2 = [f"{inp2}/{x}" for x in ["eyeglasses", "sunglasses", "no_eyewear"]]

    # Extract the specified members from both of the zip files
    contents = unpack("face-attributes-grouped.zip", root, mem1)
    contents += unpack("face-attributes-extra.zip", root, mem2)
    contents += kwargs["categories"]

    # Sequential operations with additional categorization of inp2
    categorize(**{**kwargs, "inp_dir": os.path.join(root, inp2)})
    categorize(**kwargs)
    resize_all(**kwargs)
    gen_splits(**kwargs)
    clear_keys(**kwargs)
    clean(contents, root)

    # Rename the output directory to more exact name
    os.rename(kwargs["out_dir"], kwargs["inp_dir"])


def generate_split_paths(split_info_file_paths, celeba_mapping_file_path, save_dir):
    for split_info_path in split_info_file_paths:
        # Read the first column of the the data split info file (filenames)
        file_names = np.genfromtxt(split_info_path, dtype=str, usecols=0)

        # Determine the type of split
        if "train" in split_info_path:
            train_set = {*file_names}
            subdir = "train"
        elif "val" in split_info_path:
            val_set = {*file_names}
            subdir = "val"
        elif "test" in split_info_path:
            test_set = {*file_names}
            subdir = "test"

        # Create image and mask directories as well while looping
        os.makedirs(os.path.join(save_dir, subdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, subdir, "masks"), exist_ok=True)

    # Init split info
    split_info = {}

    with open(celeba_mapping_file_path, "r") as f:
        for line in f.readlines()[1:]:
            # Read Celeba Mask HQ index and CelebA file name
            [idx, _, orig_file] = line.split()

            if orig_file in train_set:
                # If CelebA file name belongs to train dataset
                split_info[int(idx)] = os.path.join(save_dir, "train")
            elif orig_file in val_set:
                # If CelebA file name belongs to val dataset
                split_info[int(idx)] = os.path.join(save_dir, "val")
            elif orig_file in test_set:
                # If CelebA file name belongs to test dataset
                split_info[int(idx)] = os.path.join(save_dir, "test")

    return split_info


def walk_through_masks(mask_dir, img_dir, split_info, resize):
    # Count the total number of files in the directory tree, init tqdm
    total = sum(len(files) for _, _, files in os.walk(mask_dir))
    pbar = tqdm(desc="    * Selecting masks with glasses", total=total)

    for root, _, files in os.walk(mask_dir):
        for file in files:
            # Update pbar
            pbar.update(1)

            if "eye_g" not in file:
                # Ignore no-glasses
                continue

            # Get the train/val/test type
            idx = int(file.split("_")[0])
            parent_path = split_info[idx]

            # Create the full path to original files
            mask_path = os.path.join(root, file)
            image_path = os.path.join(img_dir, str(idx) + ".jpg")

            # Create a save path of original files to train/val/test location
            image_save_path = os.path.join(parent_path, "images", str(idx) + ".jpg")
            mask_save_path = os.path.join(
                parent_path, "masks", file.replace(".png", ".jpg")
            )

            # Open the image, convert mask to black/white
            image = Image.open(image_path).resize(resize)
            mask = Image.open(mask_path).resize(resize)
            mask = Image.fromarray((np.array(mask) > 0).astype(np.uint8) * 255)

            # Save the mask and the image
            image.save(image_save_path)
            mask.save(mask_save_path)


def prepare_celeba_mask_hq(**kwargs):
    # Generate title to show in terminal
    generate_title("Celeba Mask HQ")

    # Get root, update kwargs
    root = kwargs["root"]
    size = kwargs["size"]

    contents = unpack("CelebAMask-HQ.zip", root)
    contents += unpack("annotations.zip", root)

    # Create train/val/test split info
    split_info = generate_split_paths(
        [os.path.join(root, f"{x}_label.txt") for x in ["train", "val", "test"]],
        os.path.join(root, "CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"),
        os.path.join(root, "celeba-mask-hq"),
    )

    # Walk through samples and process
    walk_through_masks(
        os.path.join(root, "CelebAMask-HQ/CelebAMask-HQ-mask-anno"),
        os.path.join(root, "CelebAMask-HQ/CelebA-HQ-img"),
        split_info,
        size,
    )

    # Clean up data dir
    clean(contents, root)


def prepare_glasses_segmentation_synthetic(**kwargs):
    # Generate title to show in terminal
    generate_title("Glasses Segmentation Synthetic")

    # Get root
    root = kwargs["root"]
    frac = kwargs.get("sunglasses_fraction", 0.5)

    # Unpack the contents and
    unpack("glasses-segmentation-synthetic.zip", root)

    root = os.path.join(root, "glasses-segmentation-synthetic")
    total = sum(len(files) for _, _, files in os.walk(root))
    pbar = tqdm(desc=f"    * Selecting images and masks", total=total)

    for split_type in ["train", "val", "test"]:
        # Get all the filepaths, sort them, compute number of sunglasses
        filenames = sorted([f.name for f in os.scandir(os.path.join(root, split_type))])
        num_sunglasses = int((len(filenames) // 8) * frac)
        keep, i = set(), 0

        for filename in filenames:
            if "-seg" in filename:
                # Always add frame
                keep.add(filename)

            if (i >= num_sunglasses and "-all" in filename) or (
                i < num_sunglasses and "-sunglasses" in filename
            ):
                # Add either regular eyeglasses or sunglasses
                keep.add(filename)
                i += 1

        # Create 2 directories: for images and masks
        os.makedirs(os.path.join(root, split_type, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split_type, "masks"), exist_ok=True)

        for filename in filenames:
            # Get the filename in the current split
            split_path = os.path.join(root, split_type)
            filepath = os.path.join(split_path, filename)

            if filename not in keep:
                # Remove if not needed
                os.remove(filepath)
            elif "-seg" in filename:
                # Get the mask name without suffix, move to masks
                filename = filename.replace("-seg", "")
                shutil.move(filepath, os.path.join(split_path, "masks", filename))
            else:
                # Get the image name without suffix, move to images
                filename = filename.replace("-all", "").replace("-sunglasses", "")
                shutil.move(filepath, os.path.join(split_path, "images", filename))

            # Update other half
            pbar.update(1)

    # Remove the zip dir
    os.remove(root + ".zip")


def parse_kwargs():
    # Init parser for command-line interface
    parser = argparse.ArgumentParser()

    # Add the possible arguments
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        choices=[
            "eyeglasses-classification",
            "sunglasses-classification",
            "full-glasses-segmentation",
            "glass-frames-segmentation",
        ],
        help="The type of task to generate data splits for.",
    )
    parser.add_argument(
        "-r",
        "--root",
        required=True,
        help="The path to the directory with all the unzipped data files.",
    ),
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        nargs="+",
        default=[256, 256],
        help=f"The desired size (width, height) of cropped image faces. If "
        f"provided as a single number, the same value is used for both "
        f"width and height. Defaults to [256, 256].",
    )
    parser.add_argument(
        "-vs",
        "--val-size",
        type=float,
        default=0.15,
        help=f"The fraction of images to use for validation set. Note that "
        f"for some datasets this is ignored as default splits are known.",
    )
    parser.add_argument(
        "-ts",
        "--test-size",
        type=float,
        default=0.15,
        help=f"The fraction of images to use for test set. Note that for some "
        f"datasets this is ignored because default splits are known.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="",
        help=f"The device on which to perform preprocessing. Can be, for "
        f"example, 'cpu', 'cuda'. If not specified the device is chosen "
        f"CUDA or MPS, if either is available. Defaults to ''.",
    )

    # Parse the acquired args as kwargs
    kwargs = vars(parser.parse_args())

    # Add custom arguments
    match kwargs["task"]:
        case "eyeglasses-classification":
            kwargs["categories"] = ["eyeglasses", "no_eyeglasses"]
        case "sunglasses-classification":
            kwargs["categories"] = ["sunglasses", "no_sunglasses"]

    # Automatically determine the device
    if kwargs["device"] == "" and torch.cuda.is_available():
        kwargs["device"] = torch.device("cuda")
    elif kwargs["device"] == "" and torch.backends.mps.is_available():
        kwargs["device"] = torch.device("mps")
    elif kwargs["device"] == "":
        kwargs["device"] = torch.device("cpu")

    return kwargs


if __name__ == "__main__":
    # Get command-line args
    kwargs = parse_kwargs()

    match kwargs.pop("task"):
        case "eyeglasses-classification":
            prepare_glasses_and_coverings(**kwargs)
            prepare_face_attributes_grouped(**kwargs)
            # raise NotImplementedError("Sorry, not implemented yet!")
        case "sunglasses-classification":
            prepare_specs_on_faces(**kwargs)
            prepare_cmu_face_images(**kwargs)
            prepare_glasses_and_coverings(**kwargs)
            prepare_face_attributes_grouped(**kwargs)
            prepare_sunglasses_no_sunglasses(**kwargs)
        case "full-glasses-segmentation":
            prepare_celeba_mask_hq(**kwargs)
        case "glass-frames-segmentation":
            prepare_glasses_segmentation_synthetic(**kwargs)

    print()
