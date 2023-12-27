import argparse
import glob
import os
import random
import shutil
import sys
import tarfile
import warnings
import zipfile
from multiprocessing import cpu_count
from typing import Generator

import numpy as np
import rarfile
import torch
from face_crop_plus import Cropper
from PIL import Image
from pycocotools.coco import COCO
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


########################################################################
##########################                   ###########################
##########################     UTILITIES     ###########################
##########################                   ###########################
########################################################################


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


def get_extractable(
    filenames: dict[str, set[str] | dict] | list[str],
    root: str = ".",
) -> Generator[
    tuple[zipfile.ZipFile | rarfile.RarFile | tarfile.TarFile, str], None, None
]:
    if isinstance(filenames, list):
        # Convert to dict if a list of filenames is passed
        filenames = {filename: set() for filename in filenames}

    for filename, members in filenames.items():
        if not os.path.exists(file_path := os.path.join(root, filename)):
            warnings.warn(f"Please ensure {file_path} is a valid path.")
            continue

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

            for member in list(iterable):
                if len(members) > 0 and member not in members:
                    continue

                # Yield extractable
                yield file, member

                if isinstance(members, dict) and any(
                    member.endswith(ext) for ext in {".zip", ".rar", ".tar.gz"}
                ):
                    # If the member is archive, recurse
                    new_root = os.path.join(root, member)
                    yield from get_extractable(members[member], new_root)


def unpack_v2(
    filenames: list[str] | dict[str, set[str] | dict],
    root: str = ".",
    unpack_dir: str = ".",
    pbar: tqdm = None,
):
    # Create a directory to unpack files to
    unpack_path = os.path.join(root, unpack_dir)
    os.makedirs(unpack_path, exist_ok=True)

    # Calculate total number of files to extract
    total = sum(1 for _ in get_extractable(filenames, root))

    if pbar is None:
        # Initialize progress bar
        pbar = tqdm(total=total)
    else:
        # Update total and description
        pbar.total = pbar.total + total
        pbar.refresh()

    # Update description to indicate extracting
    pbar.set_description("Processing" if pbar.desc is None else pbar.desc)
    pbar.set_description(f"{pbar.desc} (extracting)")

    for file, member in get_extractable(filenames, root):
        # Update pbar
        pbar.update(1)

        if os.path.exists(os.path.join(root, member)):
            # Not extracting if it is already extracted
            continue

        try:
            # Extract and print error is failed
            file.extract(member=member, path=unpack_path)
        except zipfile.error | rarfile.Error | tarfile.TarError as e:
            print(e)

    # Update the description not to indicate extracting anymore
    pbar.set_description(pbar.desc.replace(" (extracting)", ""))


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


########################################################################
#########################                    ###########################
#########################   CLASSIFICATION   ###########################
#########################                    ###########################
########################################################################


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

    # Init landmarks, is_glasses set, metadata path and get_name fn
    names, coords, landmarks, is_glasses_set = [], [], {}, set()
    get_name = lambda path: "_".join(os.path.basename(path).split("_")[:4])
    set_type = [2, 3] if kwargs["categories"][0] == "sunglasses" else [1]
    mat_path = os.path.join(root, "metadata", "metadata.mat")

    for sample in loadmat(mat_path)["metadata"][0]:
        # Add landmarks to the dictionary
        name = sample[-1][0][0][0][:-2]
        landmarks[name] = np.array(sample[12][0]).reshape(-1, 2)

        if sample[10][0][0] in set_type:
            # If glasses exist, add
            is_glasses_set.add(name)

    for filename in os.listdir(kwargs["inp_dir"]):
        # Append filenames and landms
        names.append(filename)
        coords.append(landmarks[get_name(filename)])

    # Create landmarks required to align and center-crop images
    kwargs["landmarks"] = np.stack(coords)[:, [3, 0, 14, 7, 6]], np.array(names)
    kwargs["criteria_fn"] = lambda path: get_name(path) in is_glasses_set
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

    # Define the criteria function
    target_dir = kwargs["categories"][0]  # "eyeglasses" or "sunglasses"
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

    # Define the criteria function
    target_dir = kwargs["categories"][0]  # "eyeglasses" or "sunglasses"
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


########################################################################
#########################                    ###########################
#########################    SEGMENTATION    ###########################
#########################                    ###########################
########################################################################


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


########################################################################
##########################                   ###########################
##########################     DETECTION     ###########################
##########################                   ###########################
########################################################################


# def parse_tensorflow_csv(
#     csv_path: str,
#     class_map: dict[str : list[str]] = {},
#     size: tuple[int, int] = (256, 256),
# ):
#     # E.g., {
#     #     "Glasses": ["data/detection/worn/test", "data/classification/eyeglasses/test", "data/classification/anyglasses/test", "data/classification/no_sunglasses/test"],
#     #     "Sunglasses": ["data/detection/worn/test", "data/classification/sunglasses/test", "data/classification/anyglasses/test", "data/classification/no_eyeglasses/test"],
#     #     "No Glasses": ["data/detection/eyes/test", "data/classification/no_eyeglasses/test", "data/classification/no_sunglasses/test", "data/classification/no_anyglasses/test"],
#     # }
#     class_map

#     # Get the path to image directory
#     img_dir = os.path.dirname(csv_path)

#     with open(csv_path, "r") as f:
#         for line in f.readlines()[1:]:
#             # Get the image name, size, class name, and the bounding box
#             [filename, w, h, class_name, x1, y1, x2, y2] = line.split(",")

#             if class_name not in class_map.keys():
#                 # Skip if not needed
#                 continue

#             # Open the image and resize if needed
#             img = Image.open(os.path.join(img_dir, filename)).resize(size)

#             for path in class_map[class_name]:
#                 if "classification" in path:
#                     # Save the resized image only for classification
#                     img.save(os.path.join(path, "images", filename))
#                     continue

#                 # Copy the image and create .txt annotation filename
#                 img.save(os.path.join(path, "images", filename))
#                 txt = filename.split(".")[0] + ".txt"

#                 # Compute the normalized bounding box
#                 bbox = [
#                     (int(x1) + int(x2)) / 2 / size[0],
#                     (int(y1) + int(y2)) / 2 / size[1],
#                     (int(x2) - int(x1)) / size[0],
#                     (int(y2) - int(y1)) / size[1],
#                 ]

#                 with open(os.path.join(path, "annotations", txt), "w") as f:
#                     # Save the bounding box to a .txt
#                     f.write(" ".join(map(str, bbox)))


def parse_coco_json(
    json_path: str,
    class_map: dict[str : list[str]] = {},
    size: tuple[int, int] = (256, 256),
    pbar: tqdm = None,
):
    # Get img dir, load COCO annotations
    img_dir = os.path.dirname(json_path)

    # Save original std and redirect stdout and stderr to os.devnull
    original_stdout, sys.stdout = sys.stdout, open(os.devnull, "w")
    original_stderr, sys.stderr = sys.stderr, open(os.devnull, "w")

    # Load COCO annotations
    coco = COCO(json_path)

    # Restore the original stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Create a dictionary to map category ids to category names
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.dataset["categories"]}

    for img_id in list(coco.imgs.keys()):
        # Get the image info and class name
        img_info = coco.loadImgs(img_id)[0]

        # Get annotation id for the image
        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            # Get the class name of the current annotation
            class_name = cat_id_to_name[ann["category_id"]]

            if class_name not in class_map.keys():
                continue

            # Load the image and resize if needed
            img = Image.open(os.path.join(img_dir, img_info["file_name"]))
            img = img.resize(size)

            for path in class_map[class_name]:
                if "classification" in path:
                    # Save the resized image only for classification
                    img.save(os.path.join(path, img_info["file_name"]))
                elif "detection" in path:
                    # Normalize bbox (x_center, y_center, width, height)
                    x = (ann["bbox"][0] + ann["bbox"][2] / 2) / img_info["width"]
                    y = (ann["bbox"][1] + ann["bbox"][3] / 2) / img_info["height"]
                    w = ann["bbox"][2] / img_info["width"]
                    h = ann["bbox"][3] / img_info["height"]

                    # Copy the image and create .txt annotation filename
                    img.save(os.path.join(path, "images", img_info["file_name"]))
                    txt = img_info["file_name"].rsplit(".", 1)[0] + ".txt"

                    with open(os.path.join(path, "annotations", txt), "w") as f:
                        # Write the bounding box
                        f.write(f"{x} {y} {w} {h}")
                elif "segmentation" in path:
                    # Generate the mask from annotation and resize it
                    mask = np.zeros((img_info["height"], img_info["width"]))
                    mask = np.maximum(coco.annToMask(ann), mask) * 255
                    msk = Image.fromarray(mask).convert("1").resize(size)

                    # Save the mask and the image to corresponding dirs
                    img.save(os.path.join(path, "images", img_info["file_name"]))
                    msk.save(os.path.join(path, "masks", img_info["file_name"]))

        # Update pbar
        pbar.update(1)


def walk_coco_splits(
    data_file: str,
    save_name: str,
    root: str = "data",
    class_map: dict[str, dict[str, list[str]]] = {},
    size: tuple[int, int] = (256, 256),
    delete_original: bool = False,
    tasks: list[str] = [],
    **kwargs,
):
    for key, val in list(class_map.items()):
        # Filter out unwanted task preprocessing
        class_map[key] = {k: v for k, v in val.items() if k in tasks}

    for class_name, task_map in class_map.items():
        for task_name, task_cats in task_map.items():
            for task_cat in task_cats:
                # Check how many files are in save_dir
                task_cat = task_cat.replace("no_", "")
                save_path = os.path.join(root, task_name, task_cat, save_name)

                if not os.path.exists(save_path):
                    continue

                if sum(len(files) for _, _, files in os.walk(save_path)) > 0:
                    print(f"* Skipping {save_name} (already processed)")
                    return

    # Initialize tqdm progress bar for current dataset
    pbar_desc = f"* Processing {save_name}"
    pbar = tqdm(desc=pbar_desc, total=0)

    # Unpack the data files
    unpack_v2(
        [data_file],
        root=root,
        unpack_dir="tmp",
        pbar=pbar,
    )

    # Compute total files to process, update pbar
    extracted_path = os.path.join(root, "tmp")
    total = sum(len(files) for _, _, files in os.walk(extracted_path))
    pbar.total += total + 1
    pbar.refresh()
    pbar.set_description(f"{pbar_desc} (categorizing)")

    for split_type in os.scandir(extracted_path):
        if not split_type.is_dir():
            pbar.update(1)
            continue

        # Get the split name + the path to the JSON file, init class map
        split_name = split_type.name if split_type.name != "valid" else "val"
        json_path = glob.glob(os.path.join(split_type.path, "*.json"))[0]
        _class_map = {key: [] for key in class_map.keys()}
        pbar.update(1)

        for class_name, task_map in class_map.items():
            for task_name, task_cats in task_map.items():
                for task_cat in task_cats:
                    if task_name == "classification":
                        # Binary dir name is given
                        bin_name = task_cat
                        task_cat = task_cat.replace("no_", "")
                    else:
                        # Not classification
                        bin_name = None

                    # Create the output directory
                    join = [root, task_name, task_cat, save_name, split_name]
                    join += [bin_name] if bin_name is not None else []
                    output_dir = os.path.join(*join)

                    if task_name == "classification":
                        # Create positive or negative sub-dir
                        os.makedirs(output_dir, exist_ok=True)
                    elif task_name == "detection":
                        # Create images and annotations sub-dirs
                        img_dir = os.path.join(output_dir, "images")
                        ann_dir = os.path.join(output_dir, "annotations")
                        os.makedirs(img_dir, exist_ok=True)
                        os.makedirs(ann_dir, exist_ok=True)
                    elif task_name == "segmentation":
                        # Create images and masks sub-dirs
                        img_dir = os.path.join(output_dir, "images")
                        msk_dir = os.path.join(output_dir, "masks")
                        os.makedirs(img_dir, exist_ok=True)
                        os.makedirs(msk_dir, exist_ok=True)

                    # Append output parent dir to the class map
                    _class_map[class_name].append(output_dir)

        # Parse the COCO JSON file
        parse_coco_json(json_path, _class_map, size, pbar)

    # Init deletable files and dirs and clean them up
    pbar.set_description(f"{pbar_desc} (cleaning)")
    deletable = ["tmp"] + ([data_file] if delete_original else [])
    clean(deletable, root)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (done)")


def prepare_ai_pass(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "AI-Pass.v6i.coco.zip"
    kwargs["save_name"] = "ai-pass"
    kwargs["class_map"] = {
        "glasses": {
            "detection": ["worn"],
            "classification": ["eyeglasses", "no_sunglasses"],
        },
        "sunglasses": {
            "detection": ["worn"],
            "classification": ["sunglasses", "no_eyeglasses"],
        },
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_pex5(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "PEX5.v4i.coco.zip"
    kwargs["save_name"] = "pex5"
    kwargs["class_map"] = {
        "glasses": {
            "detection": ["worn"],
            "classification": ["eyeglasses", "no_sunglasses"],
        },
        "sunglasses": {
            "detection": ["worn"],
            "classification": ["sunglasses", "no_eyeglasses"],
        },
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_sunglasses_glasses_detect(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "sunglasses_glasses_detect.v1i.coco.zip"
    kwargs["save_name"] = "sunglasses-glasses-detect"
    kwargs["class_map"] = {
        "glasses": {
            "detection": ["worn"],
            "classification": ["eyeglasses", "no_sunglasses"],
        },
        "sunglasses": {
            "detection": ["worn"],
            "classification": ["sunglasses", "no_eyeglasses"],
        },
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_glasses_detection(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "Glasses Detection.v2i.coco.zip"
    kwargs["save_name"] = "glasses-detection"
    kwargs["class_map"] = {
        "Glasses": {
            "detection": ["worn"],
            "classification": ["eyeglasses", "anyglasses", "no_sunglasses"],
        },
        "Sunglasses": {
            "detection": ["worn"],
            "classification": ["sunglasses", "anyglasses", "no_eyeglasses"],
        },
        "No Glasses": {
            "detection": ["eyes"],
            "classification": ["no_eyeglasses", "no_sunglasses", "no_anyglasses"],
        },
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_glasses_image_dataset(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "glasses.v1-glasses_2022-04-01-8-12pm.coco.zip"
    kwargs["save_name"] = "glasses-image-dataset"
    kwargs["class_map"] = {
        "glasses": {
            "detection": ["worn"],
            "classification": ["eyeglasses", "anyglasses", "no_sunglasses"],
        },
        "sun_glasses": {
            "detection": ["worn"],
            "classification": ["sunglasses", "anyglasses", "no_eyeglasses"],
        },
        "no_glasses": {
            "detection": ["eyes"],
            "classification": ["no_eyeglasses", "no_sunglasses", "no_anyglasses"],
        },
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_ex07(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "Ex07.v1i.coco.zip"
    kwargs["save_name"] = "ex07"
    kwargs["class_map"] = {
        "eyesglass": {
            "detection": ["worn"],
            "classification": ["anyglasses"],
        },
        "without_eyesglass": {
            "detection": ["eyes"],
            "classification": ["no_anyglasses"],
        },
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


########################################################################
##########################                   ###########################
##########################        CLI        ###########################
##########################                   ###########################
########################################################################


def parse_kwargs():
    # Init parser for command-line interface
    parser = argparse.ArgumentParser()

    # Add the possible arguments
    # parser.add_argument(
    #     "-t",
    #     "--task",
    #     required=True,
    #     choices=[
    #         "eyeglasses-classification",
    #         "sunglasses-classification",
    #         "full-glasses-segmentation",
    #         "glass-frames-segmentation",
    #     ],
    #     help="The type of task to generate data splits for.",
    # )
    parser.add_argument(
        "-t",
        "--tasks",
        type=str,
        nargs="+",
        default=["classification", "segmentation", "detection"],
        help="The type of tasks to generate data splits for.",
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="data",
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
        f"for some datasets this is ignored since default splits are known.",
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
    parser.add_argument(
        "-del",
        "--delete-original",
        action="store_true",
        help=f"Whether to delete the original zip file after unpacking. ",
    )

    # Parse the acquired args as kwargs
    kwargs = vars(parser.parse_args())

    # Add custom arguments
    # match kwargs["task"]:
    #     case "eyeglasses-classification":
    #         kwargs["categories"] = ["eyeglasses", "no_eyeglasses"]
    #     case "sunglasses-classification":
    #         kwargs["categories"] = ["sunglasses", "no_sunglasses"]

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

    # match kwargs.pop("task"):
    #     case "eyeglasses-classification":
    #         prepare_specs_on_faces(**kwargs)
    #         prepare_glasses_and_coverings(**kwargs)
    #         prepare_face_attributes_grouped(**kwargs)
    #     case "sunglasses-classification":
    #         prepare_specs_on_faces(**kwargs)
    #         prepare_cmu_face_images(**kwargs)
    #         prepare_glasses_and_coverings(**kwargs)
    #         prepare_face_attributes_grouped(**kwargs)
    #         prepare_sunglasses_no_sunglasses(**kwargs)
    #     case "full-glasses-segmentation":
    #         prepare_celeba_mask_hq(**kwargs)
    #     case "glass-frames-segmentation":
    #         prepare_glasses_segmentation_synthetic(**kwargs)

    prepare_ai_pass(**kwargs)
    prepare_pex5(**kwargs)
    prepare_sunglasses_glasses_detect(**kwargs)
    prepare_glasses_detection(**kwargs)
    prepare_glasses_image_dataset(**kwargs)
    prepare_ex07(**kwargs)
