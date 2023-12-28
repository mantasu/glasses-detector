import argparse
import glob
import os
import random
import shutil
import sys
import tarfile
import warnings
import zipfile
from collections import defaultdict
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Callable, Generator

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


# def generate_title(title, pad=5):
#     # Generate title with borders and print it
#     midline = "#" * pad + " " + title + " " + "#" * pad
#     top_bot = "#" * len(midline)
#     print("\n".join(["\n" + top_bot, midline, top_bot]))


# def folder_name(filepath):
#     return os.path.basename(os.path.dirname(filepath))


# def unpack(filename, root=".", members=set()):
#     if not os.path.exists(file_path := os.path.join(root, filename)):
#         warnings.warn(f"Please ensure {file_path} is a valid path.")
#         return []

#     # Check the contents before extracting
#     contents = set(os.listdir(root))

#     # Choose a correct unpacking interface
#     if file_path.endswith(".zip"):
#         open_file = zipfile.ZipFile
#     elif file_path.endswith(".rar"):
#         open_file = rarfile.RarFile
#     elif file_path.endswith(".tar.gz"):
#         open_file = tarfile.open

#     with open_file(file_path) as file:
#         # Choose a correct method to list files inside pack
#         if isinstance(file, (zipfile.ZipFile, rarfile.RarFile)):
#             iterable = file.namelist()
#         elif isinstance(file, tarfile.TarFile):
#             iterable = file.getnames()

#         for member in tqdm(iterable, desc=f"    * Extracting '{filename}'"):
#             if len(members) > 0 and not any(member.startswith(m) for m in members):
#                 # Skip if not needed to extract
#                 continue

#             if os.path.exists(os.path.join(root, member)):
#                 # Not extracting if it is already extracted
#                 continue

#             try:
#                 # Extract and print error is failed
#                 file.extract(member=member, path=root)
#             except zipfile.error | rarfile.Error | tarfile.TarError as e:
#                 print(e)

#     return list(set(os.listdir(root)) - contents) + [filename]


def get_extractable(
    filenames: list[str] | dict[str, list[str] | dict],
    root: str = ".",
    unpack_dir: str | None = None,
    is_nested: bool = False,
) -> Generator[
    tuple[zipfile.ZipFile | rarfile.RarFile | tarfile.TarFile, str, bool], None, None
]:
    if isinstance(filenames, list):
        # Convert to dict if a list of filenames is passed
        filenames = {filename: {} for filename in filenames}

    for filename, members in filenames.items():
        if not os.path.exists(file_path := os.path.join(root, filename)):
            if not is_nested:
                # Raise a warning if the top-level archive is not found
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
                if len(members) > 0 and not any(member.startswith(m) for m in members):
                    continue

                yield file, member, is_nested

                if not any(member.endswith(ext) for ext in {".zip", ".rar", ".tar.gz"}):
                    # Skip if the current member is not an archive
                    continue

                # Nested root and filenames
                new_filenames = [member]
                new_root = root

                if unpack_dir is not None:
                    # Top archive was extracted to unpack_dir
                    new_root = os.path.join(root, unpack_dir)

                if isinstance(members, dict) and len(members) > 0:
                    # Nested archive has specific members to extract
                    new_filenames = {member: members[member]}

                yield from get_extractable(new_filenames, new_root, is_nested=True)


def unpack(
    filenames: list[str] | dict[str, list[str] | dict],
    root: str = ".",
    unpack_dir: str = ".",
    pbar: tqdm | None = None,
    update_total: bool = False,
):
    # Create a directory to unpack files to
    unpack_path = os.path.join(root, unpack_dir)
    os.makedirs(unpack_path, exist_ok=True)

    if pbar is not None and update_total:
        # Caclulate num files to extract and update pbar total
        total = sum(1 for _ in get_extractable(filenames, root))
        pbar.total = pbar.total + total
        pbar.refresh()
    elif pbar is not None:
        # Update description to indicate extracting
        pbar.set_description(f"{pbar.desc} (extracting)")

    for file, member, is_nested in get_extractable(filenames, root, unpack_dir):
        try:
            # Extract and print error if failed
            file.extract(member=member, path=unpack_path)
        except zipfile.error | rarfile.Error | tarfile.TarError as e:
            print(e)

        if pbar is not None and not update_total:
            # Update pbar
            pbar.update(1)
        elif pbar is not None and not is_nested:
            # Update pbar
            pbar.update(1)

    if pbar is not None:
        # Update the description not to indicate extracting anymore
        pbar.set_description(pbar.desc.replace(" (extracting)", ""))


# def categorize(**kwargs):
#     # Retreive items from kwargs
#     data_dir = kwargs["inp_dir"]
#     criteria_fn = kwargs["criteria_fn"]
#     categories = kwargs["categories"]

#     # Create positive and negative dirs (for sunglasses/no sunglasses)
#     pos_dir = os.path.join(os.path.dirname(data_dir), categories[0] + "/")
#     neg_dir = os.path.join(os.path.dirname(data_dir), categories[1] + "/")

#     # Make the actual directories
#     os.makedirs(pos_dir, exist_ok=True)
#     os.makedirs(neg_dir, exist_ok=True)

#     # Count the total number of files in the directory tree, init tqdm
#     basedir = os.path.basename(data_dir)
#     total = sum(len(files) for _, _, files in os.walk(data_dir))
#     pbar = tqdm(desc=f"    * Grouping images in '{basedir}'", total=total)

#     for root, _, filenames in os.walk(data_dir):
#         for filename in filenames:
#             if os.path.splitext(filename)[1] not in VALID_EXTENSIONS:
#                 # Skip non-image files
#                 pbar.update(1)
#                 continue

#             # Choose correct target directory
#             filepath = os.path.join(root, filename)
#             target_dir = pos_dir if criteria_fn(filepath) else neg_dir

#             try:
#                 # Move to target dir, ignore duplicates
#                 shutil.move(filepath, target_dir)
#             except shutil.Error:
#                 pass

#             # Update pbar
#             pbar.update(1)


# def gen_splits(**kwargs):
#     # Retrieve kwarg vals
#     root = kwargs["root"]
#     dirs = kwargs["categories"]
#     out_dir = kwargs["out_dir"]

#     # Calculate total number of file and initialize progress bar
#     total = sum(len(os.listdir(os.path.join(root, x))) for x in dirs)
#     pbar = tqdm(desc=f"    * Creating data splits", total=total)

#     for dir in dirs:
#         # List the files in the directory that has to be split and shuffle
#         filenames = list(os.listdir(os.path.join(root, dir)))
#         random.shuffle(filenames)

#         # Compute the number of val and test files
#         num_val = int(len(filenames) * kwargs["val_size"])
#         num_test = int(len(filenames) * kwargs["test_size"])

#         # Split filenames to 3 group types
#         splits = {
#             "train/": filenames[num_test + num_val :],
#             "val/": filenames[num_test : num_test + num_val],
#             "test/": filenames[:num_test],
#         }

#         for splitname, files in splits.items():
#             # Create a split directory for the dir to split
#             split_dir = os.path.join(out_dir, splitname, dir)
#             os.makedirs(split_dir, exist_ok=True)

#             for file in files:
#                 # Move all the files in this split to the created directory
#                 shutil.move(os.path.join(root, dir, file), split_dir)
#                 pbar.update(1)


# def crop_align(**kwargs):
#     # Initialize cropper
#     cropper = Cropper(
#         output_size=kwargs.get("size", (256, 256)),
#         landmarks=kwargs.get("landmarks", None),
#         output_format="jpg",
#         padding="replicate",
#         enh_threshold=None,
#         device=kwargs.get("device", "cpu"),
#         num_processes=kwargs.get("num_processes", 1),
#     )

#     for category in kwargs["categories"]:
#         # Process directory (crop and align faces inside it)
#         input_dir = os.path.join(kwargs["root"], category)
#         pbar_desc = f"    * Cropping faces for {category}"
#         cropper.process_dir(input_dir, desc=pbar_desc)

#         # Remove the original dir
#         shutil.rmtree(input_dir)
#         os.rename(input_dir + "_faces", input_dir)


# def resize_all(**kwargs):
#     # Retrieve kwarg vals
#     size = kwargs["size"]
#     root = kwargs["root"]
#     dirs = kwargs["categories"]

#     # Calculate total number of file and initialize progress bar
#     total = sum(len(os.listdir(os.path.join(root, dir))) for dir in dirs)
#     pbar = tqdm(desc=f"    * Resizing images", total=total)

#     for dir in dirs:
#         for filename in os.listdir(os.path.join(root, dir)):
#             # Generate filepath and open the image
#             filepath = os.path.join(root, dir, filename)
#             image = Image.open(filepath)

#             if image.size != size:
#                 # Resize if needed
#                 image = image.resize(size)

#             # Save under same path
#             image.save(filepath)
#             pbar.update(1)

# def clear_keys(**kwargs):
#     # Get key set and deletables
#     key_set = set(kwargs.keys())
#     to_del = ["inp_dir", "outp_dir", "criteria_fn", "landmarks", "num_processes"]

#     for key in to_del:
#         if key in key_set:
#             # Del added key
#             del kwargs[key]


def clean(contents: list[str], root: str = "."):
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


########################################################################
#########################                    ###########################
#########################   CLASSIFICATION   ###########################
#########################                    ###########################
########################################################################


def categorize_binary(
    data_files: list[str] | dict[str, dict | list[str]],
    cat_map: dict[str, Callable[[str], bool]],
    save_name: str,
    root: str = "data",
    split_fn: Callable[[str], str] | None = None,
    size: tuple[int, int] = (256, 256),
    delete_original: bool = False,
    **kwargs,
):
    for cat in os.scandir(os.path.join(root, "classification")):
        if not cat.is_dir():
            continue

        for ds_dir in os.scandir(cat.path):
            if ds_dir.name == save_name and kwargs.get("force", False):
                # Remove the dataset if exists
                shutil.rmtree(ds_dir.path)
            elif (
                ds_dir.name == save_name
                and sum(len(files) for _, _, files in os.walk(ds_dir.path)) > 0
            ):
                print(f"* Skipping {save_name} (already processed)")
                return

    # Initialize tqdm progress bar for current dataset
    pbar_desc = f"* Processing {save_name}"
    pbar_total = kwargs.get("total", 0)
    update_total = pbar_total == 0
    pbar = tqdm(desc=pbar_desc, total=pbar_total)

    # Unpack the contents from the dataset archives
    unpack(data_files, root, "tmp", pbar, update_total)
    extracted_path = os.path.join(root, "tmp")

    # Update pbar description, unpack criteria, init src2tgt
    pbar.set_description(f"{pbar_desc} (reading contents)")
    bin_count = defaultdict(lambda: defaultdict(lambda: 0))
    src_to_tgt = defaultdict(lambda: [])

    for _root, _, filenames in os.walk(extracted_path):
        for filename in filenames:
            if os.path.splitext(filename)[1] not in VALID_EXTENSIONS:
                continue

            # Choose the correct target directory
            filepath = os.path.join(_root, filename)

            # Choose the correct split type (always "train" by default)
            split = "train" if split_fn is None else split_fn(filepath)

            for cat, fn in cat_map.items():
                # Get the dirname (pos/neg) based on the criteria
                dirname = ("" if fn(filepath) else "no_") + cat

                # Create the target dir
                tgt_dir = os.path.join(
                    root,
                    "classification",
                    cat,
                    save_name,
                    split,
                    dirname,
                )

                # Create tgt_dir and update src2tgt
                os.makedirs(tgt_dir, exist_ok=True)
                src_to_tgt[filepath].append(tgt_dir)
                bin_count[cat][dirname] += 1

            # Update total
            pbar_total += 1

    if split_fn is None:
        for cat, pos_neg_count in bin_count.items():
            for dirname, count in pos_neg_count.items():
                # Generate train/val/test splits for each cat & bin
                num_val = int(count * kwargs.get("val_size", 0.0))
                num_test = int(count * kwargs.get("test_size", 0.0))
                val_cnt, test_cnt = 0, 0

                for key in sorted(src_to_tgt.keys()):
                    for i in range(len(src_to_tgt[key])):
                        # Split the target path to multiple parts
                        path = os.path.normpath(src_to_tgt[key][i]).split(os.sep)

                        if path[-1] != dirname:
                            continue

                        if val_cnt < num_val:
                            # train -> "val"
                            path[-2] = "val"
                            val_cnt += 1
                        elif test_cnt < num_test:
                            # train -> "test"
                            path[-2] = "test"
                            test_cnt += 1

                        # Update the target path and make dirs
                        src_to_tgt[key][i] = os.sep.join(path)
                        os.makedirs(src_to_tgt[key][i], exist_ok=True)

    if update_total:
        # Update progress bar total (+1 for cleaning)
        pbar.total = pbar_total + pbar.total + 1
        pbar.refresh()

    # Update pbar description to indicate categorization
    pbar.set_description(f"{pbar_desc} (categorizing)")

    for src, tgts in src_to_tgt.items():
        # Open the image and resize if needed
        image = Image.open(src).resize(size)

        for tgt in tgts:
            # Save the image to target dir (as .jpg image)
            name, _ = os.path.splitext(os.path.basename(src))
            image.save(os.path.join(tgt, name + ".jpg"), "JPEG")

        # Update pbar
        pbar.update(1)

    # Init deletable files and dirs and clean them up
    pbar.set_description(f"{pbar_desc} (cleaning)")
    files = list(data_files.keys()) if isinstance(data_files, dict) else data_files
    deletable = ["tmp"] + (files if delete_original else [])
    clean(deletable, root)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (done)")


def prepare_specs_on_faces(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "specs-on-faces"
    kwargs["data_files"] = ["original images.rar"]

    # Unpack metadata immediately (before data archive)
    unpack(["metadata.rar"], kwargs["root"], "tmp")

    # Init glasses sets, helper function and metadata path
    is_eyeglasses_set, is_sunglasses_set = set(), set()
    get_name = lambda path: "_".join(os.path.basename(path).split("_")[:4])
    mat_path = os.path.join(kwargs["root"], "tmp", "metadata", "metadata.mat")

    for sample in loadmat(mat_path)["metadata"][0]:
        # Get the name of the sample
        name = sample[-1][0][0][0][:-2]

        if sample[10][0][0] == 1:
            # If transparent
            is_eyeglasses_set.add(name)
        elif sample[10][0][0] in {2, 3}:
            # If semi-transparent/opaque
            is_sunglasses_set.add(name)

    # Remove the temporary directory (if not removed)
    shutil.rmtree(os.path.join(kwargs["root"], "tmp"))

    # Add category map
    kwargs["cat_map"] = {
        "eyeglasses": lambda path: get_name(path) in is_eyeglasses_set,
        "sunglasses": lambda path: get_name(path) in is_sunglasses_set,
    }

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_cmu_face_images(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "cmu-face-images"
    kwargs["data_files"] = {"cmu+face+images.zip": ["faces.tar.gz"]}

    # Add category map
    kwargs["cat_map"] = {
        "sunglasses": lambda path: "sunglasses" in os.path.basename(path),
        "anyglasses": lambda path: "sunglasses" in os.path.basename(path),
    }

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_sunglasses_no_sunglasses(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "sunglasses-no-sunglasses"
    kwargs["data_files"] = ["sunglasses-no-sunglasses.zip"]

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))
    ffn = lambda path: folder_name(os.path.dirname(path))

    # Add category map
    kwargs["cat_map"] = {
        "sunglasses": lambda path: folder_name(path) == "with_glasses",
        "anyglasses": lambda path: folder_name(path) == "with_glasses",
    }

    # Add split type check function
    kwargs["split_fn"] = lambda path: "val" if ffn(path) == "valid" else "train"

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_glasses_and_coverings(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "glasses-and-coverings"
    kwargs["data_files"] = {
        "glasses-and-coverings.zip": [
            "glasses-and-coverings/plain",
            "glasses-and-coverings/glasses",
            "glasses-and-coverings/sunglasses",
            "glasses-and-coverings/sunglasses-imagenet",
        ]
    }

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))

    # Add category map
    kwargs["cat_map"] = {
        "eyeglasses": lambda path: folder_name(path) == "glasses",
        "sunglasses": lambda path: folder_name(path)
        in ["sunglasses", "sunglasses-imagenet"],
        "anyglasses": lambda path: folder_name(path) != "plain",
    }

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_face_attributes_grouped(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "face-attributes-grouped"
    kwargs["data_files"] = {
        "face-attributes-grouped.zip": [
            "face-attributes-grouped/train/eyewear",
            "face-attributes-grouped/train/nowear",
            "face-attributes-grouped/val/eyewear",
            "face-attributes-grouped/val/nowear",
            "face-attributes-grouped/test/eyewear",
            "face-attributes-grouped/test/nowear",
        ]
    }

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))
    fffn = lambda path: folder_name(os.path.dirname(os.path.dirname(path)))

    # Add category map
    kwargs["cat_map"] = {
        "eyeglasses": lambda path: folder_name(path) == "eyeglasses",
        "sunglasses": lambda path: folder_name(path) == "sunglasses",
        "anyglasses": lambda path: folder_name(path) in ["eyeglasses", "sunglasses"],
    }

    # Add split type check function
    kwargs["split_fn"] = lambda path: fffn(path)

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_face_attributes_extra(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "face-attributes-extra"
    kwargs["data_files"] = {
        "face-attributes-extra.zip": [
            "face-attributes-extra/sunglasses",
            "face-attributes-extra/eyeglasses",
            "face-attributes-extra/no_eyewear",
        ]
    }

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))

    # Add category map
    kwargs["cat_map"] = {
        "eyeglasses": lambda path: folder_name(path) == "eyeglasses",
        "sunglasses": lambda path: folder_name(path) == "sunglasses",
        "anyglasses": lambda path: folder_name(path) in ["eyeglasses", "sunglasses"],
    }

    # Prepare for classification
    categorize_binary(**kwargs)


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


def prepare_eyeglass(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "eyeglass.v10i.coco-segmentation.zip"
    kwargs["save_name"] = "eyeglass"
    kwargs["class_map"] = {
        "frame": {"segmentation": ["frames"]},
        "glass": {"segmentation": ["lenses"]},
    }

    def seg_fn(class_name: str, masks: dict[str, np.ndarray]):
        # Get the masks for the current class
        mask = masks[class_name]

        if class_name == "frame" and "glass" in masks.keys():
            # Subtract glass from frame
            mask -= masks["glass"]
        elif class_name == "frame":
            # Raise KeyError if no glass mask found
            raise KeyError("No glass mask found")

        return mask

    # Add seg_fn to kwargs
    kwargs["seg_fn"] = seg_fn

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_glasses_lenses(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs[
        "data_file"
    ] = "glasses lenses segmentation.v7-sh-improvments-version.coco.zip"
    kwargs["save_name"] = "glasses-lenses"
    kwargs["class_map"] = {"glasses-lenses": {"segmentation": ["lenses"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_glasses_lens(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "glasses lens.v6i.coco-segmentation.zip"
    kwargs["save_name"] = "glasses-lens"
    kwargs["class_map"] = {"glasses": {"segmentation": ["lenses"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_glasses_segmentation_cropped_faces(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs[
        "data_file"
    ] = "glasses segmentation cropped faces.v2-segmentation_models_pytorch-s_1st_version.coco-segmentation.zip"
    kwargs["save_name"] = "glasses-segmentation-cropped-faces"
    kwargs["class_map"] = {"glasses-lenses": {"segmentation": ["lenses"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


########################################################################
##########################                   ###########################
##########################     DETECTION     ###########################
##########################                   ###########################
########################################################################


def parse_coco_json(
    json_path: str,
    class_map: dict[str : list[str]] = {},
    size: tuple[int, int] = (256, 256),
    pbar: tqdm = None,
    **kwargs,
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

        # Create a dictionary to store masks for segmentation task
        masks = defaultdict(lambda: np.zeros((img_info["height"], img_info["width"])))

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
                    # Update the mask for the current class
                    mask = masks[class_name]
                    masks[class_name] = np.maximum(coco.annToMask(ann), mask)

        if len(masks) > 0:
            # Get the mask retrieval function (value by default)
            seg_fn = kwargs.get("seg_fn", lambda k, d: d[k])

            # Save the masks for segmentation task
            for class_name in masks.keys():
                for path in class_map[class_name]:
                    if "segmentation" not in path:
                        continue

                    try:
                        # Get the mask for the current class
                        mask = seg_fn(class_name, masks)
                    except:
                        continue

                    # Generate the mask from annotation and resize it
                    msk = Image.fromarray(mask * 255).convert("1").resize(size)

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

    if sum(len(v) for v in class_map.values()) == 0:
        # If no tasks left, return
        return

    for task in tasks:
        for cat in os.scandir(os.path.join(root, task)):
            if not cat.is_dir():
                continue

            for ds_dir in os.scandir(cat.path):
                if ds_dir.name == save_name and kwargs.get("force", False):
                    # Remove the dataset if exists
                    shutil.rmtree(ds_dir.path)
                elif (
                    ds_dir.name == save_name
                    and sum(len(fs) for _, _, fs in os.walk(ds_dir.path)) > 0
                ):
                    print(f"* Skipping {save_name} (already processed)")
                    return

    # for class_name, task_map in class_map.items():
    #     for task_name, task_cats in task_map.items():
    #         for task_cat in task_cats:
    #             # Check how many files are in save_dir
    #             task_cat = task_cat.replace("no_", "")
    #             save_path = os.path.join(root, task_name, task_cat, save_name)

    #             if not os.path.exists(save_path):
    #                 continue

    #             if sum(len(files) for _, _, files in os.walk(save_path)) > 0:
    #                 print(f"* Skipping {save_name} (already processed)")
    #                 return

    # Initialize tqdm progress bar for current dataset
    pbar_desc = f"* Processing {save_name}"
    pbar_total = kwargs.get("total", 0)
    update_total = pbar_total == 0
    pbar = tqdm(desc=pbar_desc, total=0)

    # Unpack the data files
    unpack([data_file], root, "tmp", pbar, update_total)

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
        parse_coco_json(json_path, _class_map, size, pbar, **kwargs)

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


def prepare_no_eyeglasses(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "no eyeglass.v3i.coco.zip"
    kwargs["save_name"] = "no-eyeglasses"
    kwargs["class_map"] = {"No Eyeglass": {"detection": ["eyes"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_kacamata_membaca(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "Kacamata-Membaca.v1i.coco.zip"
    kwargs["save_name"] = "kacamata-membaca"
    kwargs["class_map"] = {"Kacamata-Membaca": {"detection": ["standalone"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_onlyglasses(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "onlyglasses.v1i.coco.zip"
    kwargs["save_name"] = "onlyglasses"
    kwargs["class_map"] = {"glasses": {"detection": ["standalone"]}}

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
    # parser.add_argument(
    #     "-d",
    #     "--device",
    #     type=str,
    #     default="",
    #     help=f"The device on which to perform preprocessing. Can be, for "
    #     f"example, 'cpu', 'cuda'. If not specified the device is chosen "
    #     f"CUDA or MPS, if either is available. Defaults to ''.",
    # )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help=f"Whether to force preprocessing each dataset. This flag will"
        f"delete each dataset for the specified task (all by default) if it"
        f"already exists and process it again fresh from the start.",
    )
    parser.add_argument(
        "-d",
        "--delete-original",
        action="store_true",
        help=f"Whether to delete the original data archives after unpacking.",
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
    # if kwargs["device"] == "" and torch.cuda.is_available():
    #     kwargs["device"] = torch.device("cuda")
    # elif kwargs["device"] == "" and torch.backends.mps.is_available():
    #     kwargs["device"] = torch.device("mps")
    # elif kwargs["device"] == "":
    #     kwargs["device"] = torch.device("cpu")

    return kwargs


if __name__ == "__main__":
    # Get command-line args
    kwargs = parse_kwargs()

    if "classification" in kwargs["tasks"]:
        # Main target: classification
        prepare_cmu_face_images(**kwargs)
        prepare_specs_on_faces(**kwargs)
        prepare_sunglasses_no_sunglasses(**kwargs)
        prepare_glasses_and_coverings(**kwargs)
        prepare_face_attributes_grouped(**kwargs)
        prepare_face_attributes_extra(**kwargs)

    if "segmentation" in kwargs["tasks"]:
        # Main target: segmentation
        prepare_eyeglass(**kwargs)
        prepare_glasses_lenses(**kwargs)
        prepare_glasses_lens(**kwargs)
        prepare_glasses_segmentation_cropped_faces(**kwargs)

    if "classification" in kwargs["tasks"] or "detection" in kwargs["tasks"]:
        # Main target: detection
        prepare_ai_pass(**kwargs)
        prepare_pex5(**kwargs)
        prepare_sunglasses_glasses_detect(**kwargs)
        prepare_glasses_detection(**kwargs)
        prepare_glasses_image_dataset(**kwargs)
        prepare_ex07(**kwargs)
        prepare_no_eyeglasses(**kwargs)
        prepare_kacamata_membaca(**kwargs)
        prepare_onlyglasses(**kwargs)
