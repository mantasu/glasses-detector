import argparse
import glob
import os
import shutil
import sys
import tarfile
import warnings
import zipfile
from collections import defaultdict
from copy import deepcopy
from fnmatch import fnmatch
from multiprocessing import Lock, cpu_count
from multiprocessing.pool import ThreadPool
from typing import Callable, Generator

import numpy as np
import rarfile
from PIL import Image
from pycocotools.coco import COCO
from scipy.io import loadmat
from tqdm import tqdm

LOCK = Lock()

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
                if len(members) > 0 and not any(fnmatch(member, m) for m in members):
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
        # Update pbar total (count only top-level extractables)
        total = sum(1 for _ in get_extractable(filenames, root, unpack_dir))
        pbar.total = pbar.total + total
        pbar.refresh()
    elif pbar is not None:
        # Update description to indicate extracting
        pbar.set_description(f"{pbar.desc} (extracting)")

    for file, member, is_nested in get_extractable(filenames, root, unpack_dir):
        # Extract and print error if failed
        file.extract(member=member, path=unpack_path)

        if pbar is not None and (not update_total or not is_nested):
            # Update pbar
            pbar.update(1)

    if pbar is not None:
        # Update the description not to indicate extracting anymore
        pbar.set_description(pbar.desc.replace(" (extracting)", ""))


def check_if_present(
    tasks: list[str],
    save_name: str,
    root: str = "data",
    force: bool = False,
) -> bool:
    for task in tasks:
        for cat in os.scandir(os.path.join(root, task)):
            if not cat.is_dir():
                continue

            for ds_dir in os.scandir(cat.path):
                if ds_dir.name == save_name and force:
                    # Remove the dataset if exists
                    shutil.rmtree(ds_dir.path)
                elif (
                    ds_dir.name == save_name
                    and sum(len(fs) for _, _, fs in os.walk(ds_dir.path)) > 0
                ):
                    print(f"* Skipping {save_name} (already processed)")
                    return True

    return False


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
##########################                   ###########################
##########################      HELPERS      ###########################
##########################                   ###########################
########################################################################


def process_image(args):
    # Unpack the arguments
    src, tgts, size, pbar, is_dir, to_grayscale = args

    # Open the image
    image = Image.open(src)

    if to_grayscale:
        # Convert to grayscale
        image = image.convert("L")

    if image.size != size:
        # Resize the image
        image = image.resize(size)

    if isinstance(tgts, str):
        # Standardize
        tgts = [tgts]

    for tgt in tgts:
        if is_dir:
            # Save the image to target dir (as .jpg image)
            name, _ = os.path.splitext(os.path.basename(src))
            image.save(os.path.join(tgt, name + ".jpg"), "JPEG")
        else:
            # Save the image to target file (as .jpg image)
            image.save(os.path.splitext(tgt)[0] + ".jpg", "JPEG")

    if pbar is not None:
        with LOCK:
            # Update pbar
            pbar.update(1)


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
    if check_if_present(
        ["classification"], save_name, root, kwargs.get("force", False)
    ):
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

    # Prepare the arguments for the process_image function
    args = [(s, t, size, pbar, True, False) for s, t in src_to_tgt.items()]

    with ThreadPool(processes=cpu_count()) as pool:
        for _ in pool.imap_unordered(process_image, args):
            pass

    # Init deletable files and dirs and clean them up
    pbar.set_description(f"{pbar_desc} (cleaning)")
    files = list(data_files.keys()) if isinstance(data_files, dict) else data_files
    deletable = ["tmp"] + (files if delete_original else [])
    clean(deletable, root)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (done)")


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
                # Split the path to multiple parts based on separator
                path_splits = os.path.normpath(path).split(os.sep)

                if "classification" in path_splits:
                    # Save the resized image only for classification
                    img.save(os.path.join(path, img_info["file_name"]))
                elif "detection" in path_splits:
                    # Normalize bbox (x_center, y_center, width, height)
                    # x = (ann["bbox"][0] + ann["bbox"][2] / 2) / img_info["width"]
                    # y = (ann["bbox"][1] + ann["bbox"][3] / 2) / img_info["height"]
                    # w = ann["bbox"][2] / img_info["width"]
                    # h = ann["bbox"][3] / img_info["height"]

                    # Convert to pascal_voc format (with resized bbox)
                    x1 = int(ann["bbox"][0] * size[0] / img_info["width"])
                    y1 = int(ann["bbox"][1] * size[1] / img_info["height"])
                    x2 = x1 + int(ann["bbox"][2] * size[0] / img_info["width"])
                    y2 = y1 + int(ann["bbox"][3] * size[1] / img_info["height"])

                    # Copy the image and create .txt annotation filename
                    img.save(os.path.join(path, "images", img_info["file_name"]))
                    txt = img_info["file_name"].rsplit(".", 1)[0] + ".txt"

                    with open(os.path.join(path, "annotations", txt), "w") as f:
                        # Write the bounding box
                        f.write(f"{x1} {y1} {x2} {y2}")
                elif "segmentation" in path_splits:
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

    if check_if_present(tasks, save_name, root, kwargs.get("force", False)):
        return

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


########################################################################
#########################                    ###########################
#########################   CLASSIFICATION   ###########################
#########################                    ###########################
########################################################################


def prepare_specs_on_faces(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "specs-on-faces"
    kwargs["data_files"] = ["original images.rar"]

    # Unpack metadata immediately (before data archive)
    unpack(["metadata.rar"], kwargs["root"], "tmp")

    if not os.path.exists(os.path.join(kwargs["root"], "metadata.rar")):
        # If metadata.rar is not present, ignore processing
        shutil.rmtree(os.path.join(kwargs["root"], "tmp"))
        return

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
            "glasses-and-coverings/plain/*",
            "glasses-and-coverings/glasses/*",
            "glasses-and-coverings/sunglasses/*",
            "glasses-and-coverings/sunglasses-imagenet/*",
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
            "face-attributes-grouped/train/eyewear/*",
            "face-attributes-grouped/train/nowear/*",
            "face-attributes-grouped/val/eyewear/*",
            "face-attributes-grouped/val/nowear/*",
            "face-attributes-grouped/test/eyewear/*",
            "face-attributes-grouped/test/nowear/*",
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
            "face-attributes-extra/sunglasses/*",
            "face-attributes-extra/eyeglasses/*",
            "face-attributes-extra/no_eyewear/*",
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


def prepare_glasses_no_glasses(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "glasses-no-glasses"
    kwargs["data_files"] = ["glasses-no-glasses.zip"]

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))

    # Add category map
    kwargs["cat_map"] = {
        "anyglasses": lambda path: folder_name(path) == "glasses",
    }

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_indian_facial_database(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "indian-facial-database"
    kwargs["data_files"] = {
        "An Indian facial database highlighting the Spectacle.zip": [
            "An Indian facial database highlighting the Spectacle/Version 2/*001_0.jpg",
            "An Indian facial database highlighting the Spectacle/Version 2/*001_1.jpg",
        ]
    }

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))

    # Add category map
    kwargs["cat_map"] = {"anyglasses": lambda path: folder_name(path) == "WITH"}

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_face_attribute_2(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    archive = "FaceAttribute 2.v2i.multiclass.zip"
    kwargs["save_name"] = "face-attribute-2"
    kwargs["data_files"] = [archive]
    is_glasses_set = set()

    for split_type in ["train", "valid"]:
        # Unpack the CSV file into tmp dir
        unpack({archive: [f"{split_type}/_classes.csv"]}, kwargs["root"], "tmp")
        csv_path = os.path.join(kwargs["root"], "tmp", split_type, "_classes.csv")

        if not os.path.exists(csv_path):
            # If CSV file is not present, ignore processing
            shutil.rmtree(os.path.join(kwargs["root"], "tmp"))
            return

        # Read the CSV file
        with open(csv_path, "r") as f:
            lines = f.readlines()
            class_names = lines[0].split(",")[1:]

            for line in f.readlines()[1:]:
                # Get the class name
                filename = line.split(", ")[0]
                class_name = class_names[line.split(", ")[1:].index("1")]

                if "Glasses" in class_name:
                    # If glasses are present
                    is_glasses_set.add(filename)

        # Remove the temporary directory with csv files
        shutil.rmtree(os.path.join(kwargs["root"], "tmp"))

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))

    # Add category map
    kwargs["cat_map"] = {
        "anyglasses": lambda path: os.path.basename(path) in is_glasses_set,
    }

    # Add split type check function
    kwargs["split_fn"] = lambda path: "val" if folder_name(path) == "valid" else "train"

    # Prepare for classification
    categorize_binary(**kwargs)


def prepare_glasses_shadows_synthetic(**kwargs):
    # Dataset name and files
    kwargs = deepcopy(kwargs)
    kwargs["save_name"] = "glasses-shadows-synthetic"
    kwargs["data_files"] = ["glasses-shadows-synthetic.zip"]

    # Create helper functions to check the folder name the file is in
    folder_name = lambda path: os.path.basename(os.path.dirname(path))
    ffn = lambda path: folder_name(os.path.dirname(path))

    # Add category map and split fn
    kwargs["cat_map"] = {"shadows": lambda path: folder_name(path) == "shadows"}
    kwargs["split_fn"] = lambda path: ffn(path)

    # Prepare for classification
    categorize_binary(**kwargs)


########################################################################
#########################                    ###########################
#########################    SEGMENTATION    ###########################
#########################                    ###########################
########################################################################


def celeba_mask_hq_generate_split_paths(
    split_info_file_paths: list[str],
    celeba_mapping_file_path: str,
    save_dir: str,
) -> dict[int, str]:
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


def celeba_mask_hq_walk_through_masks(
    mask_dir: str,
    img_dir: str,
    split_info: dict[int, str],
    resize: tuple[int, int],
    pbar: tqdm,
    update_total: bool = False,
):
    if update_total:
        # Count the total number of files in the directory tree
        total = sum(len(files) for _, _, files in os.walk(mask_dir))
        pbar.total = pbar.total + total
        pbar.refresh()

    pbar.set_description(f"{pbar.desc} (categorizing)")

    for root, _, files in os.walk(mask_dir):
        for file in files:
            if "eye_g" not in file:
                # Ignore no-glasses
                pbar.update(1)
                continue

            # Get the train/val/test type
            idx = int(file.split("_")[0])
            parent_path = split_info[idx]

            # Create the full path to original files
            mask_path = os.path.join(root, file)
            image_path = os.path.join(img_dir, str(idx) + ".jpg")

            # Create a save path of original files to train/val/test location
            image_save_path = os.path.join(parent_path, "images", str(idx) + ".jpg")
            mask_save_path = os.path.join(parent_path, "masks", str(idx) + ".jpg")

            # Open the image, convert mask to black/white
            image = Image.open(image_path).resize(resize)
            mask = Image.open(mask_path).resize(resize)
            mask = Image.fromarray((np.array(mask) > 0).astype(np.uint8) * 255)

            # Save the mask and the image
            image.save(image_save_path)
            mask.save(mask_save_path)

            # Update pbar
            pbar.update(1)


def prepare_celeba_mask_hq(**kwargs):
    # Dataset name and files
    save_name = "celeba-mask-hq"
    data_files = ["CelebAMask-HQ.zip", "annotations.zip"]
    root = kwargs.get("root", "data")
    size = kwargs.get("size", (256, 256))
    delete_original = kwargs.get("delete_original", False)

    if check_if_present(["segmentation"], save_name, root, kwargs.get("force", False)):
        return

    # Initialize tqdm progress bar for current dataset
    pbar_desc = f"* Processing {save_name}"
    pbar_total = kwargs.get("total", 0)
    update_total = pbar_total == 0
    pbar = tqdm(desc=pbar_desc, total=pbar_total)

    # Unpack the data files into "tmp" dir
    unpack(data_files, root, "tmp", pbar, update_total)

    if update_total:
        # Update progress bar total
        pbar.total = pbar.total + 2
        pbar.refresh()

    # Update pbar description
    pbar.set_description(f"{pbar_desc} (reading contents)")

    # Create train/val/test split info dictionary
    split_info = celeba_mask_hq_generate_split_paths(
        [os.path.join(root, "tmp", f"{x}_label.txt") for x in ["train", "val", "test"]],
        os.path.join(root, "tmp", "CelebAMask-HQ", "CelebA-HQ-to-CelebA-mapping.txt"),
        os.path.join(root, "segmentation", "full", "celeba-mask-hq"),
    )

    # Update pbar
    pbar.update(1)
    pbar.set_description(pbar_desc)

    # Walk through samples and process
    celeba_mask_hq_walk_through_masks(
        os.path.join(root, "tmp", "CelebAMask-HQ", "CelebAMask-HQ-mask-anno"),
        os.path.join(root, "tmp", "CelebAMask-HQ", "CelebA-HQ-img"),
        split_info,
        size,
        pbar,
        update_total,
    )

    # Init deletable files and dirs and clean them up
    pbar.set_description(f"{pbar_desc} (cleaning)")
    deletable = ["tmp"] + (data_files if delete_original else [])
    clean(deletable, root)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (done)")


def prepare_glasses_segmentation_synthetic(**kwargs):
    # Dataset name and files
    save_name = "glasses-segmentation-synthetic"
    data_files = {
        "glasses-segmentation-synthetic.zip": [
            "*-seg.png",
            "*-shseg.png",
            "*-all.png",
            "*-sunglasses.png",
        ]
    }
    root = kwargs.get("root", "data")
    size = kwargs.get("size", (256, 256))
    frac = kwargs.get("sunglasses_fraction", 0.5)
    force = kwargs.get("force", False)
    delete_original = kwargs.get("delete_original", False)

    if check_if_present(["segmentation"], save_name, root, force):
        return

    # Initialize tqdm progress bar for current dataset
    pbar_desc = f"* Processing {save_name}"
    pbar_total = kwargs.get("total", 0)
    update_total = pbar_total == 0
    pbar = tqdm(desc=pbar_desc, total=pbar_total)

    # Unpack the data files into "tmp" dir
    unpack(data_files, root, "tmp", pbar, update_total)

    if update_total:
        # Update progress bar total
        total = sum(len(files) for _, _, files in os.walk(os.path.join(root, "tmp")))
        pbar.total = pbar.total + total // 4 * 3 + 2
        pbar.refresh()

    # Update pbar description
    pbar.set_description(f"{pbar_desc} (reading contents)")
    src_to_tgt = defaultdict(lambda: [])

    for split_type in ["train", "val", "test"]:
        # Get all the filepaths, sort them, compute number of sunglasses
        filenames = sorted(os.listdir(os.path.join(root, "tmp", save_name, split_type)))
        num_sunglasses = int((len(filenames) // 4) * frac)
        i = 0

        # Create 2 segmentation categories: frames and shadows
        fra_dir = os.path.join(root, "segmentation", "frames", save_name, split_type)
        sha_dir = os.path.join(root, "segmentation", "shadows", save_name, split_type)

        for subdir in ["images", "masks"]:
            # Create 2 sub-directories: for frames and shadows
            os.makedirs(os.path.join(fra_dir, subdir), exist_ok=True)
            os.makedirs(os.path.join(sha_dir, subdir), exist_ok=True)

        for filename in filenames:
            # Get the full path to the original extracted image file
            filepath = os.path.join(root, "tmp", save_name, split_type, filename)

            if filename.endswith("-seg.png"):
                # Add the frames mask to the corresponding category
                tgt = os.path.join(fra_dir, "masks", filename.replace("-seg", ""))
                src_to_tgt[filepath].append(tgt)

            if filename.endswith("-shseg.png"):
                # Add the shadows mask to the corresponding category
                tgt = os.path.join(sha_dir, "masks", filename.replace("-shseg", ""))
                src_to_tgt[filepath].append(tgt)

            if (i >= num_sunglasses and filename.endswith("-all.png")) or (
                i < num_sunglasses and filename.endswith("-sunglasses.png")
            ):
                # Add either regular eyeglasses or sunglasses
                tgt_name = filename.replace("-all", "").replace("-sunglasses", "")
                tgt_1 = os.path.join(fra_dir, "images", tgt_name)
                tgt_2 = os.path.join(sha_dir, "images", tgt_name)
                src_to_tgt[filepath].extend([tgt_1, tgt_2])
                i += 1

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (categorizing)")

    # Prepare the arguments for the process_image function
    args = [(s, t, size, pbar, False, len(t) == 1) for s, t in src_to_tgt.items()]

    with ThreadPool(processes=cpu_count()) as pool:
        for _ in pool.imap_unordered(process_image, args):
            pass

    # Init deletable files and dirs and clean them up
    pbar.set_description(f"{pbar_desc} (cleaning)")
    deletable = ["tmp"] + (data_files if delete_original else [])
    clean(deletable, root)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (done)")


def prepare_face_synthetics_glasses(**kwargs):
    # Dataset name and files
    save_name = "face-synthetics-glasses"
    data_files = {
        "face-synthetics-glasses.zip": [
            "face-synthetics-glasses/train/images/*",
            "face-synthetics-glasses/train/masks/*",
            "face-synthetics-glasses/val/images/*",
            "face-synthetics-glasses/val/masks/*",
            "face-synthetics-glasses/test/images/*",
            "face-synthetics-glasses/test/masks/*",
        ]
    }
    root = kwargs.get("root", "data")
    size = kwargs.get("size", (256, 256))
    force = kwargs.get("force", False)
    delete_original = kwargs.get("delete_original", False)

    if check_if_present(["segmentation"], save_name, root, force):
        return

    # Initialize tqdm progress bar for current dataset
    pbar_desc = f"* Processing {save_name}"
    pbar_total = kwargs.get("total", 0)
    update_total = pbar_total == 0
    pbar = tqdm(desc=pbar_desc, total=pbar_total)

    # Unpack the data files into "tmp" dir
    unpack(data_files, root, "tmp", pbar, update_total)

    if update_total:
        # Update progress bar total
        total = sum(len(files) for _, _, files in os.walk(os.path.join(root, "tmp")))
        pbar.total = pbar.total + total + 2
        pbar.refresh()

    # Update pbar description and init src_to_tgt mapping
    pbar.set_description(f"{pbar_desc} (reading contents)")
    src_to_tgt = {}

    for _root, _, files in os.walk(os.path.join(root, "tmp", save_name)):
        for file in files:
            # Get path to the original file and create one for target
            path = os.path.normpath(src := os.path.join(_root, file)).split(os.sep)
            tgt = os.path.join(
                root,
                "segmentation",
                "smart",
                *path[path.index("tmp") + 1 :],
            )
            # Update src to tgt
            src_to_tgt[src] = tgt
            os.makedirs(os.path.dirname(tgt), exist_ok=True)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (categorizing)")

    # Prepare arguments (masks are already grayscale, no need to convert)
    args = [(s, t, size, pbar, False, False) for s, t in src_to_tgt.items()]

    with ThreadPool(processes=cpu_count()) as pool:
        for _ in pool.imap_unordered(process_image, args):
            pass

    # Init deletable files and dirs and clean them up
    pbar.set_description(f"{pbar_desc} (cleaning)")
    deletable = ["tmp"] + (data_files if delete_original else [])
    clean(deletable, root)

    # Update pbar
    pbar.update(1)
    pbar.set_description(f"{pbar_desc} (done)")


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
    kwargs["data_file"] = (
        "glasses lenses segmentation.v7-sh-improvments-version.coco.zip"
    )
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
    kwargs["data_file"] = (
        "glasses segmentation cropped faces.v2-segmentation_models_pytorch-s_1st_version.coco-segmentation.zip"
    )
    kwargs["save_name"] = "glasses-segmentation-cropped-faces"
    kwargs["class_map"] = {"glasses-lenses": {"segmentation": ["lenses"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_spects_segmentation(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "Spects Segementation.v3i.coco-segmentation.zip"
    kwargs["save_name"] = "spects-segmentation"
    kwargs["class_map"] = {"Spectacles": {"segmentation": ["full"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_kinh(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "kinh.v1i.coco.zip"
    kwargs["save_name"] = "kinh"
    kwargs["class_map"] = {"kinh": {"segmentation": ["full"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_capstone_mini_2(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "CAPSTONE_MINI_2.v1i.coco-segmentation.zip"
    kwargs["save_name"] = "capstone-mini-2"
    kwargs["class_map"] = {"Temple": {"segmentation": ["legs"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_sunglasses_color_detection(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = (
        "Sunglasses Color detection roboflow.v2i.coco-segmentation.zip"
    )
    kwargs["save_name"] = "sunglasses-color-detection"
    kwargs["class_map"] = {
        "Lens": {"segmentation": ["lenses"]},
        "Asta": {"segmentation": ["legs"]},
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_sunglasses_color_detection_2(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "Sunglasses Color detection 2.v3i.coco-segmentation.zip"
    kwargs["save_name"] = "sunglasses-color-detection-2"
    kwargs["class_map"] = {
        "Lens": {"segmentation": ["lenses"]},
        "Asta": {"segmentation": ["legs"]},
    }

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_glass_color(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "Glass-Color.v1i.coco-segmentation.zip"
    kwargs["save_name"] = "glass-color"
    kwargs["class_map"] = {"Lens": {"segmentation": ["lenses"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


########################################################################
##########################                   ###########################
##########################     DETECTION     ###########################
##########################                   ###########################
########################################################################


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
    kwargs["class_map"] = {"Kacamata-Membaca": {"detection": ["solo"]}}

    # Process the data splits
    walk_coco_splits(**kwargs)


def prepare_onlyglasses(**kwargs):
    # Update the kwargs (file path, dataset name, class map)
    kwargs["data_file"] = "onlyglasses.v1i.coco.zip"
    kwargs["save_name"] = "onlyglasses"
    kwargs["class_map"] = {"glasses": {"detection": ["solo"]}}

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
        prepare_glasses_no_glasses(**kwargs)
        prepare_indian_facial_database(**kwargs)
        prepare_face_attribute_2(**kwargs)
        prepare_glasses_shadows_synthetic(**kwargs)

    if "segmentation" in kwargs["tasks"]:
        # Main target: segmentation
        prepare_celeba_mask_hq(**kwargs)
        prepare_glasses_segmentation_synthetic(**kwargs)
        prepare_face_synthetics_glasses(**kwargs)
        prepare_eyeglass(**kwargs)
        prepare_glasses_lenses(**kwargs)
        prepare_glasses_lens(**kwargs)
        prepare_glasses_segmentation_cropped_faces(**kwargs)
        prepare_spects_segmentation(**kwargs)
        prepare_kinh(**kwargs)
        prepare_capstone_mini_2(**kwargs)
        prepare_sunglasses_color_detection(**kwargs)
        prepare_sunglasses_color_detection_2(**kwargs)
        prepare_glass_color(**kwargs)

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
