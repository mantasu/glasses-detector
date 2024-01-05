import os
import random
from collections import defaultdict
from typing import Callable

import torch
from torch.utils.data import Dataset

from .mixins import DataLoaderMixin, ImageLoaderMixin


class ImageSegmentationDataset(Dataset, ImageLoaderMixin, DataLoaderMixin):
    def __init__(
        self,
        root: str = ".",
        split_type: str = "train",
        img_dirname: str = "images",
        name_map_fn: dict[
            str, Callable[[str], str]
        ] = {},  # maps mask name to image name
        seed: int = 0,
    ):
        super().__init__()

        self.data = []
        cat2paths = defaultdict(lambda: {"names": [], "paths": []})

        for dir in os.listdir(root):
            if not os.path.isdir(dir := os.path.join(root, dir, split_type)):
                continue

            for cat in os.scandir(dir):
                # Read the list of names and paths to images/masks
                name_fn = name_map_fn.get(cat.name, lambda x: x)
                names = list(map(name_fn, os.listdir(cat.path)))
                paths = [f.path for f in os.scandir(cat.path)]

                # Extend the lists of image/mask names + paths
                cat2paths[cat.name]["names"].extend(names)
                cat2paths[cat.name]["paths"].extend(paths)

        # Get the names of image files and paths to them
        img_names = cat2paths[img_dirname]["names"]
        img_paths = cat2paths[img_dirname]["paths"]

        for img_name, img_path in zip(img_names, img_paths):
            # Add the default image entry
            self.data.append({"image": img_path})

            for mask_dirname, names_and_paths in cat2paths.items():
                if mask_dirname == img_dirname:
                    # Skip if it's image folder
                    continue

                if img_name in names_and_paths["names"]:
                    # Get the index of corresponding mask and add it
                    i = names_and_paths["names"].index(img_name)
                    mask_path = names_and_paths["paths"][i]
                    self.data[-1][mask_dirname] = mask_path
                else:
                    # No mask for this category is present
                    self.data[-1][mask_dirname] = None

        # Sort & shuffle
        self.data.sort(key=lambda x: x["image"])
        random.seed(seed)
        random.shuffle(self.data)

        # Create image augmentation pipeline based on split type
        self.transform = self.create_transform(split_type == "train")

    @property
    def name2idx(self):
        return dict(zip(self.data[0].keys()), range(len(self.data[0])))

    @property
    def idx2name(self):
        return dict(zip(range(len(self.data[0]), self.data[0].keys())))

    def __getitem__(self, index):
        # Load the image and the masks
        image = self.data[index]["image"]
        masks = list(self.data[index].values())[1:]
        image, masks = self.load_image(image, masks, self.transform)

        return image, torch.stack(masks, dim=0).to(torch.float32)

    def __len__(self):
        return len(self.data)
