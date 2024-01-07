import os
import random
from collections import defaultdict
from typing import Callable

import albumentations as A
import torch
from torch.utils.data import Dataset

from .mixins import DataLoaderMixin, ImageLoaderMixin


class ImageDetectionDataset(Dataset, ImageLoaderMixin, DataLoaderMixin):
    # It's more efficient to implement a specific dataset for each task
    # And it is very unlikely that multiple tasks will be considered at
    # once, meaning a generic dataset is not needed
    def __init__(
        self,
        root: str = ".",
        split_type: str = "train",
        img_folder: str = "images",
        ann2img_fn: dict[str, Callable[[str], str]] = {},
        # for each annotation folder name, a function that maps annotation file name to the image file name it belongs
        seed: int = 0,
    ):
        super().__init__()

        self.data = []
        cat2paths = defaultdict(lambda: {"names": [], "paths": []})

        for dataset in os.listdir(root):
            if not os.path.isdir(p := os.path.join(root, dataset, split_type)):
                continue

            for cat in os.scandir(p):
                # Read the list of names and paths to images/masks
                name_fn = ann2img_fn.get(cat.name, lambda x: x.replace(".txt", ".jpg"))
                names = list(map(name_fn, os.listdir(cat.path)))
                paths = [f.path for f in os.scandir(cat.path)]

                # Extend the lists of image/annot names + paths
                cat2paths[cat.name]["names"].extend(names)
                cat2paths[cat.name]["paths"].extend(paths)

        # Pop the non-category folder (get image names and paths)
        img_names, img_paths = cat2paths.pop(img_folder).values()

        for img_name, img_path in zip(img_names, img_paths):
            # Add the default image entry
            self.data.append({"image": img_path})

            for cat_dirname, names_and_paths in cat2paths.items():
                if img_name in names_and_paths["names"]:
                    # Get the index of corresponding annotation
                    i = names_and_paths["names"].index(img_name)
                    annotation_path = names_and_paths["paths"][i]
                    self.data[-1][cat_dirname] = annotation_path
                else:
                    # No annotation but add for equally sized batches
                    self.data[-1][cat_dirname] = None

        # Sort & shuffle
        self.data.sort(key=lambda x: x["image"])
        random.seed(seed)
        random.shuffle(self.data)

        # Create image augmentation pipeline based on split type
        p = A.BboxParams(format="pascal_voc", label_fields=["classes"])
        self.transform = self.create_transform(split_type == "train", bbox_params=p)

    @property
    def name2idx(self):
        return dict(zip(self.data[0].keys()), range(len(self.data[0])))

    @property
    def idx2name(self):
        return dict(zip(range(len(self.data[0]), self.data[0].keys())))

    def __getitem__(self, index):
        # Load the image, bboxes and classes
        image = self.data[index]["image"]
        bboxes = list(self.data[index].values())[1:]
        labels = [1] * len(bboxes)
        # labels = [self.cat2label(k) for k in list(self.data[index].keys())[1:]]

        (image, bboxes, labels) = self.load_image(
            image=image,
            bboxes=bboxes,
            classes=labels,
            transform=self.transform,
        )

        # TODO: create cat2label map and map class names to labels
        # TODO: there may be more bboxes read than classes after loading
        # the transformed image so consider adding either a max_bbox
        # argument or implement a custom collate function for dataloader

        if len(bboxes) == 0:
            bboxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)

        annotations = {"boxes": bboxes, "labels": labels}

        return image, annotations

    def __len__(self):
        return len(self.data)
