from typing import Any

import albumentations as A
import numpy
import PIL.Image as Image
import skimage.transform as st
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


class ImageLoaderMixin:
    @staticmethod
    def create_transform(is_train: bool = False, **kwargs) -> A.Compose:
        # Default augmentation
        transform = [
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(),
            A.OneOf(
                [
                    A.RandomResizedCrop(256, 256, p=0.5),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                    A.PiecewiseAffine(),
                    A.Perspective(),
                    A.GridDistortion(),
                ]
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3
                    ),
                    A.RandomGamma(),
                    A.CLAHE(),
                    A.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    A.HueSaturationValue(),
                ]
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=3),
                    A.GaussianBlur(),
                    A.MedianBlur(),
                    A.GaussNoise(),
                ]
            ),
            A.Normalize(),
            ToTensorV2(),
        ]

        if "bbox_params" not in kwargs:
            transform.insert(-2, A.CoarseDropout(max_holes=5, p=0.3))

        if not is_train:
            # Only keep the last two
            transform = transform[-2:]

        return A.Compose(transform, **kwargs)

    @staticmethod
    def load_image(
        image: str | Image.Image | numpy.ndarray,
        masks: list[str | Image.Image | numpy.ndarray] = [],
        bboxes: list[str | list[int | float | str]] = [],  # x_min, y_min, x_max, y_max
        classes: list[Any] = [],  # one for each bbox
        resize: tuple[int, int] | None = None,
        transform: A.Compose | bool = False,
    ) -> torch.Tensor:
        def open_image_file(image_file, is_mask=False):
            if isinstance(image_file, str):
                # If the image is provided as a path
                image_file = Image.open(image_file)

            if isinstance(image_file, Image.Image):
                # If the image is not a numpy array
                image_file = numpy.array(image_file)

            if is_mask and image_file.ndim > 2:
                # Convert image to black & white, ensure only 1 channel
                image_file = (image_file > 127).any(axis=2).astype(numpy.uint8)
            elif is_mask:
                # Convert image to black & white of type UINT8
                image_file = (image_file > 127).astype(numpy.uint8)
            elif image_file.ndim == 2:
                # Image is not a mask, so convert it to RGB
                image_file = numpy.stack([image_file] * 3, axis=-1)

            if resize is not None:
                # Resize image to new (w, h)
                size = resize[1], resize[0]
                image_file = st.resize(image_file, size)

            return image_file

        def open_bbox_files(bbox_files, classes):
            # Init new
            _bboxes, _classes = [], []

            for i, bbox_file in enumerate(bbox_files):
                if isinstance(bbox_file, str):
                    with open(bbox_file, "r") as f:
                        # Each line is bbox: "x_min y_min x_max y_max"
                        batch = [xyxy.strip().split() for xyxy in f.readlines()]
                else:
                    # bbox_file is a single bbox (list[str | int | float])
                    batch = [bbox_file]

                batch = [list(map(float, xyxy)) for xyxy in batch]

                for i, xyxy in enumerate(batch):
                    if xyxy[2] <= xyxy[0]:
                        batch[i][0] = min(xyxy[0], image.shape[1] - 1)
                        batch[i][2] = batch[i][0] + 1

                    if xyxy[3] <= xyxy[1]:
                        batch[i][1] = min(xyxy[1], image.shape[0] - 1)
                        batch[i][3] = batch[i][1] + 1

                if resize is not None:
                    # Get old and new width, height
                    old_h, old_w = image.shape[:2]
                    new_w, new_h = resize

                    # Convert bboxes to new (w, h)
                    batch = [
                        [
                            xyxy[0] * new_w / old_w,
                            xyxy[1] * new_h / old_h,
                            xyxy[2] * new_w / old_w,
                            xyxy[3] * new_h / old_h,
                        ]
                        for xyxy in batch
                    ]

                # Add to list
                _bboxes.extend(batch)

                if classes != []:
                    # If classes are provided, add them
                    _classes.extend([classes[i]] * len(batch))

            return _bboxes, _classes

        kwargs = {}

        if isinstance(transform, bool):
            if bboxes != []:
                kwargs.update(
                    {
                        "bbox_params": A.BboxParams(
                            format="pascal_voc",
                            label_fields=["classes"] if classes != [] else None,
                        )
                    }
                )

            # Load transform (train/test is based on bool)
            transform = ImageLoaderMixin.create_transform(transform, **kwargs)

        # Load image, mask, bbox files
        image = open_image_file(image)
        masks = [open_image_file(m, True) for m in masks]
        bboxes, classes = open_bbox_files(bboxes, classes)

        # Create transform kwargs
        kwargs["image"] = image
        kwargs.update({"masks": masks} if masks != [] else {})
        kwargs.update({"bboxes": bboxes} if bboxes != [] else {})
        kwargs.update({"classes": classes} if classes != [] else {})

        # Transform everything, init returns
        transformed = transform(**kwargs)
        return_list = [transformed["image"]]

        if masks != []:
            # TODO: check if transformation is converted to a tensor
            return_list.append(transformed["masks"])

        if bboxes != []:
            bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            return_list.append(bboxes)

        if classes != []:
            classes = torch.tensor(transformed["classes"], dtype=torch.int64)
            return_list.append(classes)

        if len(return_list) == 1:
            return return_list[0]

        return tuple(return_list)


class DataLoaderMixin:
    @classmethod
    def create_loader(cls, **kwargs) -> DataLoader:
        # Split all the given kwargs to dataset (cls) and loader kwargs
        cls_kwargs_set = {
            "root",
            "dirs",
            "split_type",
            "label_type",
            "img_dirname",
            "name_map_fn",
            "seed",
        }
        set_kwargs = {k: v for k, v in kwargs.items() if k in cls_kwargs_set}
        ldr_kwargs = {k: v for k, v in kwargs.items() if k not in cls_kwargs_set}

        # Define default loader kwargs
        default_loader_kwargs = {
            "dataset": cls(**set_kwargs),
            "batch_size": 64,
            "num_workers": 12,
            "pin_memory": True,
            "shuffle": set_kwargs.get("split_type", "train") == "train",
        }

        # Update default loader kwargs with custom
        default_loader_kwargs.update(ldr_kwargs)

        return DataLoader(**default_loader_kwargs)

    @classmethod
    def create_loaders(cls, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
        # Create train, validationa and test loaders
        train_loader = cls.create_loader(split_type="train", **kwargs)
        val_loader = cls.create_loader(split_type="val", **kwargs)
        test_loader = cls.create_loader(split_type="test", **kwargs)

        return train_loader, val_loader, test_loader
