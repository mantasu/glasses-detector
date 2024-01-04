import albumentations as A
import numpy
import PIL.Image as Image
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


class ImageLoaderMixin:
    @staticmethod
    def create_transform(is_train: bool = False) -> A.Compose:
        # Default augmentation
        transform = [
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(),
            A.OneOf(
                [
                    A.RandomResizedCrop(256, 256, p=0.5),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                    A.PiecewiseAffine(),
                    A.Perspective(),
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
            A.CoarseDropout(max_holes=5, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]

        if not is_train:
            # Only keep the last two
            transform = transform[-2:]

        return A.Compose(transform)

    @staticmethod
    def load_image(
        image: str | Image.Image | numpy.ndarray,
        masks: list[str | Image.Image | numpy.ndarray] = [],
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

            return image_file

        if isinstance(transform, bool):
            # Load transform (train/test is based on bool)
            transform = ImageLoaderMixin.create_transform(transform)

        # Load image and mask files
        image = open_image_file(image)
        masks = [open_image_file(m, True) for m in masks]

        if masks == []:
            return transform(image=image)["image"]

        # Transform the image and masks
        transformed = transform(image=image, masks=masks)
        image, masks = transformed["image"], transformed["masks"]

        return image, masks


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
