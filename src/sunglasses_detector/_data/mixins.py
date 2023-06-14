import numpy
import torch
import PIL.Image as Image
import albumentations as A

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

class ImageLoaderMixin():
    @staticmethod
    def create_transform(is_train: bool = False) -> A.Compose:
        # Default augmentation
        transform = [
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(),
            A.OneOf([
                A.RandomResizedCrop(256, 256, p=0.5),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
                A.PiecewiseAffine(),
                A.Perspective()
            ]),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(),
                A.CLAHE(),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                A.HueSaturationValue(),
            ]),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussianBlur(),
                A.MedianBlur(),
                A.GaussNoise(),
            ]),
            A.CoarseDropout(max_holes=5, p=0.3),
            A.Normalize(),
            ToTensorV2()
        ]

        if not is_train:
            # Only keep the last two
            transofrm = transofrm[:-2]
        
        return A.Compose(transform)

    @classmethod
    def load_image(
        cls, 
        image: str | Image.Image | numpy.ndarray,
        transform: A.Compose | bool = False,
    ) -> torch.Tensor:
        if isinstance(transform, bool):
            # Load transform (only normalize)
            transform = cls.create_transform(transform)
        
        if isinstance(image, str):
            # If the image is provided as a path to it
            image = Image.open(image).convert("RGB")
        
        if isinstance(image, Image.Image):
            # If the image is not ndarray
            image = numpy.array(image)
        
        # Apply augmentation to the image
        image = transform(image=image)["image"]

        return image

class DataLoaderMixin():
    @classmethod
    def create_loader(cls, **kwargs):
        # Split all the given kwargs to dataset (cls) and loader kwargs
        cls_kwargs_set = {"root", "dirs", "split_type", "label_type", "seed"}
        set_kwargs = {k: v for k, v in kwargs.items() if k in cls_kwargs_set} 
        ldr_kwargs = {k: v for k, v in kwargs.items() if k not in cls_kwargs_set}

        # Define default loader kwargs
        default_loader_kwargs = {
            "dataset": cls(**set_kwargs),
            "batch_size": 64,
            "num_workers": 12,
            "pin_memory": True,
            "shuffle": set_kwargs.get("data_split", "train") == "train"
        }

        # Update default loader kwargs with custom
        default_loader_kwargs.update(ldr_kwargs)

        return DataLoader(**default_loader_kwargs)

    @classmethod
    def create_loaders(cls, **kwargs):
        # Create train, validationa and test loaders
        train_loader = cls.create_loader(split_type="train", **kwargs)
        val_loader = cls.create_loader(split_type="val", **kwargs)
        test_loader = cls.create_loader(split_type="test", **kwargs)

        return train_loader, val_loader, test_loader