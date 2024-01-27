from copy import deepcopy

import albumentations as A
import numpy as np
import skimage.transform as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ToTensor(ToTensorV2):
    def apply_to_mask(self, mask, **params):
        return torch.from_numpy((mask > 0).astype(np.float32))


class AugmenterMixin:
    @staticmethod
    def default_augmentations() -> list[A.BasicTransform]:
        return [
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
            A.CoarseDropout(max_holes=5, p=0.3),
            A.Normalize(),
            ToTensor(),
            # ToTensorV2(),
        ]

    @classmethod
    def create_transform(
        cls,
        is_train: bool = False,
        **kwargs,
    ) -> A.Compose:
        # Get the list of default augmentations
        transform = cls.default_augmentations()
        kwargs = deepcopy(kwargs)

        if kwargs.pop("has_bbox", False):
            # Add bbox params
            kwargs.setdefault(
                "bbox_params",
                A.BboxParams(
                    format="pascal_voc",
                    label_fields=["bbcats"],
                    min_visibility=0.1,
                    **kwargs.pop("bbox_kwargs", {}),
                ),
            )
            if isinstance(transform[-3], A.CoarseDropout):
                # CoarseDropout not supported with bbox_params
                transform.pop(-3)

        if kwargs.pop("has_keys", False):
            # Add keypoint params
            kwargs.setdefault(
                "keypoint_params",
                A.KeypointParams(
                    format="xy",
                    label_fields=["kpcats"],
                    remove_invisible=False,
                    **kwargs.pop("keys_kwargs", {}),
                ),
            )

        if not is_train:
            # Only keep the last two
            transform = transform[-2:]

        return A.Compose(transform, **kwargs)

    @staticmethod
    def load_image(
        image: str | Image.Image | np.ndarray,
        resize: tuple[int, int] | None = None,
        is_mask: bool = False,
        return_orig_size: bool = False,
    ) -> np.ndarray:
        if isinstance(image, str):
            # Image is given as path
            image = Image.open(image)

        if isinstance(image, Image.Image):
            # Image is not a numpy array
            image = np.array(image)

        if is_mask:
            # Convert image to black & white and ensure only 1 channel
            image = ((image > 127).any(2) if image.ndim > 2 else (image > 127)) * 255
        elif image.ndim == 2:
            # Image isn't a mask, convert it to RGB
            image = np.stack([image] * 3, axis=-1)

        # Convert image to UINT8
        image = image.astype(np.uint8)
        orig_size = image.shape[:2][::-1]

        if resize is not None:
            # Resize image to new (w, h)
            image = st.resize(image, resize[::-1])

        if return_orig_size:
            # Original size as well
            return image, orig_size

        return image

    @staticmethod
    def load_boxes(
        boxes: str | list[list[int | float | str]],
        resize: tuple[int, int] | None = None,
        img_size: tuple[int, int] | None = None,
    ) -> list[list[float]]:
        if isinstance(boxes, str):
            with open(boxes, "r") as f:
                # Each line is bounding box: "x_min y_min x_max y_max"
                boxes = [xyxy.strip().split() for xyxy in f.readlines()]

        # Convert each coordinate in each bbox to float
        boxes = [list(map(float, xyxy)) for xyxy in boxes]

        if img_size is None:
            if resize is not None:
                raise ValueError("img_size must be provided if resize is not None")

            return boxes

        for i, box in enumerate(boxes):
            if box[2] <= box[0]:
                # Ensure x_min < x_max <= img_size[0]
                boxes[i][0] = min(box[0], img_size[0] - 1)
                boxes[i][2] = boxes[i][0] + 1

            if box[3] <= box[1]:
                # Ensure y_min < y_max <= img_size[1]
                boxes[i][1] = min(box[1], img_size[1] - 1)
                boxes[i][3] = boxes[i][1] + 1

        if resize is not None:
            # Convert boxes to new (w, h)
            boxes = [
                [
                    box[0] * resize[0] / img_size[0],
                    box[1] * resize[1] / img_size[1],
                    box[2] * resize[0] / img_size[0],
                    box[3] * resize[1] / img_size[1],
                ]
                for box in boxes
            ]

        return boxes

    @staticmethod
    def load_keypoints(
        keypoints: str | list[list[int | float | str]],
        resize: tuple[int, int] | None = None,
        img_size: tuple[int, int] | None = None,
    ) -> list[list[float]]:
        if isinstance(keypoints, str):
            with open(keypoints, "r") as f:
                # Each line is keypoint: "x y"
                keypoints = [xy.strip().split() for xy in f.readlines()]

        # Convert each coordinate in each keypoint to float
        keypoints = [list(map(float, xy)) for xy in keypoints]

        if img_size is None:
            if resize is not None:
                raise ValueError("img_size must be provided if resize is not None")

            return keypoints

        if resize is not None:
            # Convert keypoints to new (w, h)
            keypoints = [
                [
                    keypoint[0] * resize[0] / img_size[0],
                    keypoint[1] * resize[1] / img_size[1],
                ]
                for keypoint in keypoints
            ]

        return keypoints

    @classmethod
    def load_transform(
        cls,
        image: str | Image.Image | np.ndarray,
        masks: list[str | Image.Image | np.ndarray] = [],
        boxes: list[str | list[list[int | float | str]]] = [],
        bcats: list[str] = [],
        keys: list[str | list[list[int | float | str]]] = [],
        kcats: list[str] = [],
        resize: tuple[int, int] | None = None,
        transform: A.Compose | bool = False,  # False means test/val
    ) -> torch.Tensor | tuple[torch.Tensor]:
        # Load the image and resize if needed (also return original size)
        image, orig_size = cls.load_image(image, resize, return_orig_size=True)
        transform_kwargs = {"image": image}

        if isinstance(transform, bool):
            # Load transform (train or test is based on bool val)
            transform = cls.create_transform(is_train=transform)

        if masks != []:
            # Load masks and add to transform kwargs
            masks = [cls.load_image(m, resize, is_mask=True) for m in masks]
            transform_kwargs.update({"masks": masks})

        if boxes != []:
            # Initialize flat boxes and cats
            flat_boxes, flat_bcats = [], []

            for i, b in enumerate(boxes):
                # Load boxes and add to transform kwargs
                b = cls.load_boxes(b, resize, orig_size)
                flat_boxes.extend(b)

                if bcats != []:
                    # For each box, add corresponding cat
                    flat_bcats.extend([bcats[i]] * len(b))

            # Add boxes to transform kwargs
            transform_kwargs.update({"bboxes": flat_boxes})

            if bcats != []:
                # Also add cats to transform kwargs
                transform_kwargs.update({"bbcats": flat_bcats})

        if keys != []:
            # Initialize flat keypoints and cats
            flat_keys, flat_kcats = [], []

            for i, k in enumerate(keys):
                # Load keypoints and add to transform kwargs
                k = cls.load_keypoints(k, resize, orig_size)
                flat_keys.extend(k)

                if kcats != []:
                    # For each keypoint, add corresponding cat
                    flat_kcats.extend([kcats[i]] * len(k))

            # Add keypoints to transform kwargs, update cats
            transform_kwargs.update({"keypoints": flat_keys})

            if kcats != []:
                # Also add cats to transform kwargs
                transform_kwargs.update({"kpcats": flat_kcats})

        # Transform everything, generate return list
        transformed = transform(**transform_kwargs)
        return_list = [transformed["image"]]

        for key in ["masks", "bboxes", "bbcats", "keypoints", "kpcats"]:
            if key not in transformed:
                continue

            if key in {"bboxes", "keypoints", "bbcats", "kpcats"}:
                # Convert to torch tensor if key is category or bbox/keypoint
                dtype = torch.long if key in ["bbcats", "kpcats"] else torch.float32
                transformed[key] = torch.tensor(transformed[key], dtype=dtype)

            # Add to the return list
            return_list.append(transformed[key])

        if len(return_list) == 1:
            return return_list[0]

        return tuple(return_list)
