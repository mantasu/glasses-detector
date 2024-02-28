import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from os.path import basename, splitext
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from .augmenter_mixin import AugmenterMixin


class BaseCategorizedDataset(ABC, Dataset, AugmenterMixin):
    def __init__(
        self,
        root: str = ".",
        split_type: str = "train",
        img_folder: str = "images",
        label_type: str = "enum",
        cat2idx_fn: Callable[[str], int] | dict[str | int] | None = None,
        pth2idx_fn: Callable[[str], str] = lambda x: splitext(basename(x))[0],
        seed: int = 0,
    ):
        super().__init__()

        self.label_type = label_type
        self.img_folder = img_folder
        self.data = defaultdict(lambda: {})
        self.cats = []

        for dataset in os.listdir(root):
            if not os.path.isdir(p := os.path.join(root, dataset, split_type)):
                # No split
                continue

            for cat in os.scandir(p):
                if cat.name != img_folder and cat.name not in self.cats:
                    # Expand category list
                    self.cats.append(cat.name)

                for file in os.scandir(cat.path):
                    # Add image/annotation path under file name as key
                    self.data[pth2idx_fn(file.path)][cat.name] = file.path

        # Shuffle only values (sort first for reproducibility)
        self.data = [v for _, v in sorted(self.data.items())]
        random.seed(seed)
        random.shuffle(self.data)

        # Sort cats as well
        self.cats = sorted(
            self.cats,
            key=(
                None
                if cat2idx_fn is None
                else cat2idx_fn.get if isinstance(cat2idx_fn, dict) else cat2idx_fn
            ),
        )

        # Create a default transformation
        self.transform = self.create_transform(split_type == "train")
        self.__post_init__()

    @cached_property
    def cat2idx(self) -> dict[str, int]:
        return dict(zip(self.cats, range(len(self.cats))))

    @cached_property
    def idx2cat(self) -> dict[int, str]:
        return dict(zip(range(len(self.cats)), self.cats))

    @classmethod
    def create_loader(cls, **kwargs) -> DataLoader:
        # Get argument names from DataLoader
        fn_code = DataLoader.__init__.__code__
        init_arg_names = fn_code.co_varnames[: fn_code.co_argcount]

        # Split all the given kwargs to dataset (cls) and loader kwargs
        set_kwargs = {k: v for k, v in kwargs.items() if k not in init_arg_names}
        ldr_kwargs = {k: v for k, v in kwargs.items() if k in init_arg_names}

        # Define default loader kwargs
        default_loader_kwargs = {
            "dataset": cls(**set_kwargs),
            "batch_size": 64,
            "num_workers": 12,
            "pin_memory": True,
            "drop_last": True,
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

    def cat2tensor(self, cat: str | list[str]) -> torch.Tensor:
        # Convert category name(-s) to the index list
        cat = cat if isinstance(cat, list) else [cat]
        indices = list(map(self.cat2idx.get, cat))

        match self.label_type:
            case "enum":
                label = torch.tensor(indices)
            case "onehot":
                label = torch.eye(len(self.cats))[indices]
            case "multihot":
                label = torch.any(torch.eye(len(self.cats))[indices], 0, True)
            case _:
                raise ValueError(f"Unknown label type: {self.label_type}")

        return label.to(torch.long)

    def tensor2cat(self, tensor: torch.Tensor) -> str | list[str] | list[list[str]]:
        match self.label_type:
            case "enum":
                # Add a batch dimension if tensor is a scalar, get cats
                ts = tensor.unsqueeze(0) if tensor.ndim == 0 else tensor
                cat = [self.idx2cat[i.item()] for i in ts]
                return cat[0] if tensor.ndim == 0 or len(tensor) == 0 else cat
            case "onehot":
                # Get cats directly (works for both 1D and 2D tensors)
                cat = [self.idx2cat[i.item()] for i in torch.where(tensor)[0]]
                return cat[0] if tensor.ndim == 1 else cat
            case "multihot":
                # Add a batch dimension if tensor is a 1D list, get cats
                ts = tensor if tensor.ndim > 1 else tensor.unsqueeze(0)
                cat = [[self.idx2cat[i.item()] for i in torch.where[t][0]] for t in ts]
                return cat[0] if tensor.ndim == 1 else cat
            case _:
                raise ValueError(f"Unknown label type: {self.label_type}")

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, index: int) -> Any: ...

    def __post_init__(self):
        pass
