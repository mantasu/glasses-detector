import os
import torch
import random
import pytorch_lightning as pl
import torchvision.transforms as T

from PIL import Image
from typing import Any
from torch.utils.data import Dataset, DataLoader

class SunglassesOrNotDataset(Dataset):
    def __init__(self,
                 data_path: str | os.PathLike = "data",
                 target: str ="train",
                 transform: T.Compose | None = None,
                 load_in_memory: bool = False
                ):
        super().__init__()

        # Initialize some dataset attributes
        self.load_in_memory = load_in_memory
        self.samples = []

        # Assign transform or make default
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5]),
        ]) if transform is None else transform

        for root, _, files in os.walk(data_path):
            if target in root:
                # Append all sample-label pairs from the target directory
                self.samples.extend([self.make_sample(root, f) for f in files])
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # Get the stored sample and label
        sample, label = self.samples[index]

        if not self.load_in_memory:
            # Load sample if it's not in memory
            sample = self.load_sample(sample)
            label = torch.tensor(label, dtype=torch.long)
        
        return sample, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def make_sample(self, root: str, file: str) -> str | torch.Tensor:
        # Sample is a path, label - bool
        sample = os.path.join(root, file)
        label = f"not_sunglasses" not in root

        if self.load_in_memory:
            # Load the tensor to memory
            sample = self.load_sample(sample)
            label = torch.tensor(label, dtype=torch.long)
        
        return sample, label

    def load_sample(self, path_to_image: str | os.PathLike) -> torch.Tensor:
        # Read image and convert to tensor
        image = Image.open(path_to_image)
        image = self.transform(image)

        return image  

class SunglassesOrNotJointModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str | os.PathLike,
                 train_transform: T.Compose | None = None,
                 test_transform: T.Compose | None = None,
                 load_in_memory: bool = False,
                 **loader_kwargs
                ):
        super().__init__()
        # Assign dataset attributes
        self.data_path = data_path
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.load_in_memory = load_in_memory
        self.loader_kwargs = loader_kwargs

    def train_dataloader(self) -> DataLoader:
        # Create train dataset and return loader
        train_dataset = SunglassesOrNotDataset(
            data_path=self.data_path,
            target="train",
            transform=self.train_transform,
            load_in_memory=self.load_in_memory
        )
        return DataLoader(train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Create val dataset and return loader
        val_dataset = SunglassesOrNotDataset(
            data_path=self.data_path,
            target="val",
            transform=self.test_transform,
            load_in_memory=self.load_in_memory
        )
        return DataLoader(val_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = SunglassesOrNotDataset(
            data_path=self.data_path,
            target="test",
            transform=self.test_transform,
            load_in_memory=self.load_in_memory
        )
        return DataLoader(test_dataset, shuffle=True, **self.loader_kwargs)