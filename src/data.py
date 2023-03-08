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
                 seed: int | None = None,
                ):
        super().__init__()

        # Initialize samples
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
        
        if seed is not None:
            # Sort the samples and shuffle them deterministically
            self.samples = sorted(self.samples, key=lambda x: x[0])
            random.seed(seed)
            random.shuffle(self.samples)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the stored sample and label
        image_path, label = self.samples[index]
        
        # Load image and label to tensors
        sample = self.load_sample(image_path)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def make_sample(self, root: str, file: str) -> tuple[str, bool]:
        # Sample is a path, label - bool
        sample = os.path.join(root, file)
        label = f"no_sunglasses" not in root
        
        return sample, label

    def load_sample(self, image_path: str | os.PathLike) -> torch.Tensor:
        # Read image, convert to RGB (if grayscale) and to tensor
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image  

class SunglassesOrNotModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str | os.PathLike,
                 augment_train: bool = True,
                 **loader_kwargs
                ):
        super().__init__()
        # Assign dataset attributes
        self.data_path = data_path
        self.train_transform = None
        self.loader_kwargs = loader_kwargs

        if augment_train:
            self.train_transform = T.Compose(self.create_augmentation())

        # Set some default data loader arguments
        self.loader_kwargs.setdefault("batch_size", 10)
        self.loader_kwargs.setdefault("num_workers", 24)
        self.loader_kwargs.setdefault("pin_memory", True)
    
    def create_augmentation(self) -> list[torch.nn.Module]:
        return [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomInvert(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize([.5, .5, .5], [.5, .5, .5])
        ]

    def train_dataloader(self) -> DataLoader:
        # Create train dataset and return loader
        train_dataset = SunglassesOrNotDataset(
            data_path=self.data_path,
            target="train",
            transform=self.train_transform
        )
        return DataLoader(train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Create val dataset and return loader
        val_dataset = SunglassesOrNotDataset(
            data_path=self.data_path,
            target="val",
            seed=0
        )
        return DataLoader(val_dataset, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        # Create test dataset and return loader
        test_dataset = SunglassesOrNotDataset(
            data_path=self.data_path,
            target="test",
            seed=0
        )
        return DataLoader(test_dataset, **self.loader_kwargs)
