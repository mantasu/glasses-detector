import os
import torch
import random

from typing import Iterable, Any
from collections import defaultdict
from torch.utils.data import Dataset
from mixins import ImageLoaderMixin, DataLoaderMixin

AVAILABLE_CLASSIFICATION_DIRS = [
    "cmu-face-images",
    "specs-on-faces",
    "sunglasses-no-sunglasses",
    "glasses-and-coverings",
    "face-attributes-grouped",
]

class ImageClassificationDataset(Dataset, ImageLoaderMixin, DataLoaderMixin):
    def __init__(
        self, 
        root: str = '.',
        dirs: Iterable[str] = [''],
        split_type: str = "train",
        label_type: str | dict[str, Any] = "onehot", # enum, onehot, {} !vals must be immutable objects
        seed: int = 0,
    ):
        super().__init__()

        # Init attributes and local vars
        self.data, self.label2name = [], {}
        cat2paths = defaultdict(lambda: [])
        
        for dir in dirs:
            for cat in os.scandir(os.path.join(root, dir, split_type)):
                # Add path to the image under category of the dir's name
                cat2paths[cat].extend([f.path for f in os.scandir(cat.path)])

        for i, (key, val) in enumerate(cat2paths.items()):
            # Create a correct label
            if isinstance(label_type, dict):
                label = label_type[key]
            elif label_type == "enum":
                label = i
            elif label_type == "onehot":
                label = (int(i == j) for j in range(len(cat2paths)))
            
            # Update mapping and data
            self.label2name[label] = key
            self.data.extend([(img_path, label) for img_path in val])
        
        # Sort & shuffle
        self.data.sort()
        random.seed(seed)
        random.shuffle(self.data)

        # Create image augmentation pipeline based on split type
        self.transform = self.create_transform(split_type)
    
    @property
    def name2label(self, name):
        return (k for k, v in self.label2name.items() if v == name)[0]

    def __getitem__(self, index):
        # Get the data and tensorize
        img_path, label = self.data[index]
        x = self.load_image(img_path, self.transform)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.data)
