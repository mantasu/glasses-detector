from typing import override

import torch

from .base_categorized_dataset import BaseCategorizedDataset


class BinaryClassificationDataset(BaseCategorizedDataset):
    def __post_init__(self):
        # Flatten (some image names may have been the same across cats)
        self.data = [dict([cat]) for d in self.data for cat in d.items()]

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cat, pth = next(iter(self.data[index].items()))
        label = self.cat2tensor(cat)
        image = self.load_transform(image=pth, transform=self.transform)

        return image, label
