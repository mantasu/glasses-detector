from typing import override

import torch

from .base_categorized_dataset import BaseCategorizedDataset


class BinarySegmentationDataset(BaseCategorizedDataset):
    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        image, masks = self.load_transform(
            image=sample[self.img_folder],
            masks=[sample["masks"]],
            transform=self.transform,
        )

        return image, torch.stack(masks)
