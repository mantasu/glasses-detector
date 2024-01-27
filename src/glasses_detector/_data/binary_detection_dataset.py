from typing import override

import albumentations as A
import torch
from torch.utils.data import DataLoader

from .base_categorized_dataset import BaseCategorizedDataset


class BinaryDetectionDataset(BaseCategorizedDataset):
    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        images = [item[0] for item in batch]
        annots = [{"boxes": item[1], "labels": item[2]} for item in batch]

        return images, annots

    @override
    @classmethod
    def create_loader(cls, **kwargs) -> DataLoader:
        kwargs.setdefault("collate_fn", cls.collate_fn)
        return super().create_loader(**kwargs)

    @override
    def create_transform(self, is_train: bool, **kwargs) -> A.Compose:
        kwargs.setdefault("has_bbox", True)
        return super().create_transform(is_train, **kwargs)

    @override
    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        image, bboxes, bbcats = self.load_transform(
            image=sample[self.img_folder],
            boxes=[sample["annotations"]],
            bcats=[1],  # 0 - background
            transform=self.transform,
        )

        if len(bboxes) == 0:
            bboxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            bbcats = torch.tensor([0], dtype=torch.int64)

        return image, bboxes, bbcats
