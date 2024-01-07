import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops import box_iou


class BinaryDetector(pl.LightningModule):
    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()

        # Assign attributes
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize some metrics to monitor the performance
        self.label_metric = torchmetrics.F1Score(task="binary")
        self.boxes_metric = torchmetrics.R2Score(num_outputs=4)

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        # Forward propagate and compute loss
        # imgs = [img for img in batch[0]]
        # annotations = [{k: v for k, v in t.items()} for t in batch[1]]
        imgs = [*batch[0]]
        annotations = [
            {"boxes": b, "labels": l}
            for b, l in zip(batch[1]["boxes"], batch[1]["labels"])
        ]
        loss_dict = self(imgs, annotations)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(self, batch, prefix=""):
        # Forward pass and compute loss
        # imgs = [img for img in batch[0]]
        # annotations = [{k: v for k, v in t.items()} for t in batch[1]]

        imgs = [*batch[0]]
        annotations = [
            {"boxes": b, "labels": l}
            for b, l in zip(batch[1]["boxes"], batch[1]["labels"])
        ]

        with torch.inference_mode():
            self.train()
            loss_dict = self(imgs, annotations)
            self.eval()

        loss = sum(loss for loss in loss_dict.values())

        # Get actual labels and predictions
        y_labels = torch.stack([ann["labels"] for ann in annotations])
        y_boxes = torch.stack([ann["boxes"] for ann in annotations])
        y_hat = self(imgs)
        y_hat_labels = torch.stack([pred["labels"] for pred in y_hat])
        y_hat_boxes = torch.stack([pred["boxes"] for pred in y_hat])

        # Compute metrics
        f1_score = self.label_metric(y_hat_labels, y_labels)
        r2_score = self.boxes_metric(y_hat_boxes.squeeze(), y_boxes.squeeze())
        ious = [
            box_iou(pred_box, target_box)
            for pred_box, target_box in zip(y_hat_boxes, y_boxes)
        ]
        mean_iou = torch.stack([iou.mean() for iou in ious]).mean()

        # Log the loss and the metrics
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_f1", f1_score, prog_bar=True)
        self.log(f"{prefix}_r2", r2_score, prog_bar=True)
        self.log(f"{prefix}_iou", mean_iou, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, prefix="test")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
