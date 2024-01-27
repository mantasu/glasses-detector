import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .metrics import BoxClippedR2, BoxIoU, BoxMSLE


class BinaryDetector(pl.LightningModule):
    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()

        # Assign attributes
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = 1e-3

        # Initialize val_loss metric (just the mean)
        self.val_loss = torchmetrics.MeanMetric()

        # Create F1 score to monitor average label performance
        self.label_metrics = torchmetrics.MetricCollection(
            [torchmetrics.F1Score(task="binary")]
        )

        # Initialize some metrics to monitor bbox performance
        self.boxes_metrics = torchmetrics.MetricCollection(
            [BoxMSLE(), BoxIoU(), BoxClippedR2()]
        )

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        # Forward propagate and compute loss
        loss_dict = self(batch[0], batch[1])
        loss = sum(loss for loss in loss_dict.values()) / len(batch[0])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(self, batch):
        # Forward pass and compute loss
        with torch.inference_mode():
            self.train()
            loss = sum(loss for loss in self(batch[0], batch[1]).values())
            self.val_loss.update(loss / len(batch[0]))
            self.eval()

        # Update all the metrics
        self.boxes_metrics.update(self(batch[0]), batch[1], self.label_metrics)

    def on_eval_epoch_end(self, prefix=""):
        # Compute total loss and metrics
        loss = self.val_loss.compute()
        label_metrics = self.label_metrics.compute()
        boxes_metrics = self.boxes_metrics.compute()

        # Log the metrics and the learning rate
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_f1", label_metrics["BinaryF1Score"], prog_bar=True)
        self.log(f"{prefix}_msle", boxes_metrics["BoxMSLE"], prog_bar=True)
        self.log(f"{prefix}_r2", boxes_metrics["BoxClippedR2"], prog_bar=True)
        self.log(f"{prefix}_iou", boxes_metrics["BoxIoU"], prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch)

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(prefix="val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch)

    def on_test_epoch_end(self):
        self.on_eval_epoch_end(prefix="test")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        # Initialize AdamW optimizer and Reduce On Plateau scheduler
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
