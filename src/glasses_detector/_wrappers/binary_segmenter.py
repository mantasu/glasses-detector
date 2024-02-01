import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

from .tools import CosineAnnealingWarmerRestarts


class BinarySegmenter(pl.LightningModule):
    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()

        # Assign attributes
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Create loss function and account for imbalance of classes
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.val_loss = torchmetrics.MeanMetric()

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MatthewsCorrCoef(task="binary"),
                torchmetrics.F1Score(task="binary"),  # same as Dice
                torchmetrics.JaccardIndex(task="binary"),  # IoU
            ]
        )

    @property
    def pos_weight(self):
        if self.train_loader is None:
            # Not known
            return None

        # Init counts
        pos, neg = 0, 0

        for _, mask in self.train_loader:
            # Update pos and neg sums
            pos += mask.sum()
            neg += (1 - mask).sum()

        return neg / pos

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        # Forward propagate and compute loss
        loss = self.criterion(self(batch[0]), batch[1])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(self, batch, prefix=""):
        # Forward pass
        x, y = batch
        y_hat = self(x)

        # Compute the loss and the metrics
        self.val_loss.update(self.criterion(y_hat, y))
        self.metrics.update(y_hat.sigmoid(), y.long())
        # self.log(f"{prefix}_loss", self.criterion(y_hat, y), prog_bar=True)

    def on_eval_epoch_end(self, prefix=""):
        # Compute total loss and metrics
        loss = self.val_loss.compute()
        metrics = self.metrics.compute()

        # Reset the loss and the metrics
        self.val_loss.reset()
        self.metrics.reset()

        # Log the loss and the metrics
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_mcc", metrics["BinaryMatthewsCorrCoef"], prog_bar=True)
        self.log(f"{prefix}_f1", metrics["BinaryF1Score"], prog_bar=True)
        self.log(f"{prefix}_iou", metrics["BinaryJaccardIndex"], prog_bar=True)

        if not isinstance(opt := self.optimizers(), list):
            # Log the learning rate of a single optimizer
            self.log("lr", opt.param_groups[0]["lr"], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, prefix="val")

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(prefix="val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, prefix="test")

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
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, threshold=0.01
        )

        # scheduler = ExponentialLR(optimizer, gamma=0.98)

        # Multiply eta_max by gamma after every restart (warmer restart)
        # optimizer = AdamW(self.parameters(), lr=5e-3, weight_decay=1e-2)
        # scheduler = CosineAnnealingWarmerRestarts(optimizer, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
