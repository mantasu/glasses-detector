import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BinaryClassifier(pl.LightningModule):
    def __init__(self, model, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()

        # Assign attributes
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Create loss function and account for imbalance of classes
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Create F1 score and ROC-AUC metrics to monitor
        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(task="binary"),
                torchmetrics.AUROC(task="binary"),  # ROC-AUC
                torchmetrics.AveragePrecision(task="binary"),  # PR-AUC
            ]
        )

    @property
    def pos_weight(self):
        if self.train_loader is None:
            # Not known
            return None

        # Calculate the positive weight to account for class imbalance
        targets = np.array([y for _, y in iter(self.train_loader.dataset.data)])
        pos_count = targets.sum()
        neg_count = len(targets) - pos_count

        return torch.tensor(neg_count / pos_count)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Forward propagate and compute loss
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def eval_step(self, batch, prefix=""):
        # Forward pass
        x, y = batch
        y_hat = self(x)

        # Compute the loss and the metrics
        loss = self.criterion(y_hat, y)
        metrics = self.metrics(y_hat, y.long())

        # Log the loss and the metrics
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_f1", metrics["BinaryF1Score"], prog_bar=True)
        self.log(f"{prefix}_roc_auc", metrics["BinaryAUROC"], prog_bar=True)
        self.log(f"{prefix}_pr_auc", metrics["BinaryAveragePrecision"], prog_bar=True)

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
        # Initialize AdamW optimizer and Cosine Annealing scheduler
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=0.1)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
