import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..classifier import SunglassesClassifier
from .._data.classification_dataset import ImageClassificationDataset

class SunglassesClassifierWrapper(pl.LightningModule):
    def __init__(self, base_model, **kwargs):
        super().__init__()

        # Create the classifier and criterion instances
        self.classifier = SunglassesClassifier(base_model=base_model)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Create F1 score and ROC-AUC metrics to monitor
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task="binary"),
            torchmetrics.AUROC(task="binary")
        ])

        # Create the data loaders from
        loaders = ImageClassificationDataset.create_loaders(**kwargs)
        self.train_loader, self.val_loader, self.test_loader = loaders
    
    @property
    def pos_weight(self):
        # Calculate the positive weight to account for class imbalance
        targets = np.concatenate([y for _, y in iter(self.train_loader)])
        pos_count = targets.sum()
        neg_count = len(targets) - pos_count

        return torch.tensor(neg_count / pos_count)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        # Forward propagate and compute loss
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def eval_step(self, batch, prefix=''):
        # Forward pass
        x, y = batch
        y_hat = self(x)

        # Compute the loss and the metrics
        loss = self.criterion(y_hat, y)
        metrics = self.metrics(y_hat, y.long())

        # Log the loss and the metrics
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_f1", metrics["F1Score"], prog_bar=True)
        self.log(f"{prefix}_roc_auc", metrics["AUROC"], prog_bar=True)
    
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
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6)

        return [optimizer], [scheduler]