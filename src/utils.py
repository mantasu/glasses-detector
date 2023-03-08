import os
import torch
import random
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the F1 score given two tensors.
    
    Parameters:
        y_true (torch.Tensor): True labels tensor.
        y_pred (torch.Tensor): Predicted labels tensor.
    
    Returns:
        float: The F1 score.
    """
    # Compute true positives, false positives, and false negatives
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    return f1_score.item()

def compute_gamma(num_epochs: int, start_lr: float = 1e-3, end_lr: float = 5e-5) -> float:
    if num_epochs < 1:
        return 1
    
    return (end_lr / start_lr) ** (1 / num_epochs)

def get_checkpoint_callback() -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath="checkpoints",
        filename="sunglasses-classifier-{epoch:02d}",
        save_last=False,
        every_n_epochs=1,
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )

def seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
