import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as T

from PIL import Image
from utils import f1_score, compute_gamma
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights

class SunglassesClssifier(pl.LightningModule):
    def __init__(self, num_epochs: int = -1):
        super().__init__()
        self.gamma = compute_gamma(num_epochs, start_lr=3e-3, end_lr=6e-4)
        self.features = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        self.classifier = nn.Sequential(nn.Linear(1000, 1), nn.Flatten(0))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
    
    def predict(self, img: str | Image.Image) -> str:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        transform = T.Compose([T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)])

        x = transform(img).unsqueeze(0)
        y_hat = self(x).sigmoid().item()

        prediction = "wears sunglasses" if round(y_hat) else "no sunglasses"
        confidence = y_hat if round(y_hat) else 1 - y_hat

        return prediction, confidence
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)

        return torch.stack((y.squeeze(), y_hat.squeeze()))
    
    def validation_epoch_end(self, validation_step_outputs: list):
        [y, y_hat] = torch.cat(validation_step_outputs, dim=1)

        loss = self.loss_fn(y_hat, y).item()
        f1 = f1_score(y, y_hat.sigmoid().round())

        self.log_dict({"val_loss": loss, "val_f1": f1}, prog_bar=True)
        

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)

        return torch.stack((y.squeeze(), y_hat.squeeze()))
    
    def test_epoch_end(self, test_step_outputs: list):
        [y, y_hat] = torch.cat(test_step_outputs, dim=1)

        loss = self.loss_fn(y_hat, y).item()
        f1 = f1_score(y, y_hat.sigmoid().round())

        self.log_dict({"test_loss": loss, "test_f1": f1}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.gamma)

        return [optimizer], [scheduler]
