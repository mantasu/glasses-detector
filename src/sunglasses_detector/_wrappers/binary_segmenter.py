import sys
import torch
import numpy as np
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms.functional import to_tensor, normalize, to_pil_image
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.lraspp import LRASPPHead

from torchvision.models.resnet import BasicBlock

from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    fcn_resnet50, FCN_ResNet50_Weights,
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
)

sys.path.append("src")

# from utils.training import train
# from data.sunglasses_segmentation_data import SunglassesSegmentationDataModule


class BinarySegmenter(pl.LightningModule):
    def __init__(
        self,
        base_model: str = "deeplab",        
        is_base_pretrained: bool = False,
    ):
        super().__init__()
        # Load and replace the last layer with a binary segmentation head
        self.base_model = self.load_base_model(base_model, is_base_pretrained)

        # Initialize some metrics to monitor the performance
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task="binary"),
            torchmetrics.Dice()
        ])

        # Define a binary cross-entropy loss function + a scheduling param
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
    
    def load_base_model(self, model_name: str, is_pretrained: bool):
        match model_name:
            case "deeplab":
                w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                m = deeplabv3_resnet50(weights=w if is_pretrained else None)
                m.classifier = DeepLabHead(2048, 1)
                m.aux_classifier = None
            case "fcn":
                w = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                m = fcn_resnet50(weights=w if is_pretrained else None)
                m.classifier[-1] = nn.Conv2d(512, 1, 1)
                m.aux_classifier = None
            case "lraspp":
                w = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
                m = lraspp_mobilenet_v3_large(weights=w if is_pretrained else None)
                m.classifier = LRASPPHead(40, 960, 1, 128)
            case "mini":
                m = MiniSunglassesSegmenter()
            case _:
                raise NotImplementedError(f"{model_name} is not a valid choice!")
        
        return m

    def forward(self, x):
        # Pass the input through the segmentation model
        out = self.base_model(x)["out"]

        return out
    
    @torch.no_grad()
    def predict(self, x: Image.Image | np.ndarray) -> Image.Image:
        # Image to tensor
        x = to_tensor(x)

        # Normalize to standard augmentation values if not already norm
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Compute the mask prediction, round off
        x = self(x.unsqueeze(0)).squeeze().round()

        return to_pil_image(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        # Mini-batch
        x, y = batch

        # Forward pass + loss computation
        loss = self.loss_fn(self(x), y)

        # Log mini-batch train loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> dict[str, torch.Tensor]:
        # Get samples and predict
        x, y = batch
        y_hat = self(x)

        # Compute the mini-batch loss
        loss = self.loss_fn(y_hat, y)

        return {"loss": loss, "y_hat": y_hat, "y": y}
    
    def validation_epoch_end(self, outputs: dict[str, torch.Tensor], is_val: bool = True):
        # Concatinate all the computed losses to compute the average
        loss_mean = torch.stack([out["loss"] for out in outputs]).mean()

        # Concatinate y_hats and ys and apply the metrics
        y_hat = torch.cat([out["y_hat"] for out in outputs])
        y = torch.cat([out["y"] for out in outputs])
        metrics = list(self.metrics(y_hat, y.long()).values())

        if is_val:
            # If it's validation step, also show the learning rate
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("lr", lr, prog_bar=True)
        
        # Log validation or test MSE, SSIM, and PSNR to the progress bar
        self.log(f"{'val' if is_val else 'test'}_loss", loss_mean, True)
        self.log(f"{'val' if is_val else 'test'}_f1", metrics[0], True)
        self.log(f"{'val' if is_val else 'test'}_dice", metrics[1], True)

    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int) -> dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs: dict[str, torch.Tensor]):
        self.validation_epoch_end(outputs, is_val=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6)

        return [optimizer], [scheduler]


class MiniSunglassesSegmenter(nn.Module):
    class Down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            self.pool0 = nn.MaxPool2d(2)
            self.conv1 = Conv2dNormActivation(in_channels, out_channels)
            self.conv2 = Conv2dNormActivation(out_channels, out_channels)
        
        def forward(self, x):
            return self.conv2(self.conv1(self.pool0(x)))
        
    class Up(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            half_channels = in_channels // 2
            self.conv0 = nn.ConvTranspose2d(half_channels, half_channels, 2, 2)
            self.conv1 = Conv2dNormActivation(in_channels, out_channels)
            self.conv2 = Conv2dNormActivation(out_channels, out_channels)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor):
            x1 = self.conv0(x1)

            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2))
            
            x = torch.cat([x2, x1], dim=1)

            return self.conv2(self.conv1(x))
    
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            Conv2dNormActivation(3, 16),
            Conv2dNormActivation(16, 16),
        )

        self.down1 = self.Down(16, 32)
        self.down2 = self.Down(32, 64)
        self.down3 = self.Down(64, 128)
        self.down4 = self.Down(128, 128)

        self.up1 = self.Up(256, 64)
        self.up2 = self.Up(128, 32)
        self.up3 = self.Up(64, 16)
        self.up4 = self.Up(32, 16)

        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x1 = self.first(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.last(x)

        return {"out": out}

# def run_train(base_model: str = "test", **kwargs):

#     max_epochs = kwargs.pop("max_epochs", 310)
#     pretrained = kwargs.pop("is_base_pretrained", True)
#     model_name = kwargs.pop("model_name", "sunglasses-segmenter-" + base_model)

#     model = GlassesSegmenter("mini", pretrained)
#     datamodule = SunglassesSegmentationDataModule()

#     train(
#         model=model,
#         datamodule=datamodule,
#         model_name=model_name,
#         max_epochs=max_epochs,
#         **kwargs
#     )

# def run_test(base_model: str = "mini", ckpt_path: str = "checkpoints/unused/sunglasses-segmenter-fcn-epoch=291-val_loss=0.05329.ckpt"):
#     model = GlassesSegmenter.load_from_checkpoint(ckpt_path, base_model=base_model)
    
#     datamodule = SunglassesSegmentationDataModule()

#     trainer = pl.Trainer(accelerator="gpu")
#     trainer.test(model, datamodule=datamodule)

# if __name__ == "__main__":
#     run_test()