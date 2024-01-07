import torch
import torch.nn as nn


class TinyBinaryDetector(nn.Module):
    """Tiny binary detector.

    This is a custom detector created with the aim to contain very few
    parameters while maintaining a reasonable accuracy. It only has
    several sequential convolutional and pooling blocks (with
    batch-norm in between).
    """

    def __init__(self):
        super().__init__()

        # Several convolutional blocks
        self.features = nn.Sequential(
            self._create_block(3, 5, 3),
            self._create_block(5, 10, 3),
            self._create_block(10, 15, 3),
            self._create_block(15, 20, 3),
            self._create_block(20, 25, 3),
            self._create_block(25, 80, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # FC for bounding box prediction
        self.fc = nn.Linear(80, 4)

    def _create_block(self, num_in, num_out, filter_size):
        return nn.Sequential(
            nn.Conv2d(num_in, num_out, filter_size, 1, "valid", bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_out),
            nn.MaxPool2d(2, 2),
        )

    def forward(
        self,
        imgs: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            imgs (list[torch.Tensor]): A list of images.
            annotations (list[dict[str, torch.Tensor]], optional): A
                list of annotations for each image. Each annotation is a
                dictionary that contains the bounding boxes and labels
                for all objects in the image. If ``None``, the network
                is in inference mode.

        Returns:
            dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
            During training, returns a dictionary containing the
            classification and regression losses for each image in the
            batch. During inference, returns a list of dictionaries, one
            for each input image. Each dictionary contains the predicted
            boxes, labels, and scores for all detections in the image.
        """
        preds = [*self.fc(self.features(torch.stack(imgs)))[:, None, :]]

        if targets is not None:
            return self.compute_loss(preds, targets, imgs[0].size()[-2:])
        else:
            return [
                {
                    "boxes": pred,
                    "labels": torch.ones(1, dtype=torch.int64, device=pred.device),
                    "scores": torch.ones(1, device=pred.device),
                }
                for pred in preds
            ]

    def compute_loss(
        self,
        preds: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Compute the loss for the predicted bounding boxes.

        Args:
            preds (list[torch.Tensor]): A list of predicted bounding
                boxes for each image.
            targets (list[dict[str, torch.Tensor]]): A list of targets
                for each image.

        Returns:
            dict[str, torch.Tensor]: A dictionary of losses for each
            image in the batch.
        """
        criterion = nn.MSELoss()
        loss_dict = {}

        # Use to divide (x_min, y_min, x_max, y_max) by (w, h, w, h)
        size = torch.tensor([[*size[::-1], *size[::-1]]], device=preds[0].device)

        for i, pred in enumerate(preds):
            loss = criterion(pred / size, targets[i]["boxes"] / size)
            loss_dict[i] = loss

        return loss_dict
