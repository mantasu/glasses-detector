import torch
import torch.nn as nn


class TinyBinaryDetector(nn.Module):
    """Tiny binary detector.

    This is a custom detector created with the aim to contain very few
    parameters while maintaining a reasonable accuracy. It only has
    several sequential convolutional and pooling blocks (with
    batch-norm in between).

    Note:
        I tried varying the architecture, including activations,
        convolution behavior (groups and stride), pooling, and layer
        structure. This also includes residual and dense connections,
        as well as combinations. Turns out, they do not perform as well
        as the current architecture which is just a bunch of
        CONV-RELU-BN-MAXPOOL blocks with no paddings.
    """

    def __init__(self):
        super().__init__()

        # Several convolutional blocks
        self.features = nn.Sequential(
            self._create_block(3, 6, 15),
            self._create_block(6, 12, 7),
            self._create_block(12, 24, 5),
            self._create_block(24, 48, 3),
            self._create_block(48, 96, 3),
            self._create_block(96, 192, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Fully connected layer
        self.fc = nn.Linear(192, 4)

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

        This takes a list of images and returns a list of predictions
        for each image or a loss dictionary if the targets are provided.
        This is to match the API of the PyTorch *torchvision* models,
        which specify that:

            "During training, returns a dictionary containing the
            classification and regression losses for each image in the
            batch. During inference, returns a list of dictionaries, one
            for each input image. Each dictionary contains the predicted
            boxes, labels, and scores for all detections in the image."

        Args:
            imgs (list[torch.Tensor]): A list of images.
            annotations (list[dict[str, torch.Tensor]], optional): A
                list of annotations for each image. Each annotation is a
                dictionary that contains:

                1. ``"boxes"``: the bounding boxes for each object
                2. ``"labels"``: labels
                   for all objects in the image. If ``None``, the
                   network is in inference mode.


        Returns:
            dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
            A dictionary with only a single "regression" loss entry if
            ``targets`` were specified. Otherwise, a list of
            dictionaries with the predicted bounding boxes, labels, and
            scores for all detections in each image.
        """
        # Forward pass; insert a new dimension to indicate a single bbox
        preds = self.fc(self.features(torch.stack(imgs)))

        # Get width and height
        h, w = imgs[0].shape[-2:]

        # Convert to (x_min, y_min, x_max, y_max)
        preds[:, 0] = preds[:, 0] * w
        preds[:, 1] = preds[:, 1] * h
        preds[:, 2] = preds[:, 0] + preds[:, 2] * w
        preds[:, 3] = preds[:, 1] + preds[:, 3] * h

        if targets is None:
            # Clamp the coordinates to the image size
            preds[:, 0] = torch.clamp(preds[:, 0], 0, w)
            preds[:, 1] = torch.clamp(preds[:, 1], 0, h)
            preds[:, 2] = torch.clamp(preds[:, 2], 0, w)
            preds[:, 3] = torch.clamp(preds[:, 3], 0, h)

        # Convert to shape (N, 1, 4)
        preds = [*preds[:, None, :]]

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

        This computes the MSE loss between the predicted bounding boxes
        and the target bounding boxes. The returned dictionary contains
        only one key: "regression".

        Args:
            preds (list[torch.Tensor]): A list of predicted bounding
                boxes for each image.
            targets (list[dict[str, torch.Tensor]]): A list of targets
                for each image.

        Returns:
            dict[str, torch.Tensor]: A dictionary with only one key:
            "regression" which contains the regression MSE loss.
        """
        # Initialize criterion, loss dictionary, and device
        criterion, loss_dict, device = nn.MSELoss(), {}, preds[0].device

        # Use to divide (x_min, y_min, x_max, y_max) by (w, h, w, h)
        size = torch.tensor([[*size[::-1], *size[::-1]]], device=device)

        for i, pred in enumerate(preds):
            # Compute the loss (normalize the coordinates before that)
            loss = criterion(pred / size, targets[i]["boxes"][:1] / size)
            loss_dict[i] = loss

        return loss_dict
