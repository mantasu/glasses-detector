from abc import abstractmethod
from typing import override

import torch
import torchmetrics
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import mean_squared_log_error, r2_score
from torchvision.ops import box_iou


class BoxMetric(torchmetrics.Metric):
    def __init__(self, name="sum", is_min=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # Select min/max mode
        self.is_min = is_min
        self.name = name

    @abstractmethod
    def compute_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor: ...

    def update(
        self,
        preds: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
        *classification_metrics,
    ):
        # Initialize flat target labels and predicted labels
        pred_l = torch.empty(0, dtype=torch.long, device=preds[0]["labels"].device)
        target_l = torch.empty(0, dtype=torch.long, device=preds[0]["labels"].device)

        for pred, target in zip(preds, targets):
            if len(pred["boxes"]) == 0:
                pred["boxes"] = torch.tensor(
                    [[0, 0, 0, 0]], dtype=torch.float32, device=pred["boxes"].device
                )
                pred["labels"] = torch.tensor(
                    [0], dtype=torch.long, device=pred["labels"].device
                )

            if len(target["boxes"]) == 0:
                target["boxes"] = torch.tensor(
                    [[0, 0, 0, 0]], dtype=torch.float32, device=target["boxes"].device
                )
                target["labels"] = torch.tensor(
                    [0], dtype=torch.long, device=target["labels"].device
                )

            # Compute the matrix of similarities and select best
            similarities = self.compute_matrix(pred["boxes"], target["boxes"])
            cost_matrix = similarities if self.is_min else 1 - similarities
            pred_idx, target_idx = linear_sum_assignment(cost_matrix.cpu())
            best_sims = similarities[pred_idx, target_idx]

            # Add the labels of matched predictions
            target_l = torch.cat([target_l, target["labels"][target_idx]])
            pred_l = torch.cat([pred_l, pred["labels"][pred_idx]])

            if (remain := list(set(range(len(pred["boxes"]))) - set(pred_idx))) != []:
                # Add the labels of unmatched predictions, set as bg
                pred_l = torch.cat([pred_l, pred["labels"][remain]])
                padded = torch.tensor(len(remain) * [0], device=target_l.device)
                target_l = torch.cat([target_l, padded])

            if (
                remain := list(set(range(len(target["boxes"]))) - set(target_idx))
            ) != []:
                # Add the labels of unmatched predictions, set as bg
                target_l = torch.cat([target_l, target["labels"][remain]])
                padded = torch.tensor(len(remain) * [0], device=pred_l.device)
                pred_l = torch.cat([pred_l, torch.tensor(len(remain) * [0])])

            # Update the bbox metric
            setattr(self, self.name, getattr(self, self.name) + best_sims.sum())
            self.total += max(len(pred["boxes"]), len(target["boxes"]))

        for metric in classification_metrics:
            # Update classification metrics
            metric.update(pred_l, target_l)

    def compute(self):
        return getattr(self, self.name) / self.total


class BoxIoU(BoxMetric):
    def __init__(self, **kwargs):
        kwargs["name"] = "iou_sum"
        super().__init__(**kwargs)

    @override
    def compute_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return box_iou(preds, targets)


class BoxClippedR2(BoxMetric):
    def __init__(self, **kwargs):
        # Get r2 kwargs, remove them from kwargs
        r2_args = r2_score.__code__.co_varnames
        self.r2_kwargs = {k: v for k, v in kwargs.items() if k in r2_args}
        kwargs = {k: v for k, v in kwargs.items() if k not in r2_args}
        kwargs["name"] = "r2_sum"
        super().__init__(**kwargs)

    @override
    def compute_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    max(0, r2_score(p.view(-1), t.view(-1), **self.r2_kwargs))
                    for t in targets
                ]
                for p in preds
            ]
        )


class BoxMSLE(BoxMetric):
    def __init__(self, **kwargs):
        kwargs.setdefault("is_min", True)
        kwargs["name"] = "msle_sum"
        super().__init__(**kwargs)

    @override
    def compute_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                [mean_squared_log_error(p.view(-1), t.view(-1)) for t in targets]
                for p in preds
            ]
        )
