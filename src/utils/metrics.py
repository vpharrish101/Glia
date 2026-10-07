import torch
import numpy as np

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

_dice_metric=None
_iou_metric=None


def _get_dice_metric(device):
    global _dice_metric
    if _dice_metric is None:
        _dice_metric=BinaryF1Score(threshold=0.5).to(device)
    return _dice_metric

def _get_iou_metric(device):
    global _iou_metric
    if _iou_metric is None:
        _iou_metric=BinaryJaccardIndex(threshold=0.5).to(device)
    return _iou_metric


def dice_coefficient(pred: torch.Tensor,
                     target: torch.Tensor,)->torch.Tensor:
    """
    Batch-mean Dice (F1) for binary segmentation.

    Args:
        pred: Raw logits, shape (B, 1, H, W).
        target: Binary mask, shape (B, 1, H, W).
        threshold: Binarization threshold applied after sigmoid.
    """

    metric=_get_dice_metric(pred.device)
    pred_flat=torch.sigmoid(pred).reshape(-1)
    target_flat=target.reshape(-1).long()
    return metric(pred_flat, target_flat)


def iou_score(pred: torch.Tensor,
              target: torch.Tensor,
              threshold: float=0.5,)->torch.Tensor:
    """
    Batch-mean IoU (Jaccard) for binary segmentation.

    Args:
        pred: Raw logits, shape (B, 1, H, W).
        target: Binary mask, shape (B, 1, H, W).
        threshold: Binarization threshold applied after sigmoid.
    """
    metric=_get_iou_metric(pred.device)
    pred_flat=torch.sigmoid(pred).reshape(-1)
    target_flat=target.reshape(-1).long()
    return metric(pred_flat, target_flat)


def dice_coefficient_numpy(pred: np.ndarray, 
                           target: np.ndarray,
                             smooth: float=1e-6)->float:
    """
    NumPy Dice for evaluation scripts.
    """
    p,t=pred.flatten().astype(float),target.flatten().astype(float)
    intersection=(p*t).sum()
    return float((2.0*intersection+smooth)/(p.sum()+t.sum()+smooth))


def iou_score_numpy(pred: np.ndarray, 
                    target: np.ndarray, 
                    smooth: float=1e-6)->float:
    """
    NumPy IoU for evaluation scripts.
    """
    p,t=pred.flatten().astype(float),target.flatten().astype(float)
    intersection=(p*t).sum()
    union=p.sum()+t.sum()-intersection
    return float((intersection+smooth)/(union+smooth))
