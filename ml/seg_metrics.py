"""Pixel metrics for a binary damage mask."""

from __future__ import annotations

import numpy as np


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def pixel_iou(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_b = _binary(truth)
    pred_b = _binary(pred)
    intersection = np.logical_and(truth_b, pred_b).sum()
    union = np.logical_or(truth_b, pred_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def dice_score(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_b = _binary(truth)
    pred_b = _binary(pred)
    intersection = np.logical_and(truth_b, pred_b).sum()
    total = truth_b.sum() + pred_b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def pixel_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_b = _binary(truth)
    pred_b = _binary(pred)
    if truth_b.size == 0:
        return 0.0
    return float(np.equal(truth_b, pred_b).mean())
