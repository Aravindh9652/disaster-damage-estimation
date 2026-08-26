from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from labels import CLASSES, SEVERITY_WEIGHTS


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    return {name: float(scores[i]) for i, name in enumerate(CLASSES)}


def evaluation_report(y_true, y_pred) -> dict:
    return {
        "weighted_f1": weighted_f1(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": per_class_f1(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(CLASSES), zero_division=0, output_dict=True
        ),
    }


def tile_severity_index(labels: list[str]) -> float:
    """0–100 score: share of maximum possible destruction on a tile."""
    if not labels:
        return 0.0
    total = sum(SEVERITY_WEIGHTS[name] for name in labels)
    return float(100.0 * total / len(labels))


def class_histogram(labels: list[str]) -> dict[str, int]:
    return {name: sum(1 for x in labels if x == name) for name in CLASSES}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
