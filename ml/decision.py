"""Configurable model-comparison and image-quality decision rules."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from .labels import CLASSES
except ImportError:
    from labels import CLASSES


def _float_setting(name: str, default: float) -> float:
    value = float(os.getenv(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


@dataclass(frozen=True)
class DecisionRules:
    min_confidence: float = 0.60
    no_damage_max_area: float = 0.02
    minor_max_area: float = 0.25
    major_max_area: float = 0.60
    minimum_class_gap: int = 1
    minimum_image_dimension: int = 128
    minimum_gray_stddev: float = 0.025

    @classmethod
    def from_environment(cls) -> "DecisionRules":
        rules = cls(
            min_confidence=_float_setting("MODEL_DISAGREEMENT_MIN_CONFIDENCE", 0.60),
            no_damage_max_area=_float_setting("UNET_NO_DAMAGE_MAX_AREA", 0.02),
            minor_max_area=_float_setting("UNET_MINOR_MAX_AREA", 0.25),
            major_max_area=_float_setting("UNET_MAJOR_MAX_AREA", 0.60),
            minimum_class_gap=int(os.getenv("MODEL_DISAGREEMENT_MIN_CLASS_GAP", "1")),
            minimum_image_dimension=int(os.getenv("IMAGE_MIN_DIMENSION", "128")),
            minimum_gray_stddev=_float_setting("IMAGE_MIN_GRAY_STDDEV", 0.025),
        )
        if not (
            rules.no_damage_max_area
            < rules.minor_max_area
            < rules.major_max_area
        ):
            raise ValueError("U-Net area thresholds must be strictly increasing.")
        if rules.minimum_image_dimension < 1:
            raise ValueError("IMAGE_MIN_DIMENSION must be positive.")
        if rules.minimum_class_gap < 1 or rules.minimum_class_gap > len(CLASSES) - 1:
            raise ValueError("MODEL_DISAGREEMENT_MIN_CLASS_GAP must be between 1 and 3.")
        return rules

    def public(self) -> dict:
        return {
            "minimum_confidence_for_confirmation": self.min_confidence,
            "unet_area_class_thresholds": {
                "no_damage_max": self.no_damage_max_area,
                "minor_damage_max": self.minor_max_area,
                "major_damage_max": self.major_max_area,
                "destroyed": f">{self.major_max_area}",
            },
            "minimum_class_gap": self.minimum_class_gap,
            "minimum_image_dimension": self.minimum_image_dimension,
            "minimum_gray_stddev": self.minimum_gray_stddev,
        }


def _class_from_area(area: float, rules: DecisionRules) -> str:
    if area <= rules.no_damage_max_area:
        return "no-damage"
    if area <= rules.minor_max_area:
        return "minor-damage"
    if area <= rules.major_max_area:
        return "major-damage"
    return "destroyed"


def compare_models(
    prediction: str | None,
    confidence: float | None,
    damage_area_ratio: float | None,
    segmentation_method: str | None,
    rules: DecisionRules,
) -> dict:
    """Compare ResNet class with U-Net area bins only when both trained models ran."""
    if prediction is None or confidence is None:
        return {"available": False, "reason": "Classifier prediction is unavailable."}
    if segmentation_method != "unet" or damage_area_ratio is None:
        return {
            "available": False,
            "reason": "A trained U-Net result is unavailable; heuristic masks are not compared as U-Net output.",
        }
    if prediction not in CLASSES:
        raise ValueError(f"Unknown classifier prediction: {prediction}")
    if not 0 <= confidence <= 1 or not 0 <= damage_area_ratio <= 1:
        raise ValueError("Confidence and damage area must be between 0 and 1.")

    segmentation_class = _class_from_area(damage_area_ratio, rules)
    class_gap = abs(CLASSES.index(prediction) - CLASSES.index(segmentation_class))
    return {
        "available": True,
        "classifier_class": prediction,
        "unet_area_class": segmentation_class,
        "class_gap": class_gap,
        "disagreement": class_gap >= rules.minimum_class_gap,
        "comparison_note": (
            "Agreement is not evidence that either model is correct."
            if class_gap == 0
            else "Prediction and U-Net-derived area class differ; human review is recommended."
        ),
    }


def image_quality_warnings(image: Image.Image, rules: DecisionRules, source: str) -> list[str]:
    warnings: list[str] = []
    if min(image.size) < rules.minimum_image_dimension:
        warnings.append(
            f"{source} image is smaller than {rules.minimum_image_dimension} px on at least one side."
        )
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if float(gray.std()) < rules.minimum_gray_stddev:
        warnings.append(f"{source} image has very low contrast; inspect image quality.")
    return warnings


def review_reasons(
    confidence: float,
    disagreement: dict,
    quality_warnings: list[str],
    rules: DecisionRules,
) -> list[str]:
    reasons = []
    if confidence < rules.min_confidence:
        reasons.append("low-confidence")
    if disagreement.get("available") and disagreement.get("disagreement"):
        reasons.append("model-disagreement")
    if quality_warnings:
        reasons.append("image-quality")
    return reasons
