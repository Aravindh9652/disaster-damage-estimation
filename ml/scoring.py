"""Configurable, project-defined decision-support score.

Priority = 100 × (w_s × severity + w_a × damage area + w_i × severity × impact factor).
All inputs are normalized to [0, 1]. Prediction confidence is deliberately kept
separate so uncertainty does not automatically lower a building's priority.
"""

from __future__ import annotations

import os

try:
    from .labels import SEVERITY_WEIGHTS
except ImportError:
    from labels import SEVERITY_WEIGHTS

# Share of a building footprint treated as damaged for each Joint Damage Scale class.
# Distinct from the ordinal severity weights so area and severity are not the same number.
AREA_WEIGHTS = {
    "no-damage": 0.02,
    "minor-damage": 0.25,
    "major-damage": 0.60,
    "destroyed": 0.95,
}

PRIORITY_WEIGHTS = {
    "severity": float(os.getenv("PRIORITY_WEIGHT_SEVERITY", "0.5")),
    "damage_area": float(os.getenv("PRIORITY_WEIGHT_DAMAGE_AREA", "0.3")),
    "impact": float(os.getenv("PRIORITY_WEIGHT_IMPACT", "0.2")),
}
if any(value < 0 for value in PRIORITY_WEIGHTS.values()) or not abs(sum(PRIORITY_WEIGHTS.values()) - 1) < 1e-9:
    raise ValueError("Priority weights must be non-negative and sum to 1.")

PRIORITY_CUTOFFS = {
    "critical": float(os.getenv("PRIORITY_CUTOFF_CRITICAL", "60")),
    "high": float(os.getenv("PRIORITY_CUTOFF_HIGH", "35")),
    "moderate": float(os.getenv("PRIORITY_CUTOFF_MODERATE", "15")),
}
if not 0 <= PRIORITY_CUTOFFS["moderate"] <= PRIORITY_CUTOFFS["high"] <= PRIORITY_CUTOFFS["critical"] <= 100:
    raise ValueError("Priority cutoffs must satisfy 0 <= moderate <= high <= critical <= 100.")


def priority_configuration() -> dict:
    return {
        "formula": (
            "100 * (w_severity * normalized_severity + "
            "w_damage_area * normalized_damage_area + "
            "w_impact * normalized_severity * normalized_impact_factor)"
        ),
        "weights": PRIORITY_WEIGHTS.copy(),
        "impact_factor": {
            "minimum": 0.0,
            "maximum": 1.0,
            "neutral_default": 1.0,
            "meaning": "Contextual impact multiplier; 1.0 is neutral, and values below 1 reduce the impact contribution.",
        },
        "cutoffs": PRIORITY_CUTOFFS.copy(),
        "confidence_affects_score": False,
        "label": "Project-defined decision-support score, not an official emergency standard.",
    }


def priority_level(score: float) -> str:
    if score >= PRIORITY_CUTOFFS["critical"]:
        return "critical"
    if score >= PRIORITY_CUTOFFS["high"]:
        return "high"
    if score >= PRIORITY_CUTOFFS["moderate"]:
        return "moderate"
    return "low"


def priority_score(severity: float, damage_area: float, impact_factor: float = 1.0) -> dict:
    """Score normalized severity/area with a neutral [0, 1] impact multiplier."""
    severity_frac = severity / 100.0 if severity > 1 else float(severity)
    area = min(1.0, max(0.0, float(damage_area)))
    impact_multiplier = min(1.0, max(0.0, float(impact_factor)))
    severity_frac = min(1.0, max(0.0, severity_frac))
    impact = severity_frac * impact_multiplier
    score = 100.0 * (
        PRIORITY_WEIGHTS["severity"] * severity_frac
        + PRIORITY_WEIGHTS["damage_area"] * area
        + PRIORITY_WEIGHTS["impact"] * impact
    )
    rounded = round(score, 2)
    level = priority_level(rounded)
    return {
        "priority_score": rounded,
        "priority_level": level,
        "components": {
            "severity": round(severity_frac, 4),
            "damage_area": round(area, 4),
            "impact": round(impact, 4),
            "impact_factor": round(impact_multiplier, 4),
            "weights": PRIORITY_WEIGHTS.copy(),
        },
        "formula": (
            "100 * (w_severity * normalized_severity + "
            "w_damage_area * normalized_damage_area + "
            "w_impact * normalized_severity * normalized_impact_factor)"
        ),
        "priority_rationale": (
            f"{level.title()} project-defined decision-support score "
            f"({rounded:.2f}/100; cutoffs: {PRIORITY_CUTOFFS})."
        ),
    }


def zone_damage_area(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return float(sum(AREA_WEIGHTS[label] for label in labels) / len(labels))


def building_severity(label: str) -> float:
    return float(SEVERITY_WEIGHTS[label])


def human_class_for_review(
    status: str, prediction: str, corrected_class: str | None
) -> str | None:
    if status == "confirmed":
        return prediction
    if status == "corrected":
        if corrected_class is None:
            raise ValueError("A corrected review requires a human damage class.")
        return corrected_class
    if status == "unresolved":
        return None
    raise ValueError(f"Unknown review status: {status}")
