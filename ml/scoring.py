"""Transparent building and zone priority score.

Priority = 100 × (0.5 × severity + 0.3 × damage area + 0.2 × impact)

severity and damage area are fractions in [0, 1]. Impact is severity × confidence,
so an uncertain severe call ranks below a confident one. This is a project score
for ordering inspections, not an official emergency-management standard.
"""

from __future__ import annotations

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


def priority_level(score: float) -> str:
    if score >= 60:
        return "critical"
    if score >= 35:
        return "high"
    if score >= 15:
        return "moderate"
    return "low"


def priority_score(severity: float, damage_area: float, confidence: float) -> dict:
    """All inputs are fractions in [0, 1], except severity may also be passed as 0–100."""
    severity_frac = severity / 100.0 if severity > 1 else float(severity)
    area = min(1.0, max(0.0, float(damage_area)))
    conf = min(1.0, max(0.0, float(confidence)))
    severity_frac = min(1.0, max(0.0, severity_frac))
    impact = severity_frac * conf
    score = 100.0 * (0.5 * severity_frac + 0.3 * area + 0.2 * impact)
    rounded = round(score, 2)
    return {
        "priority_score": rounded,
        "priority_level": priority_level(rounded),
        "components": {
            "severity": round(severity_frac, 4),
            "damage_area": round(area, 4),
            "impact": round(impact, 4),
            "weights": {"severity": 0.5, "damage_area": 0.3, "impact": 0.2},
        },
    }


def zone_damage_area(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return float(sum(AREA_WEIGHTS[label] for label in labels) / len(labels))


def building_severity(label: str) -> float:
    return float(SEVERITY_WEIGHTS[label])
