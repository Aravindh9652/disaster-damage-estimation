"""Joint Damage Scale used by xBD / xView2."""

CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# Ordinal severity weights (0..1) used to roll building counts into a tile score.
SEVERITY_WEIGHTS = {
    "no-damage": 0.0,
    "minor-damage": 1.0 / 3.0,
    "major-damage": 2.0 / 3.0,
    "destroyed": 1.0,
}

CLASS_COLORS = {
    "no-damage": "#22c55e",
    "minor-damage": "#eab308",
    "major-damage": "#f97316",
    "destroyed": "#ef4444",
}
