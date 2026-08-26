#!/usr/bin/env python3
"""Roll building-level Joint Damage Scale labels into tile severity maps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from labels import CLASSES
from metrics import class_histogram, tile_severity_index


def load_predictions(path: Path) -> list[dict]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "predictions" in payload:
            return payload["predictions"]
        if isinstance(payload, list):
            return payload
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict], group_key: str = "tile") -> list[dict]:
    groups: dict[str, list[str]] = defaultdict(list)
    disasters: dict[str, str] = {}
    for row in rows:
        label = row.get("pred") or row.get("label")
        key = row.get(group_key) or row.get("disaster") or "unknown"
        groups[key].append(label)
        disasters[key] = row.get("disaster") or key
    out = []
    for key, labels in sorted(groups.items()):
        idx = tile_severity_index(labels)
        if idx >= 60:
            band = "critical"
        elif idx >= 35:
            band = "severe"
        elif idx >= 15:
            band = "moderate"
        else:
            band = "low"
        out.append(
            {
                "id": key,
                "disaster": disasters[key],
                "buildings": len(labels),
                "severity_index": round(idx, 2),
                "band": band,
                "counts": class_histogram(labels),
            }
        )
    return sorted(out, key=lambda x: -x["severity_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("../web/public/demo/metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("../web/public/demo/severity.json"))
    args = parser.parse_args()
    rows = load_predictions(args.predictions)
    summary = summarize(rows)
    payload = {
        "scale": list(CLASSES),
        "formula": "100 * mean(ordinal_weight) where no-damage=0, minor=1/3, major=2/3, destroyed=1",
        "tiles": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
