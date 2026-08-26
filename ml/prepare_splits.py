#!/usr/bin/env python3
"""Stratified train/val/test split of building crops.

Prefers grouping by disaster so test tiles are not mixed into training.
Falls back to a plain stratified split when a disaster has too few samples.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from labels import CLASSES


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def split_rows(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    by_disaster: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_disaster[row.get("disaster") or "unknown"].append(row)

    disasters = sorted(by_disaster)
    if len(disasters) >= 3:
        train_d, hold = train_test_split(disasters, test_size=0.35, random_state=seed)
        val_d, test_d = train_test_split(hold, test_size=0.5, random_state=seed)
        mapping = {d: "train" for d in train_d}
        mapping.update({d: "val" for d in val_d})
        mapping.update({d: "test" for d in test_d})
        buckets = {"train": [], "val": [], "test": []}
        for row in rows:
            buckets[mapping[row.get("disaster") or "unknown"]].append(row)
        # Guard against a missing class in train.
        train_labels = {r["label"] for r in buckets["train"]}
        if train_labels != set(CLASSES):
            return _stratified(rows, seed)
        return buckets
    return _stratified(rows, seed)


def _stratified(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    labels = [r["label"] for r in rows]
    train, rest = train_test_split(rows, test_size=0.3, stratify=labels, random_state=seed)
    rest_labels = [r["label"] for r in rest]
    val, test = train_test_split(rest, test_size=0.5, stratify=rest_labels, random_state=seed)
    return {"train": train, "val": val, "test": test}


def write_split(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["uid"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/crops/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    splits = split_rows(rows, args.seed)
    for name, part in splits.items():
        dest = args.out / f"{name}.csv"
        write_split(dest, part)
        counts = {c: sum(1 for r in part if r["label"] == c) for c in CLASSES}
        print(f"{name:5s} {len(part):5d}  {counts}")


if __name__ == "__main__":
    main()
