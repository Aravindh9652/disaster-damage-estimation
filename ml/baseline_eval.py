#!/usr/bin/env python3
"""Fast CPU baseline on downloaded xBD crops.

This is not a replacement for ResNet50. It gives honest numbers on YOUR
extracted chips while `train.py` (ResNet50) runs on GPU/Colab.

Features: 32-bin RGB histograms + mean/std per channel.
Model: class-weighted RandomForest.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from labels import CLASS_COLORS, CLASS_TO_IDX, CLASSES
from metrics import class_histogram, evaluation_report, tile_severity_index


def features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 256), density=True)
        hist.append(h)
        hist.append(np.array([arr[:, :, c].mean() / 255.0, arr[:, :, c].std() / 255.0]))
    return np.concatenate(hist)


def load_rows(manifest: Path, limit_per_class: int, seed: int) -> list[dict]:
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)
    by = defaultdict(list)
    for row in rows:
        by[row["label"]].append(row)
    out = []
    for label in CLASSES:
        part = by[label]
        rng.shuffle(part)
        out.extend(part[:limit_per_class] if limit_per_class else part)
    rng.shuffle(out)
    return out


def resolve(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    p2 = Path(rel)
    if p2.exists():
        return p2
    # manifest paths are relative to repo root
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("../data/xbd/manifest.csv"))
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--limit-per-class", type=int, default=800)
    parser.add_argument("--out", type=Path, default=Path("../web/public/demo/xbd_metrics.json"))
    parser.add_argument("--gallery", type=Path, default=Path("../web/public/demo/xbd-crops"))
    parser.add_argument("--gallery-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.manifest, args.limit_per_class, args.seed)
    print(f"loading {len(rows)} images")
    X, y, keep = [], [], []
    for row in rows:
        path = resolve(args.root, row["path"])
        if not path.exists():
            continue
        try:
            X.append(features(path))
            y.append(CLASS_TO_IDX[row["label"]])
            keep.append((row, path))
        except Exception:
            continue
    X = np.stack(X)
    y = np.array(y)
    idx = np.arange(len(y))
    train_i, test_i = train_test_split(idx, test_size=0.25, stratify=y, random_state=args.seed)
    clf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    clf.fit(X[train_i], y[train_i])
    pred = clf.predict(X[test_i])
    proba = clf.predict_proba(X[test_i])
    report = evaluation_report(y[test_i], pred)

    # gallery of held-out crops
    args.gallery.mkdir(parents=True, exist_ok=True)
    gallery = []
    by_label = defaultdict(list)
    for i, pi in enumerate(test_i):
        row, path = keep[pi]
        by_label[row["label"]].append((i, row, path, int(pred[i]), proba[i]))
    rng = random.Random(args.seed)
    for label in CLASSES:
        part = by_label[label]
        rng.shuffle(part)
        dest_dir = args.gallery / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, row, path, pidx, pr in part[: args.gallery_n]:
            dest = dest_dir / path.name
            Image.open(path).convert("RGB").resize((128, 128)).save(dest, quality=85)
            gallery.append(
                {
                    "uid": row["uid"],
                    "label": row["label"],
                    "pred": CLASSES[pidx],
                    "crop": f"/demo/xbd-crops/{label}/{path.name}",
                    "probs": {CLASSES[k]: round(float(pr[k]), 4) for k in range(len(CLASSES))},
                    "correct": CLASSES[pidx] == row["label"],
                }
            )

    full_counts = {c: 0 for c in CLASSES}
    with args.manifest.open() as f:
        for row in csv.DictReader(f):
            full_counts[row["label"]] += 1

    payload = {
        "source": "https://drive.google.com/drive/folders/1Yeb7vfp4bOTb2i1gcT0HBOBPpEE0uMHH",
        "dataset": "xBD building crops (resnet_dataset)",
        "n_total": sum(full_counts.values()),
        "counts": full_counts,
        "baseline": "RandomForest on RGB histograms (CPU stand-in; run train.py for ResNet50)",
        "n_eval": int(len(test_i)),
        "metrics": {
            "weighted_f1": round(report["weighted_f1"], 4),
            "macro_f1": round(report["macro_f1"], 4),
            "per_class_f1": {k: round(v, 4) for k, v in report["per_class_f1"].items()},
            "confusion_matrix": report["confusion_matrix"],
        },
        "classes": [
            {"id": name, "ordinal": i, "color": CLASS_COLORS[name]}
            for i, name in enumerate(CLASSES)
        ],
        "gallery": gallery,
        "severity_if_all_predicted": round(
            tile_severity_index([CLASSES[int(p)] for p in pred]), 2
        ),
        "severity_if_all_labeled": round(
            tile_severity_index([CLASSES[int(t)] for t in y[test_i]]), 2
        ),
        "class_histogram_eval_true": class_histogram([CLASSES[int(t)] for t in y[test_i]]),
        "class_histogram_eval_pred": class_histogram([CLASSES[int(p)] for p in pred]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("weighted F1", payload["metrics"]["weighted_f1"])
    print("per-class", payload["metrics"]["per_class_f1"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
