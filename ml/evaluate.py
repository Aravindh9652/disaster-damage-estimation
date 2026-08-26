#!/usr/bin/env python3
"""Evaluate a trained ResNet50 checkpoint and write a severity-ready JSON report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BuildingCropDataset, eval_transforms
from labels import CLASSES, IDX_TO_CLASS
from metrics import class_histogram, evaluation_report, tile_severity_index
from model import build_resnet50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--split", type=Path, default=Path("../data/splits/test.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../models/resnet50_best.pt"))
    parser.add_argument("--out", type=Path, default=Path("../web/public/demo/metrics.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = BuildingCropDataset(args.split, args.root, eval_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ys, ps, records = [], [], []
    by_tile: dict[str, list[str]] = defaultdict(list)
    with torch.no_grad():
        idx = 0
        for images, labels, uids in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            pred = logits.argmax(dim=1).cpu()
            for i in range(len(uids)):
                row = ds.rows[idx + i]
                y, p = int(labels[i]), int(pred[i])
                ys.append(y)
                ps.append(p)
                pred_name = IDX_TO_CLASS[p]
                by_tile[row.get("tile") or row.get("disaster") or "tile"].append(pred_name)
                records.append(
                    {
                        "uid": uids[i],
                        "true": CLASSES[y],
                        "pred": pred_name,
                        "probs": {CLASSES[k]: float(probs[i, k]) for k in range(len(CLASSES))},
                        "disaster": row.get("disaster"),
                        "tile": row.get("tile"),
                    }
                )
            idx += len(uids)

    report = evaluation_report(ys, ps)
    tiles = []
    for tile, labels in sorted(by_tile.items()):
        tiles.append(
            {
                "tile": tile,
                "severity_index": round(tile_severity_index(labels), 2),
                "counts": class_histogram(labels),
                "buildings": len(labels),
            }
        )
    payload = {
        "split": str(args.split),
        "checkpoint": str(args.checkpoint),
        "metrics": report,
        "tiles": tiles,
        "predictions": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("weighted F1:", round(report["weighted_f1"], 4))
    print("per-class F1:", {k: round(v, 3) for k, v in report["per_class_f1"].items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
