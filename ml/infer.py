#!/usr/bin/env python3
"""Run the trained ResNet50 on a folder of building crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dataset import eval_transforms
from labels import CLASSES
from model import build_resnet50


def predict_image(model, path: Path, device, tfm) -> dict:
    image = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    pred = int(probs.argmax())
    return {
        "path": str(path),
        "pred": CLASSES[pred],
        "confidence": float(probs[pred]),
        "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("../models/resnet50_best.pt"))
    parser.add_argument("--out", type=Path, default=Path("../models/inference.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tfm = eval_transforms()

    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.png"))
    results = [predict_image(model, p, device, tfm) for p in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    for row in results:
        print(f"{row['pred']:14s}  {row['confidence']:.2f}  {row['path']}")


if __name__ == "__main__":
    main()
