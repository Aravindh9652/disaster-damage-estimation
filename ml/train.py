#!/usr/bin/env python3
"""Fine-tune ResNet50 on extracted xBD building crops.

Typical continuation after extraction::

    python extract_buildings.py --images ../data/raw/images --labels ../data/raw/labels
    python prepare_splits.py
    python train.py --epochs 8 --freeze-backbone
    python evaluate.py --checkpoint ../models/resnet50_best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BuildingCropDataset, eval_transforms, train_transforms
from labels import CLASSES
from metrics import evaluation_report
from model import OrdinalCrossEntropy, build_resnet50


def class_weights_from_loader(dataset: BuildingCropDataset, device: torch.device) -> torch.Tensor:
    counts = Counter(row["label"] for row in dataset.rows)
    freqs = torch.tensor([counts.get(c, 1) for c in CLASSES], dtype=torch.float32)
    weights = freqs.sum() / (len(CLASSES) * freqs)
    return weights.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []
    for images, labels, _uids in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        ys.extend(labels.cpu().tolist())
        ps.extend(preds.cpu().tolist())
    report = evaluation_report(ys, ps)
    report["loss"] = total_loss / max(len(loader.dataset), 1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--splits", type=Path, default=Path("../data/splits"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("../models"))
    args = parser.parse_args()

    freeze = not args.unfreeze_backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = BuildingCropDataset(args.splits / "train.csv", args.root, train_transforms())
    val_ds = BuildingCropDataset(args.splits / "val.csv", args.root, eval_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_resnet50(pretrained=True, freeze_backbone=freeze).to(device)
    weights = class_weights_from_loader(train_ds, device)
    criterion = OrdinalCrossEntropy(class_weights=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_rep = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_rep = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_rep["loss"],
                "val_loss": val_rep["loss"],
                "train_weighted_f1": train_rep["weighted_f1"],
                "val_weighted_f1": val_rep["weighted_f1"],
            }
        )
        print(
            f"epoch {epoch:02d}  train_f1={train_rep['weighted_f1']:.3f}  "
            f"val_f1={val_rep['weighted_f1']:.3f}  val_loss={val_rep['loss']:.3f}"
        )
        if val_rep["weighted_f1"] >= best_f1:
            best_f1 = val_rep["weighted_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": CLASSES,
                    "val": val_rep,
                    "epoch": epoch,
                },
                args.out / "resnet50_best.pt",
            )

    (args.out / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"best val weighted F1: {best_f1:.3f}")
    print(f"checkpoint: {args.out / 'resnet50_best.pt'}")


if __name__ == "__main__":
    main()
