#!/usr/bin/env python3
"""Build a self-contained demo dataset that looks like xBD continuation output.

The original xBD download is huge and gated. This script synthesizes four
disaster tiles, extracts building crops the same way extract_buildings.py
would, and writes dashboard JSON as if ResNet50 had been evaluated on them.
Confusion between minor and major damage is injected on purpose — that is
the hard pair on the Joint Damage Scale.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from labels import CLASS_COLORS, CLASS_TO_IDX, CLASSES
from metrics import class_histogram, evaluation_report, tile_severity_index

ROOT = Path(__file__).resolve().parents[1]
TILE_SIZE = 640
CROP_SIZE = 96
RNG = random.Random(7)
NP_RNG = np.random.default_rng(7)

DISASTERS = [
    {
        "id": "hurricane-harvey-houston",
        "name": "Hurricane Harvey",
        "place": "Houston, Texas",
        "type": "flood",
        "date": "2017-08-29",
        "mix": {"no-damage": 0.42, "minor-damage": 0.22, "major-damage": 0.20, "destroyed": 0.16},
        "n": 48,
    },
    {
        "id": "santa-rosa-wildfire",
        "name": "Santa Rosa Wildfire",
        "place": "Santa Rosa, California",
        "type": "fire",
        "date": "2017-10-09",
        "mix": {"no-damage": 0.28, "minor-damage": 0.14, "major-damage": 0.18, "destroyed": 0.40},
        "n": 40,
    },
    {
        "id": "palu-tsunami",
        "name": "Palu Tsunami",
        "place": "Palu, Indonesia",
        "type": "tsunami",
        "date": "2018-09-28",
        "mix": {"no-damage": 0.34, "minor-damage": 0.18, "major-damage": 0.24, "destroyed": 0.24},
        "n": 44,
    },
    {
        "id": "mexico-city-earthquake",
        "name": "Mexico City Earthquake",
        "place": "Mexico City",
        "type": "earthquake",
        "date": "2017-09-19",
        "mix": {"no-damage": 0.55, "minor-damage": 0.20, "major-damage": 0.17, "destroyed": 0.08},
        "n": 52,
    },
]


def terrain(kind: str) -> Image.Image:
    h = NP_RNG.normal(0, 1, (TILE_SIZE, TILE_SIZE))
    for _ in range(6):
        h = (
            h
            + np.roll(h, 1, 0)
            + np.roll(h, -1, 0)
            + np.roll(h, 1, 1)
            + np.roll(h, -1, 1)
        ) / 5
    h = (h - h.min()) / (h.max() - h.min() + 1e-6)
    yy, xx = np.mgrid[0:TILE_SIZE, 0:TILE_SIZE]
    if kind == "flood":
        base = np.dstack(
            (
                40 + 30 * h + 10 * np.sin(xx / 40),
                70 + 50 * h,
                55 + 35 * h,
            )
        )
        water = (h < 0.38) | ((xx + 0.4 * yy) % 180 < 28)
        base[water] = np.array([32, 72, 118]) + 18 * h[water][..., None]
    elif kind == "fire":
        base = np.dstack((70 + 50 * h, 62 + 40 * h, 38 + 20 * h))
        burn = h < 0.45
        base[burn] = np.array([48, 36, 28]) + 40 * h[burn][..., None]
        ash = h < 0.22
        base[ash] = np.array([28, 24, 22])
    elif kind == "tsunami":
        base = np.dstack((90 + 40 * h, 95 + 35 * h, 70 + 20 * h))
        shore = xx > TILE_SIZE * 0.62 + 18 * np.sin(yy / 30)
        base[shore] = np.array([28, 78, 128])
        mud = (~shore) & (h < 0.4)
        base[mud] = np.array([92, 78, 58])
    else:
        base = np.dstack((78 + 40 * h, 82 + 38 * h, 70 + 25 * h))
        grid = ((xx % 48) < 3) | ((yy % 48) < 3)
        base[grid] *= 0.78
    noise = NP_RNG.integers(-8, 9, base.shape)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(0.6))
    return img


def sample_label(mix: dict[str, float]) -> str:
    r = RNG.random()
    acc = 0.0
    for name, p in mix.items():
        acc += p
        if r <= acc:
            return name
    return "no-damage"


def building_spec(label: str, kind: str) -> dict:
    w = RNG.randint(18, 36)
    h = RNG.randint(16, 32)
    if label == "destroyed":
        roof = (96, 72, 58) if kind != "fire" else (32, 28, 26)
        debris = True
    elif label == "major-damage":
        roof = (150, 92, 70)
        debris = RNG.random() < 0.5
    elif label == "minor-damage":
        roof = (168, 140, 112)
        debris = False
    else:
        roof = (186, 168, 148) if kind != "fire" else (170, 150, 130)
        debris = False
    return {"w": w, "h": h, "roof": roof, "debris": debris}


def place_buildings(n: int, mix: dict[str, float], kind: str) -> list[dict]:
    buildings = []
    tries = 0
    while len(buildings) < n and tries < 4000:
        tries += 1
        spec = building_spec(sample_label(mix), kind)
        x = RNG.randint(24, TILE_SIZE - spec["w"] - 24)
        y = RNG.randint(24, TILE_SIZE - spec["h"] - 24)
        poly = [
            (x, y),
            (x + spec["w"], y + RNG.randint(-2, 2)),
            (x + spec["w"] + RNG.randint(-2, 3), y + spec["h"]),
            (x + RNG.randint(-2, 2), y + spec["h"]),
        ]
        cx = sum(p[0] for p in poly) / 4
        cy = sum(p[1] for p in poly) / 4
        overlap = False
        for other in buildings:
            if abs(cx - other["cx"]) < 28 and abs(cy - other["cy"]) < 28:
                overlap = True
                break
        if overlap:
            continue
        uid = f"{kind[:3]}-{len(buildings):03d}"
        buildings.append({**spec, "poly": poly, "cx": cx, "cy": cy, "uid": uid})
    # Re-assign labels with exact mix after placement
    labels = []
    for name, p in mix.items():
        labels.extend([name] * int(round(p * len(buildings))))
    while len(labels) < len(buildings):
        labels.append("no-damage")
    labels = labels[: len(buildings)]
    RNG.shuffle(labels)
    for b, lab in zip(buildings, labels):
        spec = building_spec(lab, kind)
        b["label"] = lab
        b["roof"] = spec["roof"]
        b["debris"] = spec["debris"]
        b["w"] = spec["w"]
        b["h"] = spec["h"]
    return buildings


def draw_building(draw: ImageDraw.ImageDraw, b: dict) -> None:
    draw.polygon(b["poly"], fill=b["roof"], outline=(40, 36, 32))
    if b["label"] == "destroyed":
        for _ in range(8):
            px = int(b["cx"] + RNG.randint(-18, 18))
            py = int(b["cy"] + RNG.randint(-14, 14))
            draw.rectangle((px, py, px + 4, py + 3), fill=(70, 58, 48))
    elif b["label"] == "major-damage":
        x0, y0 = b["poly"][0]
        x1, y1 = b["poly"][2]
        draw.line((x0 + 4, y0 + 4, x1 - 4, y1 - 4), fill=(90, 40, 30), width=2)
    elif b["label"] == "minor-damage":
        x0, y0 = b["poly"][0]
        draw.line((x0 + 3, y0 + 6, x0 + 14, y0 + 6), fill=(110, 80, 40), width=2)


def crop_building(tile: Image.Image, b: dict) -> Image.Image:
    xs = [p[0] for p in b["poly"]]
    ys = [p[1] for p in b["poly"]]
    pad = 8
    box = (
        max(min(xs) - pad, 0),
        max(min(ys) - pad, 0),
        min(max(xs) + pad, TILE_SIZE),
        min(max(ys) + pad, TILE_SIZE),
    )
    crop = tile.crop(box).convert("RGB")
    mask = Image.new("L", crop.size, 0)
    local = [(x - box[0], y - box[1]) for x, y in b["poly"]]
    ImageDraw.Draw(mask).polygon(local, fill=255)
    bg = Image.new("RGB", crop.size, (24, 28, 32))
    return Image.composite(crop, bg, mask).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)


CONFUSION = {
    "no-damage": [0.78, 0.16, 0.05, 0.01],
    "minor-damage": [0.22, 0.46, 0.26, 0.06],
    "major-damage": [0.08, 0.28, 0.44, 0.20],
    "destroyed": [0.04, 0.10, 0.22, 0.64],
}


def predict(true_label: str) -> tuple[str, list[float]]:
    weights = CONFUSION[true_label]
    pred = RNG.choices(CLASSES, weights=weights, k=1)[0]
    probs = [p * 0.55 for p in weights]
    probs[CLASS_TO_IDX[pred]] += 0.45
    jitter = [RNG.uniform(-0.03, 0.03) for _ in CLASSES]
    probs = [max(0.01, p + j) for p, j in zip(probs, jitter)]
    s = sum(probs)
    probs = [p / s for p in probs]
    return pred, probs


def legend(img: Image.Image, title: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle((12, 12, 330, 52), fill=(12, 16, 22))
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((22, 24), title, fill=(235, 240, 245), font=font)
    return img


def main() -> None:
    public = ROOT / "web" / "public" / "demo"
    crops_dir = ROOT / "data" / "crops"
    splits_dir = ROOT / "data" / "splits"
    tiles_dir = public / "tiles"
    crop_pub = public / "crops"
    for d in (public, crops_dir, splits_dir, tiles_dir, crop_pub):
        d.mkdir(parents=True, exist_ok=True)

    events = []
    all_buildings = []
    y_true, y_pred = [], []
    history = []
    f1_curve = [0.28, 0.37, 0.45, 0.51, 0.55, 0.57, 0.59, 0.60]
    loss_curve = [1.48, 1.22, 1.05, 0.94, 0.86, 0.81, 0.77, 0.74]
    for epoch, (f1, loss) in enumerate(zip(f1_curve, loss_curve), start=1):
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(loss - 0.08, 3),
                "val_loss": round(loss, 3),
                "train_weighted_f1": round(min(f1 + 0.05, 0.82), 3),
                "val_weighted_f1": round(f1, 3),
            }
        )

    for disaster in DISASTERS:
        tile = terrain(disaster["type"])
        draw = ImageDraw.Draw(tile)
        buildings = place_buildings(disaster["n"], disaster["mix"], disaster["type"])
        for b in buildings:
            draw_building(draw, b)
        tile = legend(tile, f"{disaster['name']}  ·  post-disaster")
        tile_name = f"{disaster['id']}_post.png"
        tile.save(tiles_dir / tile_name)

        event_buildings = []
        for b in buildings:
            crop = crop_building(tile, b)
            rel = f"crops/{b['label']}/{disaster['id']}_{b['uid']}.png"
            dest = crops_dir / b["label"] / f"{disaster['id']}_{b['uid']}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dest)
            pub = crop_pub / b["label"]
            pub.mkdir(parents=True, exist_ok=True)
            crop.save(pub / f"{disaster['id']}_{b['uid']}.png")

            pred, probs = predict(b["label"])
            y_true.append(CLASS_TO_IDX[b["label"]])
            y_pred.append(CLASS_TO_IDX[pred])
            record = {
                "uid": f"{disaster['id']}_{b['uid']}",
                "disaster_id": disaster["id"],
                "disaster": disaster["name"],
                "tile": tile_name,
                "path": rel,
                "crop": f"/demo/crops/{b['label']}/{disaster['id']}_{b['uid']}.png",
                "label": b["label"],
                "pred": pred,
                "label_idx": CLASS_TO_IDX[b["label"]],
                "probs": {name: round(p, 4) for name, p in zip(CLASSES, probs)},
                "polygon": [{"x": p[0], "y": p[1]} for p in b["poly"]],
                "centroid": {"x": round(b["cx"], 1), "y": round(b["cy"], 1)},
            }
            event_buildings.append(record)
            all_buildings.append(record)

        pred_labels = [b["pred"] for b in event_buildings]
        true_labels = [b["label"] for b in event_buildings]
        events.append(
            {
                **{k: disaster[k] for k in ("id", "name", "place", "type", "date")},
                "tile": f"/demo/tiles/{tile_name}",
                "width": TILE_SIZE,
                "height": TILE_SIZE,
                "buildings": event_buildings,
                "counts_true": class_histogram(true_labels),
                "counts_pred": class_histogram(pred_labels),
                "severity_true": round(tile_severity_index(true_labels), 2),
                "severity_pred": round(tile_severity_index(pred_labels), 2),
            }
        )

    report = evaluation_report(y_true, y_pred)
    # Write splits (by disaster)
    by_split = {"train": [], "val": [], "test": []}
    assignment = {
        "hurricane-harvey-houston": "train",
        "mexico-city-earthquake": "train",
        "palu-tsunami": "val",
        "santa-rosa-wildfire": "test",
    }
    for row in all_buildings:
        split = assignment[row["disaster_id"]]
        by_split[split].append(
            {
                "uid": row["uid"],
                "path": f"data/{row['path']}",
                "label": row["label"],
                "label_idx": row["label_idx"],
                "disaster": row["disaster_id"],
                "tile": row["tile"],
            }
        )
    for name, rows in by_split.items():
        dest = splits_dir / f"{name}.csv"
        with dest.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = crops_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["uid", "path", "label", "label_idx", "disaster", "tile"]
        )
        w.writeheader()
        for row in all_buildings:
            w.writerow(
                {
                    "uid": row["uid"],
                    "path": f"data/{row['path']}",
                    "label": row["label"],
                    "label_idx": row["label_idx"],
                    "disaster": row["disaster_id"],
                    "tile": row["tile"],
                }
            )

    catalog = {
        "project": "AI Disaster Damage Mapping and Severity Estimation",
        "continuation": "Building extraction → ResNet50 classification → tile severity maps",
        "classes": [
            {
                "id": name,
                "ordinal": i,
                "color": CLASS_COLORS[name],
                "meaning": {
                    "no-damage": "Roof intact, no debris, structure appears habitable.",
                    "minor-damage": "Partial roof damage, water staining, or limited debris.",
                    "major-damage": "Partial collapse, large roof loss, or heavy flooding.",
                    "destroyed": "Washed away, burned to foundation, or fully collapsed.",
                }[name],
            }
            for i, name in enumerate(CLASSES)
        ],
        "pipeline": [
            {"step": 1, "name": "Download xBD subset", "status": "done"},
            {"step": 2, "name": "Extract building polygons", "status": "done"},
            {"step": 3, "name": "Balance & split crops", "status": "done"},
            {"step": 4, "name": "Fine-tune ResNet50", "status": "done"},
            {"step": 5, "name": "Evaluate Joint Damage Scale", "status": "done"},
            {"step": 6, "name": "Map tile severity", "status": "done"},
        ],
        "metrics": {
            "weighted_f1": round(report["weighted_f1"], 4),
            "macro_f1": round(report["macro_f1"], 4),
            "per_class_f1": {k: round(v, 4) for k, v in report["per_class_f1"].items()},
            "confusion_matrix": report["confusion_matrix"],
            "n_buildings": len(all_buildings),
            "n_events": len(events),
        },
        "history": history,
        "events": events,
        "note": (
            "Demo tiles are synthesized so the dashboard runs without the multi-GB xBD download. "
            "Point extract_buildings.py at your saved xBD subset and run train.py to replace these numbers."
        ),
    }
    (public / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print("events", len(events), "buildings", len(all_buildings))
    print("weighted F1", round(report["weighted_f1"], 3))
    print("wrote", public / "catalog.json")


if __name__ == "__main__":
    main()
