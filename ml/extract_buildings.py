#!/usr/bin/env python3
"""Crop individual buildings from xBD tiles using label polygons.

This is the step you already completed. Keep it here so the rest of the
pipeline can be run from a single folder.

Expected xBD layout (subset is fine)::

    data/raw/
      images/   *_pre_disaster.png  *_post_disaster.png
      labels/   *_pre_disaster.json *_post_disaster.json

Output::

    data/crops/{no-damage,minor-damage,major-damage,destroyed}/{uid}.png
    data/crops/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

from labels import CLASS_TO_IDX, CLASSES


WKT_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    nums = [float(x) for x in WKT_NUM.findall(wkt)]
    return list(zip(nums[0::2], nums[1::2]))


def polygon_from_feature(feature: dict) -> list[tuple[float, float]]:
    if "wkt" in feature:
        return parse_wkt_polygon(feature["wkt"])
    geom = feature.get("geometry") or {}
    if geom.get("type") == "Polygon":
        ring = geom["coordinates"][0]
        return [(float(x), float(y)) for x, y in ring]
    raise ValueError("Unsupported feature geometry")


def crop_with_padding(
    image: Image.Image,
    pts: list[tuple[float, float]],
    pad: int = 8,
    size: int = 224,
    mask_background: bool = True,
) -> Image.Image:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    left = max(int(min(xs)) - pad, 0)
    top = max(int(min(ys)) - pad, 0)
    right = min(int(max(xs)) + pad, image.width)
    bottom = min(int(max(ys)) + pad, image.height)
    if right <= left or bottom <= top:
        raise ValueError("empty crop")

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    if mask_background:
        mask = Image.new("L", crop.size, 0)
        local = [(x - left, y - top) for x, y in pts]
        ImageDraw.Draw(mask).polygon(local, fill=255)
        bg = Image.new("RGB", crop.size, (20, 24, 28))
        crop = Image.composite(crop, bg, mask)
    return crop.resize((size, size), Image.Resampling.BILINEAR)


def iter_xy_features(label: dict) -> list[dict]:
    features = label.get("features", {})
    if isinstance(features, dict):
        return list(features.get("xy") or [])
    if isinstance(features, list):
        return features
    return []


def extract_from_pair(post_image: Path, post_label: Path, out_dir: Path, size: int) -> list[dict]:
    image = Image.open(post_image).convert("RGB")
    label = json.loads(post_label.read_text())
    meta = label.get("metadata") or {}
    disaster = meta.get("disaster") or post_image.stem.split("_")[0]
    rows: list[dict] = []

    for feat in iter_xy_features(label):
        props = feat.get("properties") or feat
        if props.get("feature_type") not in (None, "building"):
            continue
        subtype = props.get("subtype") or props.get("damage")
        if subtype not in CLASS_TO_IDX:
            continue
        uid = str(props.get("uid") or f"{post_image.stem}_{len(rows)}")
        try:
            pts = polygon_from_feature(feat)
            crop = crop_with_padding(image, pts, size=size)
        except (ValueError, KeyError):
            continue
        class_dir = out_dir / subtype
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / f"{uid}.png"
        crop.save(dest)
        rows.append(
            {
                "uid": uid,
                "path": str(dest.relative_to(out_dir.parent.parent)),
                "label": subtype,
                "label_idx": CLASS_TO_IDX[subtype],
                "disaster": disaster,
                "tile": post_image.name,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/labels"))
    parser.add_argument("--out", type=Path, default=Path("data/crops"))
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    posts = sorted(args.images.glob("*_post_disaster.png"))
    if not posts:
        posts = sorted(args.images.glob("*post*.png"))

    all_rows: list[dict] = []
    for img in posts:
        label = args.labels / img.with_suffix(".json").name
        if not label.exists():
            alt = args.labels / img.name.replace(".png", ".json")
            label = alt if alt.exists() else label
        if not label.exists():
            print(f"skip (no label): {img.name}")
            continue
        rows = extract_from_pair(img, label, args.out, args.size)
        print(f"{img.name}: {len(rows)} buildings")
        all_rows.extend(rows)

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["uid", "path", "label", "label_idx", "disaster", "tile"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    counts = {c: sum(1 for r in all_rows if r["label"] == c) for c in CLASSES}
    print(f"wrote {len(all_rows)} crops -> {args.out}")
    print("class counts:", counts)


if __name__ == "__main__":
    main()
