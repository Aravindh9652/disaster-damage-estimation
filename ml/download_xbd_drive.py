#!/usr/bin/env python3
"""Download the shared xBD building-crop folder from Google Drive.

Source: https://drive.google.com/drive/folders/1Yeb7vfp4bOTb2i1gcT0HBOBPpEE0uMHH
Layout on Drive: resnet_dataset/{destroyed,major_damage,minor_damage,no_damage}/*.jpg

Writes ImageFolder-style crops using xBD class names (hyphenated):

    data/xbd/crops/{no-damage,minor-damage,major-damage,destroyed}/
    data/xbd/filelist.json
    data/xbd/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from labels import CLASS_TO_IDX, CLASSES

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1Yeb7vfp4bOTb2i1gcT0HBOBPpEE0uMHH"
API_KEY = "AIzaSyC1qbk75NzWBvSaDh6KnsjjA9pIrP4lYIE"
FOLDER_IDS = {
    "destroyed": "1c22fPsUwcA71RwFY310ewHVhkES-VSrQ",
    "major_damage": "1BfF-JVYLdIVa5WTomHL1am0MTh2l-Kv7",
    "minor_damage": "16HnpJpGpzc_S_IRDpbT03ETs7l28QpQ5",
    "no_damage": "1536K4U5Ac-Oo388GV4vJWwSgln05GKEE",
}
DRIVE_TO_XBD = {
    "no_damage": "no-damage",
    "minor_damage": "minor-damage",
    "major_damage": "major-damage",
    "destroyed": "destroyed",
}


def list_folder(fid: str) -> list[dict]:
    files: list[dict] = []
    token = None
    while True:
        params = {
            "q": f"'{fid}' in parents and trashed = false",
            "key": API_KEY,
            "pageSize": 1000,
            "fields": "files(id,name,size,mimeType),nextPageToken",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if token:
            params["pageToken"] = token
        url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Referer": "https://drive.google.com/"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        files.extend(data.get("files") or [])
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.1)
    return [
        f
        for f in files
        if f.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def download_one(file_id: str, dest: Path, retries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 500:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                blob = r.read()
            if blob[:2] != b"\xff\xd8" and blob[:8] != b"\x89PNG\r\n\x1a\n":
                # HTML interstitials happen under rate limits.
                time.sleep(1.5 * (attempt + 1))
                continue
            dest.write_bytes(blob)
            return True
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("../data/xbd"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--per-class", type=int, default=0, help="0 = all files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    listing: dict[str, list[dict]] = {}
    cached = args.out / "filelist.json"
    if cached.exists():
        listing = json.loads(cached.read_text())
        print("loaded cached file list")
    else:
        for drive_name, fid in FOLDER_IDS.items():
            print(f"listing {drive_name}...")
            listing[drive_name] = list_folder(fid)
            print(f"  {len(listing[drive_name])} images")
        args.out.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(listing))

    jobs = []
    for drive_name, files in listing.items():
        xbd = DRIVE_TO_XBD[drive_name]
        chosen = list(files)
        rng.shuffle(chosen)
        if args.per_class > 0:
            chosen = chosen[: args.per_class]
        for meta in chosen:
            dest = args.out / "crops" / xbd / meta["name"]
            jobs.append((meta["id"], dest, xbd, meta["name"]))

    print(f"downloading {len(jobs)} crops with {args.workers} workers")
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, fid, dest): (dest, xbd, name) for fid, dest, xbd, name in jobs}
        done = 0
        for fut in as_completed(futs):
            dest, xbd, name = futs[fut]
            done += 1
            if fut.result():
                ok += 1
            else:
                fail += 1
            if done % 250 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  ok={ok} fail={fail}", flush=True)

    rows = []
    for xbd in CLASSES:
        folder = args.out / "crops" / xbd
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rows.append(
                {
                    "uid": path.stem,
                    "path": str(path.relative_to(args.out.parent.parent)),
                    "label": xbd,
                    "label_idx": CLASS_TO_IDX[xbd],
                    "disaster": "xbd-shared",
                    "tile": path.name,
                }
            )
    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["uid"])
        w.writeheader()
        w.writerows(rows)
    counts = {c: sum(1 for r in rows if r["label"] == c) for c in CLASSES}
    print("manifest", manifest, "n", len(rows), counts)
    print("source", DRIVE_FOLDER)


if __name__ == "__main__":
    main()
