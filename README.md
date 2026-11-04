# AI Disaster Damage Mapping and Severity Estimation

Continuation of an **xBD / xView2** student project after dataset download and building extraction.

The original files (`AI_Disaster_Damage_Mapping_Severity_Estimation.docx`, `xBD_ResNet50_Damage_Classification.ipynb`) lived on a local Downloads folder and were not copied into this repo. This project picks up at the next scientific step: **classify each extracted building with ResNet50, then map tile-level severity**.

## What was already done

1. Download a subset of [xBD](https://arxiv.org/abs/1911.09296) (pre/post satellite tiles + JSON polygons).
2. Extract building footprints from post-disaster imagery into per-class crops.

## What this repo adds

| Piece | Role |
| --- | --- |
| `ml/extract_buildings.py` | Same extraction step, kept so the pipeline is one folder |
| `ml/prepare_splits.py` | Disaster-aware train/val/test split |
| `ml/train.py` | ImageNet ResNet50 + ordinal-aware loss |
| `ml/evaluate.py` | Weighted F1, confusion matrix, per-building probs |
| `ml/severity.py` | Tile severity index (0–100) |
| `notebooks/02_resnet50_damage_classification.ipynb` | Colab-friendly continuation notebook |
| `web/` | Dashboard: damage map, model scores, how to rerun on your crops |

Joint Damage Scale (ordinal):

- **no-damage** (0)
- **minor-damage** (1/3)
- **major-damage** (2/3)
- **destroyed** (1)

Tile severity = `100 × mean(ordinal weight)` over buildings on that tile. Rank neighborhoods with it; do not treat it as a dollar-loss model.

xBD is class-imbalanced. **Weighted F1** is the metric that matters. Minor vs major damage is the hard pair.

## Your xBD data (Google Drive)

The shared folder is already **extracted building chips**, not raw 1024×1024 tiles:

https://drive.google.com/drive/folders/1Yeb7vfp4bOTb2i1gcT0HBOBPpEE0uMHH

`resnet_dataset/` contains `no_damage`, `minor_damage`, `major_damage`, `destroyed` JPEGs (~13k crops).

```bash
cd ml
python download_xbd_drive.py --out ../data/xbd
python prepare_splits.py --manifest ../data/xbd/manifest.csv --out ../data/splits
python baseline_eval.py --manifest ../data/xbd/manifest.csv
python train.py --root .. --epochs 8 --freeze-backbone   # ResNet50, GPU preferred
```

Crops are large in file count; `data/xbd/crops/` is gitignored. Re-run the downloader on another machine.

## Dashboard (demo tiles)

The multi-GB xBD archive is not bundled. `ml/generate_demo_artifacts.py` synthesizes four disaster tiles in the same format as extraction output so the UI can run immediately. Replace them with your crops when you train.

```bash
cd web
npm install
npm run dev
```

Opens on [http://127.0.0.1:43147](http://127.0.0.1:43147).

## Live inference portal

The `/run` page accepts a post-disaster building crop and an optional pre-disaster
image. The API classifies the crop, estimates a damage mask, and returns a
priority score in the same response. Nothing is written to a database.

| Module | Role |
| --- | --- |
| `ml/preprocess.py` | Resize and phase-correlation alignment of a pre/post pair |
| `ml/unet.py`, `ml/segment.py` | U-Net damage mask, or change/texture estimate when no mask checkpoint exists |
| `ml/scoring.py`, `ml/zones.py` | Priority score and in-memory zone ranking from the demo catalog |
| `ml/seg_metrics.py` | IoU, Dice, and pixel accuracy for binary masks |

Priority = `100 × (0.5 × severity + 0.3 × damage area + 0.2 × severity × confidence)`.

```bash
GET /api/damage-map
GET /api/priority-zones
GET /api/buildings/{uid}
POST /api/analyze          # post image only
POST /api/analyze-pair     # post_image + pre_image
```

`GET /api/analysis/{id}` is omitted on purpose: results are not stored.

Start the backend before using the upload form:

```bash
pip install -r ml/requirements.txt
uvicorn ml.api:app --host 0.0.0.0 --port 8000
```

Then start the Next.js app from `web/`. Classification uses the ResNet50 checkpoint.
The damage mask uses `models/unet_damage.pt` when that file exists, and otherwise
a pre/post change mask or a texture estimate. Critical assessments require human review.

Open `/map` for the Leaflet view. Building polygons are drawn in longitude and latitude around each event city (Houston, Santa Rosa, Palu, Mexico City) at 0.8 m per pixel. **Download GeoJSON** saves that layer. No coordinates are written to a database.

Open `/priority` for zone ranking. Docker runs the API only, with the model folder mounted read-only:

```bash
docker compose up --build
```

## Train on your extracted buildings

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt

# If you only have tiles, extract crops:
python ml/extract_buildings.py --images data/raw/images --labels data/raw/labels --out data/crops

python ml/prepare_splits.py --manifest data/crops/manifest.csv --out data/splits
cd ml
python train.py --root .. --epochs 8 --freeze-backbone
python evaluate.py --root .. --checkpoint ../models/resnet50_best.pt
python severity.py
python train_unet.py --data ../data/masks   # only if you have image/mask pairs
```

On Colab, use a GPU runtime and `--unfreeze-backbone` after the classification head has settled.

## Project layout

```
ml/           training and evaluation
notebooks/    recap + continuation
web/          Next.js severity dashboard
data/crops/   demo (or your) building chips
web/public/demo/   tiles + catalog.json for the UI
```
