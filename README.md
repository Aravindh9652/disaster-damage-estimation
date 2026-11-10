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

## Inference, evidence reports, and human review

The `/run` page accepts a post-disaster crop and an optional pre-disaster image.
The API keeps the ResNet50 classification, segmentation method, and human review
as separate evidence. A mask is identified as `unet` only when the trained
`models/unet_damage.pt` checkpoint was loaded; otherwise change/texture methods
are explicitly labeled heuristic estimates.

| Module | Role |
| --- | --- |
| `ml/preprocess.py` | Resize and phase-correlation alignment of a pre/post pair |
| `ml/unet.py`, `ml/segment.py` | U-Net damage mask, or change/texture estimate when no mask checkpoint exists |
| `ml/scoring.py`, `ml/zones.py` | Configurable decision-support score and demo-catalog zone ranking |
| `ml/seg_metrics.py` | IoU, Dice, and pixel accuracy for binary masks |
| `ml/decision.py` | Configurable model-disagreement and image-quality review signals |
| `ml/persistence.py`, `ml/migrations/` | PostgreSQL/PostGIS persistence, migrations, reports, and review history |

Model disagreement is computed only if a ResNet50 prediction and trained U-Net
mask are both available. The mask area is converted to a class using configured
area thresholds; a class gap at or above the configured minimum creates a review
signal. Heuristic segmentation is not compared as if it were U-Net output.
Agreement is not evidence that either model is correct.

The default priority score is
`100 × (0.5 × normalized severity + 0.3 × normalized damaged area + 0.2 × normalized severity × impact factor)`.
The impact factor is a normalized multiplier with a neutral default of `1.0`.
The `/run` form accepts a documented contextual impact factor per assessment;
use the neutral value when no supported context is available.
Confidence/uncertainty is reported separately and does not lower priority.
This project-defined score is for decision support, not an official emergency
standard.

```bash
GET /api/damage-map
GET /api/priority-zones
GET /api/buildings/{uid}
POST /api/analyze          # image plus optional event/building/coverage/location/time metadata
POST /api/analyze-pair     # post_image + pre_image plus optional metadata
GET /api/assessments       # persistent history with filters
GET /api/reviews           # items needing human review
POST /api/assessments/{id}/review
GET /api/assessments/map   # saved point assessments; AI and human fields are separate
GET /api/assessments/{id}/comparisons
```

The versioned SQL migration is applied at API startup when `DATABASE_URL` is
configured. Original uploaded images and masks are stored as files under
`data/assessments/`; PostgreSQL stores image metadata and relative paths. For
privacy, these local files are gitignored. Review decisions are audit-recorded
separately. Corrections are not automatically used as training examples.

Persisted map points require coordinates explicitly supplied with the analysis;
the application does not infer a building's real-world location. The catalog
map is synthetic/demo data with approximate event-city anchors, not original
xBD georeferencing. Repeated-assessment comparison is only enabled when event,
building, caller-supplied stable coverage ID, acquisition time, location,
dimensions, and model versions match. Even then, model-output differences do
not prove physical damage worsened or improved.

### Windows / VS Code local setup

1. Open the repository folder in VS Code. Use PowerShell from the repository root.
2. Start the PostGIS database (Docker Desktop must be running):

```powershell
docker compose up -d db
```

3. Create a Python environment, install dependencies, and point the API at the local database:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ml\requirements.txt
$env:DATABASE_URL = "postgresql+psycopg://damagemap:local-development-only@localhost:5432/damagemap"
uvicorn ml.api:app --host 127.0.0.1 --port 8000
```

4. In a second terminal, install and start the frontend:

```powershell
Set-Location web
npm install
npm run dev
```

Open `http://localhost:43147`. The first backend startup creates PostGIS tables
from `ml/migrations/001_assessments.sql`. For Dockerized API + database, run
`docker compose up --build`; start the Next.js frontend separately as above.
Compose's default database credentials are for local development only and must
be changed before any non-local use.

If you do not configure `DATABASE_URL`, inference responses are explicitly
marked as not saved, and history/review endpoints return HTTP 503. Configure the
following optional environment variables to adjust decision thresholds and score:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `MODEL_DISAGREEMENT_MIN_CONFIDENCE` | `0.60` | Below this confidence, enqueue human review |
| `MODEL_DISAGREEMENT_MIN_CLASS_GAP` | `1` | Minimum ordinal class gap to flag disagreement |
| `UNET_NO_DAMAGE_MAX_AREA` | `0.02` | Maximum U-Net area ratio binned as no damage |
| `UNET_MINOR_MAX_AREA` | `0.25` | Maximum ratio binned as minor damage |
| `UNET_MAJOR_MAX_AREA` | `0.60` | Maximum ratio binned as major damage |
| `IMAGE_MIN_DIMENSION` | `128` | Image dimension below which a quality warning is raised |
| `IMAGE_MIN_GRAY_STDDEV` | `0.025` | Low-contrast warning threshold |
| `PRIORITY_WEIGHT_SEVERITY` | `0.5` | Normalized severity contribution |
| `PRIORITY_WEIGHT_DAMAGE_AREA` | `0.3` | Normalized damaged-area contribution |
| `PRIORITY_WEIGHT_IMPACT` | `0.2` | Severity × impact-factor contribution |

Priority weights must be non-negative and sum to 1. Cutoffs can be adjusted with
`PRIORITY_CUTOFF_CRITICAL` (`60`), `PRIORITY_CUTOFF_HIGH` (`35`), and
`PRIORITY_CUTOFF_MODERATE` (`15`).

The reviewer's name is not authenticated in this prototype; run it only in a
trusted local environment. Image retention, access control, and authenticated
reviewer identity must be addressed before operational deployment.

### Model and data prerequisites

Classification requires `models/resnet50_best.pt`; otherwise `/api/analyze`
returns HTTP 503. A trained segmentation comparison requires
`models/unet_damage.pt`; without it, the API reports comparison as unavailable
and uses only explicitly labeled change/texture estimates. The demo catalog is
not a substitute for real model predictions, real geospatial building
footprints, or field verification. Only real geospatial input supplied by the
operator is stored as a PostGIS point.

For manual backend-only startup without PostGIS:

```bash
uvicorn ml.api:app --host 127.0.0.1 --port 8000
```

Open `/map` for the Leaflet view. Synthetic demo polygons are drawn around approximate city anchors; they do not represent verified xBD coordinates. Saved analyses with operator-provided coordinates appear as a distinct layer with AI and human-review values shown separately. **Download GeoJSON** exports the selected demo layer.

Open `/priority` for demo-catalog zone ranking. Docker Compose runs PostGIS and the API, with model weights mounted read-only:

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
