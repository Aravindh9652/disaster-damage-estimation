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

## Dashboard (demo tiles)

The multi-GB xBD archive is not bundled. `ml/generate_demo_artifacts.py` synthesizes four disaster tiles in the same format as extraction output so the UI can run immediately. Replace them with your crops when you train.

```bash
cd web
npm install
npm run dev
```

Opens on [http://127.0.0.1:43147](http://127.0.0.1:43147).

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
