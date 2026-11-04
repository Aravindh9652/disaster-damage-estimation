#!/usr/bin/env python3
"""FastAPI service for damage classification, segmentation, and priority ranking.

Analysis results are returned in the response and are not stored.
Map and zone routes read the demo catalog from disk on each call.

Run from the repository root:
    uvicorn ml.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .dataset import eval_transforms
from .labels import CLASSES, SEVERITY_WEIGHTS
from .model import build_resnet50
from .scoring import building_severity, priority_score
from .segment import estimate_damage_mask, load_unet, mask_png_base64
from .zones import damage_map, find_building, load_catalog, priority_zones

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "resnet50_best.pt"
UNET_CHECKPOINT = ROOT / "models" / "unet_damage.pt"
CATALOG = ROOT / "web" / "public" / "demo" / "catalog.json"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def _load_classifier(device: torch.device):
    if not CHECKPOINT.is_file():
        return None
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.model = _load_classifier(device)
    app.state.transform = eval_transforms()
    app.state.unet = load_unet(UNET_CHECKPOINT, device)
    yield


app = FastAPI(title="DamageMap inference API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:43147", "http://127.0.0.1:43147"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _read_catalog() -> dict:
    if not CATALOG.is_file():
        raise HTTPException(status_code=404, detail="Demo catalog was not found.")
    return load_catalog(CATALOG)


async def _read_image(upload: UploadFile) -> Image.Image:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a PNG, JPEG, or WebP image.")
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB upload limit.")
    try:
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from exc


def _classify(image: Image.Image) -> tuple[str, float, dict[str, float]]:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"ResNet50 checkpoint not found at {CHECKPOINT.name}. Train or copy the weights, then restart.",
        )
    tensor = app.state.transform(image).unsqueeze(0).to(app.state.device)
    with torch.inference_mode():
        probs = torch.softmax(app.state.model(tensor), dim=1)[0].cpu().tolist()
    predicted_index = int(torch.tensor(probs).argmax())
    predicted_class = CLASSES[predicted_index]
    probabilities = {label: round(float(probs[i]), 6) for i, label in enumerate(CLASSES)}
    return predicted_class, float(probs[predicted_index]), probabilities


def _assess(post: Image.Image, pre: Image.Image | None, filename: str) -> dict:
    predicted_class, confidence, probabilities = _classify(post)
    severity = building_severity(predicted_class)
    region = estimate_damage_mask(post, pre, unet=app.state.unet, device=app.state.device)
    scored = priority_score(severity, region["damage_area_ratio"], confidence)
    return {
        "filename": filename,
        "prediction": predicted_class,
        "confidence": round(confidence, 6),
        "probabilities": probabilities,
        "severity_score": round(severity * 100, 1),
        "damage_area_ratio": region["damage_area_ratio"],
        "damage_mask_png": mask_png_base64(region["mask"]),
        "segmentation_method": region["method"],
        "registration_shift": region["registration_shift"],
        **scored,
        "model": CHECKPOINT.name,
        "stored": False,
        "notice": (
            "Decision-support result only. The damage mask is an estimate. "
            "Confirm critical assessments with qualified human review. Nothing from this request is saved."
        ),
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ready" if app.state.model is not None else "catalog-only",
        "classifier": app.state.model is not None,
        "segmentation": "unet" if app.state.unet is not None else "change-or-texture",
        "device": str(app.state.device),
        "persistence": "none",
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    """Classify one post-disaster building crop and estimate its damaged area."""
    source = await _read_image(image)
    return _assess(source, None, image.filename or "uploaded-image")


@app.post("/api/analyze-pair")
async def analyze_pair(
    post_image: UploadFile = File(...),
    pre_image: UploadFile = File(...),
) -> dict:
    """Align a pre/post pair, then classify the post image and segment the change."""
    post = await _read_image(post_image)
    pre = await _read_image(pre_image)
    return _assess(post, pre, post_image.filename or "post-image")


@app.get("/api/damage-map")
def get_damage_map(event: str | None = Query(default=None)) -> dict:
    return damage_map(_read_catalog(), event)


@app.get("/api/buildings/{uid}")
def get_building(uid: str) -> dict:
    building = find_building(_read_catalog(), uid)
    if building is None:
        raise HTTPException(status_code=404, detail="Building not found in the demo catalog.")
    return building


@app.get("/api/priority-zones")
def get_priority_zones(disaster_type: str | None = Query(default=None)) -> dict:
    zones = priority_zones(_read_catalog(), disaster_type)
    return {
        "formula": "100 * (0.5 * severity + 0.3 * damage_area + 0.2 * severity * confidence)",
        "stored": False,
        "zones": zones,
    }
