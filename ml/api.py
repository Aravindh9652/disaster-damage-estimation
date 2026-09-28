#!/usr/bin/env python3
"""FastAPI service for ResNet50 building-damage inference.

Run from the repository root:
    uvicorn ml.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .dataset import eval_transforms
from .labels import CLASSES, SEVERITY_WEIGHTS
from .model import build_resnet50

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "resnet50_best.pt"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Model checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    app.state.model = model
    app.state.device = device
    app.state.transform = eval_transforms()
    yield


app = FastAPI(title="DamageMap inference API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:43147", "http://127.0.0.1:43147"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ready", "model": CHECKPOINT.name, "device": str(app.state.device)}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    """Classify one post-disaster building crop into the xBD four-class scale."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a PNG, JPEG, or WebP image.")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB upload limit.")
    try:
        source = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from exc

    tensor = app.state.transform(source).unsqueeze(0).to(app.state.device)
    with torch.inference_mode():
        probs = torch.softmax(app.state.model(tensor), dim=1)[0].cpu().tolist()
    predicted_index = int(torch.tensor(probs).argmax())
    predicted_class = CLASSES[predicted_index]
    probabilities = {label: round(float(probs[i]), 6) for i, label in enumerate(CLASSES)}
    severity_score = round(SEVERITY_WEIGHTS[predicted_class] * 100, 1)

    return {
        "filename": image.filename or "uploaded-image",
        "prediction": predicted_class,
        "confidence": round(float(probs[predicted_index]), 6),
        "probabilities": probabilities,
        "severity_score": severity_score,
        "model": CHECKPOINT.name,
        "notice": "Decision-support result only. Confirm critical assessments with qualified human review.",
    }
