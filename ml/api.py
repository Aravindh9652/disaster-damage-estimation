#!/usr/bin/env python3
"""FastAPI service for damage inference, PostGIS assessment history, and ranking.

The demo map and zones remain catalog-backed; submitted analyses are persisted
when DATABASE_URL is configured.

Run from the repository root:
    uvicorn ml.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import hashlib
import binascii
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from .dataset import eval_transforms
from .decision import DecisionRules, compare_models, image_quality_warnings, review_reasons
from .labels import CLASSES
from .model import build_resnet50
from .scoring import building_severity, priority_score
from .segment import estimate_damage_mask, load_unet, mask_png_base64
from . import persistence
from .scoring import priority_configuration
from .zones import damage_map, find_building, load_catalog, priority_zones

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "resnet50_best.pt"
UNET_CHECKPOINT = ROOT / "models" / "unet_damage.pt"
CATALOG = ROOT / "web" / "public" / "demo" / "catalog.json"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
IMAGE_MIME_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
ImageUpload = tuple[Image.Image, bytes, dict]


class ReviewSubmission(BaseModel):
    status: Literal["confirmed", "corrected", "unresolved"]
    corrected_class: Literal["no-damage", "minor-damage", "major-damage", "destroyed"] | None = None
    reviewer: str = Field(min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_correction(self):
        if self.status == "corrected" and self.corrected_class is None:
            raise ValueError("corrected_class is required when status is corrected.")
        if self.status != "corrected" and self.corrected_class is not None:
            raise ValueError("corrected_class is only allowed when status is corrected.")
        return self


def _load_classifier(device: torch.device):
    if not CHECKPOINT.is_file():
        return None
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_resnet50(pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.damage_map_version = str(
        checkpoint.get("model_version", checkpoint.get("epoch", CHECKPOINT.name))
    )
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.model = _load_classifier(device)
    app.state.transform = eval_transforms()
    app.state.unet = load_unet(UNET_CHECKPOINT, device)
    app.state.decision_rules = DecisionRules.from_environment()
    app.state.persistence_enabled = persistence.database_enabled()
    if app.state.persistence_enabled:
        persistence.initialize_database()
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


async def _read_image(upload: UploadFile) -> ImageUpload:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a PNG, JPEG, or WebP image.")
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel processing limit.",
                )
            if source.format not in IMAGE_MIME_TYPES:
                raise HTTPException(status_code=415, detail="Only PNG, JPEG, and WebP images are supported.")
            source.verify()
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
            metadata = {
                "filename": upload.filename or "uploaded-image",
                "content_type": IMAGE_MIME_TYPES[source.format],
                "uploaded_content_type": upload.content_type,
                "format": source.format,
                "width": image.width,
                "height": image.height,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        return image, payload, metadata
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
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


def _assess(
    post: Image.Image,
    pre: Image.Image | None,
    filename: str,
    impact_factor: float = 1.0,
) -> dict:
    predicted_class, confidence, probabilities = _classify(post)
    severity = building_severity(predicted_class)
    region = estimate_damage_mask(post, pre, unet=app.state.unet, device=app.state.device)
    disagreement = compare_models(
        predicted_class,
        confidence,
        region["damage_area_ratio"],
        region["method"],
        app.state.decision_rules,
    )
    quality_warnings = image_quality_warnings(post, app.state.decision_rules, "Post-disaster")
    if pre is not None:
        quality_warnings.extend(
            image_quality_warnings(pre, app.state.decision_rules, "Pre-disaster")
        )
    reasons = review_reasons(
        confidence, disagreement, quality_warnings, app.state.decision_rules
    )
    scored = priority_score(severity, region["damage_area_ratio"], impact_factor)
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
        "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_disagreement": disagreement,
        "decision_rules": app.state.decision_rules.public(),
        "image_quality_warnings": quality_warnings,
        "review_reasons": reasons,
        "uncertainty": round(1.0 - confidence, 6),
        "human_verification": None,
        **scored,
        "model": getattr(app.state.model, "damage_map_version", CHECKPOINT.name),
        "segmentation_model": UNET_CHECKPOINT.name if region["method"] == "unet" else None,
        "stored": False,
        "notice": (
            "Project-defined decision-support result, not an official emergency standard. "
            "Model agreement does not prove correctness. Confirm critical assessments with qualified human review."
        ),
    }


def _persist(
    report: dict,
    post: ImageUpload,
    pre: ImageUpload | None,
    event_id: str | None,
    building_uid: str | None,
    coverage_id: str | None,
    latitude: float | None,
    longitude: float | None,
    imagery_acquired_at: datetime | None,
) -> dict:
    if not persistence.database_enabled():
        report["stored"] = False
        report["review_status"] = "pending" if report["review_reasons"] else "unreviewed"
        report["persistence_warning"] = "Not saved: DATABASE_URL is not configured."
        report.update(
            {
                "event_id": event_id,
                "building_uid": building_uid,
                "coverage_id": coverage_id,
                "imagery_acquired_at": imagery_acquired_at.isoformat() if imagery_acquired_at else None,
            }
        )
        return report

    try:
        return persistence.persist_assessment(
            report,
            post[1],
            post[2],
            pre[1] if pre else None,
            pre[2] if pre else None,
            event_id,
            building_uid,
            coverage_id,
            latitude,
            longitude,
            imagery_acquired_at,
        )
    except (SQLAlchemyError, OSError, ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=503, detail=f"Assessment could not be persisted: {exc}") from exc


def _require_persistence() -> None:
    if not persistence.database_enabled():
        raise HTTPException(
            status_code=503,
            detail="Assessment history and review require DATABASE_URL and a migrated PostGIS database.",
        )


def _validate_location(latitude: float | None, longitude: float | None) -> None:
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=422, detail="Provide both latitude and longitude, or neither.")
    if latitude is not None and not -90 <= latitude <= 90:
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90.")
    if longitude is not None and not -180 <= longitude <= 180:
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180.")


def _validate_impact_factor(impact_factor: float) -> None:
    if not 0 <= impact_factor <= 1:
        raise HTTPException(status_code=422, detail="Impact factor must be between 0 and 1.")


@app.get("/api/health")
def health() -> dict:
    warnings: list[str] = []
    if app.state.model is None:
        warnings.append(f"Classifier checkpoint not found at {CHECKPOINT.name}.")
    if app.state.unet is None:
        warnings.append("No U-Net checkpoint found; change detection or texture estimation will be used.")

    return {
        "status": "ready" if app.state.model is not None else "catalog-only",
        "classifier": app.state.model is not None,
        "segmentation": "unet" if app.state.unet is not None else "change-or-texture",
        "device": str(app.state.device),
        "persistence": "postgresql-postgis" if persistence.database_enabled() else "disabled",
        "model_path": str(CHECKPOINT),
        "unet_path": str(UNET_CHECKPOINT),
        "warnings": warnings,
        "decision_rules": app.state.decision_rules.public(),
        "priority_configuration": priority_configuration(),
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    event_id: str | None = Form(default=None, max_length=120),
    building_uid: str | None = Form(default=None, max_length=120),
    coverage_id: str | None = Form(default=None, max_length=160),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    imagery_acquired_at: datetime | None = Form(default=None),
    impact_factor: float = Form(default=1.0),
) -> dict:
    """Classify one post-disaster building crop and estimate its damaged area."""
    _validate_location(latitude, longitude)
    _validate_impact_factor(impact_factor)
    post = await _read_image(image)
    report = _assess(post[0], None, post[2]["filename"], impact_factor)
    return _persist(
        report, post, None, event_id, building_uid, coverage_id, latitude, longitude, imagery_acquired_at
    )


@app.post("/api/analyze-pair")
async def analyze_pair(
    post_image: UploadFile = File(...),
    pre_image: UploadFile = File(...),
    event_id: str | None = Form(default=None, max_length=120),
    building_uid: str | None = Form(default=None, max_length=120),
    coverage_id: str | None = Form(default=None, max_length=160),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    imagery_acquired_at: datetime | None = Form(default=None),
    impact_factor: float = Form(default=1.0),
) -> dict:
    """Align a pre/post pair, then classify the post image and segment the change."""
    _validate_location(latitude, longitude)
    _validate_impact_factor(impact_factor)
    post = await _read_image(post_image)
    pre = await _read_image(pre_image)
    report = _assess(post[0], pre[0], post[2]["filename"], impact_factor)
    return _persist(
        report, post, pre, event_id, building_uid, coverage_id, latitude, longitude, imagery_acquired_at
    )


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
        **priority_configuration(),
        "stored": False,
        "zones": zones,
    }


@app.get("/api/assessments")
def get_assessments(
    event_id: str | None = Query(default=None, max_length=120),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    severity: Literal["no-damage", "minor-damage", "major-damage", "destroyed"] | None = None,
    review_status: Literal["unreviewed", "pending", "confirmed", "corrected", "unresolved"] | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    _require_persistence()
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not be after date_to.")
    rows = persistence.list_assessments(
        event_id,
        datetime.combine(date_from, time.min, tzinfo=timezone.utc) if date_from else None,
        datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc) if date_to else None,
        severity,
        review_status,
        limit,
    )
    return {"count": len(rows), "assessments": rows}


@app.get("/api/reviews")
def get_review_queue(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    _require_persistence()
    rows = persistence.list_assessments(review_status="pending", limit=limit)
    return {"count": len(rows), "assessments": rows}


@app.get("/api/assessments/map")
def get_assessment_map(event_id: str | None = Query(default=None, max_length=120)) -> dict:
    _require_persistence()
    return persistence.assessment_map_features(event_id)


@app.get("/api/assessments/{assessment_id}")
def get_assessment_report(assessment_id: UUID) -> dict:
    _require_persistence()
    report = persistence.get_assessment(str(assessment_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    has_pre_image = report.get("pre_image_path") is not None
    report.pop("post_image_path", None)
    report.pop("pre_image_path", None)
    report.pop("mask_path", None)
    report["images"] = {
        "post": f"/api/assessments/{assessment_id}/images/post",
        "pre": f"/api/assessments/{assessment_id}/images/pre" if has_pre_image else None,
        "mask": f"/api/assessments/{assessment_id}/images/mask",
    }
    return report


@app.get("/api/assessments/{assessment_id}/comparisons")
def compare_assessment_history(assessment_id: UUID) -> dict:
    _require_persistence()
    result = persistence.compare_assessments(str(assessment_id))
    if result is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return result


@app.get("/api/assessments/{assessment_id}/images/{image_type}")
def get_assessment_image(assessment_id: UUID, image_type: Literal["post", "pre", "mask"]):
    _require_persistence()
    report = persistence.get_assessment(str(assessment_id))
    if report is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    try:
        path = persistence.image_path(report, image_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Requested image is not available.")
    media_type = "image/png" if image_type == "mask" else (
        (report.get("pre_image_metadata") if image_type == "pre" else report.get("post_image_metadata")) or {}
    ).get("content_type") or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.post("/api/assessments/{assessment_id}/review")
def review_assessment(assessment_id: UUID, submission: ReviewSubmission) -> dict:
    _require_persistence()
    reviewer = submission.reviewer.strip()
    if not reviewer:
        raise HTTPException(status_code=422, detail="Reviewer name cannot be blank.")
    result = persistence.submit_review(
        str(assessment_id),
        submission.status,
        submission.corrected_class,
        reviewer,
        submission.notes.strip() if submission.notes else None,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    result.pop("post_image_path", None)
    result.pop("pre_image_path", None)
    result.pop("mask_path", None)
    return result
