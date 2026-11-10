"""PostgreSQL/PostGIS persistence for assessments and human review history."""

from __future__ import annotations

import base64
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from .scoring import human_class_for_review

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("DATABASE_URL")
ASSESSMENT_STORAGE = Path(
    os.getenv("ASSESSMENT_STORAGE_DIR", str(ROOT / "data" / "assessments"))
).resolve()
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def database_enabled() -> bool:
    return engine is not None


def require_database():
    if engine is None:
        raise RuntimeError("Assessment persistence is disabled; configure DATABASE_URL.")
    return engine


def initialize_database() -> None:
    db = require_database()
    with db.begin() as connection:
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS postgis")
        connection.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        applied = set(
            connection.execute(text("SELECT version FROM schema_migrations")).scalars()
        )
        for migration in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")):
            if migration.name in applied:
                continue
            for statement in migration.read_text(encoding="utf-8").split(";"):
                if statement.strip():
                    connection.exec_driver_sql(statement)
            connection.execute(
                text("INSERT INTO schema_migrations (version) VALUES (:version)"),
                {"version": migration.name},
            )


def persist_assessment(
    report: dict[str, Any],
    post_bytes: bytes,
    post_metadata: dict[str, Any],
    pre_bytes: bytes | None,
    pre_metadata: dict[str, Any] | None,
    event_id: str | None,
    building_uid: str | None,
    coverage_id: str | None,
    latitude: float | None,
    longitude: float | None,
    imagery_acquired_at: datetime | None,
) -> dict[str, Any]:
    db = require_database()
    assessment_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    review_status = "pending" if report["review_reasons"] else "unreviewed"
    relative_dir = Path(str(assessment_id))
    file_dir = ASSESSMENT_STORAGE / relative_dir
    post_path = relative_dir / "post-image"
    pre_path = relative_dir / "pre-image" if pre_bytes is not None else None
    mask_path = relative_dir / "damage-mask.png"

    file_dir.mkdir(parents=True, exist_ok=False)
    try:
        (ASSESSMENT_STORAGE / post_path).write_bytes(post_bytes)
        if pre_bytes is not None and pre_path is not None:
            (ASSESSMENT_STORAGE / pre_path).write_bytes(pre_bytes)
        (ASSESSMENT_STORAGE / mask_path).write_bytes(
            base64.b64decode(report["damage_mask_png"], validate=True)
        )
        with db.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO assessments (
                        id, created_at, event_id, building_uid, coverage_id, imagery_acquired_at, location,
                        filename, pre_filename, post_image_path, pre_image_path, mask_path,
                        post_image_metadata, pre_image_metadata, model_version,
                        segmentation_model_version, prediction, confidence, probabilities,
                        severity_score, damage_area_ratio, segmentation_method,
                        model_disagreement, decision_rules, image_quality_warnings, review_reasons,
                        priority_score, priority_level, priority_components, priority_formula,
                        priority_rationale, review_status
                    ) VALUES (
                        CAST(:id AS uuid), :created_at, :event_id, :building_uid, :coverage_id, :imagery_acquired_at,
                        CASE WHEN :latitude IS NULL THEN NULL
                             ELSE ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326) END,
                        :filename, :pre_filename, :post_image_path, :pre_image_path, :mask_path,
                        CAST(:post_image_metadata AS jsonb), CAST(:pre_image_metadata AS jsonb),
                        :model_version, :segmentation_model_version, :prediction, :confidence,
                        CAST(:probabilities AS jsonb), :severity_score, :damage_area_ratio,
                        :segmentation_method, CAST(:model_disagreement AS jsonb),
                        CAST(:decision_rules AS jsonb),
                        CAST(:image_quality_warnings AS jsonb), CAST(:review_reasons AS jsonb),
                        :priority_score, :priority_level, CAST(:priority_components AS jsonb),
                        :priority_formula, :priority_rationale, :review_status
                    )
                    """
                ),
                {
                    "id": str(assessment_id),
                    "created_at": created_at,
                    "event_id": event_id,
                    "building_uid": building_uid,
                    "coverage_id": coverage_id,
                    "imagery_acquired_at": imagery_acquired_at,
                    "latitude": latitude,
                    "longitude": longitude,
                    "filename": report["filename"],
                    "pre_filename": pre_metadata.get("filename") if pre_metadata else None,
                    "post_image_path": post_path.as_posix(),
                    "pre_image_path": pre_path.as_posix() if pre_path else None,
                    "mask_path": mask_path.as_posix(),
                    "post_image_metadata": json.dumps(post_metadata),
                    "pre_image_metadata": json.dumps(pre_metadata) if pre_metadata else None,
                    "model_version": report["model"],
                    "segmentation_model_version": report["segmentation_model"],
                    "prediction": report["prediction"],
                    "confidence": report["confidence"],
                    "probabilities": json.dumps(report["probabilities"]),
                    "severity_score": report["severity_score"],
                    "damage_area_ratio": report["damage_area_ratio"],
                    "segmentation_method": report["segmentation_method"],
                    "model_disagreement": json.dumps(report["model_disagreement"]),
                    "decision_rules": json.dumps(report["decision_rules"]),
                    "image_quality_warnings": json.dumps(report["image_quality_warnings"]),
                    "review_reasons": json.dumps(report["review_reasons"]),
                    "priority_score": report["priority_score"],
                    "priority_level": report["priority_level"],
                    "priority_components": json.dumps(report["components"]),
                    "priority_formula": report["formula"],
                    "priority_rationale": report["priority_rationale"],
                    "review_status": review_status,
                },
            )
    except Exception:
        shutil.rmtree(file_dir, ignore_errors=True)
        raise

    report.update(
        {
            "assessment_id": str(assessment_id),
            "assessment_timestamp": created_at.isoformat(),
            "stored": True,
            "review_status": review_status,
            "images": {
                "post": f"/api/assessments/{assessment_id}/images/post",
                "pre": f"/api/assessments/{assessment_id}/images/pre" if pre_path else None,
                "mask": f"/api/assessments/{assessment_id}/images/mask",
            },
            "event_id": event_id,
            "building_uid": building_uid,
            "imagery_acquired_at": imagery_acquired_at.isoformat() if imagery_acquired_at else None,
            "human_verification": None,
        }
    )
    return report


def _json_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    return json.loads(value)


def list_assessments(
    event_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    severity: str | None = None,
    review_status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    db = require_database()
    query = text(
        """
        SELECT id::text, created_at, event_id, building_uid, coverage_id, filename, prediction, confidence,
               probabilities, severity_score, damage_area_ratio, segmentation_method,
               image_quality_warnings, review_reasons, model_disagreement, decision_rules,
               priority_score, priority_level, priority_components, priority_formula,
               priority_rationale,
               review_status, human_damage_class, reviewer, review_notes, reviewed_at,
               post_image_metadata, pre_image_metadata, model_version,
               segmentation_model_version, imagery_acquired_at
        FROM assessments
        WHERE (:event_id IS NULL OR event_id = :event_id)
          AND (:date_from IS NULL OR created_at >= :date_from)
          AND (:date_to IS NULL OR created_at < :date_to)
          AND (:severity IS NULL OR prediction = :severity)
          AND (:review_status IS NULL OR review_status = :review_status)
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    with db.connect() as connection:
        rows = connection.execute(
            query,
            {
                "event_id": event_id,
                "date_from": date_from,
                "date_to": date_to,
                "severity": severity,
                "review_status": review_status,
                "limit": limit,
            },
        ).mappings()
        return [_format_row(dict(row)) for row in rows]


def get_assessment(assessment_id: str) -> dict[str, Any] | None:
    db = require_database()
    query = text(
        """
        SELECT id::text, created_at, event_id, building_uid, coverage_id, filename, prediction, confidence,
               probabilities, severity_score, damage_area_ratio, segmentation_method,
               model_disagreement, decision_rules, image_quality_warnings, review_reasons, priority_score,
               priority_level, priority_components, priority_formula, priority_rationale,
               review_status, human_damage_class,
               reviewer, review_notes, reviewed_at, post_image_metadata, pre_image_metadata,
               model_version, segmentation_model_version, imagery_acquired_at,
               post_image_path, pre_image_path, mask_path, ST_Y(location) AS latitude,
               ST_X(location) AS longitude
        FROM assessments WHERE id = CAST(:id AS uuid)
        """
    )
    with db.connect() as connection:
        row = connection.execute(query, {"id": assessment_id}).mappings().first()
    return _format_row(dict(row)) if row else None


def image_path(assessment: dict[str, Any], image_type: str) -> Path | None:
    column = {"post": "post_image_path", "pre": "pre_image_path", "mask": "mask_path"}.get(image_type)
    if column is None:
        raise ValueError("Image type must be post, pre, or mask.")
    relative_path = assessment.get(column)
    if not relative_path:
        return None
    path = (ASSESSMENT_STORAGE / relative_path).resolve()
    if not path.is_relative_to(ASSESSMENT_STORAGE):
        raise ValueError("Stored image path is outside the configured assessment directory.")
    return path


def submit_review(
    assessment_id: str,
    status: str,
    corrected_class: str | None,
    reviewer: str,
    notes: str | None,
) -> dict[str, Any] | None:
    db = require_database()
    now = datetime.now(timezone.utc)
    with db.begin() as connection:
        prediction = connection.execute(
            text("SELECT prediction FROM assessments WHERE id = CAST(:id AS uuid) FOR UPDATE"),
            {"id": assessment_id},
        ).scalar_one_or_none()
        if prediction is None:
            return None
        human_damage_class = human_class_for_review(status, prediction, corrected_class)
        connection.execute(
            text(
                """
                UPDATE assessments
                SET review_status = :status, human_damage_class = :human_damage_class,
                    reviewer = :reviewer, review_notes = :notes, reviewed_at = :reviewed_at
                WHERE id = CAST(:id AS uuid)
                """
            ),
            {
                "id": assessment_id,
                "status": status,
                "human_damage_class": human_damage_class,
                "reviewer": reviewer,
                "notes": notes,
                "reviewed_at": now,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO assessment_reviews
                    (assessment_id, status, human_damage_class, reviewer, notes, reviewed_at)
                VALUES (CAST(:id AS uuid), :status, :human_damage_class, :reviewer, :notes, :reviewed_at)
                """
            ),
            {
                "id": assessment_id,
                "status": status,
                "human_damage_class": human_damage_class,
                "reviewer": reviewer,
                "notes": notes,
                "reviewed_at": now,
            },
        )
    return get_assessment(assessment_id)


def assessment_map_features(event_id: str | None = None) -> dict[str, Any]:
    db = require_database()
    query = text(
        """
        SELECT id::text, event_id, building_uid, prediction, human_damage_class,
               review_status, confidence, priority_score,
               ST_X(location) AS longitude,
               ST_Y(location) AS latitude
        FROM assessments
        WHERE location IS NOT NULL AND (:event_id IS NULL OR event_id = :event_id)
        ORDER BY created_at DESC
        """
    )
    with db.connect() as connection:
        rows = connection.execute(query, {"event_id": event_id}).mappings()
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]],
                },
                "properties": {
                    "id": row["id"],
                    "event_id": row["event_id"],
                    "building_uid": row["building_uid"],
                    "ai_prediction": row["prediction"],
                    "human_damage_class": row["human_damage_class"],
                    "review_status": row["review_status"],
                    "confidence": row["confidence"],
                    "priority_score": row["priority_score"],
                },
            }
            for row in rows
        ]
    return {"type": "FeatureCollection", "features": features}


def compare_assessments(assessment_id: str) -> dict[str, Any] | None:
    current = get_assessment(assessment_id)
    if current is None:
        return None
    required = ("event_id", "building_uid", "coverage_id", "imagery_acquired_at", "latitude", "longitude")
    missing = [key for key in required if current.get(key) is None]
    if missing:
        return {
            "comparable": False,
            "reason": f"Comparison requires event, building, stable coverage ID, acquisition time, and supplied coordinates; missing: {', '.join(missing)}.",
        }
    db = require_database()
    query = text(
        """
        SELECT id::text, created_at, imagery_acquired_at, prediction, damage_area_ratio,
               model_version, segmentation_method, segmentation_model_version
        FROM assessments
        WHERE id <> CAST(:id AS uuid)
          AND event_id = :event_id AND building_uid = :building_uid
          AND coverage_id = :coverage_id
          AND imagery_acquired_at < :imagery_acquired_at
          AND ST_DWithin(location::geography, ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography, 1)
          AND post_image_metadata->>'width' = :width
          AND post_image_metadata->>'height' = :height
          AND model_version = :model_version
          AND segmentation_method = :segmentation_method
          AND COALESCE(segmentation_model_version, '') = COALESCE(:segmentation_model_version, '')
        ORDER BY imagery_acquired_at DESC LIMIT 1
        """
    )
    metadata = current.get("post_image_metadata") or {}
    with db.connect() as connection:
        previous = connection.execute(
            query,
            {
                "id": assessment_id,
                "event_id": current["event_id"],
                "building_uid": current["building_uid"],
                "coverage_id": current["coverage_id"],
                "imagery_acquired_at": datetime.fromisoformat(current["imagery_acquired_at"]),
                "longitude": current["longitude"],
                "latitude": current["latitude"],
                "width": str(metadata.get("width")),
                "height": str(metadata.get("height")),
                "model_version": current["model_version"],
                "segmentation_method": current["segmentation_method"],
                "segmentation_model_version": current["segmentation_model_version"],
            },
        ).mappings().first()
    if previous is None:
        return {
            "comparable": False,
            "reason": "No earlier record matches the event, building, explicit coverage ID, location, image dimensions, acquisition ordering, and model versions.",
        }
    return {
        "comparable": True,
        "previous": {
            **dict(previous),
            "created_at": previous["created_at"].isoformat(),
            "imagery_acquired_at": previous["imagery_acquired_at"].isoformat(),
        },
        "current": {
            "id": current["id"],
            "imagery_acquired_at": current["imagery_acquired_at"],
            "prediction": current["prediction"],
            "damage_area_ratio": current["damage_area_ratio"],
        },
        "conclusion": (
            "Model outputs are comparable under the supplied coverage metadata. "
            "Differences do not establish that physical damage worsened or improved."
        ),
    }


def _format_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "probabilities",
        "review_reasons",
        "model_disagreement",
        "decision_rules",
        "image_quality_warnings",
        "priority_components",
        "post_image_metadata",
        "pre_image_metadata",
    ):
        if key in row:
            row[key] = _json_value(row[key])
    for key in ("created_at", "reviewed_at", "imagery_acquired_at"):
        if row.get(key) is not None:
            row[key] = row[key].isoformat()
    row["human_verification"] = (
        {
            "status": row["review_status"],
            "damage_class": row.get("human_damage_class"),
            "reviewer": row.get("reviewer"),
            "notes": row.get("review_notes"),
            "reviewed_at": row.get("reviewed_at"),
        }
        if row.get("reviewed_at")
        else None
    )
    row["uncertainty"] = round(1 - row["confidence"], 6)
    return row
