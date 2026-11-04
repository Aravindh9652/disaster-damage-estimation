"""Build map features and priority zones from the in-memory demo catalog.

Nothing is written back. The catalog file is read on each request.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .geo import event_origin, pixel_to_lonlat
    from .scoring import priority_score, zone_damage_area
except ImportError:
    from geo import event_origin, pixel_to_lonlat
    from scoring import priority_score, zone_damage_area


def load_catalog(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ring(event: dict, polygon: list[dict]) -> tuple[list[list[float]], str]:
    origin = event_origin(event)
    width = float(event.get("width") or 0)
    height = float(event.get("height") or 0)
    if origin and width and height:
        latitude, longitude = origin
        ring = [
            list(
                pixel_to_lonlat(
                    float(point["x"]),
                    float(point["y"]),
                    latitude,
                    longitude,
                    width,
                    height,
                )
            )
            for point in polygon
        ]
        kind = "EPSG:4326"
    else:
        ring = [[float(point["x"]), float(point["y"])] for point in polygon]
        kind = "tile-pixels"
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring, kind


def damage_map(catalog: dict, event_id: str | None = None) -> dict:
    features = []
    crs = "EPSG:4326"
    for event in catalog.get("events", []):
        if event_id and event["id"] != event_id:
            continue
        for building in event["buildings"]:
            ring, kind = _ring(event, building["polygon"])
            crs = kind
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {
                        "uid": building["uid"],
                        "event_id": event["id"],
                        "disaster": event["name"],
                        "disaster_type": event["type"],
                        "label": building["label"],
                        "prediction": building["pred"],
                        "confidence": round(max(building["probs"].values()), 4),
                    },
                }
            )
    return {
        "type": "FeatureCollection",
        "crs": crs,
        "note": (
            "Longitude and latitude place each demo tile on its city at 0.8 m per pixel. "
            "This is not the original xBD georeferencing, and nothing is stored."
        ),
        "features": features,
    }


def find_building(catalog: dict, uid: str) -> dict | None:
    for event in catalog.get("events", []):
        for building in event["buildings"]:
            if building["uid"] == uid:
                return {
                    "uid": building["uid"],
                    "event_id": event["id"],
                    "disaster": event["name"],
                    "disaster_type": event["type"],
                    "place": event["place"],
                    "label": building["label"],
                    "prediction": building["pred"],
                    "probabilities": building["probs"],
                    "polygon": building["polygon"],
                }
    return None


def priority_zones(catalog: dict, disaster_type: str | None = None) -> list[dict]:
    zones = []
    for event in catalog.get("events", []):
        if disaster_type and event["type"] != disaster_type:
            continue
        labels = [building["pred"] for building in event["buildings"]]
        confidences = [max(building["probs"].values()) for building in event["buildings"]]
        confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        scored = priority_score(event["severity_pred"], zone_damage_area(labels), confidence)
        zones.append(
            {
                "id": event["id"],
                "name": event["name"],
                "type": event["type"],
                "place": event["place"],
                "date": event["date"],
                "buildings": len(labels),
                "severity_index": event["severity_pred"],
                "confidence": round(confidence, 4),
                **scored,
            }
        )
    return sorted(zones, key=lambda zone: -zone["priority_score"])
