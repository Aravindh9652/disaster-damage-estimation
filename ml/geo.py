"""Project demo tile pixels into longitude and latitude.

Each event is anchored on the city it represents. A pixel is 0.8 m, so a 640 px
tile covers about half a kilometer. These are map positions for the demo, not
the original xBD image georeferencing.
"""

from __future__ import annotations

import math

METERS_PER_PIXEL = 0.8

# City centers for the four demo events.
ANCHORS: dict[str, tuple[float, float]] = {
    "hurricane-harvey-houston": (29.7604, -95.3698),
    "santa-rosa-wildfire": (38.4404, -122.7141),
    "palu-tsunami": (-0.8917, 119.8707),
    "mexico-city-earthquake": (19.4326, -99.1332),
}


def event_origin(event: dict) -> tuple[float, float] | None:
    if "latitude" in event and "longitude" in event:
        return float(event["latitude"]), float(event["longitude"])
    return ANCHORS.get(event.get("id", ""))


def pixel_to_lonlat(
    x: float,
    y: float,
    latitude: float,
    longitude: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Return (longitude, latitude). Pixel y grows downward."""
    east_m = (x - width / 2.0) * METERS_PER_PIXEL
    north_m = (height / 2.0 - y) * METERS_PER_PIXEL
    lat = latitude + north_m / 111_320.0
    cos_lat = math.cos(math.radians(latitude)) or 1e-6
    lon = longitude + east_m / (111_320.0 * cos_lat)
    return round(lon, 6), round(lat, 6)
