"""Checks for segmentation, scoring, registration, and zone ranking."""

import numpy as np
from PIL import Image

from preprocess import register_pair
from scoring import human_class_for_review, priority_score, zone_damage_area
from seg_metrics import dice_score, pixel_accuracy, pixel_iou
from segment import change_mask, estimate_damage_mask, texture_mask
from unet import build_unet
from zones import damage_map, priority_zones


def test_priority_bounds():
    low = priority_score(0, 0, 1)
    high = priority_score(100, 1, 1)
    assert low["priority_score"] == 0
    assert low["priority_level"] == "low"
    assert high["priority_score"] == 100
    assert high["priority_level"] == "critical"


def test_priority_formula():
    scored = priority_score(0.5, 0.2, 1.0)
    expected = 100 * (0.5 * 0.5 + 0.3 * 0.2 + 0.2 * (0.5 * 1.0))
    assert abs(scored["priority_score"] - round(expected, 2)) < 0.02
    assert scored["components"]["impact_factor"] == 1.0


def test_priority_does_not_depend_on_confidence():
    assert priority_score(0.8, 0.4)["priority_score"] == priority_score(0.8, 0.4, 1.0)["priority_score"]


def test_human_verification_is_separate_from_prediction():
    assert human_class_for_review("confirmed", "major-damage", None) == "major-damage"
    assert human_class_for_review("corrected", "major-damage", "minor-damage") == "minor-damage"
    assert human_class_for_review("unresolved", "major-damage", None) is None


def test_area_weights_differ_from_a_single_class_mix():
    area = zone_damage_area(["no-damage", "destroyed"])
    assert 0.4 < area < 0.6


def test_pixel_metrics():
    truth = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [0, 0]])
    assert pixel_iou(truth, pred) == 0.5
    assert dice_score(truth, truth) == 1.0
    assert pixel_accuracy(truth, pred) == 0.75


def test_identical_pair_has_no_shift():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    aligned, shift = register_pair(image, image, size=64)
    assert shift == (0, 0)
    assert aligned.size == (64, 64)


def test_masks_are_binary_and_sized():
    rng = np.random.default_rng(1)
    pre = Image.fromarray(rng.integers(0, 40, (80, 80, 3), dtype=np.uint8))
    post_arr = rng.integers(0, 40, (80, 80, 3), dtype=np.uint8)
    post_arr[20:50, 20:50] = 220
    post = Image.fromarray(post_arr)
    mask = change_mask(pre.resize((64, 64)), post.resize((64, 64)))
    assert mask.dtype == bool
    assert mask.shape == (64, 64)
    assert mask.mean() > 0.05
    texture = texture_mask(post)
    assert texture.shape == (256, 256)
    estimate = estimate_damage_mask(post, pre)
    assert estimate["method"] == "pre-post-change"
    assert 0 <= estimate["damage_area_ratio"] <= 1


def test_unet_forward_shape():
    import torch

    model = build_unet()
    logits = model(torch.zeros(1, 3, 64, 64))
    assert logits.shape == (1, 1, 64, 64)


def test_zones_from_catalog():
    catalog = {
        "events": [
            {
                "id": "a",
                "name": "Flood",
                "type": "flood",
                "place": "Coast",
                "date": "2018",
                "severity_pred": 80,
                "buildings": [
                    {
                        "uid": "b1",
                        "label": "destroyed",
                        "pred": "destroyed",
                        "probs": {"destroyed": 0.9, "no-damage": 0.1},
                        "polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}],
                    }
                ],
            },
            {
                "id": "b",
                "name": "Calm",
                "type": "flood",
                "place": "Inland",
                "date": "2018",
                "severity_pred": 2,
                "buildings": [
                    {
                        "uid": "b2",
                        "label": "no-damage",
                        "pred": "no-damage",
                        "probs": {"no-damage": 0.8},
                        "polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
                    }
                ],
            },
        ]
    }
    zones = priority_zones(catalog)
    assert zones[0]["id"] == "a"
    assert zones[0]["priority_level"] == "critical"
    geo = damage_map(catalog, event_id="a")
    assert len(geo["features"]) == 1
    ring = geo["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert geo["crs"] == "tile-pixels"


def test_geojson_uses_longitude_latitude():
    catalog = {
        "events": [
            {
                "id": "hurricane-harvey-houston",
                "name": "Hurricane Harvey",
                "type": "flood",
                "width": 640,
                "height": 640,
                "buildings": [
                    {
                        "uid": "b1",
                        "label": "destroyed",
                        "pred": "major-damage",
                        "probs": {"major-damage": 0.7},
                        "polygon": [{"x": 320, "y": 320}, {"x": 340, "y": 320}, {"x": 340, "y": 340}],
                    }
                ],
            }
        ]
    }
    geo = damage_map(catalog)
    ring = geo["features"][0]["geometry"]["coordinates"][0]
    longitudes = [point[0] for point in ring]
    latitudes = [point[1] for point in ring]
    assert geo["crs"] == "EPSG:4326"
    assert all(-95.4 < lon < -95.3 for lon in longitudes)
    assert all(29.7 < lat < 29.8 for lat in latitudes)


if __name__ == "__main__":
    test_priority_bounds()
    test_priority_formula()
    test_priority_does_not_depend_on_confidence()
    test_human_verification_is_separate_from_prediction()
    test_area_weights_differ_from_a_single_class_mix()
    test_pixel_metrics()
    test_identical_pair_has_no_shift()
    test_masks_are_binary_and_sized()
    test_unet_forward_shape()
    test_zones_from_catalog()
    test_geojson_uses_longitude_latitude()
    print("ok")
