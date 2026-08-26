import json
from pathlib import Path

from labels import CLASSES
from metrics import evaluation_report, tile_severity_index


def test_severity_all_destroyed():
    assert tile_severity_index(["destroyed", "destroyed"]) == 100.0


def test_severity_none():
    assert tile_severity_index(["no-damage", "no-damage"]) == 0.0


def test_severity_mixed():
    score = tile_severity_index(["no-damage", "destroyed"])
    assert 49.0 < score < 51.0


def test_perfect_classifier_f1():
    y = list(range(4)) * 5
    report = evaluation_report(y, y)
    assert report["weighted_f1"] == 1.0
    assert report["confusion_matrix"][0][0] == 5


def test_classes_order():
    assert CLASSES[0] == "no-damage"
    assert CLASSES[-1] == "destroyed"


if __name__ == "__main__":
    test_severity_all_destroyed()
    test_severity_none()
    test_severity_mixed()
    test_perfect_classifier_f1()
    test_classes_order()
    print("ok")
