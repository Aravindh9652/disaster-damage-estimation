import unittest

from PIL import Image

try:
    from .decision import DecisionRules, compare_models, image_quality_warnings, review_reasons
except ImportError:
    from decision import DecisionRules, compare_models, image_quality_warnings, review_reasons


class DecisionRulesTests(unittest.TestCase):
    def setUp(self):
        self.rules = DecisionRules()

    def test_comparison_requires_trained_unet(self):
        result = compare_models("destroyed", 0.95, 0.8, "pre-post-change", self.rules)
        self.assertFalse(result["available"])
        self.assertIn("trained U-Net", result["reason"])

    def test_disagreement_is_flagged_on_class_gap(self):
        result = compare_models("no-damage", 0.95, 0.8, "unet", self.rules)
        self.assertTrue(result["available"])
        self.assertEqual(result["unet_area_class"], "destroyed")
        self.assertTrue(result["disagreement"])

    def test_close_prediction_is_not_flagged(self):
        result = compare_models("minor-damage", 0.95, 0.1, "unet", self.rules)
        self.assertFalse(result["disagreement"])
        self.assertIn("not evidence", result["comparison_note"])

    def test_quality_and_confidence_add_review_reasons(self):
        warnings = image_quality_warnings(Image.new("RGB", (32, 32), "white"), self.rules, "Post")
        comparison = {"available": False}
        self.assertEqual(
            review_reasons(0.2, comparison, warnings, self.rules),
            ["low-confidence", "image-quality"],
        )


if __name__ == "__main__":
    unittest.main()
