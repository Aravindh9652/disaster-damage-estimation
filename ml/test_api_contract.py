import unittest
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from ml.api import CHECKPOINT, ReviewSubmission, _validate_impact_factor, _validate_location, app
from fastapi import HTTPException


class APIContractTests(unittest.TestCase):
    def test_corrected_review_requires_class(self):
        with self.assertRaises(ValueError):
            ReviewSubmission(status="corrected", reviewer="reviewer")
        self.assertEqual(
            ReviewSubmission(
                status="corrected", corrected_class="major-damage", reviewer="reviewer"
            ).corrected_class,
            "major-damage",
        )

    def test_location_requires_a_valid_pair(self):
        with self.assertRaises(HTTPException):
            _validate_location(40.0, None)
        with self.assertRaises(HTTPException):
            _validate_location(91.0, 0.0)
        _validate_location(40.0, -75.0)

    def test_impact_factor_is_normalized(self):
        _validate_impact_factor(1)
        with self.assertRaises(HTTPException):
            _validate_impact_factor(1.1)

    def test_history_reports_disabled_database(self):
        from ml import persistence

        if persistence.database_enabled():
            self.skipTest("Test requires DATABASE_URL to be unset.")
        with TestClient(app) as client:
            response = client.get("/api/assessments")
            self.assertEqual(response.status_code, 503)
            self.assertIn("DATABASE_URL", response.json()["detail"])

    def test_health_exposes_missing_trained_models_and_score_configuration(self):
        with TestClient(app) as client:
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("priority_configuration", data)
            self.assertFalse(data["priority_configuration"]["confidence_affects_score"])
            if not data["classifier"]:
                self.assertIn("Classifier checkpoint", " ".join(data["warnings"]))
            self.assertEqual(data["decision_rules"]["minimum_class_gap"], 1)

    def test_classifier_and_segmentation_readiness_are_reported_separately(self):
        with TestClient(app) as client:
            data = client.get("/api/health").json()
        self.assertEqual(data["model_path"], str(CHECKPOINT))
        self.assertEqual(data["unet_path"], str(CHECKPOINT.with_name("unet_damage.pt")))
        self.assertEqual(data["classifier"], app.state.model is not None)
        self.assertEqual(data["segmentation"] == "unet", app.state.unet is not None)
        if not CHECKPOINT.is_file():
            self.assertFalse(data["classifier"])
            self.assertTrue(any("Classifier checkpoint" in warning for warning in data["warnings"]))
        if not CHECKPOINT.with_name("unet_damage.pt").is_file():
            self.assertEqual(data["segmentation"], "change-or-texture")
            self.assertTrue(any("U-Net checkpoint" in warning for warning in data["warnings"]))

    def test_demo_map_and_priority_endpoints_disclose_catalog_scope(self):
        with TestClient(app) as client:
            map_response = client.get("/api/damage-map")
            priority_response = client.get("/api/priority-zones")
        self.assertEqual(map_response.status_code, 200)
        self.assertIn("not the original xBD georeferencing", map_response.json()["note"])
        self.assertGreater(len(map_response.json()["features"]), 0)
        self.assertEqual(priority_response.status_code, 200)
        self.assertFalse(priority_response.json()["stored"])
        self.assertGreater(len(priority_response.json()["zones"]), 0)

    def test_database_backed_history_and_review_routes_fail_explicitly_without_database(self):
        from ml import persistence

        if persistence.database_enabled():
            self.skipTest("Test requires DATABASE_URL to be unset.")
        with TestClient(app) as client:
            responses = [
                client.get("/api/assessments"),
                client.get("/api/reviews"),
                client.get("/api/assessments/map"),
            ]
        self.assertEqual([response.status_code for response in responses], [503, 503, 503])
        self.assertTrue(
            all("DATABASE_URL" in response.json()["detail"] for response in responses)
        )

    def test_invalid_assessment_id_is_rejected(self):
        with TestClient(app) as client:
            response = client.get("/api/assessments/not-a-uuid")
            self.assertEqual(response.status_code, 422)

    def test_missing_classifier_does_not_return_a_fabricated_prediction(self):
        if CHECKPOINT.is_file():
            self.skipTest("Test requires the ResNet checkpoint to be absent.")
        image_buffer = BytesIO()
        Image.new("RGB", (32, 32), "white").save(image_buffer, format="PNG")
        with TestClient(app) as client:
            response = client.post(
                "/api/analyze",
                files={"image": ("crop.png", image_buffer.getvalue(), "image/png")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("checkpoint not found", response.json()["detail"])

    def test_unsupported_image_format_is_rejected(self):
        image_buffer = BytesIO()
        Image.new("RGB", (32, 32), "white").save(image_buffer, format="BMP")
        with TestClient(app) as client:
            response = client.post(
                "/api/analyze",
                files={"image": ("crop.bmp", image_buffer.getvalue(), "image/png")},
            )
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
