import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import app
from data.pe_extractor import STATIC_FEATURES


class FakeModel:
    def predict_proba(self, _values):
        return [[0.8, 0.2]]


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_reports_model_availability(self):
        with patch("backend.app.STATIC_MODEL_PATH", SimpleNamespace(exists=lambda: False)), \
             patch("backend.app.DYNAMIC_MODEL_PATH", SimpleNamespace(exists=lambda: False)):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["static_model_ready"])

    def test_missing_upload_is_a_client_error(self):
        response = self.client.post("/api/analyze")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Select a Windows PE file", response.json["error"])

    def test_static_only_request_does_not_invent_dynamic_evidence(self):
        feature_values = {name: 0.0 for name in STATIC_FEATURES}
        feature_values["entropy"] = 6.2
        shap = [{"feature": "entropy", "value": 6.2, "impact": 0.1}]
        with patch("backend.app.extract_pe_features", return_value=feature_values), \
             patch("backend.app._load_model", return_value=FakeModel()), \
             patch("backend.app.explain_tree_model", return_value=shap):
            response = self.client.post(
                "/api/analyze",
                data={"file": (io.BytesIO(b"MZ mock PE"), "sample.exe")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json["dynamic"])
        self.assertEqual(response.json["weights"], {"static": 1.0, "dynamic": 0.0})
        self.assertEqual(response.json["analysis_source"], "local")


if __name__ == "__main__":
    unittest.main()
