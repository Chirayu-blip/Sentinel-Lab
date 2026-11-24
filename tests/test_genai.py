import unittest
import json
from unittest.mock import patch

from backend.genai import generate_explanation


class GenAIExplanationTests(unittest.TestCase):
    def test_missing_provider_uses_grounded_local_summary(self):
        evidence = {
            "verdict": "SUSPICIOUS",
            "score": 0.5,
            "entropy": 6.2,
            "weights": {"static": 0.5, "dynamic": 0.5},
            "static": {"score": 0.2, "shap": [
                {"feature": "suspicious_imports", "value": 2, "impact": 0.1}
            ]},
            "dynamic": None,
        }
        with patch.dict("os.environ", {}, clear=True):
            explanation, source = generate_explanation(evidence)
        self.assertEqual(source, "local")
        self.assertIn("SUSPICIOUS", explanation)
        self.assertIn("suspicious_imports", explanation)
        self.assertIn("No dynamic log", explanation)
        self.assertIn("not proof", explanation)

    def test_configured_provider_receives_only_structured_evidence(self):
        response_body = json.dumps({
            "choices": [{"message": {"content": "Signals support a cautious review."}}]
        }).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return response_body

        result = {
            "verdict": "BENIGN",
            "score": 0.1,
            "entropy": 5.2,
            "weights": {"static": 1.0, "dynamic": 0.0},
            "static": {"score": 0.1, "shap": []},
            "dynamic": None,
            "file_name": "private-sample.exe",
            "file_bytes": b"MZ sensitive content",
        }
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True), \
             patch("backend.genai.urlopen", return_value=FakeResponse()) as urlopen:
            explanation, source = generate_explanation(result)
        self.assertEqual(source, "genai")
        self.assertEqual(explanation, "Signals support a cautious review.")
        request = urlopen.call_args.args[0]
        submitted = json.loads(request.data)["messages"][1]["content"]
        self.assertNotIn("private-sample.exe", submitted)
        self.assertNotIn("MZ sensitive content", submitted)
        self.assertIn('"verdict":"BENIGN"', submitted)


if __name__ == "__main__":
    unittest.main()
