"""Flask API for static/dynamic malware triage and grounded explanations."""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from backend.genai import generate_explanation
from data.parse_apimon import DYNAMIC_FEATURES, parse_apimon_csv
from data.pe_extractor import STATIC_FEATURES, extract_pe_features
from models.explain import explain_tree_model
from models.fusion import fuse_scores

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
STATIC_MODEL_PATH = ROOT / "models" / "static_model.pkl"
DYNAMIC_MODEL_PATH = ROOT / "models" / "dynamic_model.pkl"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGIN", "http://127.0.0.1:5173,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "Request exceeds the 100 MiB upload limit"}), 413


def _load_model(path: Path, expected_features: list[str]):
    if not path.exists():
        raise RuntimeError(
            f"Model artifact is missing: {path.name}. Train it using the instructions in README.md."
        )
    artifact = joblib.load(path)
    if artifact.get("features") != expected_features:
        raise RuntimeError(f"{path.name} feature schema does not match this app")
    model = artifact["model"]
    model._sentinel_demo = bool(artifact.get("demo", False))
    return model


def _demo_mode(path: Path) -> bool:
    return bool(path.exists() and joblib.load(path).get("demo", False))


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "static_model_ready": STATIC_MODEL_PATH.exists(),
        "dynamic_model_ready": DYNAMIC_MODEL_PATH.exists(),
        "demo_mode": _demo_mode(STATIC_MODEL_PATH) or _demo_mode(DYNAMIC_MODEL_PATH),
        "genai_enabled": bool(os.getenv("OPENAI_API_KEY")),
    })


@app.post("/api/analyze")
def analyze():
    pe_file = request.files.get("file")
    if pe_file is None:
        return jsonify({"error": "Select a Windows PE file in the file field"}), 400
    if Path(pe_file.filename or "").suffix.lower() not in {".exe", ".dll", ".sys"}:
        return jsonify({"error": "Upload a .exe, .dll, or .sys Windows PE file"}), 400
    content = pe_file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File exceeds the 100 MiB upload limit"}), 413
    if not content.startswith(b"MZ"):
        return jsonify({"error": "The uploaded file does not have a Windows PE signature"}), 400

    try:
        static_values = extract_pe_features(content)
        static_model = _load_model(STATIC_MODEL_PATH, STATIC_FEATURES)
        static_score = float(static_model.predict_proba(
            pd.DataFrame([[static_values[name] for name in STATIC_FEATURES]], columns=STATIC_FEATURES)
        )[0][1])
        static_evidence = {
            "score": static_score,
            "shap": explain_tree_model(static_model, STATIC_FEATURES, static_values),
        }

        dynamic_file = request.files.get("log")
        dynamic_score = None
        dynamic_evidence = None
        dynamic_values = None
        if dynamic_file and dynamic_file.filename:
            if Path(dynamic_file.filename).suffix.lower() != ".csv":
                return jsonify({"error": "The dynamic log must be a .csv file"}), 400
            dynamic_values = parse_apimon_csv(dynamic_file.stream)
            dynamic_model = _load_model(DYNAMIC_MODEL_PATH, DYNAMIC_FEATURES)
            dynamic_score = float(dynamic_model.predict_proba(
                pd.DataFrame([[dynamic_values[name] for name in DYNAMIC_FEATURES]], columns=DYNAMIC_FEATURES)
            )[0][1])
            dynamic_evidence = {
                "score": dynamic_score,
                "shap": explain_tree_model(dynamic_model, DYNAMIC_FEATURES, dynamic_values),
            }

        entropy = static_values["entropy"]
        fusion = fuse_scores(static_score, dynamic_score, entropy)
        result = {
            **fusion,
            "entropy": entropy,
            "static": static_evidence,
            "dynamic": dynamic_evidence,
            "file_name": Path(pe_file.filename).name,
            "demo_mode": bool(
                getattr(static_model, "_sentinel_demo", False)
                or getattr(dynamic_model, "_sentinel_demo", False)
            ) if dynamic_score is not None else bool(
                getattr(static_model, "_sentinel_demo", False)
            ),
        }
        explanation, source = generate_explanation(result)
        result["analysis"] = explanation
        result["analysis_source"] = source
        return jsonify(result)
    except (ValueError, RuntimeError) as exc:
        status = 503 if isinstance(exc, RuntimeError) else 400
        return jsonify({"error": str(exc)}), status
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
