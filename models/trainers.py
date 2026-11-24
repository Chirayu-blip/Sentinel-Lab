"""Training and persistence helpers for static and dynamic tree models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from data.parse_apimon import DYNAMIC_FEATURES
from data.pe_extractor import STATIC_FEATURES

MODEL_DIR = Path(__file__).resolve().parent


def _validate_training_frame(frame: pd.DataFrame, features: list[str]) -> tuple:
    required = features + ["label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")
    labeled = frame[required].dropna()
    if labeled["label"].nunique() != 2:
        raise ValueError("Training data must contain both label classes (0 and 1)")
    return labeled[features].astype(float), labeled["label"].astype(int)


def train_static(frame: pd.DataFrame, demo: bool = False) -> Path:
    x, y = _validate_training_frame(frame, STATIC_FEATURES)
    model = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    model.fit(x, y)
    destination = MODEL_DIR / "static_model.pkl"
    joblib.dump({"model": model, "features": STATIC_FEATURES, "demo": demo}, destination)
    return destination


def train_dynamic(frame: pd.DataFrame, demo: bool = False) -> Path:
    x, y = _validate_training_frame(frame, DYNAMIC_FEATURES)
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42,
    )
    model.fit(x, y)
    destination = MODEL_DIR / "dynamic_model.pkl"
    joblib.dump({"model": model, "features": DYNAMIC_FEATURES, "demo": demo}, destination)
    return destination
