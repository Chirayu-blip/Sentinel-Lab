"""Evaluate modalities and entropy-adaptive fusion on paired feature rows."""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data.parse_apimon import DYNAMIC_FEATURES
from data.pe_extractor import STATIC_FEATURES
from models.fusion import dynamic_weight


def _metrics(labels, scores):
    predictions = np.asarray(scores) >= 0.5
    return {
        "auc": roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan"),
        "accuracy": accuracy_score(labels, predictions),
    }


def evaluate(path: str) -> None:
    frame = pd.read_csv(path)
    required = STATIC_FEATURES + DYNAMIC_FEATURES + ["label"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Paired evaluation CSV is missing: {', '.join(missing)}")
    frame = frame[required].dropna()
    if frame["label"].nunique() != 2:
        raise ValueError("Paired evaluation set must include both label classes")

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    static_train, static_test, dynamic_train, dynamic_test, labels_train, labels_test = (
        train_test_split(
            frame[STATIC_FEATURES], frame[DYNAMIC_FEATURES], frame["label"].astype(int),
            test_size=0.25, random_state=42, stratify=frame["label"],
        )
    )
    static_model = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(static_train, labels_train)
    dynamic_model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42,
    ).fit(dynamic_train, labels_train)
    p_static = static_model.predict_proba(static_test)[:, 1]
    p_dynamic = dynamic_model.predict_proba(dynamic_test)[:, 1]
    entropy = static_test["entropy"].to_numpy()
    print(f"Paired held-out rows: {len(labels_test):,}")
    for name, scores in (
        ("Static", p_static),
        ("Dynamic", p_dynamic),
        ("Fixed 0.5/0.5", 0.5 * p_static + 0.5 * p_dynamic),
    ):
        print(f"{name:16} { _metrics(labels_test, scores) }")
    adaptive = np.asarray([
        (1 - dynamic_weight(value)) * p_s + dynamic_weight(value) * p_d
        for value, p_s, p_d in zip(entropy, p_static, p_dynamic)
    ])
    print(f"{'Adaptive (6.2)':16} { _metrics(labels_test, adaptive) }")

    for name, mask in (
        ("H < 5", entropy < 5),
        ("5 <= H <= 7", (entropy >= 5) & (entropy <= 7)),
        ("H > 7", entropy > 7),
    ):
        if mask.any():
            print(f"{name:16} { _metrics(labels_test.to_numpy()[mask], adaptive[mask]) }")
        else:
            print(f"{name:16} no rows")

    print("\nEntropy threshold sweep (adaptive fusion):")
    for threshold in np.arange(5.0, 7.01, 0.2):
        scores = np.asarray([
            (1 - dynamic_weight(value, threshold)) * p_s
            + dynamic_weight(value, threshold) * p_d
            for value, p_s, p_d in zip(entropy, p_static, p_dynamic)
        ])
        metrics = _metrics(labels_test, scores)
        print(f"theta={threshold:.1f}  AUC={metrics['auc']:.4f}  accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("paired_csv")
    evaluate(argument_parser.parse_args().paired_csv)
