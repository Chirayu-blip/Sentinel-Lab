"""Train static and/or dynamic classifiers from labeled feature data."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data.ember_loader import load_ember
from models.trainers import train_dynamic, train_static


def report_holdout(frame: pd.DataFrame, features: list[str], model_factory, name: str):
    x = frame[features].astype(float)
    y = frame["label"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = model_factory()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(f"{name} held-out ROC AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"{name} held-out accuracy: {accuracy_score(y_test, probabilities >= 0.5):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ember", nargs="+", help="EMBER raw train_features_*.jsonl files")
    parser.add_argument("--dynamic", help="Labeled dynamic API Monitor feature CSV")
    parser.add_argument("--static-csv", help="Optional precomputed static feature CSV")
    args = parser.parse_args()
    if not any((args.ember, args.dynamic, args.static_csv)):
        parser.error("Provide at least one of --ember, --static-csv, or --dynamic")

    if args.ember or args.static_csv:
        if args.ember:
            frame = load_ember(args.ember)
            Path("data/ember_features.csv").parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv("data/ember_features.csv", index=False)
        else:
            frame = pd.read_csv(args.static_csv)
        from data.pe_extractor import STATIC_FEATURES
        from sklearn.ensemble import RandomForestClassifier

        report_holdout(
            frame, STATIC_FEATURES,
            lambda: RandomForestClassifier(
                n_estimators=300, max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            "Static",
        )
        print(f"Saved static model to {train_static(frame)}")

    if args.dynamic:
        from data.parse_apimon import DYNAMIC_FEATURES
        from sklearn.ensemble import GradientBoostingClassifier

        frame = pd.read_csv(args.dynamic)
        report_holdout(
            frame, DYNAMIC_FEATURES,
            lambda: GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42,
            ),
            "Dynamic",
        )
        print(f"Saved dynamic model to {train_dynamic(frame)}")


if __name__ == "__main__":
    main()
