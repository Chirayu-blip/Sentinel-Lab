"""Create clearly labeled synthetic demo models for testing the upload flow."""

import numpy as np
import pandas as pd

from data.parse_apimon import DYNAMIC_FEATURES
from data.pe_extractor import STATIC_FEATURES
from models.trainers import train_dynamic, train_static


def _static_demo_frame(seed: int = 17, rows_per_class: int = 500) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        for _ in range(rows_per_class):
            row = {
                "file_size": float(random.lognormal(12.0 if label == 0 else 13.0, 1.0)),
                "entropy": float(random.normal(5.2 if label == 0 else 6.7, 0.65)),
                "num_imports": float(random.integers(10, 160)),
                "num_exports": float(random.integers(0, 20)),
                "num_sections": float(random.integers(3, 10)),
                "has_debug": float(random.random() < (0.25 if label == 0 else 0.08)),
                "has_tls": float(random.random() < (0.04 if label == 0 else 0.12)),
                "has_resources": float(random.random() < 0.75),
                "virtual_size": float(random.lognormal(13.0, 1.0)),
                "raw_size": float(random.lognormal(12.5, 1.0)),
                "section_entropy_mean": float(random.normal(4.6 if label == 0 else 6.0, 0.8)),
                "suspicious_imports": float(random.poisson(0.15 if label == 0 else 2.2)),
                "packer_detected": float(random.random() < (0.03 if label == 0 else 0.38)),
                "timestamp_anomaly": float(random.random() < (0.02 if label == 0 else 0.2)),
                "label": label,
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=STATIC_FEATURES + ["label"])


def _dynamic_demo_frame(seed: int = 23, rows_per_class: int = 500) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        for _ in range(rows_per_class):
            features = {name: 0.0 for name in DYNAMIC_FEATURES}
            risky_scale = 1 if label == 0 else 8
            features.update({
                "VirtualAlloc": float(random.poisson(0.25 if label == 0 else 2.5)),
                "WriteProcessMemory": float(random.poisson(0.03 if label == 0 else 1.8)),
                "CreateRemoteThread": float(random.poisson(0.02 if label == 0 else 0.9)),
                "OpenProcess": float(random.poisson(0.3 if label == 0 else 2.2)),
                "InternetOpenUrl": float(random.poisson(1.0 if label == 0 else 5.0)),
                "RegSetValue": float(random.poisson(0.4 if label == 0 else 2.2)),
                "CreateService": float(random.poisson(0.02 if label == 0 else 0.7)),
                "CryptEncrypt": float(random.poisson(0.15 if label == 0 else 3.0)),
                "ShellExecute": float(random.poisson(0.2 if label == 0 else 1.2)),
                "GetAsyncKeyState": float(random.poisson(0.02 if label == 0 else 1.5)),
                "CreateFile": float(random.poisson(8 if label == 0 else 14)),
                "DeleteFile": float(random.poisson(0.15 if label == 0 else 2.5)),
                "unique_api_count": float(random.integers(8, 35) + label * 8),
                "total_api_calls": float(random.integers(20, 180) + label * 25),
            })
            features["network_activity"] = features["InternetOpenUrl"]
            features["registry_writes"] = features["RegSetValue"]
            features["file_deletions"] = features["DeleteFile"]
            features["label"] = label
            rows.append(features)
    return pd.DataFrame(rows, columns=DYNAMIC_FEATURES + ["label"])


def main() -> None:
    static_path = train_static(_static_demo_frame(), demo=True)
    dynamic_path = train_dynamic(_dynamic_demo_frame(), demo=True)
    print("Created SYNTHETIC DEMO models (not trained on EMBER or real captures):")
    print(f"  static:  {static_path}")
    print(f"  dynamic: {dynamic_path}")
    print("Use only to test the upload/UI flow. Do not use the scores for malware decisions.")


if __name__ == "__main__":
    main()
