"""Convert EMBER 2018 raw JSONL files into compact feature rows."""

import argparse
import json
from pathlib import Path

import pandas as pd

from data.pe_extractor import STATIC_FEATURES, features_from_ember


def load_ember(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                label = record.get("label")
                if label not in (0, 1, 0.0, 1.0, "0", "1"):
                    continue
                row = features_from_ember(record)
                row["label"] = int(label)
                rows.append(row)
    if not rows:
        raise ValueError("No labeled EMBER records (label 0 or 1) were found")
    return pd.DataFrame(rows, columns=STATIC_FEATURES + ["label"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="EMBER train_features_*.jsonl paths")
    parser.add_argument("--output", default="data/ember_features.csv")
    args = parser.parse_args()
    frame = load_ember(args.inputs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} labeled rows to {args.output}")


if __name__ == "__main__":
    main()
