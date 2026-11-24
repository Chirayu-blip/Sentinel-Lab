import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.parse_apimon import DYNAMIC_FEATURES
from data.pe_extractor import STATIC_FEATURES
from models.trainers import train_dynamic, train_static


def labeled_frame(features):
    rows = []
    for index in range(40):
        label = index % 2
        row = {feature: float(index % 7 + label) for feature in features}
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


class TrainerTests(unittest.TestCase):
    def test_static_and_dynamic_trainers_write_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("models.trainers.MODEL_DIR", Path(temporary_directory)):
                static_path = train_static(labeled_frame(STATIC_FEATURES))
                dynamic_path = train_dynamic(labeled_frame(DYNAMIC_FEATURES))
            self.assertTrue(static_path.is_file())
            self.assertTrue(dynamic_path.is_file())


if __name__ == "__main__":
    unittest.main()
