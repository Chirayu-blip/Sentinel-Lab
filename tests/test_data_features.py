import io
import tempfile
import unittest
from pathlib import Path

from data.ember_loader import load_ember
from data.parse_apimon import DYNAMIC_FEATURES, load_labeled_captures, parse_apimon_csv
from data.pe_extractor import STATIC_FEATURES, features_from_ember


class DataFeatureTests(unittest.TestCase):
    def test_api_monitor_parser_counts_calls_and_aliases(self):
        csv_data = io.StringIO(
            "API Name,Result\n"
            "KERNEL32.dll!VirtualAlloc,OK\n"
            "VirtualAlloc,OK\n"
            "WriteProcessMemory,OK\n"
            "InternetOpenUrlA,OK\n"
            "DeleteFileW,OK\n"
        )
        features = parse_apimon_csv(csv_data)
        self.assertEqual(list(features), DYNAMIC_FEATURES)
        self.assertEqual(features["VirtualAlloc"], 2)
        self.assertEqual(features["WriteProcessMemory"], 1)
        self.assertEqual(features["network_activity"], 1)
        self.assertEqual(features["file_deletions"], 1)
        self.assertEqual(features["total_api_calls"], 5)

    def test_ember_mapping_has_expected_schema_and_histogram_entropy(self):
        record = {
            "label": 1,
            "general": {"size": 20, "vsize": 30},
            "histogram": [1, 1],
            "imports": {"KERNEL32.dll": ["VirtualAlloc", "CreateFileW"]},
            "exports": ["ExportedSymbol"],
            "sections": {"sections": [
                {"name": ".text", "entropy": 5.0, "vsize": 12, "size": 8}
            ]},
            "datadirectories": [{"name": "IMAGE_DIRECTORY_ENTRY_DEBUG"}],
            "header": {"timestamp": 1600000000},
        }
        features = features_from_ember(record)
        self.assertEqual(list(features), STATIC_FEATURES)
        self.assertEqual(features["entropy"], 1.0)
        self.assertEqual(features["suspicious_imports"], 1.0)
        self.assertEqual(features["has_debug"], 1.0)
        self.assertEqual(features["file_size"], 20.0)

    def test_capture_manifest_builds_labeled_dynamic_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "capture.csv").write_text(
                "API Name\nCreateFileW\nDeleteFileW\n", encoding="utf-8"
            )
            (root / "manifest.csv").write_text(
                "capture_path,label\ncapture.csv,1\n", encoding="utf-8"
            )
            frame = load_labeled_captures(str(root / "manifest.csv"))
        self.assertEqual(list(frame.columns), DYNAMIC_FEATURES + ["label"])
        self.assertEqual(frame.iloc[0]["label"], 1)
        self.assertEqual(frame.iloc[0]["CreateFile"], 1.0)
        self.assertEqual(frame.iloc[0]["DeleteFile"], 1.0)


if __name__ == "__main__":
    unittest.main()
