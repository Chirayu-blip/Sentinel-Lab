import math
import unittest

from models.fusion import dynamic_weight, fuse_scores


class FusionTests(unittest.TestCase):
    def test_entropy_threshold_gives_equal_weights(self):
        self.assertAlmostEqual(dynamic_weight(6.2), 0.5)

    def test_dynamic_weight_increases_with_entropy(self):
        self.assertLess(dynamic_weight(5.0), 0.5)
        self.assertGreater(dynamic_weight(7.0), 0.5)
        self.assertAlmostEqual(dynamic_weight(6.2), 1 / (1 + math.exp(0)))

    def test_missing_dynamic_score_is_static_only(self):
        result = fuse_scores(0.7)
        self.assertEqual(result["weights"], {"static": 1.0, "dynamic": 0.0})
        self.assertEqual(result["score"], 0.7)
        self.assertEqual(result["verdict"], "MALWARE")

    def test_dynamic_score_uses_entropy_weight(self):
        result = fuse_scores(0.2, 0.8, entropy=6.2)
        self.assertEqual(result["weights"], {"static": 0.5, "dynamic": 0.5})
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["verdict"], "SUSPICIOUS")

    def test_dynamic_score_requires_entropy(self):
        with self.assertRaises(ValueError):
            fuse_scores(0.2, 0.8)


if __name__ == "__main__":
    unittest.main()
