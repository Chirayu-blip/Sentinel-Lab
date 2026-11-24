import unittest

from sklearn.ensemble import RandomForestClassifier

from models.explain import explain_tree_model


class ShapExplanationTests(unittest.TestCase):
    def test_tree_explanation_returns_ranked_numeric_contributions(self):
        model = RandomForestClassifier(n_estimators=8, random_state=4)
        model.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 0, 1, 1])
        explanation = explain_tree_model(
            model, ["small", "large"], {"small": 2.0, "large": 2.0}
        )
        self.assertEqual(len(explanation), 2)
        self.assertTrue(all(isinstance(item["impact"], float) for item in explanation))
        self.assertEqual({item["feature"] for item in explanation}, {"small", "large"})


if __name__ == "__main__":
    unittest.main()
