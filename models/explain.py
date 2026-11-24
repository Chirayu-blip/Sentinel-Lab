"""Generate per-sample SHAP contributions for tree classifiers."""

import numpy as np


def explain_tree_model(model, feature_names: list[str], values: dict[str, float], top_k: int = 8):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required to explain model predictions") from exc

    row = np.asarray([[float(values[name]) for name in feature_names]], dtype=float)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    if isinstance(shap_values, list):
        contributions = np.asarray(shap_values[-1])[0]
    else:
        array = np.asarray(shap_values)
        if array.ndim == 3:
            contributions = array[0, :, -1]
        else:
            contributions = array[0]
    order = np.argsort(np.abs(contributions))[::-1][:top_k]
    return [
        {"feature": feature_names[index], "value": float(row[0, index]),
         "impact": float(contributions[index])}
        for index in order
    ]
