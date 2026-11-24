"""Entropy-adaptive fusion and verdict mapping."""

import math

MALWARE_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.40
DEFAULT_ENTROPY_THRESHOLD = 6.2


def dynamic_weight(entropy: float, threshold: float = DEFAULT_ENTROPY_THRESHOLD) -> float:
    exponent = max(-60.0, min(60.0, (float(entropy) - threshold) * 2.0))
    return 1.0 / (1.0 + math.exp(-exponent))


def fuse_scores(
    static_score: float,
    dynamic_score: float | None = None,
    entropy: float | None = None,
    threshold: float = DEFAULT_ENTROPY_THRESHOLD,
) -> dict:
    if dynamic_score is None:
        dynamic_weight_value = 0.0
    else:
        if entropy is None:
            raise ValueError("Entropy is required when a dynamic score is provided")
        dynamic_weight_value = dynamic_weight(entropy, threshold)
    static_weight_value = 1.0 - dynamic_weight_value
    score = static_weight_value * float(static_score)
    if dynamic_score is not None:
        score += dynamic_weight_value * float(dynamic_score)
    verdict = (
        "MALWARE" if score >= MALWARE_THRESHOLD
        else "SUSPICIOUS" if score >= SUSPICIOUS_THRESHOLD
        else "BENIGN"
    )
    return {
        "score": float(score),
        "verdict": verdict,
        "weights": {"static": static_weight_value, "dynamic": dynamic_weight_value},
        "threshold": threshold,
    }
