"""Evidence-grounded narrative generation for classifier results."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _local_explanation(result: dict) -> str:
    verdict = result["verdict"]
    score = result["score"]
    static_score = result["static"]["score"]
    dynamic = result.get("dynamic")
    lead = (
        f"The fused malware score is {score:.1%}, producing a {verdict} verdict. "
        f"The static model estimates {static_score:.1%}."
    )
    if dynamic is not None:
        lead += (
            f" The dynamic model estimates {dynamic['score']:.1%}; "
            f"fusion assigns {result['weights']['dynamic']:.1%} weight to dynamic evidence "
            f"at entropy {result['entropy']:.2f}."
        )
    else:
        lead += " No dynamic log was supplied, so this is a static-only result."

    explanations = []
    for modality in ("static", "dynamic"):
        evidence = result.get(modality)
        if not evidence:
            continue
        top = evidence.get("shap", [])[:3]
        for item in top:
            direction = "increases" if item["impact"] > 0 else "decreases"
            explanations.append(
                f"{item['feature']} ({item['value']:g}) {direction} the "
                f"{modality} model's malware score"
            )
    if explanations:
        lead += " Key model signals: " + "; ".join(explanations) + "."
    lead += " SHAP describes model influence, not proof of malicious behavior."
    return lead


def generate_explanation(result: dict) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _local_explanation(result), "local"

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    evidence = {
        key: result[key]
        for key in ("verdict", "score", "entropy", "weights", "static", "dynamic")
        if key in result
    }
    request_body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a defensive malware-triage analyst. Explain only the supplied "
                    "classifier result and SHAP evidence in plain language. The verdict and "
                    "score are authoritative: do not change them, infer unsupported behavior, "
                    "or claim SHAP proves intent. Mention uncertainty and recommend isolated "
                    "review for suspicious or malware results. Do not provide execution advice."
                ),
            },
            {
                "role": "user",
                "content": "Write a concise analyst summary from this JSON evidence:\n"
                + json.dumps(evidence, separators=(",", ":"), allow_nan=False),
            },
        ],
    }
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"].strip()
        if not text:
            raise RuntimeError("The configured GenAI provider returned an empty explanation")
        return text, "genai"
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GenAI explanation request failed: {exc}") from exc
