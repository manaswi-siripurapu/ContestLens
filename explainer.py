"""
explainer.py — Generate a human-readable explanation using Groq AI.

Uses llama-3.3-70b-versatile (fast + cheap on Groq) with a constrained prompt.
Falls back to a rule-based explanation if the API key is not set or fails.

To get a Groq API key: https://console.groq.com
Set env var:  export GROQ_API_KEY=gsk_...
"""

import os
from typing import Optional


def _build_prompt(username: str, score_result: dict, features: list[dict]) -> str:
    """Build the user-turn prompt."""
    lines = [
        f"LeetCode username: {username}",
        f"Anomaly index: {score_result['score']}/10  ({score_result['verdict']} anomaly level)",
        f"Data quality: {score_result['data_quality']}",
        "",
        "Feature breakdown:",
    ]
    for feat in features:
        conf = feat.get("confidence", "low")
        if conf == "low":
            continue
        lines.append(
            f"  • {feat['label']} — score {feat['score']}/10 "
            f"({conf} confidence): {feat['description']}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a neutral data analyst presenting findings from an automated LeetCode \
profile analysis tool. Your job is to explain what the data shows — not to \
accuse anyone of cheating.

Rules:
1. Write exactly 3–5 sentences.
2. Reference specific feature values (rating jumps, solve times, etc.) when they \
   are anomalous. Skip features that are normal.
3. Use hedged, analytical language: "the data shows", "this pattern is consistent \
   with", "statistically unusual", "worth further investigation".
4. NEVER use the words: cheat, cheater, fraud, fake, bot, automated, guilty, \
   dishonest, plagiarism.
5. If the anomaly score is below 3.0, lead with a statement that the profile \
   appears consistent with a legitimate user.
6. If data quality is "insufficient", note that the analysis is limited by \
   available data.
7. End with: "This analysis is automated and should not be used as definitive evidence."
"""

GROQ_MODEL = "llama-3.3-70b-versatile"

def generate_explanation(
    username: str,
    score_result: dict,
    features: list[dict],
    api_key: Optional[str] = None,
) -> str:
    """
    Returns a plain-text explanation string.
    Falls back to deterministic explanation if Groq is unavailable.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    if not key:
        return _fallback_explanation(username, score_result, features)

    try:
        from groq import Groq

        client = Groq(api_key=key)

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(username, score_result, features)},
            ],
            temperature=0.3,     # low temp = more factual, less hallucination
            max_tokens=300,
            top_p=1,
            stream=False,
        )

        text = completion.choices[0].message.content.strip()
        return text if text else _fallback_explanation(username, score_result, features)

    except ImportError:
        return (
            "[groq package not installed — run: pip install groq]\n"
            + _fallback_explanation(username, score_result, features)
        )
    except Exception as exc:
        return (
            f"[Explanation unavailable: {type(exc).__name__}]\n"
            + _fallback_explanation(username, score_result, features)
        )


def _fallback_explanation(username: str, score_result: dict, features: list[dict]) -> str:
    """Rule-based fallback — no API needed."""
    score   = score_result["score"]
    quality = score_result["data_quality"]

    if quality == "insufficient":
        return (
            f"The analysis of {username} could not be completed fully due to "
            "limited publicly available data. "
            "This profile has not attended enough contests, or the required data "
            "fields were unavailable. "
            "This analysis is automated and should not be used as definitive evidence."
        )

    top_features = sorted(
        [f for f in features if f.get("confidence") != "low" and f.get("score", 0) >= 4.0],
        key=lambda x: -x["score"],
    )[:3]

    if score < 3.0:
        base = (
            f"The profile for {username} appears broadly consistent with a legitimate user. "
            "No statistically unusual patterns were detected across the evaluated signals. "
        )
    elif score < 5.0:
        base = (
            f"The profile for {username} shows mild anomalies in {len(top_features)} "
            f"signal(s), resulting in an anomaly index of {score}/10. "
        )
    elif score < 7.5:
        base = (
            f"The profile for {username} shows statistically unusual patterns across "
            f"multiple signals, resulting in an anomaly index of {score}/10. "
        )
    else:
        base = (
            f"The profile for {username} exhibits multiple statistically anomalous "
            f"patterns (anomaly index {score}/10) that warrant closer examination. "
        )

    details = "".join(f"{f['description']}. " for f in top_features)
    caveat  = "This analysis is automated and should not be used as definitive evidence."
    return (base + details + caveat).strip()
