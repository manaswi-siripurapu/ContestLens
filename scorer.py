"""
scorer.py — Weighted aggregation of feature scores into a final anomaly score.

Output is an "Anomaly Index" 0-10.
We deliberately do NOT call it a "probability of being fake."

Weights are tuned heuristically. Run cohort_test.py and evaluate.py
to validate and move toward data-backed weights in a future version.
"""

from __future__ import annotations

METHODOLOGY_VERSION = "v1.0-heuristic"
"""
Scoring methodology version. Increment when weights or signal logic changes.
  v1.0-heuristic : weights are reasoned, not empirically trained.
  v1.1-calibrated: (next) weights validated against labeled dataset.
To upgrade: run cohort_test.py + evaluate.py, adjust weights, bump version.
"""

FEATURE_WEIGHTS: dict[str, float] = {
    "Solve time vs field":            0.20,
    "Rating velocity":                0.10,
    "Contest solve speed":            0.14,
    "Ranking consistency":            0.11,
    "Percentile vs problems solved":  0.16,
    "Profile depth":                  0.12,
    "Hidden submission graph":        0.06,
    "Submission pattern entropy":     0.04,
    "Early contest performance":      0.06,
    "Language distribution":          0.01,
}

CONFIDENCE_MULTIPLIERS: dict[str, float] = {
    "high":   1.0,
    "medium": 0.7,
    "low":    0.0,
}

VERDICT_THRESHOLDS = [
    (8.0, "Very high",  "danger"),
    (6.0, "High",       "warning"),
    (4.0, "Moderate",   "caution"),
    (2.5, "Low",        "normal"),
    (1.0, "Normal",   "normal"),
]


def compute_anomaly_score(features: list[dict]) -> dict:
    """
    Aggregate feature scores into a final Anomaly Index 0-10.
    Low-confidence features are excluded (multiplier = 0).
    An interaction bonus fires when solve speed + profile depth both score >= 5.
    """
    weighted_sum   = 0.0
    total_weight   = 0.0
    feature_details = []

    for feat in features:
        label      = feat.get("label", "Unknown")
        raw_score  = float(feat.get("score", 0.0))
        confidence = feat.get("confidence", "low")

        base_weight  = FEATURE_WEIGHTS.get(label, 0.0)
        conf_mult    = CONFIDENCE_MULTIPLIERS.get(confidence, 0.0)
        eff_weight   = base_weight * conf_mult
        contribution = raw_score * eff_weight

        feature_details.append({
            "label":        label,
            "score":        round(raw_score, 2),
            "weight":       base_weight,
            "eff_weight":   round(eff_weight, 3),
            "contribution": round(contribution, 3),
            "confidence":   confidence,
            "description":  feat.get("description", ""),
        })

        weighted_sum += contribution
        total_weight += eff_weight

    if total_weight < 0.001:
        final_score  = 0.0
        data_quality = "insufficient"
    else:
        final_score  = min(10.0, weighted_sum / total_weight)
        if total_weight >= 0.6:
            data_quality = "good"
        elif total_weight >= 0.25:
            data_quality = "partial"
        else:
            data_quality = "insufficient"

    # Cross-feature interaction: fast solve + shallow profile amplify each other.
    # Only fires when both signals are independently suspicious (>= 5.0).
    # Max bonus: (10-5) * (10-5) * 0.04 = +1.0 point.
    by_label       = {f["label"]: f["score"] for f in feature_details}
    speed_score    = by_label.get("Contest solve speed", 0.0)
    depth_score    = by_label.get("Profile depth", 0.0)
    interaction_bonus = 0.0
    if speed_score >= 5.0 and depth_score >= 5.0:
        interaction_bonus = (speed_score - 5.0) * (depth_score - 5.0) * 0.04
        final_score = min(10.0, final_score + interaction_bonus)

    final_score = round(final_score, 2)

    verdict, severity = "Very low", "normal"
    for threshold, label, sev in VERDICT_THRESHOLDS:
        if final_score >= threshold:
            verdict, severity = label, sev
            break

    active_features = sum(
        1 for f in features
        if f.get("confidence") != "low" and f.get("score", 0) > 0
    )

    top_signals = sorted(
        [f for f in feature_details if f["eff_weight"] > 0],
        key=lambda x: -x["score"],
    )[:2]

    if top_signals and final_score >= 3.0:
        signal_names = " and ".join(f["label"].lower() for f in top_signals)
        summary = (
            f"Anomaly index {final_score}/10. "
            f"Strongest signals: {signal_names}."
        )
    elif final_score < 3.0:
        summary = (
            f"Anomaly index {final_score}/10 — "
            "profile looks consistent with a legitimate user."
        )
    else:
        summary = f"Anomaly index {final_score}/10."

    return {
        "score":             final_score,
        "verdict":           verdict,
        "severity":          severity,
        "feature_scores":    feature_details,
        "data_quality":      data_quality,
        "active_features":   active_features,
        "interaction_bonus": round(interaction_bonus, 3),
        "methodology":       METHODOLOGY_VERSION,
        "summary":           summary,
    }
