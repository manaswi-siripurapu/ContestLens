"""
scorer.py — Weighted aggregation of feature scores into a final anomaly score.

Output is an "Anomaly Index" 0-10.
We deliberately do NOT call it a "probability of being fake."

Weights are tuned so that high-signal features (solve speed, rating velocity)
dominate, while low-confidence features are discounted further.
"""

from __future__ import annotations

# Feature weights (must sum to 1.0 when all features have full confidence)
FEATURE_WEIGHTS: dict[str, float] = {
    "Solve time vs field":            0.20,
    "Rating velocity":                0.11,
    "Contest solve speed":            0.14,
    "Ranking consistency":            0.12,
    "Percentile vs problems solved":  0.16,
    "Profile depth":                  0.09,
    "Hidden submission graph":        0.07,
    "Submission pattern entropy":     0.04,
    "Early contest performance":      0.06,
    "Language distribution":          0.01,
}

CONFIDENCE_MULTIPLIERS: dict[str, float] = {
    "high":   1.0,
    "medium": 0.7,
    "low":    0.0,   # low-confidence features are fully excluded
}

# Thresholds for the final verdict label
VERDICT_THRESHOLDS = [
    (8.0, "Very high",  "danger"),
    (6.0, "High",       "warning"),
    (4.0, "Moderate",   "caution"),
    (2.5, "Low",        "normal"),
    (0.0, "Very low",   "normal"),
]


def compute_anomaly_score(features: list[dict]) -> dict:
    """
    Given a list of feature dicts (from features.compute_all_features),
    return:
      {
        "score":         float  0-10,
        "verdict":       str    ("Very high" | "High" | …),
        "severity":      str    ("danger" | "warning" | "caution" | "normal"),
        "feature_scores": list of {label, score, weight, contribution},
        "data_quality":  "good" | "partial" | "insufficient",
        "active_features": int,
        "summary":       str,
      }
    """
    weighted_sum = 0.0
    total_weight = 0.0
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

        weighted_sum  += contribution
        total_weight  += eff_weight

    # Normalise to 0-10
    if total_weight < 0.001:
        # No usable features at all
        final_score = 0.0
        data_quality = "insufficient"
    else:
        final_score  = min(10.0, (weighted_sum / total_weight))
        active_count = sum(1 for f in features if f.get("confidence") != "low")
        if total_weight >= 0.6:
            data_quality = "good"
        elif total_weight >= 0.25:
            data_quality = "partial"
        else:
            data_quality = "insufficient"

    # -----------------------------------------------------------------------
    # Cross-feature interaction: solve speed x profile depth
    #
    # Fast contest solves are MORE suspicious when the user has few practice
    # problems. Grinding 500+ problems explains fast solves — 20 problems does not.
    #
    # Bonus = (speed - 5) * (depth - 5) * 0.04
    # Max bonus at scores (10, 10): 5 * 5 * 0.04 = +1.0 point
    # Only fires when BOTH signals are >= 5.0 (both genuinely suspicious).
    # Intentionally modest — tips borderline cases, doesn't manufacture them.
    # -----------------------------------------------------------------------
    by_label       = {f["label"]: f["score"] for f in feature_details}
    speed_score    = by_label.get("Contest solve speed", 0.0)
    depth_score    = by_label.get("Profile depth", 0.0)
    interaction_bonus = 0.0
    if speed_score >= 5.0 and depth_score >= 5.0:
        interaction_bonus = (speed_score - 5.0) * (depth_score - 5.0) * 0.04
        final_score = min(10.0, final_score + interaction_bonus)

    final_score = round(final_score, 2)

    # Determine verdict
    verdict, severity = "Very low", "normal"
    for threshold, label, sev in VERDICT_THRESHOLDS:
        if final_score >= threshold:
            verdict, severity = label, sev
            break

    # Count active features
    active_features = sum(
        1 for f in features
        if f.get("confidence") != "low" and f.get("score", 0) > 0
    )

    # Build a one-line summary
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
        summary = f"Anomaly index {final_score}/10 — profile looks consistent with a legitimate user."
    else:
        summary = f"Anomaly index {final_score}/10."

    return {
        "score":            final_score,
        "verdict":          verdict,
        "severity":         severity,
        "feature_scores":   feature_details,
        "data_quality":     data_quality,
        "active_features":  active_features,
        "interaction_bonus": round(interaction_bonus, 3),
        "summary":          summary,
    }
