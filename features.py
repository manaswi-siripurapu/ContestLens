"""
features.py — Convert raw LeetCode API data into anomaly signals.

Each feature returns a dict:
    {
      "score":       float  0-10  (10 = maximally suspicious),
      "value":       <raw metric>,
      "label":       str   (human-readable metric name),
      "description": str   (what was found),
      "confidence":  "high" | "medium" | "low",
    }

A feature returns confidence="low" when there is insufficient data.
The scorer weights low-confidence features down automatically.
"""

import math
import statistics
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0

def _safe_stdev(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0

def _clamp(val: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, val))

def _shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy in bits. Returns 0 for empty / all-zero input."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


# ---------------------------------------------------------------------------
# Feature 1 — Rating Velocity
# ---------------------------------------------------------------------------

def feature_rating_velocity(contests: list[dict]) -> dict:
    """
    Single-contest rating gain.
    Bots copying solutions often show monster jumps (+300 to +600) in one go.
    Normal elite grinder: +50 to +150 per contest on average.
    """
    attended = [c for c in contests if c.get("attended", True)]

    if len(attended) < 2:
        return {
            "score": 0.0,
            "value": None,
            "label": "Rating velocity",
            "description": "Not enough contest data (need ≥ 2 contests).",
            "confidence": "low",
        }

    deltas = [
        attended[i]["rating"] - attended[i - 1]["rating"]
        for i in range(1, len(attended))
    ]
    max_gain = max(deltas)
    avg_gain = _safe_mean(deltas)

    if max_gain >= 500:
        score = _clamp(8.0 + (max_gain - 400) / 200)
    elif max_gain >= 350:
        score = _clamp(5.0 + (max_gain - 250) / 100)
    elif max_gain >= 150:
        score = _clamp(2.0 + (max_gain - 150) / 100)
    else:
        score = _clamp(max_gain / 150 * 2.0)

    return {
        "score": round(score, 2),
        "value": round(max_gain, 1),
        "avg_gain": round(avg_gain, 1),
        "all_deltas": [round(d, 1) for d in deltas],
        "label": "Rating velocity",
        "description": (
            f"Max single-contest rating gain: {max_gain:+.0f}  "
            f"(avg {avg_gain:+.1f} over {len(deltas)} contests)"
        ),
        "confidence": "high" if len(attended) >= 5 else "medium",
    }


# ---------------------------------------------------------------------------
# Feature 2 — Early-Contest Performance
# ---------------------------------------------------------------------------

def feature_early_performance(contests: list[dict]) -> dict:
    """
    How did they do in their first 3 contests?
    Real humans usually rank mediocre at first; bots often go straight to top 1%.
    """
    attended = [c for c in contests if c.get("attended", True)]

    if len(attended) < 4:
        return {
            "score": 0.0,
            "value": None,
            "label": "Early contest performance",
            "description": "Fewer than 3 contests attended — cannot evaluate.",
            "confidence": "low",
        }

    first_three = attended[:4]
    ranks = [c["ranking"] for c in first_three if c.get("ranking", 0) > 0]
    if not ranks:
        return {
            "score": 0.0,
            "value": None,
            "label": "Early contest performance",
            "description": "Ranking data unavailable for early contests.",
            "confidence": "low",
        }

    avg_early_rank = _safe_mean(ranks)

    if avg_early_rank < 100:
        score = 6.0
        desc = (
            f"Avg rank in first 3 contests: {avg_early_rank:.0f} "
            f"— elite-level debut (legitimate for prodigies; also a bot signal)"
        )
    elif avg_early_rank < 250:
        score = 4.5
        desc = f"Avg rank in first 3 contests: {avg_early_rank:.0f} — very strong debut"
    elif avg_early_rank < 1000:
        score = 3.5
        desc = f"Avg rank in first 3 contests: {avg_early_rank:.0f} — solid debut"
    else:
        score = 0.5
        desc = f"Avg rank in first 3 contests: {avg_early_rank:.0f} — typical newcomer curve"

    return {
        "score": round(score, 2),
        "value": round(avg_early_rank, 0),
        "label": "Early contest performance",
        "description": desc,
        "confidence": "medium",
    }


# ---------------------------------------------------------------------------
# Feature 3 — Solve Speed
# ---------------------------------------------------------------------------

def feature_solve_speed(contests: list[dict], profile: dict = None) -> dict:
    """
    finishTimeInSeconds: how long (from contest start) until all problems submitted.
    Sub-15-minute completion of a 4-problem contest = extremely suspicious.
    Score is gated by total_solved — a heavy grinder (800+) explaining
    fast clears is plausible; someone with 30 problems is not.
    """
    total_solved = (profile or {}).get("problems", {}).get("All", 0)
    attended = [c for c in contests if c.get("attended", True)]
    usable = [
        c for c in attended
        if c.get("finish_seconds") is not None
        and c.get("finish_seconds", 0) > 0
        and c.get("problems_solved", 0) > 0
    ]

    if not usable:
        return {
            "score": 0.0,
            "value": None,
            "label": "Contest solve speed",
            "description": "No finish-time data available for this profile.",
            "confidence": "low",
        }

    full_solves = [
        c for c in usable
        if c.get("problems_solved", 0) >= c.get("total_problems", 4)
    ]

    if not full_solves:
        speeds = [
            c["finish_seconds"] / max(c["problems_solved"], 1)
            for c in usable
        ]
        min_per_problem = min(speeds) / 60
        count_used = len(usable)
        full_clear = False
    else:
        speeds = [c["finish_seconds"] for c in full_solves]
        min_per_problem = min(speeds) / 60
        count_used = len(full_solves)
        full_clear = True

    if full_clear:
        if min_per_problem < 15 and total_solved < 800:
            score = 9.0
            desc = f"Fastest full contest clear: {min_per_problem:.1f} min — superhuman speed"
        elif min_per_problem < 20 and total_solved < 400:
            score = 7.5
            desc = f"Fastest full contest clear: {min_per_problem:.1f} min — extremely fast"
        elif min_per_problem < 40:
            score = 3.5
            desc = f"Fastest full contest clear: {min_per_problem:.1f} min — unusually fast"
        elif min_per_problem < 60:
            score = 1.0
            desc = f"Fastest full contest clear: {min_per_problem:.1f} min — fast but plausible"
        else:
            score = 0.2
            desc = f"Fastest full contest clear: {min_per_problem:.1f} min — normal range"
    else:
        if min_per_problem < 5:
            score = 7.0
            desc = f"Min time per problem: {min_per_problem:.1f} min — very fast per problem"
        elif min_per_problem < 10:
            score = 3.5
            desc = f"Min time per problem: {min_per_problem:.1f} min — fast per problem"
        else:
            score = 1.0
            desc = f"Min time per problem: {min_per_problem:.1f} min — normal"

    return {
        "score": round(score, 2),
        "value": round(min_per_problem, 1),
        "count_used": count_used,
        "full_clear": full_clear,
        "label": "Contest solve speed",
        "description": desc,
        "confidence": "high" if count_used >= 3 else "medium",
    }


# ---------------------------------------------------------------------------
# Feature 4 — Profile Depth vs Rating
# ---------------------------------------------------------------------------

def feature_profile_depth(profile: dict, contests: list[dict]) -> dict:
    """
    Compare lifetime problem count against contest rating.
    A 2500-rated coder with only 30 problems solved is a red flag.
    """
    total_solved   = profile.get("problems", {}).get("All", 0)
    hard_solved    = profile.get("problems", {}).get("Hard", 0)
    contest_rating = profile.get("contest_rating", 0.0)

    if contest_rating < 1500:
        return {
            "score": 0.0,
            "value": total_solved,
            "label": "Profile depth",
            "description": f"Contest rating {contest_rating:.0f} — too low to evaluate depth anomaly.",
            "confidence": "low",
        }

    expected = {
        1500: 50,
        1800: 150,
        2000: 250,
        2200: 400,
        2400: 550,
        2600: 700,
    }
    threshold = 80
    for rating_floor, min_problems in sorted(expected.items()):
        if contest_rating >= rating_floor:
            threshold = min_problems

    if total_solved <= 60:
        score = 10.0
        desc = f"{total_solved} problems solved at rating {contest_rating:.0f} — extremely shallow"
    elif total_solved <= 100:
        score = 9.0
        desc = f"{total_solved} problems solved at rating {contest_rating:.0f} — extremely shallow"
    elif total_solved < threshold * 0.3:
        score = 8.0
        desc = (
            f"Only {total_solved} problems solved at rating {contest_rating:.0f} "
            f"(expected ≥ {threshold})"
        )
    elif total_solved < threshold * 0.6:
        score = 5.0
        desc = (
            f"{total_solved} problems solved at rating {contest_rating:.0f} "
            f"— significantly below expected ({threshold}+)"
        )
    elif total_solved < threshold:
        score = 2.5
        desc = f"{total_solved} problems solved — slightly below expected {threshold}+"
    else:
        score = 0.0
        desc = f"{total_solved} problems solved — consistent with rating {contest_rating:.0f}"

    if hard_solved > 40 and score > 3:
        score = max(0.5, score - 2.0)
        desc += f" (mitigated: {hard_solved} hards solved)"

    return {
        "score": round(_clamp(score), 2),
        "value": total_solved,
        "hard_solved": hard_solved,
        "expected_min": threshold,
        "label": "Profile depth",
        "description": desc,
        "confidence": "medium",
    }


# ---------------------------------------------------------------------------
# Feature 5 — Submission Pattern Entropy
# ---------------------------------------------------------------------------

def feature_submission_entropy(profile: dict) -> dict:
    """
    Day-of-week Shannon entropy of submission activity.
    Very low entropy = cron-job-like schedule.
    Very high (perfectly uniform) entropy = also robotic.
    """
    calendar: dict = profile.get("calendar", {})

    if not calendar:
        return {
            "score": 0.0,
            "value": None,
            "label": "Submission pattern entropy",
            "description": "No submission calendar data available.",
            "confidence": "low",
        }

    dow_counts = [0] * 7
    total_entries = 0
    for ts_str, count in calendar.items():
        try:
            ts = int(ts_str)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            dow_counts[dt.weekday()] += int(count)
            total_entries += int(count)
        except (ValueError, OSError, OverflowError):
            continue

    if total_entries < 30:
        return {
            "score": 0.0,
            "value": None,
            "label": "Submission pattern entropy",
            "description": f"Too few submissions ({total_entries}) for reliable entropy analysis.",
            "confidence": "low",
        }

    entropy = _shannon_entropy(dow_counts)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dominant_day = days[dow_counts.index(max(dow_counts))]
    dominant_pct = max(dow_counts) / total_entries * 100

    if entropy < 1.0:
        score = 7.0
        desc = (
            f"Very low day-of-week entropy ({entropy:.2f} bits) — "
            f"{dominant_pct:.0f}% of submissions on {dominant_day}s. "
            "Highly repetitive schedule."
        )
    elif entropy < 1.5:
        score = 4.0
        desc = (
            f"Low day-of-week entropy ({entropy:.2f} bits) — "
            f"submissions heavily concentrated on {dominant_day}s."
        )
    elif entropy > 3.0:
        score = 3.0
        desc = (
            f"Suspiciously uniform entropy ({entropy:.2f} bits) — "
            "submissions spread almost perfectly across all 7 days."
        )
    elif entropy > 2.5:
        score = 1.5
        desc = f"Slightly high entropy ({entropy:.2f} bits) — unusually uniform spread."
    else:
        score = 0.0
        desc = f"Normal entropy ({entropy:.2f} bits) — typical human submission pattern."

    return {
        "score": round(score, 2),
        "value": round(entropy, 3),
        "dow_counts": dow_counts,
        "dow_labels": days,
        "total_submissions": total_entries,
        "label": "Submission pattern entropy",
        "description": desc,
        "confidence": "high" if total_entries >= 100 else "medium",
    }


# ---------------------------------------------------------------------------
# Feature 6 — Contest Ranking Consistency
# ---------------------------------------------------------------------------

def feature_ranking_consistency(contests: list[dict]) -> dict:
    """
    Variance in rank across contests.
    Suspiciously consistent top-N finishes across many contests is a red flag.
    Real coders have off days.
    """
    attended = [c for c in contests if c.get("attended", True)]

    if len(attended) < 4:
        return {
            "score": 0.0,
            "value": None,
            "label": "Ranking consistency",
            "description": "Fewer than 4 contests — consistency analysis skipped.",
            "confidence": "low",
        }

    ranks = [c["ranking"] for c in attended if c.get("ranking", 0) > 0]
    if not ranks:
        return {
            "score": 0.0,
            "value": None,
            "label": "Ranking consistency",
            "description": "No valid ranking data found.",
            "confidence": "low",
        }

    top_5_count  = sum(1 for r in ranks if r <= 5)
    top_50_count = sum(1 for r in ranks if r <= 50)
    cv = _safe_stdev(ranks) / _safe_mean(ranks) if _safe_mean(ranks) > 0 else 0

    score = 0.0
    details = []

    if top_5_count >= 3 and len(ranks) < 20:
        score += 4.0
        details.append(f"Ranked top 5 in {top_5_count} contests")
    elif top_5_count >= 2 and len(ranks) < 40:
        score += 2.0
        details.append(f"Ranked top 5 in {top_5_count} contest(s)")

    if len(ranks) >= 5:
        if cv < 0.15 and _safe_mean(ranks) < 500:
            score += 3.5
            details.append(f"Suspiciously low ranking variance (CV={cv:.2f})")
        elif cv < 0.25 and _safe_mean(ranks) < 1000:
            score += 1.5
            details.append(f"Very consistent rankings (CV={cv:.2f})")

    score = _clamp(score)
    desc = "; ".join(details) if details else (
        f"Normal ranking variance (CV={cv:.2f}, avg rank {_safe_mean(ranks):.0f})"
    )

    return {
        "score": round(score, 2),
        "value": round(_safe_mean(ranks), 0),
        "cv": round(cv, 3),
        "top_5_count": top_5_count,
        "top_50_count": top_50_count,
        "all_ranks": ranks,
        "label": "Ranking consistency",
        "description": desc,
        "confidence": "high" if len(ranks) >= 8 else "medium",
    }


# ---------------------------------------------------------------------------
# Feature 7 — Language Distribution
# ---------------------------------------------------------------------------

def feature_language_switching(profile: dict) -> dict:
    """
    Near-zero weight signal (1%). Kept for completeness but scores are
    zeroed out — evaluation showed negative separation (-0.89) meaning
    legitimate users triggered this more than suspicious ones.
    """
    langs: dict[str, int] = profile.get("languages", {})

    if not langs or sum(langs.values()) < 50:
        return {
            "score": 0.0,
            "value": None,
            "label": "Language distribution",
            "description": "Insufficient problem data to evaluate language patterns.",
            "confidence": "low",
        }

    total = sum(langs.values())
    sorted_langs = sorted(langs.items(), key=lambda x: -x[1])
    top_lang, top_count = sorted_langs[0]
    top_pct = top_count / total * 100

    if len(sorted_langs) >= 2:
        second_lang, second_count = sorted_langs[1]
        second_pct = second_count / total * 100
    else:
        second_lang, second_pct = None, 0

    if top_pct < 40 and second_pct > 30:
        score = 0.0
        desc = f"Language split: {top_lang} ({top_pct:.0f}%) / {second_lang} ({second_pct:.0f}%) — near-zero weight signal"
    elif top_pct < 55:
        score = 0.0
        desc = f"Two primary languages ({top_lang} {top_pct:.0f}%) — near-zero weight signal"
    else:
        score = 0.0
        desc = f"Dominant language: {top_lang} ({top_pct:.0f}%) — normal"

    return {
        "score": round(score, 2),
        "value": top_pct,
        "top_language": top_lang,
        "language_breakdown": sorted_langs[:5],
        "label": "Language distribution",
        "description": desc,
        "confidence": "medium",
    }


# ---------------------------------------------------------------------------
# Feature 8 — Hidden Submission Graph
# ---------------------------------------------------------------------------

def feature_hidden_graph(profile: dict, contests: list[dict]) -> dict:
    """
    Transparency indicator — kept at low-to-medium weight (7%).

    IMPORTANT: Hiding a submission graph is a legitimate privacy choice.
    This signal does NOT penalise privacy. It flags a specific combination:
    an account performing at elite contest level with no visible daily practice
    AND a shallow problem history. That combination is statistically unusual.

    This signal should never be the sole or primary reason for a high score.
    It is a weak corroborating indicator — meaningful only alongside other signals.

    Scoring:
      - Hidden + high rating (≥ 2000) + ≥ 5 contests + < 400 problems: 8.0
        (shallow practice + hidden graph + elite performance = harder to explain)
      - Hidden + moderate activity: 4.5
      - Hidden + 50+ problems: 1.0 (likely privacy preference)
      - Hidden + low activity: 0.0 (almost certainly just privacy)
    """
    calendar_hidden = profile.get("calendar_hidden", False)
    contest_rating  = profile.get("contest_rating", 0.0)
    attended_count  = profile.get("attended_count", 0)
    total_solved    = profile.get("problems", {}).get("All", 0)

    if not calendar_hidden:
        return {
            "score":       0.0,
            "value":       False,
            "label":       "Hidden submission graph",
            "description": "Submission graph is visible — no signal.",
            "confidence":  "high" if profile.get("calendar") else "low",
        }

    if contest_rating >= 2000 and attended_count >= 5 and total_solved < 400:
        score = 8.0
        desc = (
            f"Submission graph hidden. Rating {contest_rating:.0f}, "
            f"{attended_count} contests, only {total_solved} problems solved — "
            "elite performance with no visible practice trail."
        )
    elif contest_rating >= 1500 or attended_count >= 3:
        score = 2.0
        desc = (
            f"Submission graph hidden. Active contest participant "
            f"(rating {contest_rating:.0f}, {attended_count} contests) "
            "with no visible practice history."
        )
    elif total_solved > 50:
        score = 1.0
        desc = (
            f"Submission graph hidden despite {total_solved} problems solved. "
            "Likely a privacy preference — low weight signal."
        )
    else:
        score = 0.0
        desc = (
            "Submission graph hidden. Very low activity — "
            "almost certainly a privacy preference, not an anomaly."
        )

    return {
        "score":       round(score, 2),
        "value":       True,
        "label":       "Hidden submission graph",
        "description": desc,
        "confidence":  "medium",
    }


# ---------------------------------------------------------------------------
# Feature 9 — Percentile vs Problems Solved
# ---------------------------------------------------------------------------

def feature_percentile_vs_problems(profile: dict) -> dict:
    """
    Directly relates contest top-percentile to total problems solved.

    Top percentile is harder to inflate than rating — it reflects where you
    stand against every human who competed that specific week. Being top 1%
    globally with 24 problems solved is a near-impossible combination.

    Bands (percentile ceiling → expected minimum problems):
      Top 1.5% → ≥ 700  (refined from community analysis)
      Top 5%   → ≥ 200
      Top 10%  → ≥ 100
      Top 25%  → ≥ 50
    """
    top_pct      = profile.get("top_percentage", 100.0)
    total_solved = profile.get("problems", {}).get("All", 0)
    attended     = profile.get("attended_count", 0)

    if attended < 3 or top_pct is None:
        return {
            "score":       0.0,
            "value":       None,
            "label":       "Percentile vs problems solved",
            "description": "Fewer than 3 contests — percentile not yet meaningful.",
            "confidence":  "low",
        }

    if top_pct >= 99.0:
        return {
            "score":       0.0,
            "value":       None,
            "label":       "Percentile vs problems solved",
            "description": "No percentile data available.",
            "confidence":  "low",
        }

    bands = [
        (1.5,  700),
        (5.0,  200),
        (10.0, 100),
        (25.0,  50),
    ]

    expected_min = None
    band_label   = None
    for ceiling, min_problems in bands:
        if top_pct <= ceiling:
            expected_min = min_problems
            band_label   = f"top {ceiling:.0f}%"
            break

    if expected_min is None:
        return {
            "score":       0.0,
            "value":       top_pct,
            "label":       "Percentile vs problems solved",
            "description": f"Top percentile {top_pct:.1f}% — outside suspicious range.",
            "confidence":  "medium",
        }

    if total_solved == 0:
        deficit_ratio = 1.0
    else:
        deficit_ratio = max(0.0, 1.0 - (total_solved / expected_min))

    if total_solved <= 50:
        score = 10.0
        desc  = (
            f"Only {total_solved} problems solved but ranked {band_label} globally "
            f"({top_pct:.1f}%). Organically impossible."
        )
    elif deficit_ratio >= 0.85:
        score = 9.0
        desc  = (
            f"Only {total_solved} problems solved while ranked {band_label} "
            f"({top_pct:.1f}%). Expected ≥ {expected_min} — "
            f"{int(deficit_ratio*100)}% below minimum."
        )
    elif deficit_ratio >= 0.65:
        score = 6.5
        desc  = (
            f"{total_solved} problems solved at {band_label} "
            f"({top_pct:.1f}%). Expected ≥ {expected_min} — "
            f"significantly below expected depth."
        )
    elif deficit_ratio >= 0.40:
        score = 4.0
        desc  = (
            f"{total_solved} problems solved at {band_label} "
            f"({top_pct:.1f}%). Somewhat below expected {expected_min}+."
        )
    elif deficit_ratio > 0:
        score = 1.5
        desc  = (
            f"{total_solved} problems at {band_label} — "
            f"slightly below expected {expected_min}+, within normal range."
        )
    else:
        score = 0.0
        desc  = (
            f"{total_solved} problems solved at {band_label} "
            f"({top_pct:.1f}%) — consistent with expected depth."
        )

    return {
        "score":        round(_clamp(score), 2),
        "value":        top_pct,
        "total_solved": total_solved,
        "expected_min": expected_min,
        "deficit_pct":  round(deficit_ratio * 100, 1),
        "band":         band_label,
        "label":        "Percentile vs problems solved",
        "description":  desc,
        "confidence":   "high" if attended >= 5 else "medium",
    }


# ---------------------------------------------------------------------------
# Master feature extractor
# ---------------------------------------------------------------------------

def compute_all_features(
    profile: dict,
    contests: list[dict],
    username: str = "",
    include_contest_timing: bool = False,
) -> list[dict]:
    """
    Run all feature functions and return the full list.
    Pass username + include_contest_timing=True to enable the
    live contest ranking API check (slower — ~15s per lookup).
    """
    feats = [
        feature_rating_velocity(contests),
        feature_early_performance(contests),
        feature_solve_speed(contests, profile),
        feature_profile_depth(profile, contests),
        feature_percentile_vs_problems(profile),
        feature_submission_entropy(profile),
        feature_ranking_consistency(contests),
        feature_language_switching(profile),
        feature_hidden_graph(profile, contests),
    ]

    if include_contest_timing and username and contests:
        try:
            from feature_contest_timing import feature_solve_time_vs_field
            feats.append(feature_solve_time_vs_field(username, contests))
        except Exception:
            pass

    return feats
