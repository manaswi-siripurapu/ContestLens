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
    """Shannon entropy in bits.  Returns 0 for empty / all-zero input."""
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

    # Scoring thresholds (rating points gained in a single contest)
    # > 400 → near-certain anomaly; > 250 → suspicious; < 100 → normal
    if max_gain >= 400:
        score = _clamp(8.0 + (max_gain - 400) / 200)
    elif max_gain >= 250:
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

    # Ranking percentile: lower number = better rank = more suspicious for a newcomer
    # We only flag if we have ranking AND a total-participants reference
    # Fall back to absolute ranking if percentile unavailable
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
    final_rating = attended[-1]["rating"]

    # Heuristic: if first 3 avg rank < 500 AND they eventually reached high rating
    # that's perfectly fine for a prodigy — lower weight on this feature
    if avg_early_rank < 100:
        score = 8.0
        desc = (
            f"Avg rank in first 3 contests: {avg_early_rank:.0f} "
            f"— elite-level debut (legitimate for prodigies; also a bot signal)"
        )
    elif avg_early_rank < 250:
        score = 6.5
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
    """
    attended = [c for c in contests if c.get("attended", True)]
    usable = [
        c for c in attended
        if c.get("finish_seconds") is not None
        and c.get("finish_seconds", 0) > 0
        and c.get("problems_solved", 0) > 0
    ]

    total_solved = profile.get("problems", {}).get("All", 0)

    if not usable:
        return {
            "score": 0.0,
            "value": None,
            "label": "Contest solve speed",
            "description": "No finish-time data available for this profile.",
            "confidence": "low",
        }

    # Only examine contests where all problems were solved
    full_solves = [
        c for c in usable
        if c.get("problems_solved", 0) >= c.get("total_problems", 4)
    ]

    if not full_solves:
        # Partial solves: check fastest relative to problems solved
        speeds = [
            c["finish_seconds"] / max(c["problems_solved"], 1)
            for c in usable
        ]
        min_per_problem = min(speeds) / 60  # convert to minutes
        count_used = len(usable)
        full_clear = False
    else:
        speeds = [c["finish_seconds"] for c in full_solves]
        min_per_problem = min(speeds) / 60  # total minutes for full clear
        count_used = len(full_solves)
        full_clear = True

    # Thresholds for full 4-problem clear (typical LeetCode contest)
    # < 15 min full clear → extremely suspicious (world-record territory)
    # < 25 min → very suspicious
    # < 40 min → mildly suspicious
    # > 60 min → normal
    if full_clear:
        if min_per_problem < 15 and total_solved < 800:
            score = 9.0
            desc = f"Fastest full contest clear: {min_per_problem:.1f} min — superhuman speed"
        elif min_per_problem < 20 and total_solved <400:
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
        # Per-problem speed
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
    (But a new ICPC medalist might legitimately have few LeetCode problems.)
    """
    total_solved = profile.get("problems", {}).get("All", 0)
    hard_solved  = profile.get("problems", {}).get("Hard", 0)
    contest_rating = profile.get("contest_rating", 0.0)

    if contest_rating < 1500:
        # Unranked or beginner — not enough signal
        return {
            "score": 0.0,
            "value": total_solved,
            "label": "Profile depth",
            "description": f"Contest rating {contest_rating:.0f} — too low to evaluate depth anomaly.",
            "confidence": "low",
        }

    # Expected minimum problems for a given rating bracket
    # (rough community heuristic, not official)
    expected = {
        1500: 50,
        1800: 150,
        2000: 250,
        2200: 350,
        2400: 450,
        2600: 700,
    }
    threshold = 80
    for rating_floor, min_problems in sorted(expected.items()):
        if contest_rating >= rating_floor:
            threshold = min_problems

    if total_solved <= 70:
        score = 9.0
        desc = f"0 problems solved, but contest rating {contest_rating:.0f} — impossible organically"
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

    # Hard-problem bonus: solving many hards is harder to fake
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
    Bots tend to submit uniformly (7-day spread) OR on suspiciously narrow days.
    Humans have moderate entropy with weekend/weeknight peaks.

    Max theoretical entropy for 7 days = log2(7) ≈ 2.81 bits.
    We flag both extremes:
      - Very low (<1.0 bits): always submits on 1-2 days → bot or script cron job
      - Very high and uniform (>2.75 bits): perfectly uniform over all 7 days → bot
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

    # Map timestamps to day of week (0=Mon … 6=Sun)
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
    max_entropy = math.log2(7)  # ≈ 2.807

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dominant_day = days[dow_counts.index(max(dow_counts))]
    dominant_pct = max(dow_counts) / total_entries * 100

    # Flag criteria
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
        desc = (
            f"Normal entropy ({entropy:.2f} bits) — "
            f"typical human submission pattern."
        )

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
    Look at the variance in rank percentile across contests.
    Real coders have moderate variance (off days, hard contests, etc).
    Bots / paid solvers tend to be suspiciously consistent in the top 0.x%.

    We also check for impossible rankings:
    ranking=1 or ranking < 5 multiple times → extreme outlier.
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

    # Multiple rank=1 or rank<5 finishes
    if top_5_count >= 3 and len(ranks) < 20:
        score += 4.0
        details.append(f"Ranked top 5 in {top_5_count} contests")
    elif top_5_count >= 2 and len(ranks) < 40:
        score += 2.0
        details.append(f"Ranked top 5 in {top_5_count} contest(s)")

    # Low coefficient of variation = robot-like consistency
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
# Feature 7 — Language Switching
# ---------------------------------------------------------------------------

def feature_language_switching(profile: dict) -> dict:
    """
    Bots / cheaters sometimes solve using completely different languages in
    different periods because they're copying from multiple solvers.
    We can only check the overall language distribution — not per-contest.
    Suspicious: top-2 languages have nearly equal problem counts (split personality).
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

    # Normal: one dominant language (>70%).
    # Suspicious: two languages neck-and-neck at 40-60% each (potential split account).
    if top_pct < 40 and second_pct > 30:
        score = 5.0
        desc = (
            f"Language split: {top_lang} ({top_pct:.0f}%) / "
            f"{second_lang} ({second_pct:.0f}%) — abnormally even split"
        )
    elif top_pct < 55:
        score = 2.5
        desc = f"Two primary languages used roughly equally ({top_lang} {top_pct:.0f}%)"
    else:
        score = 0.5
        desc = f"Dominant language: {top_lang} ({top_pct:.0f}% of problems) — normal"

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
    A hidden submission graph is a mild but meaningful signal — especially
    when combined with high contest rating.

    Reasoning:
      - Legitimate high-rated users rarely hide their graphs because a rich
        submission history is a badge of honour in competitive programming.
      - Accounts used purely for contest manipulation have nothing to show
        in their daily practice history, so hiding it is the rational move.

    Scoring:
      - Hidden + high rating (≥ 2000): strong signal (6.0)
      - Hidden + moderate rating (1500–2000): moderate signal (3.5)
      - Hidden + low/no rating: weak signal (1.5) — many casual users hide it
      - Not hidden: 0.0

    Confidence is always "medium" because we can't distinguish "hidden" from
    "genuinely empty calendar due to API unavailability". The `calendar_hidden`
    flag in the profile already accounts for this by requiring the account to
    have solved ≥ 10 problems before flagging.
    """
    calendar_hidden  = profile.get("calendar_hidden", False)
    contest_rating   = profile.get("contest_rating", 0.0)
    attended_count   = profile.get("attended_count", 0)
    total_solved     = profile.get("problems", {}).get("All", 0)

    if not calendar_hidden:
        return {
            "score":       0.0,
            "value":       False,
            "label":       "Hidden submission graph",
            "description": "Submission graph is visible — no signal.",
            "confidence":  "high" if profile.get("calendar") else "low",
        }

    # Graph is hidden — now score based on how suspicious that is in context
    if contest_rating >= 2000 and attended_count >= 5 and total_solved < 400:
        score = 8.0
        desc  = (
            f"Submission graph is hidden. At rating {contest_rating:.0f} with "
            f"{attended_count} contests attended, hiding practice history is "
            "statistically unusual for legitimate high performers."
        )
    elif contest_rating >= 1500 or attended_count >= 3:
        score = 4.5
        desc  = (
            f"Submission graph is hidden. Active contest participant "
            f"(rating {contest_rating:.0f}, {attended_count} contests) "
            "with no visible practice history."
        )
    elif total_solved > 50:
        score = 1.0
        desc  = (
            f"Submission graph is hidden despite {total_solved} problems solved. "
            "May be a privacy preference."
        )
    else:
        score = 0.0
        desc  = (
            "Submission graph is hidden. Low activity level — "
            "likely a privacy preference rather than an anomaly signal."
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

    Why this is stronger than profile_depth alone:
      profile_depth compares problems solved against a rating BRACKET
      (e.g. "2400-rated users typically solve 250+ problems").
      But rating can be gamed in a handful of contests.

      Top percentile is harder to inflate — it reflects where you stand
      against every human who competed that specific week. Being in the
      top 1% globally with 24 problems solved is a near-impossible
      combination through legitimate practice.

    The signal is the RATIO: percentile rank / problems solved.
    A top-1% finisher who has solved 500 problems → ratio = 0.002 → normal.
    A top-1% finisher who has solved 20 problems  → ratio = 0.05  → extreme.

    Expected minimums (community heuristic):
      Top 1%   → ≥ 200 problems
      Top 5%   → ≥ 100 problems
      Top 10%  → ≥  50 problems
      Top 25%  → ≥  20 problems
    """
    top_pct      = profile.get("top_percentage", 100.0)
    total_solved = profile.get("problems", {}).get("All", 0)
    attended     = profile.get("attended_count", 0)

    # Need at least 3 contests to have a meaningful percentile
    if attended < 3 or top_pct is None:
        return {
            "score":       0.0,
            "value":       None,
            "label":       "Percentile vs problems solved",
            "description": "Fewer than 3 contests — percentile not yet meaningful.",
            "confidence":  "low",
        }

    # No percentile data (unranked / never finished)
    if top_pct >= 99.0:
        return {
            "score":       0.0,
            "value":       None,
            "label":       "Percentile vs problems solved",
            "description": "No percentile data available.",
            "confidence":  "low",
        }

    # Expected minimum problems for each percentile band
    # Format: (percentile_ceiling, expected_min_problems)
    # e.g. if top_pct <= 1.0, user is in top 1% → expect >= 200 problems
    bands = [
        (1.5,  700),
        (5.0,  200),
        (10.0,  100),
        (25.0,  50),
    ]

    expected_min = None
    band_label   = None
    for ceiling, min_problems in bands:
        if top_pct <= ceiling:
            expected_min = min_problems
            band_label   = f"top {ceiling:.0f}%"
            break

    # Not in a flaggable percentile band (worse than top 25%)
    if expected_min is None:
        return {
            "score":       0.0,
            "value":       top_pct,
            "label":       "Percentile vs problems solved",
            "description": (
                f"Top percentile {top_pct:.1f}% — outside suspicious range."
            ),
            "confidence":  "medium",
        }

    # How far below the expected minimum are they?
    if total_solved == 0:
        deficit_ratio = 1.0   # completely empty
    else:
        deficit_ratio = max(0.0, 1.0 - (total_solved / expected_min))

    # Score based on deficit
    if total_solved <= 50:
        score = 10
        desc  = (
            f"Zero problems solved but ranked {band_label} globally "
            f"({top_pct:.1f}%). Organically impossible."
        )
    elif deficit_ratio >= 0.85:
        score = 9
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
    live contest ranking API check (slower — 1 extra API call per contest).
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
            pass   # non-fatal — don't crash main flow

    return feats
