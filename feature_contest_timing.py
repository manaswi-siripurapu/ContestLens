"""
feature_contest_timing.py — Intra-contest solve time analysis  (v2)

Three independent sub-signals:

1. REVERSE SOLVE ORDER (new — highest weight)
   Real humans slow down as problems get harder: Q1 < Q2 < Q3 < Q4 in time.
   Copiers receive all solutions at once and paste them in any order, so
   their times are non-monotonic. Q4 faster than Q1 is nearly impossible
   legitimately.
   Measured as: number of inversions in the solve-time sequence.

2. FIELD MEDIAN COMPARISON
   Compares user's per-problem time against top-200 median.
   Flags Q3/Q4 solved at < 20% of field median.

3. TOTAL BURST (revised threshold: 10 minutes for full 4-problem clear)
   Catching the pathological case where everything arrives pre-solved.
"""

import time
from curl_cffi import requests as curl_requests

IMPERSONATE = "chrome124"
RANKING_URL = "https://leetcode.com/contest/api/ranking/{contest_slug}/"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_contest_slugs(contest_history: list[dict]) -> list[tuple]:
    """'Weekly Contest 420' → 'weekly-contest-420'"""
    slugs = []
    for c in contest_history[-10:]:
        title = c.get("title", "")
        slug  = title.lower().replace(" ", "-")
        if slug:
            slugs.append((slug, c))
    return slugs


def _fetch_user_row(
    session: curl_requests.Session,
    contest_slug: str,
    username: str,
    max_pages: int = 5,
) -> dict | None:
    """Find username in contest ranking pages. Returns user row or None."""
    for page in range(1, max_pages + 1):
        try:
            resp = session.get(
                RANKING_URL.format(contest_slug=contest_slug),
                params={"pagination": page, "region": "global"},
                timeout=15,
            )
            if resp.status_code != 200:
                break
            data        = resp.json()
            total_rank  = data.get("total_rank", [])
            submissions = data.get("submissions", [])
            user_num    = data.get("user_num", 0)

            for i, row in enumerate(total_rank):
                if row.get("username", "").lower() == username.lower():
                    return {
                        "rank":        row.get("rank"),
                        "finish_time": row.get("finish_time"),
                        "submissions": submissions[i] if i < len(submissions) else {},
                        "total_users": user_num,
                    }

            # If user not found and this page had fewer rows than expected,
            # they didn't attend — stop searching
            if len(total_rank) < 25:
                break

        except Exception:
            break
    return None


def _field_medians(
    session: curl_requests.Session,
    contest_slug: str,
    problem_count: int = 4,
) -> dict[int, float]:
    """Median solve time per problem index across top-200 finishers."""
    all_times: dict[int, list[float]] = {i: [] for i in range(problem_count)}
    try:
        for page in range(1, 3):
            resp = session.get(
                RANKING_URL.format(contest_slug=contest_slug),
                params={"pagination": page, "region": "global"},
                timeout=15,
            )
            if resp.status_code != 200:
                break
            for sub_row in resp.json().get("submissions", []):
                for idx in range(problem_count):
                    t = sub_row.get(str(idx), {}).get("time", 0)
                    if t and t > 0:
                        all_times[idx].append(float(t))
            time.sleep(0.5)
    except Exception:
        return {}

    return {
        idx: sorted(times)[len(times) // 2]
        for idx, times in all_times.items()
        if times
    }


def _count_inversions(times: list[float]) -> int:
    """
    Count how many (i, j) pairs have i < j but times[i] > times[j].
    For a 4-problem sequence the max is 6 inversions (completely reversed).
    Any inversion means a later (harder) problem was solved faster.

    Examples:
      [5, 7, 6, 4]  →  3 inversions  (Q4<Q1, Q4<Q2, Q4<Q3... wait)
      Let's count: (0,2): 5>6? No. (0,3): 5>4? Yes. (1,2): 7>6? Yes.
                   (1,3): 7>4? Yes. (2,3): 6>4? Yes. → 4 inversions
      [5, 7, 8, 9]  →  0 inversions  (perfectly monotonic, normal)
      [9, 8, 7, 6]  →  6 inversions  (completely reversed, max suspicious)
    """
    n   = len(times)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] > times[j]:
                inv += 1
    return inv


def _inversion_score(times: list[float]) -> tuple[float, str]:
    """
    Convert inversion count to a 0-10 anomaly score with description.
    Max inversions for 4 problems = 6.
    """
    if len(times) < 3:
        return 0.0, ""

    n          = len(times)
    max_inv    = n * (n - 1) // 2   # 6 for n=4
    inv_count  = _count_inversions(times)
    inv_ratio  = inv_count / max_inv if max_inv > 0 else 0

    # Build human-readable description of the worst inversion
    # (the one where a harder problem was solved faster than an easier one)
    worst_desc = ""
    worst_gap  = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] > times[j]:
                gap = times[i] - times[j]
                if gap > worst_gap:
                    worst_gap  = gap
                    worst_desc = (
                        f"Q{i+1} took {times[i]/60:.0f}min but "
                        f"Q{j+1} (harder) took only {times[j]/60:.0f}min"
                    )

    # Scoring:
    # 0 inversions   → 0.0
    # 1 inversion    → 2.5  (could be legitimate on easy contests)
    # 2 inversions   → 5.0
    # 3 inversions   → 7.0
    # 4+ inversions  → 8.5-10.0
    if inv_count == 0:
        score = 0.0
    elif inv_count == 1:
        score = 2.5
    elif inv_count == 2:
        score = 5.0
    elif inv_count == 3:
        score = 7.0
    else:
        score = min(10.0, 8.5 + (inv_count - 4) * 0.5)

    return round(score, 2), worst_desc


# ---------------------------------------------------------------------------
# Public feature function
# ---------------------------------------------------------------------------

def feature_solve_time_vs_field(
    username: str,
    contest_history: list[dict],
) -> dict:
    """
    Three sub-signals combined into one feature score:
      1. Reverse solve order (inversions in Q1→Q4 time sequence)
      2. Speed vs field median on hard problems
      3. Total burst (all problems within 10 minutes)
    """
    if not contest_history:
        return {
            "score":       0.0,
            "value":       None,
            "label":       "Solve time vs field",
            "description": "No contest history available.",
            "confidence":  "low",
        }

    session          = curl_requests.Session(impersonate=IMPERSONATE)
    slugs            = _get_contest_slugs(contest_history)
    contests_checked = 0

    # Accumulate evidence across contests
    inversion_hits: list[dict] = []   # reverse-order anomalies
    field_hits:     list[dict] = []   # faster than field median
    burst_hits:     list[dict] = []   # all problems in burst

    for slug, contest_meta in slugs[:5]:
        user_row = _fetch_user_row(session, slug, username)
        if not user_row:
            continue

        contests_checked += 1
        user_subs   = user_row.get("submissions", {})
        total_probs = contest_meta.get("total_problems", 4)

        # Extract per-problem solve times in order Q1, Q2, Q3, Q4
        ordered_times: list[float] = []
        for idx in range(total_probs):
            t = user_subs.get(str(idx), {}).get("time", 0)
            if t and t > 0:
                ordered_times.append(float(t))
            else:
                ordered_times.append(None)   # unsolved

        solved_times = [t for t in ordered_times if t is not None]

        # --- Signal 1: Reverse solve order ---
        if len(solved_times) >= 3:
            inv_score, inv_desc = _inversion_score(solved_times)
            if inv_score > 0:
                inversion_hits.append({
                    "contest":    contest_meta.get("title"),
                    "score":      inv_score,
                    "times_min":  [round(t/60, 1) for t in solved_times],
                    "desc":       inv_desc,
                })

        # --- Signal 2: Total burst ---
        if len(solved_times) >= 3 and len(solved_times) == total_probs:
            spread = max(solved_times) - min(solved_times)
            total  = sum(solved_times)
            if total < 600:   # all 4 problems under 10 minutes total
                burst_hits.append({
                    "contest":        contest_meta.get("title"),
                    "total_minutes":  round(total / 60, 1),
                    "spread_seconds": round(spread),
                })

        # --- Signal 3: Field comparison ---
        field_meds = _field_medians(session, slug, total_probs)
        if field_meds:
            for idx in range(total_probs):
                user_t  = user_subs.get(str(idx), {}).get("time", 0)
                field_t = field_meds.get(idx, 0)
                if not user_t or not field_t:
                    continue
                ratio = user_t / field_t
                if ratio < 0.2 and idx >= 2:
                    field_hits.append({
                        "contest":       contest_meta.get("title"),
                        "problem":       f"Q{idx+1}",
                        "user_min":      round(user_t / 60, 1),
                        "field_min":     round(field_t / 60, 1),
                    })

        time.sleep(1.0)

    if not contests_checked:
        return {
            "score":       0.0,
            "value":       None,
            "label":       "Solve time vs field",
            "description": "Could not retrieve contest ranking data.",
            "confidence":  "low",
        }

    # -------------------------------------------------------------------
    # Aggregate score
    # Reverse order is the strongest signal — use the max inversion score
    # across contests as the base, then add bonuses for the other signals.
    # -------------------------------------------------------------------
    score = 0.0
    parts = []

    # Signal 1 — inversions (max score from any single contest)
    if inversion_hits:
        best    = max(inversion_hits, key=lambda x: x["score"])
        score   = best["score"]
        times_str = " → ".join(f"{t}m" for t in best["times_min"])
        parts.append(
            f"Non-monotonic solve order in {best['contest']}: "
            f"[{times_str}]. {best['desc']}"
        )

    # Signal 2 — burst (+2.0 bonus, not double-counting if already high)
    if burst_hits:
        score = min(10.0, score + 2.0)
        b = burst_hits[0]
        parts.append(
            f"All {contest_meta.get('total_problems',4)} problems in "
            f"{b['total_minutes']}min total ({b['contest']})"
        )

    # Signal 3 — field median (+1.5 per hit, capped at +3.0)
    if field_hits:
        score = min(10.0, score + min(3.0, len(field_hits) * 1.5))
        h = field_hits[0]
        parts.append(
            f"{h['problem']} solved in {h['user_min']}min vs "
            f"field median {h['field_min']}min ({h['contest']})"
        )

    score = round(min(10.0, score), 2)
    conf  = "high" if contests_checked >= 3 else "medium"

    desc = (
        "; ".join(parts) if parts
        else f"Solve times across {contests_checked} contest(s) appear normal."
    )

    return {
        "score":             score,
        "value":             score,
        "inversion_hits":    inversion_hits,
        "burst_hits":        burst_hits,
        "field_hits":        field_hits,
        "contests_checked":  contests_checked,
        "label":             "Solve time vs field",
        "description":       desc,
        "confidence":        conf,
    }
