"""
percentile_store.py — Score distribution tracker.

Records anomaly scores from real lookups and computes
where a new score sits in the distribution of all profiles seen so far.

Tables are created inside the same cache.db used by cache.py.
"""

import sqlite3
import time
import os
from pathlib import Path

DB_PATH = Path(os.environ.get("CACHE_DB_PATH", "cache.db"))


def init_percentile_table() -> None:
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS score_distribution (
                username    TEXT  NOT NULL,
                score       REAL  NOT NULL,
                quality     TEXT  NOT NULL,
                recorded_at REAL  NOT NULL
            )
        """)


def record_score(username: str, score: float, quality: str) -> None:
    """
    Record a score into the distribution.
    Skips 'insufficient' quality profiles — they would skew the distribution
    downward since they score near zero due to missing data, not legitimacy.
    """
    if quality == "insufficient":
        return
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        c.execute(
            "INSERT INTO score_distribution (username, score, quality, recorded_at) "
            "VALUES (?, ?, ?, ?)",
            (username.lower(), score, quality, time.time()),
        )


def get_percentile(score: float) -> dict:
    """
    Return what percentile this score sits at in the stored distribution.
    Returns enough_data=False if fewer than 20 scores are stored.
    """
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        rows = c.execute("SELECT score FROM score_distribution").fetchall()

    all_scores = [r[0] for r in rows]

    if len(all_scores) < 20:
        return {
            "percentile":     None,
            "total_profiles": len(all_scores),
            "enough_data":    False,
        }

    below      = sum(1 for s in all_scores if s < score)
    percentile = round(below / len(all_scores) * 100, 1)

    return {
        "percentile":     percentile,
        "total_profiles": len(all_scores),
        "enough_data":    True,
        "higher_than":    f"{percentile:.0f}%",
        "distribution": {
            "p25": _pval(all_scores, 25),
            "p50": _pval(all_scores, 50),
            "p75": _pval(all_scores, 75),
            "p90": _pval(all_scores, 90),
        },
    }


def get_stats() -> dict:
    with sqlite3.connect(DB_PATH, check_same_thread=False) as c:
        rows = c.execute("SELECT score FROM score_distribution").fetchall()
    scores = [r[0] for r in rows]
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "mean":  round(sum(scores) / len(scores), 2),
        "min":   round(min(scores), 2),
        "max":   round(max(scores), 2),
        "p50":   _pval(scores, 50),
        "p75":   _pval(scores, 75),
        "p90":   _pval(scores, 90),
    }


def _pval(scores: list, p: int) -> float:
    s   = sorted(scores)
    idx = int(len(s) * p / 100)
    return round(s[min(idx, len(s) - 1)], 2)
