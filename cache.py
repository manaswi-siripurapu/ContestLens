"""
cache.py — SQLite-backed profile cache + session-based rate limiting.
TTL: 24 hours per username.  Rate limit: 10 lookups / hour / session.
"""

import sqlite3
import json
import time
import os
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ.get("CACHE_DB_PATH", "cache.db"))
CACHE_TTL_SECONDS = 86_400       # 24 hours
RATE_LIMIT_MAX = 10              # lookups per window
RATE_LIMIT_WINDOW = 3_600        # 1 hour in seconds


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS profile_cache (
                username   TEXT    PRIMARY KEY,
                data       TEXT    NOT NULL,
                cached_at  REAL    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rate_limits (
                session_id   TEXT  PRIMARY KEY,
                count        INTEGER NOT NULL DEFAULT 0,
                window_start REAL    NOT NULL
            );
        """)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------
# Profile cache
# ---------------------------------------------------------------------------

def get_cached(username: str) -> Optional[dict]:
    """Return cached profile dict or None if missing / expired."""
    with _conn() as c:
        row = c.execute(
            "SELECT data, cached_at FROM profile_cache WHERE username = ?",
            (username.lower(),),
        ).fetchone()

    if row is None:
        return None

    age = time.time() - row["cached_at"]
    if age > CACHE_TTL_SECONDS:
        return None  # stale — let caller re-fetch

    try:
        return json.loads(row["data"])
    except json.JSONDecodeError:
        return None


def set_cached(username: str, data: dict) -> None:
    """Persist raw API payload to cache."""
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO profile_cache (username, data, cached_at)
               VALUES (?, ?, ?)""",
            (username.lower(), json.dumps(data, default=str), time.time()),
        )


def cache_age_minutes(username: str) -> Optional[float]:
    """Return how many minutes ago this username was cached (None if not cached)."""
    with _conn() as c:
        row = c.execute(
            "SELECT cached_at FROM profile_cache WHERE username = ?",
            (username.lower(),),
        ).fetchone()
    if row is None:
        return None
    return (time.time() - row["cached_at"]) / 60


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate_limit(session_id: str) -> tuple[bool, int]:
    """
    Returns (allowed: bool, remaining: int).
    Resets the window after RATE_LIMIT_WINDOW seconds.
    """
    now = time.time()

    with _conn() as c:
        row = c.execute(
            "SELECT count, window_start FROM rate_limits WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if row is None:
            c.execute(
                "INSERT INTO rate_limits (session_id, count, window_start) VALUES (?, 1, ?)",
                (session_id, now),
            )
            return True, RATE_LIMIT_MAX - 1

        count, window_start = row["count"], row["window_start"]

        # Window expired → reset
        if now - window_start > RATE_LIMIT_WINDOW:
            c.execute(
                "UPDATE rate_limits SET count = 1, window_start = ? WHERE session_id = ?",
                (now, session_id),
            )
            return True, RATE_LIMIT_MAX - 1

        if count >= RATE_LIMIT_MAX:
            reset_in = int(RATE_LIMIT_WINDOW - (now - window_start))
            return False, reset_in   # remaining = seconds until reset

        c.execute(
            "UPDATE rate_limits SET count = count + 1 WHERE session_id = ?",
            (session_id,),
        )
        return True, RATE_LIMIT_MAX - count - 1
