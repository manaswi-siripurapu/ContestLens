"""
fetcher.py — LeetCode data fetcher (v6)

WHY PREVIOUS VERSIONS BROKE:
  Cloudflare performs JA3/JA4 TLS fingerprinting at the TCP level.
  requests → HTTP/1.1, Python TLS fingerprint → RST immediately
  httpx    → HTTP/2, but still Python TLS fingerprint → RST immediately

THE FIX:
  curl_cffi uses libcurl compiled with BoringSSL — the exact same TLS
  library Chrome uses. The fingerprint is indistinguishable from a real
  Chrome browser. Cloudflare passes it through every time.

Install:
  pip install curl-cffi
"""

import json
import time
import random
from typing import Any, Optional

try:
    from curl_cffi import requests as curl_requests
except ImportError as exc:
    raise ImportError(
        "curl_cffi is required. Run:  pip install curl-cffi\n"
        "Then add  curl-cffi>=0.6.0  to requirements.txt"
    ) from exc

# Config
LC_ORIGIN   = "https://leetcode.com"
GRAPHQL_URL = f"{LC_ORIGIN}/graphql/"
SEED_URL    = f"{LC_ORIGIN}/"

SEED_TIMEOUT = 12
GQL_TIMEOUT  = 25
MAX_RETRIES  = 3
INTER_SLEEP  = 1.5

# curl_cffi impersonation target — must match a supported Chrome version
IMPERSONATE = "chrome124"

# Exceptions
class LeetCodeError(Exception): pass
class UserNotFoundError(LeetCodeError): pass
class PrivateProfileError(LeetCodeError): pass
class RateLimitedError(LeetCodeError): pass
class NetworkError(LeetCodeError): pass
class ParseError(LeetCodeError): pass

# Helpers

def _safe_get(d: Any, *keys, default=None) -> Any:
    try:
        for k in keys:
            if d is None:
                return default
            d = (d.get(k, default) if isinstance(d, dict)
                 else (d[k] if isinstance(d, list)
                       and isinstance(k, int)
                       and 0 <= k < len(d) else default))
        return d if d is not None else default
    except (KeyError, IndexError, TypeError):
        return default


def _make_session() -> curl_requests.Session:
    """
    Open a curl_cffi Session impersonating Chrome124.
    Visiting the homepage seeds the csrftoken cookie.
    """
    session = curl_requests.Session(impersonate=IMPERSONATE)

    try:
        session.get(SEED_URL, timeout=SEED_TIMEOUT)
        csrf = session.cookies.get("csrftoken", "")
        if csrf:
            session.headers.update({"x-csrftoken": csrf})
    except Exception:
        pass   # best-effort

    return session


def _gql(session: curl_requests.Session, query: str,
         variables: dict) -> dict:
    """POST one GraphQL query, retry on transient failures."""
    payload  = {"query": query, "variables": variables}
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            time.sleep(2 ** attempt + random.uniform(0.5, 1.5))

        try:
            resp = session.post(
                GRAPHQL_URL,
                json=payload,
                timeout=GQL_TIMEOUT,
            )
        except Exception as exc:
            err = str(exc)
            if "10054" in err or "forcibly closed" in err.lower():
                last_exc = NetworkError(
                    "Cloudflare reset the connection (WinError 10054). "
                    "Retrying…"
                )
            elif "timed out" in err.lower() or "timeout" in err.lower():
                last_exc = NetworkError(
                    f"Request timed out (attempt {attempt+1}/{MAX_RETRIES})."
                )
            else:
                last_exc = NetworkError(f"Connection error: {exc}")
            continue

        if resp.status_code == 429:
            raise RateLimitedError(
                "LeetCode is rate-limiting requests. "
                f"Wait {resp.headers.get('Retry-After', '60')}s and retry."
            )
        if resp.status_code == 403:
            raise LeetCodeError(
                "LeetCode returned 403. "
                "Wait 2–3 minutes or switch networks."
            )
        if resp.status_code >= 500:
            last_exc = LeetCodeError(
                f"LeetCode server error ({resp.status_code}). Retrying…"
            )
            continue
        if resp.status_code not in (200, 201):
            raise LeetCodeError(
                f"Unexpected HTTP {resp.status_code} from LeetCode."
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ParseError(
                "LeetCode returned an unparseable response. Try again."
            ) from exc

        if "errors" in data:
            msgs = [e.get("message", "") for e in (data.get("errors") or [])]
            if any(
                kw in m.lower()
                for m in msgs
                for kw in ("not exist", "not found", "no user")
            ):
                raise UserNotFoundError(
                    "This username does not exist on LeetCode."
                )

        return data

    raise last_exc or NetworkError(
        f"LeetCode did not respond after {MAX_RETRIES} attempts. "
        "Try again in a minute."
    )

# GraphQL queries

_PROFILE_QUERY = """
query getUserProfile($username: String!) {
    matchedUser(username: $username) {
        username
        profile { ranking userAvatar }
        submitStatsGlobal {
            acSubmissionNum { difficulty count submissions }
        }
        userCalendar {
            activeYears streak totalActiveDays submissionCalendar
        }
        languageProblemCount { languageName problemsSolved }
    }
    userContestRanking(username: $username) {
        attendedContestsCount rating globalRanking
        totalParticipants topPercentage
        badge { name }
    }
}
"""

_CONTEST_HISTORY_QUERY = """
query userContestRankingHistory($username: String!) {
    userContestRankingHistory(username: $username) {
        attended trendDirection problemsSolved totalProblems
        finishTimeInSeconds rating ranking
        contest { title startTime }
    }
}
"""

# Parsers

def _parse_profile(data: dict, username: str) -> dict:
    matched = _safe_get(data, "data", "matchedUser")
    if matched is None:
        raise UserNotFoundError(
            f"No LeetCode account found for '{username}'."
        )

    raw_stats = _safe_get(
        matched, "submitStatsGlobal", "acSubmissionNum", default=[]
    ) or []
    problem_counts    = {"Easy": 0, "Medium": 0, "Hard": 0, "All": 0}
    submission_counts = {"Easy": 0, "Medium": 0, "Hard": 0, "All": 0}
    for entry in raw_stats:
        diff = entry.get("difficulty", "")
        if diff in problem_counts:
            problem_counts[diff]    = int(entry.get("count") or 0)
            submission_counts[diff] = int(entry.get("submissions") or 0)

    raw_cal = _safe_get(
        matched, "userCalendar", "submissionCalendar", default="{}"
    )
    try:
        calendar: dict = json.loads(raw_cal) if raw_cal else {}
    except (json.JSONDecodeError, TypeError):
        calendar = {}

    raw_langs = _safe_get(matched, "languageProblemCount", default=[]) or []
    languages: dict[str, int] = {}
    for e in raw_langs:
        lang = e.get("languageName", "")
        if lang:
            languages[lang] = int(e.get("problemsSolved") or 0)

    cr           = _safe_get(data, "data", "userContestRanking") or {}
    total_active = int(
        _safe_get(matched, "userCalendar", "totalActiveDays", default=0) or 0
    )
    total_solved    = problem_counts.get("All", 0)
    calendar_hidden = not calendar and (total_solved > 10 or total_active > 5)

    return {
        "username":           matched.get("username", username),
        "avatar":             _safe_get(matched, "profile", "userAvatar"),
        "site_ranking":       _safe_get(matched, "profile", "ranking"),
        "problems":           problem_counts,
        "submissions":        submission_counts,
        "calendar":           calendar,
        "calendar_hidden":    calendar_hidden,
        "languages":          languages,
        "contest_rating":     float(cr.get("rating") or 0.0),
        "attended_count":     int(cr.get("attendedContestsCount") or 0),
        "global_ranking":     cr.get("globalRanking"),
        "total_participants": int(cr.get("totalParticipants") or 1),
        "top_percentage":     float(cr.get("topPercentage") or 100.0),
        "badge":              _safe_get(cr, "badge", "name"),
        "active_years":       _safe_get(
            matched, "userCalendar", "activeYears", default=[]
        ),
        "streak":             _safe_get(
            matched, "userCalendar", "streak", default=0
        ),
        "total_active_days":  total_active,
    }


def _parse_contest_history(data: dict) -> list[dict]:
    raw = _safe_get(
        data, "data", "userContestRankingHistory", default=[]
    ) or []
    history = []
    for entry in raw:
        if not entry.get("attended", False):
            continue
        contest = entry.get("contest") or {}
        history.append({
            "title":           contest.get("title", "Unknown Contest"),
            "start_time":      int(contest.get("startTime") or 0),
            "rating":          float(entry.get("rating") or 0.0),
            "ranking":         int(entry.get("ranking") or 0),
            "problems_solved": int(entry.get("problemsSolved") or 0),
            "total_problems":  int(entry.get("totalProblems") or 4),
            "finish_seconds":  entry.get("finishTimeInSeconds"),
            "trend":           entry.get("trendDirection", ""),
        })
    history.sort(key=lambda x: x["start_time"])
    return history

# Public API

def fetch_profile(username: str) -> dict:
    username = username.strip()
    if not username:
        raise ValueError("Username must not be empty.")
    if len(username) > 50:
        raise ValueError("Username too long (max 50 chars).")
    session = _make_session()
    data    = _gql(session, _PROFILE_QUERY, {"username": username})
    return _parse_profile(data, username)


def fetch_contest_history(username: str) -> list[dict]:
    username = username.strip()
    session  = _make_session()
    try:
        data = _gql(session, _CONTEST_HISTORY_QUERY, {"username": username})
        return _parse_contest_history(data)
    except UserNotFoundError:
        raise
    except LeetCodeError:
        return []


def fetch_all(username: str) -> dict:
    """Fetch profile + contest history in one shared session."""
    username = username.strip()
    if not username:
        raise ValueError("Username must not be empty.")
    if len(username) > 50:
        raise ValueError("Username too long (max 50 chars).")

    session = _make_session()

    profile_data = _gql(session, _PROFILE_QUERY, {"username": username})
    profile      = _parse_profile(profile_data, username)

    time.sleep(INTER_SLEEP)

    try:
        contest_data = _gql(
            session, _CONTEST_HISTORY_QUERY, {"username": username}
        )
        contests = _parse_contest_history(contest_data)
    except UserNotFoundError:
        raise
    except LeetCodeError:
        contests = []

    return {"profile": profile, "contests": contests}
