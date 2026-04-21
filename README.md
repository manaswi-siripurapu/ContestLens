# ContestLens

A personal side project built out of curiosity. ContestLens looks at publicly available LeetCode contest data and surfaces statistically unusual patterns in a profile — not to accuse anyone of anything, just to see what the numbers say.

**Live demo → [contestlens-0tza.onrender.com](https://contestlens-0tza.onrender.com)**
*(free tier — first load may take ~30s-2m to wake up)*

---

## What it does

Enter a LeetCode username. ContestLens fetches their public contest history and profile data, runs it through 9 independent signals, and produces an **Anomaly Index from 0 to 10**.

A high score means the profile has unusual statistical patterns. It does not mean cheating, it does not mean anything definitive. It's a data lens — use it as a starting point for curiosity, not a basis for any judgement.

---

## How to use it

1. Open the live link above
2. Type a LeetCode username in the search box
3. Hit **Analyse**
4. Toggle **Deep Analysis** in the sidebar for the strongest signal (adds ~15s)

That's it. No login, no API key required.

---

## How it works internally

**Data layer**

LeetCode sits behind Cloudflare which blocks standard HTTP libraries at the TLS fingerprint level. ContestLens uses `curl_cffi` with `impersonate="chrome124"` — this replicates Chrome 124's exact TLS handshake so Cloudflare lets it through. Two GraphQL queries fire against `leetcode.com/graphql/` — one for profile data, one for contest history. Results are cached in SQLite for 24 hours so the same profile isn't re-fetched repeatedly.

**Feature layer**

9 independent signal functions in `features.py`, each scoring 0–10:

| Signal | Weight | What it looks at |
|---|---|---|
| Solve time vs field | 20% | Whether harder problems took longer than easier ones |
| Percentile vs problems solved | 16% | Contest percentile vs total practice volume |
| Contest solve speed | 14% | Time to complete all 4 problems |
| Ranking consistency | 12% | Variance in rank across all contests |
| Rating velocity | 11% | Size of single-contest rating jumps |
| Profile depth | 9% | Problems solved relative to rating bracket |
| Hidden submission graph | 7% | Presence of a visible practice history |
| Early performance | 6% | Rank in the first few contests |
| Submission entropy | 4% | Day-of-week distribution of submissions |

Each signal also returns a confidence level. Signals with insufficient data are excluded from scoring entirely — a new account with 2 contests doesn't get a false high score.

**Scoring layer**

`scorer.py` takes all 9 scores, applies confidence multipliers and weights, computes a weighted average, and adds an interaction bonus (up to +1.0) if both solve speed and profile depth are simultaneously flagged — those two together are statistically harder to explain than either alone.

**Explanation layer**

`explainer.py` sends the feature values to the Groq API (llama-3.3-70b) with a strictly constrained prompt — no accusatory language, 3–5 sentences, uncertainty baked in. Falls back to a deterministic rule-based explanation if no API key is set.

---

## Low level design

```
username input
    │
    ├── rate limit check  (SQLite, 10 req/hr per session)
    ├── cache check       (SQLite, 24h TTL per username)
    │
    ├── fetcher.py
    │     curl_cffi session (Chrome124 TLS)
    │     GET leetcode.com  →  seed csrftoken cookie
    │     POST /graphql/    →  profile query
    │     POST /graphql/    →  contest history query
    │
    ├── features.py        →  9 signal functions → list of feature dicts
    ├── scorer.py          →  weighted average + interaction bonus → 0-10
    ├── explainer.py       →  Groq API → plain-English summary
    │
    └── app.py             →  Streamlit renders gauge, charts, breakdown
```

**SQLite schema:**
- `profile_cache` — raw API JSON blobs, keyed by username, 24h TTL
- `rate_limits` — session-scoped lookup counter
- `score_distribution` — all scores ever computed, used for percentile ranking

---

## Validation

Evaluated on 36 manually labeled profiles — 18 known legitimate (ICPC/CF grandmasters), 18 manually identified suspicious profiles from contest leaderboard review.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 2.5 (optimal) | 100% | 83% | 0.909 |
| 5.0 (conservative) | 100% | 78% | 0.875 |

Zero false positives across all thresholds. Top discriminating signals: profile depth (+8.06 separation), percentile vs problems (+6.81), contest solve speed (+4.64).

Scoring methodology is versioned (`v1.1-calibrated`) — weights are heuristic, not trained. The labeled dataset is private and not published.

---

## Run locally

```bash
git clone https://github.com/manaswi-siripurapu/ContestLens
cd ContestLens
pip install -r requirements.txt
streamlit run app.py
```

Optional — add a Groq key for AI explanations:
```bash
export GROQ_API_KEY=gsk_...
```

---

## Accuracy evaluation tools

```bash
# Collect labeled profiles
python evaluate.py collect --suspicious user1 user2 --legitimate user3 user4 --output data/labeled.json

# Run evaluation
python evaluate.py eval --labeled data/labeled.json --report report.md

# Test known-clean profiles for calibration
python cohort_test.py
```

---

## Good to know

- Scores are statistical patterns, not verdicts
- LeetCode's GraphQL endpoint is unofficial — the tool may break if LeetCode updates their API
- Cached for 24h per username, rate-limited to 10 lookups/hour
- If you have genuine concerns about a user, report to [LeetCode directly](https://support.leetcode.com)
