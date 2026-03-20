# LeetCode Contest Anomaly Explorer

Surfaces statistically unusual patterns in LeetCode contest profiles using 7 independent signals.

---

## Quick Start

```bash
# 1. Clone / copy the project
cd leetcode_anomaly

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set your Anthropic API key for AI-powered explanations
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
streamlit run app.py
```

Open http://localhost:8501

---

## Project Structure

```
leetcode_anomaly/
├── app.py           # Streamlit UI — charts, layout, entry point
├── fetcher.py       # LeetCode GraphQL API client + error handling
├── features.py      # 7 anomaly signal computations
├── scorer.py        # Weighted aggregation → anomaly index 0-10
├── explainer.py     # Anthropic API explanation (with fallback)
├── cache.py         # SQLite cache (24h TTL) + rate limiting
└── requirements.txt
```

---

## The 7 Signals

| Signal | Weight | What it measures |
|---|---|---|
| Rating velocity | 25% | Largest single-contest rating jump |
| Contest solve speed | 22% | Fastest full contest completion time |
| Ranking consistency | 18% | Variance in rank percentile across contests |
| Profile depth | 15% | Problems solved vs expected for rating bracket |
| Submission entropy | 10% | Day-of-week Shannon entropy of submissions |
| Early performance | 6% | Rank in first 3 contests (newcomer baseline) |
| Language distribution | 4% | Abnormal language split |

---

## Output

- **Anomaly Index 0-10** (not a "probability of cheating")
- Per-signal score breakdown
- AI-generated explanation (or rule-based fallback)
- Rating trajectory chart with spike highlights
- Contest rank over time scatter
- Submission day-of-week bar chart
- Language breakdown

---

## Deployment (Railway)

```bash
# Install Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
```

Add `ANTHROPIC_API_KEY` as an environment variable in the Railway dashboard.

Create a `Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## Edge Cases Handled

- Username not found → clear error message
- User with 0 contests → signals marked low-confidence, no false score
- Network timeout → retries with exponential backoff, then user-facing error  
- LeetCode rate limiting (429) → detected and surfaced to user
- LeetCode 403 / geo-block → specific guidance to user
- Missing `finishTimeInSeconds` → solve speed feature excluded
- Calendar data empty / malformed → entropy feature excluded
- All contests unattended → empty history handled
- Anthropic API key missing → rule-based explanation fallback
- Anthropic API error → error noted, fallback used, app continues
- Very new account (< 3 contests) → low-confidence flags, disclaimer shown
- Data quality reported as "insufficient" → explicit warning in UI

---

## Legal & Ethical Notes

- Uses LeetCode's **unofficial** GraphQL endpoint. Respect their ToS.
- Cache all responses locally (24h TTL) to minimise load on LeetCode servers.
- Rate-limited to **10 lookups/hour per session** to prevent abuse.
- Results include a mandatory disclaimer on every analysis.
- Never call this tool's output "proof of cheating."
- If you believe a user is violating ToS, report to: https://support.leetcode.com

---

## Limitations

- No ground-truth labels — scoring is heuristic, not ML-trained.
- LeetCode's unofficial API may break at any time.
- Elite legitimate programmers (ICPC medalists) may score high on some signals.
- Submission timestamps are day-level only (not hour-level).
- Language-per-contest data is not available in the public API.
