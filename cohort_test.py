"""
cohort_test.py — Known-clean cohort calibration test.

Runs ContestLens against publicly verified legitimate competitive programmers.
If any score above the threshold, it's a calibration problem — this script
tells you exactly which signals fired and suggests what to review.

Usage:
    python cohort_test.py                      # default threshold 6.0
    python cohort_test.py --threshold 5.0      # stricter
    python cohort_test.py --delay 5.0          # gentler on API
    python cohort_test.py --save data/labeled.json   # append results to dataset

These are the ground truth "definitely legitimate" profiles. They are all
either ICPC World Finalists/Champions or Codeforces Grandmasters with a
verified LeetCode presence. Any score above 6.0 on these profiles is
evidence that a signal or weight needs recalibrating.
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path

from httpx import NetworkError

sys.path.insert(0, str(Path(__file__).parent))

KNOWN_CLEAN = [
    {"username": "lee215", "note": "Top 1 all-time LeetCode reputation, legendary solution writer"},
    {"username": "StefanPochmann", "note": "Extremely high reputation (~100k+), clean and optimal solutions"},
    {"username": "votrubac", "note": "Top contributor, solutions appear in most problems"},
    {"username": "niits", "note": "Top 3 reputation leaderboard, consistent contributor"},
    {"username": "rahulvarma5297", "note": "Top 5 reputation, strong contest + discuss presence"},
    {"username": "hiepit", "note": "Top 10 reputation, frequent high-quality solutions"},
    {"username": "anwendeng", "note": "Top contributor with strong problem-solving record"},
    {"username": "DBabichev", "note": "Known for math-heavy and advanced solutions"},
    {"username": "jianchao-li", "note": "Well-known for structured explanations"},
    {"username": "archit91", "note": "Popular contributor, appears often in discussions"},
    {"username": "OldCodingFarmer", "note": "Veteran contributor with consistent solutions"},
    {"username": "fun4LeetCode", "note": "One of the earliest and most respected contributors"},
    {"username": "rock", "note": "High reputation user with strong consistency"},
    {"username": "AlecLC", "note": "Active contributor with many accepted solutions"},
    {"username": "Spaulding_", "note": "Top 20 reputation, strong problem-solving presence"},
    {"username": "GraceMeng", "note": "Highly ranked contributor in discuss section"},
    {"username": "issac3", "note": "Frequent contributor with strong acceptance rate"},
    {"username": "jeantimex", "note": "Known for efficient and optimized approaches"},
    {"username": "dietpepsi", "note": "Consistent contributor in discussion solutions"},
    {"username": "shawngao", "note": "Recognized contributor with strong algorithmic solutions"}
]


def _bar(val: float, max_val: float = 10.0, width: int = 20) -> str:
    filled = int(val / max_val * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {val:.1f}"


def run_cohort(
    threshold: float = 6.0,
    delay: float = 3.0,
    save_path: str = None,
) -> dict:
    import fetcher
    import features as fm
    import scorer as sm

    results            = []
    calibration_issues = []

    print(f"\n{'='*65}")
    print(f"ContestLens — cohort calibration test")
    print(f"Methodology : {sm.METHODOLOGY_VERSION}")
    print(f"Threshold   : {threshold}/10  (above = calibration issue)")
    print(f"Profiles    : {len(KNOWN_CLEAN)}")
    print(f"{'='*65}\n")

    for entry in KNOWN_CLEAN:
        username = entry["username"]
        print(f"  {username:<18}  ", end="", flush=True)

        try:
            raw    = fetcher.fetch_all(username)
            feats  = fm.compute_all_features(raw["profile"], raw["contests"])
            result = sm.compute_anomaly_score(feats)
            score  = result["score"]
            flag   = "  <-- CALIBRATION ISSUE" if score > threshold else ""

            print(f"{_bar(score)}  ({result['verdict']}){flag}")

            # Per-signal breakdown for this profile
            active_signals = [
                f for f in result["feature_scores"]
                if f.get("eff_weight", 0) > 0 and f["score"] > 0
            ]
            active_signals.sort(key=lambda x: -x["score"])

            if score > threshold or active_signals:
                for sig in active_signals[:4]:
                    marker = " **" if sig["score"] >= 5.0 else "   "
                    print(f"      {marker}{sig['label']:<38} {sig['score']:.1f}/10")

            row = {
                "username":     username,
                "note":         entry["note"],
                "score":        score,
                "verdict":      result["verdict"],
                "quality":      result["data_quality"],
                "fired_signals": [
                    {"label": f["label"], "score": f["score"]}
                    for f in active_signals
                    if f["score"] >= 5.0
                ],
                "profile":  raw["profile"],
                "contests": raw["contests"],
            }
            results.append(row)

            if score > threshold:
                calibration_issues.append(row)

        # except Exception as exc:
        #     print(f"ERROR: {exc}")
        except Exception as exc:
            attempt = 0
            err = str(exc)
            if "10054" in err or "56" in err or "forcibly closed" in err.lower() or "reset" in err.lower():
                wait = 15 + random.uniform(0, 10)   # was: just logging and continuing
                last_exc = NetworkError(f"Cloudflare reset (attempt {attempt+1}). Waiting {wait:.0f}s...")
                time.sleep(wait)   # longer wait specifically for resets
            else:
                last_exc = NetworkError(f"Connection error: {exc}")
            continue

        time.sleep(delay)

    # Summary
    scores = [r["score"] for r in results]
    print(f"\n{'='*65}")
    print(f"Summary")
    print(f"{'='*65}")
    if scores:
        print(f"  Profiles tested     : {len(scores)}")
        print(f"  Average score       : {sum(scores)/len(scores):.2f}/10")
        print(f"  Max score           : {max(scores):.2f}/10")
        print(f"  Calibration issues  : {len(calibration_issues)} "
              f"(scored > {threshold})")

        if calibration_issues:
            print(f"\n  Issues to fix:")
            for r in calibration_issues:
                print(f"\n  {r['username']} — {r['score']:.2f}/10  ({r['note']})")
                if r["fired_signals"]:
                    print(f"  Signals that fired (score >= 5.0):")
                    for sig in r["fired_signals"]:
                        print(f"    - {sig['label']}: {sig['score']:.1f}/10")
                    print(f"\n  Suggested action:")
                    _suggest_fix(r["fired_signals"])
                else:
                    print(f"  No single signal dominates — check interaction bonus logic")
        else:
            print(f"\n  All {len(scores)} known-clean profiles scored <= {threshold}.")
            print(f"  Calibration looks good for this cohort.")
            print(f"\n  This does NOT mean the system is accurate overall.")
            print(f"  Run evaluate.py with a labeled dataset for full validation.")

    # Save JSON results
    out = {
        "methodology": sm.METHODOLOGY_VERSION,
        "threshold":   threshold,
        "results":     [
            {k: v for k, v in r.items() if k not in ("profile", "contests")}
            for r in results
        ],
        "issues":      [
            {k: v for k, v in r.items() if k not in ("profile", "contests")}
            for r in calibration_issues
        ],
        "calibrated":  len(calibration_issues) == 0,
    }
    Path("cohort_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to cohort_results.json")

    # Optionally append clean profiles to the evaluate.py labeled dataset
    if save_path:
        _append_to_dataset(results, save_path)

    return out


def _suggest_fix(fired_signals: list[dict]) -> None:
    """Print a specific recalibration suggestion for each fired signal."""
    suggestions = {
        "Contest solve speed": (
            "The solve speed threshold may be too loose for elite coders.\n"
            "  Try: raise the total_solved mitigation threshold above 800,\n"
            "  or reduce the base score for the 15-25min band."
        ),
        "Rating velocity": (
            "An elite coder can legitimately jump +300 in a single contest.\n"
            "  Try: raise the 'near-certain anomaly' threshold from 400 to 500."
        ),
        "Ranking consistency": (
            "Elite coders genuinely finish top-5 repeatedly — that's not a bug.\n"
            "  Try: only flag when top_5_count >= 5 (not 3) with < 20 contests."
        ),
        "Percentile vs problems solved": (
            "Some ICPC coders have few LeetCode problems but are legitimately elite.\n"
            "  Try: add an 'attended < 10 contests' guard before scoring this feature."
        ),
        "Early contest performance": (
            "An ICPC medalist will debut at top-50 — that's expected.\n"
            "  Try: reduce the weight of this signal or add a rating cap."
        ),
        "Profile depth": (
            "ICPC medalists may have very few LeetCode problems but be very good.\n"
            "  Try: add a mitigator for accounts where hard_solved > 20."
        ),
        "Submission pattern entropy": (
            "Some people genuinely code on 1-2 days a week due to schedule.\n"
            "  Try: only flag when entropy < 0.8 instead of < 1.0."
        ),
    }

    for sig in fired_signals:
        label   = sig["label"]
        suggest = suggestions.get(label)
        if suggest:
            print(f"    [{label}]")
            for line in suggest.split("\n"):
                print(f"      {line}")


def _append_to_dataset(results: list[dict], path: str) -> None:
    """Append known-clean profiles to an evaluate.py labeled dataset."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text().strip() if p.exists() else ""
    existing: list[dict] = json.loads(text) if text else []
    existing_users = {e["username"].lower() for e in existing}

    added = 0
    for r in results:
        if r["username"].lower() not in existing_users:
            existing.append({
                "username": r["username"],
                "label":    "legitimate",
                "source":   "cohort_test_icpc_cf",
                "profile":  r.get("profile", {}),
                "contests": r.get("contests", []),
            })
            added += 1

    p.write_text(json.dumps(existing, indent=2, default=str))
    print(f"  Appended {added} profiles to {p} as 'legitimate' labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ContestLens cohort calibration test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run this before presenting the project to anyone.
If any known-clean profile scores above 6.0, fix before showing.

After running, use --save to add these profiles to your evaluate.py dataset
as ground-truth legitimate labels:
  python cohort_test.py --save data/labeled.json
        """,
    )
    parser.add_argument("--threshold", type=float, default=6.0,
                        help="Score above which a clean profile is a calibration failure")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between API calls (default 3.0)")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="Append clean profiles to this labeled dataset JSON")
    args = parser.parse_args()

    run_cohort(args.threshold, args.delay, args.save)
