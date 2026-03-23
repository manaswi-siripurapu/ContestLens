"""
evaluate.py — Offline accuracy evaluation for ContestLens.

Two modes:

  collect  — Fetch live profiles and save to a labeled JSON dataset
  eval     — Score a labeled dataset and print full metrics + analysis

Usage:
    # Step 1: collect profiles (can run multiple times to grow dataset)
    python evaluate.py collect \\
        --suspicious user1 user2 user3 \\
        --legitimate tourist neal_wu jiangly \\
        --output data/labeled.json

    # Step 2: evaluate
    python evaluate.py eval --labeled data/labeled.json

    # Step 3: save a markdown report
    python evaluate.py eval --labeled data/labeled.json --report report.md

Labeled JSON format:
[
  {
    "username": "someuser",
    "label":    "suspicious",   // or "legitimate"
    "source":   "reddit_thread" // where you found this label
    "profile":  { ... },        // fetcher.fetch_profile() output
    "contests": [ ... ]         // fetcher.fetch_contest_history() output
  }
]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import features as fm
import scorer as sm


# Core metrics

def compute_metrics(results: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        predicted = r["score"] >= threshold
        actual    = r["label"] == "suspicious"
        if predicted and actual:       tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and actual: fn += 1
        else:                          tn += 1

    total     = tp + fp + tn + fn
    precision = tp / (tp + fp)       if (tp + fp) else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (tp + tn) / total    if total else 0.0
    specificity = tn / (tn + fp)     if (tn + fp) else 0.0

    return {
        "threshold":   threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy":    round(accuracy, 3),
        "precision":   round(precision, 3),
        "recall":      round(recall, 3),
        "f1":          round(f1, 3),
        "specificity": round(specificity, 3),
    }


def find_best_threshold(results: list[dict]) -> dict:
    """Sweep 1.0–9.0 in 0.5 steps, return threshold with best F1."""
    best = {"f1": -1.0}
    for t in [x * 0.5 for x in range(2, 19)]:
        m = compute_metrics(results, t)
        if m["f1"] > best["f1"]:
            best = m
    return best


def confusion_matrix_str(m: dict) -> str:
    spec = m["tn"] / max(m["tn"] + m["fp"], 1)
    return (
        f"\n  Confusion matrix (threshold={m['threshold']}):\n"
        f"                      Predicted suspicious  Predicted legitimate\n"
        f"  Actual suspicious       {m['tp']:>4}                  {m['fn']:>4}   "
        f"(recall {m['recall']:.0%})\n"
        f"  Actual legitimate       {m['fp']:>4}                  {m['tn']:>4}   "
        f"(specificity {spec:.0%})"
    )


# Signal analysis

def signal_importance(results: list[dict]) -> list[dict]:
    """
    For each signal, compute mean score in suspicious vs legitimate profiles.
    Higher separation = more useful discriminator.
    """
    all_signals: set[str] = set()
    for r in results:
        all_signals.update(r.get("feature_scores", {}).keys())

    rows = []
    for sig in sorted(all_signals):
        sus_scores = [
            r["feature_scores"].get(sig, 0.0)
            for r in results if r["label"] == "suspicious"
        ]
        leg_scores = [
            r["feature_scores"].get(sig, 0.0)
            for r in results if r["label"] == "legitimate"
        ]

        sus_mean = sum(sus_scores) / len(sus_scores) if sus_scores else 0.0
        leg_mean = sum(leg_scores) / len(leg_scores) if leg_scores else 0.0
        separation = sus_mean - leg_mean

        rows.append({
            "signal":      sig,
            "sus_mean":    round(sus_mean, 2),
            "leg_mean":    round(leg_mean, 2),
            "separation":  round(separation, 2),
        })

    return sorted(rows, key=lambda x: -x["separation"])


def analyse_errors(results: list[dict], threshold: float) -> dict:
    """Identify false positives and false negatives with signal breakdown."""
    fps, fns = [], []

    for r in results:
        predicted = r["score"] >= threshold
        actual    = r["label"] == "suspicious"

        if predicted and not actual:
            # False positive: legitimate profile flagged as suspicious
            top_culprits = sorted(
                r.get("feature_scores", {}).items(),
                key=lambda x: -x[1]
            )[:3]
            fps.append({
                "username": r["username"],
                "score":    r["score"],
                "source":   r.get("source", "unknown"),
                "why_flagged": [f"{s}: {v:.1f}" for s, v in top_culprits],
            })

        elif not predicted and actual:
            # False negative: suspicious profile missed
            top_missed = sorted(
                r.get("feature_scores", {}).items(),
                key=lambda x: -x[1]
            )[:3]
            fns.append({
                "username":   r["username"],
                "score":      r["score"],
                "source":     r.get("source", "unknown"),
                "top_signals": [f"{s}: {v:.1f}" for s, v in top_missed],
            })

    return {"false_positives": fps, "false_negatives": fns}


# Score all profiles in a labeled dataset

def score_all(labeled: list[dict]) -> list[dict]:
    results = []
    for item in labeled:
        username = item.get("username", "unknown")
        label    = item.get("label", "unknown")
        profile  = item.get("profile", {})
        contests = item.get("contests", [])

        if not profile:
            print(f"  ⚠  Skipping {username} — no profile data")
            continue

        try:
            feats        = fm.compute_all_features(profile, contests)
            score_result = sm.compute_anomaly_score(feats)
            results.append({
                "username":     username,
                "label":        label,
                "score":        score_result["score"],
                "verdict":      score_result["verdict"],
                "data_quality": score_result["data_quality"],
                "source":       item.get("source", "manual"),
                "feature_scores": {
                    f["label"]: round(f["score"], 2)
                    for f in score_result["feature_scores"]
                    if f.get("eff_weight", 0) > 0
                },
                "interaction_bonus": score_result.get("interaction_bonus", 0.0),
            })
        except Exception as exc:
            print(f"  ERROR scoring {username}: {exc}")

    return results


# Markdown report generator

def build_report(
    results: list[dict],
    best_m: dict,
    signal_rows: list[dict],
    errors: dict,
    dataset_path: str,
) -> str:
    n_sus = sum(1 for r in results if r["label"] == "suspicious")
    n_leg = sum(1 for r in results if r["label"] == "legitimate")
    scores_sus = [r["score"] for r in results if r["label"] == "suspicious"]
    scores_leg = [r["score"] for r in results if r["label"] == "legitimate"]

    lines = [
        f"# ContestLens — Evaluation Report",
        f"",
        f"**Methodology:** `{sm.METHODOLOGY_VERSION}`  ",
        f"**Dataset:** `{dataset_path}` — {len(results)} profiles "
        f"({n_sus} suspicious, {n_leg} legitimate)  ",
        f"",
        f"---",
        f"",
        f"## Score distributions",
        f"",
        f"| Group | Mean | Min | Max |",
        f"|---|---|---|---|",
        f"| Suspicious | {sum(scores_sus)/len(scores_sus):.2f} | "
        f"{min(scores_sus):.2f} | {max(scores_sus):.2f} |" if scores_sus else
        "| Suspicious | — | — | — |",
        f"| Legitimate | {sum(scores_leg)/len(scores_leg):.2f} | "
        f"{min(scores_leg):.2f} | {max(scores_leg):.2f} |" if scores_leg else
        "| Legitimate | — | — | — |",
        f"",
        f"---",
        f"",
        f"## Best threshold: {best_m['threshold']}",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Accuracy | {best_m['accuracy']:.1%} |",
        f"| Precision | {best_m['precision']:.1%} |",
        f"| Recall | {best_m['recall']:.1%} |",
        f"| F1 | {best_m['f1']:.3f} |",
        f"| Specificity | {best_m['specificity']:.1%} |",
        f"| TP / FP / TN / FN | {best_m['tp']} / {best_m['fp']} / "
        f"{best_m['tn']} / {best_m['fn']} |",
        f"",
        f"---",
        f"",
        f"## Signal discrimination power",
        f"",
        f"Higher separation = better discriminator between suspicious and legitimate.",
        f"",
        f"| Signal | Suspicious mean | Legitimate mean | Separation |",
        f"|---|---|---|---|",
    ]

    for row in signal_rows:
        lines.append(
            f"| {row['signal']} | {row['sus_mean']:.2f} | "
            f"{row['leg_mean']:.2f} | **{row['separation']:+.2f}** |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Error analysis",
        f"",
        f"### False positives ({len(errors['false_positives'])} legitimate profiles flagged)",
        f"",
    ]

    if errors["false_positives"]:
        for e in errors["false_positives"]:
            lines.append(f"**{e['username']}** (score={e['score']:.2f}, source={e['source']})")
            lines.append(f"- Signals that caused the flag: {', '.join(e['why_flagged'])}")
            lines.append("")
    else:
        lines.append("None.")
        lines.append("")

    lines += [
        f"### False negatives ({len(errors['false_negatives'])} suspicious profiles missed)",
        f"",
    ]

    if errors["false_negatives"]:
        for e in errors["false_negatives"]:
            lines.append(f"**{e['username']}** (score={e['score']:.2f}, source={e['source']})")
            lines.append(f"- Top signals (still low): {', '.join(e['top_signals'])}")
            lines.append("")
    else:
        lines.append("None.")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## Per-profile scores",
        f"",
        f"| Username | Label | Score | Verdict | Quality |",
        f"|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda x: -x["score"]):
        marker = " ⚠" if (
            (r["label"] == "suspicious") != (r["score"] >= best_m["threshold"])
        ) else ""
        lines.append(
            f"| {r['username']}{marker} | {r['label']} | "
            f"{r['score']:.2f} | {r['verdict']} | {r['data_quality']} |"
        )

    lines += [
        f"",
        f"*⚠ = misclassified at best threshold*",
        f"",
        f"---",
        f"",
        f"*Generated by ContestLens `evaluate.py`*",
    ]

    return "\n".join(lines)


# Data collector

def collect_profiles(
    usernames: list[str],
    labels: list[str],
    sources: list[str],
    output_path: str,
    delay: float = 2.0,
) -> None:
    import fetcher

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if p.exists():
        existing = json.loads(p.read_text())
    existing_users = {e["username"].lower() for e in existing}

    for username, label, source in zip(usernames, labels, sources):
        if username.lower() in existing_users:
            print(f"  skip  {username} (already in dataset)")
            continue

        print(f"  fetch {username:<20} label={label:<12} source={source}", end="  ", flush=True)
        try:
            raw = fetcher.fetch_all(username)
            existing.append({
                "username": username,
                "label":    label,
                "source":   source,
                "profile":  raw["profile"],
                "contests": raw["contests"],
            })
            p.write_text(json.dumps(existing, indent=2, default=str))
            print("OK")
        except Exception as exc:
            print(f"FAIL: {exc}")

        time.sleep(delay)

    print(f"\nDataset: {len(existing)} total profiles saved to {p}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ContestLens — evaluate anomaly scorer accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py collect --suspicious baduser1 baduser2 --legitimate tourist neal_wu --output data/labeled.json
  python evaluate.py eval --labeled data/labeled.json
  python evaluate.py eval --labeled data/labeled.json --threshold 5.5 --report report.md
        """,
    )
    sub = parser.add_subparsers(dest="cmd")

    # collect
    col = sub.add_parser("collect", help="Fetch profiles and save to labeled dataset")
    col.add_argument("--suspicious", nargs="*", default=[], metavar="USERNAME",
                     help="Usernames to label as suspicious")
    col.add_argument("--legitimate", nargs="*", default=[], metavar="USERNAME",
                     help="Usernames to label as legitimate")
    col.add_argument("--source", default="manual",
                     help="Where these labels come from (e.g. reddit_thread, ban_announcement)")
    col.add_argument("--output", required=True, metavar="PATH",
                     help="Output JSON file path (e.g. data/labeled.json)")
    col.add_argument("--delay", type=float, default=2.0,
                     help="Seconds between API calls (default 2.0)")

    # eval
    ev = sub.add_parser("eval", help="Score a labeled dataset and print metrics")
    ev.add_argument("--labeled", required=True, metavar="PATH",
                    help="Path to labeled JSON dataset")
    ev.add_argument("--threshold", type=float, default=None,
                    help="Score threshold for suspicious (default: auto-tune for best F1)")
    ev.add_argument("--report", default=None, metavar="PATH",
                    help="Save a markdown report to this path")
    ev.add_argument("--out", default=None, metavar="PATH",
                    help="Save per-profile JSON results to this path")

    args = parser.parse_args()

    # ---- COLLECT ----
    if args.cmd == "collect":
        usernames = args.suspicious + args.legitimate
        if not usernames:
            print("Provide at least one --suspicious or --legitimate username.")
            sys.exit(1)

        labels  = (["suspicious"] * len(args.suspicious) +
                   ["legitimate"] * len(args.legitimate))
        sources = [args.source] * len(usernames)

        collect_profiles(usernames, labels, sources, args.output, args.delay)
        return

    # ---- EVAL ----
    if args.cmd == "eval":
        p = Path(args.labeled)
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)

        labeled = json.loads(p.read_text())
        if not labeled:
            print("Dataset is empty.")
            sys.exit(1)

        n_sus = sum(1 for x in labeled if x.get("label") == "suspicious")
        n_leg = sum(1 for x in labeled if x.get("label") == "legitimate")
        print(f"\nContestLens evaluation — methodology: {sm.METHODOLOGY_VERSION}")
        print(f"Dataset: {len(labeled)} profiles ({n_sus} suspicious, {n_leg} legitimate)")
        print(f"Scoring", end="", flush=True)

        results = score_all(labeled)
        print(f" done.\n")

        if not results:
            print("No results — check dataset file.")
            sys.exit(1)

        # Per-profile table
        print(f"{'':2}{'Username':<22} {'Label':<12} {'Score':>6}  {'Verdict':<12}  Quality")
        print("  " + "-" * 65)
        for r in sorted(results, key=lambda x: -x["score"]):
            mismatch = (r["label"] == "suspicious") != (r["score"] >= (
                args.threshold or find_best_threshold(results)["threshold"]
            ))
            flag = " !" if mismatch else "  "
            print(
                f"{flag}{r['username']:<22} {r['label']:<12} "
                f"{r['score']:>6.2f}  {r['verdict']:<12}  {r['data_quality']}"
            )

        print()

        # Metrics
        if args.threshold:
            best_m = compute_metrics(results, args.threshold)
            print(f"=== Metrics at threshold {args.threshold} ===")
        else:
            best_m = find_best_threshold(results)
            print(f"=== Best threshold: {best_m['threshold']} (auto-tuned by F1) ===")

        print(f"  Accuracy    {best_m['accuracy']:.1%}")
        print(f"  Precision   {best_m['precision']:.1%}  "
              "← of profiles flagged, how many are actually suspicious")
        print(f"  Recall      {best_m['recall']:.1%}  "
              "← of all suspicious profiles, how many did we catch")
        print(f"  F1          {best_m['f1']:.3f}")
        print(f"  Specificity {best_m['specificity']:.1%}  "
              "← of legitimate profiles, how many did we correctly clear")
        print(confusion_matrix_str(best_m))

        # Threshold sweep
        print(f"\n{'Threshold':>10}  {'Accuracy':>9}  {'Precision':>9}  "
              f"{'Recall':>7}  {'F1':>7}  {'TP':>3}  {'FP':>3}  {'TN':>3}  {'FN':>3}")
        print("  " + "-" * 68)
        for t in [x * 0.5 for x in range(2, 19)]:
            m      = compute_metrics(results, t)
            marker = " <- best" if t == best_m["threshold"] else ""
            print(
                f"{t:>10.1f}  {m['accuracy']:>9.1%}  {m['precision']:>9.1%}  "
                f"{m['recall']:>7.1%}  {m['f1']:>7.3f}  "
                f"{m['tp']:>3}  {m['fp']:>3}  {m['tn']:>3}  {m['fn']:>3}{marker}"
            )

        # Signal importance
        sig_rows = signal_importance(results)
        print(f"\n{'Signal':<36} {'Sus mean':>9}  {'Leg mean':>9}  {'Sep':>6}")
        print("  " + "-" * 65)
        for row in sig_rows:
            bar = "+" * min(20, max(0, int(row["separation"] * 2)))
            print(
                f"  {row['signal']:<34} {row['sus_mean']:>9.2f}  "
                f"{row['leg_mean']:>9.2f}  {row['separation']:>+6.2f}  {bar}"
            )

        # Error analysis
        errors = analyse_errors(results, best_m["threshold"])
        fps    = errors["false_positives"]
        fns    = errors["false_negatives"]

        if fps:
            print(f"\nFalse positives ({len(fps)} — legitimate profiles incorrectly flagged):")
            for e in fps:
                print(f"  {e['username']} (score={e['score']:.2f}): {', '.join(e['why_flagged'])}")

        if fns:
            print(f"\nFalse negatives ({len(fns)} — suspicious profiles missed):")
            for e in fns:
                print(f"  {e['username']} (score={e['score']:.2f}): {', '.join(e['top_signals'])}")

        if not fps and not fns:
            print(f"\nNo misclassifications at threshold {best_m['threshold']}.")

        # Save outputs
        if args.out:
            Path(args.out).write_text(json.dumps({
                "methodology": sm.METHODOLOGY_VERSION,
                "metrics":     best_m,
                "results":     results,
                "errors":      errors,
            }, indent=2))
            print(f"\nJSON results saved to {args.out}")

        if args.report:
            md = build_report(results, best_m, sig_rows, errors, str(p))
            Path(args.report).write_text(md, encoding="utf-8")
            print(f"Markdown report saved to {args.report}")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
