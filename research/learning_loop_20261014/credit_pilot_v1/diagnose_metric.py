#!/usr/bin/env python3
"""Post-hoc exact decomposition; does not revise the registered outcome."""

import argparse
import json
from pathlib import Path

import numpy as np


def diagnose(folder):
    summary = json.loads((folder / "summary.json").read_text())
    analysis = json.loads((folder / "analysis.json").read_text())
    rows = []
    for seed, result in analysis["by_seed"].items():
        choice = result["selections"]["adaptation_gain"]
        if choice is None:
            rows.append({"seed": int(seed), "status": "NO_ELIGIBLE_CANDIDATE"})
            continue
        selected = next(
            r
            for r in summary["results"]
            if str(r["seed"]) == seed and r["candidate"] == choice["candidate"]
        )
        before, after = selected["baseline"]["audit"], selected["long"]["audit"]
        first_change = after["cold_success"] - before["cold_success"]
        late_change = float(
            np.mean(np.array(after["success_by_attempt"])[[1, 3, 5]])
            - np.mean(np.array(before["success_by_attempt"])[[1, 3, 5]])
        )
        adapt_change = after["q_adapt"] - before["q_adapt"]
        assert abs(adapt_change - (late_change - first_change)) < 1e-12
        rows.append(
            {
                "seed": int(seed),
                "candidate": choice["candidate"],
                "first_before": before["cold_success"],
                "first_after": after["cold_success"],
                "first_change": first_change,
                "later_success_change": late_change,
                "adaptation_metric_change": adapt_change,
                "total_performance_change": after["q_total"] - before["q_total"],
                "registered_long_guard": choice["long_competence_guard"],
                "short_feedback_first_change": (
                    selected["short"]["feedback"]["cold_success"]
                    - selected["baseline"]["feedback"]["cold_success"]
                ),
                "short_audit_first_change": (
                    selected["short"]["audit"]["cold_success"]
                    - selected["baseline"]["audit"]["cold_success"]
                ),
            }
        )
    complete = [r for r in rows if "first_change" in r]
    means = {
        key: float(np.mean([r[key] for r in complete]))
        for key in [
            "first_before",
            "first_after",
            "first_change",
            "later_success_change",
            "adaptation_metric_change",
            "total_performance_change",
        ]
    }
    return {
        "schema": "experience-credit-metric-diagnostic-v1",
        "status": "POST_HOC_EXACT_DECOMPOSITION_NOT_A_NEW_PASS_GATE",
        "registered_credit_supported": analysis["credit_supported"],
        "identity": "delta_adaptation = delta_later_success - delta_first_success",
        "by_seed": rows,
        "means": means,
        "scope": (
            "Three developmental Cooking seeds; first attempt itself allows interaction; "
            "no zero-interaction or recursive learning claim"
        ),
        "interpretation": (
            "Predicting a baseline-subtracted adaptation proxy does not establish useful "
            "learning improvement. First-attempt degradation contributes to this proxy; "
            "do not retune the registered gate on this panel."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.run)
    (args.run / "metric_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
