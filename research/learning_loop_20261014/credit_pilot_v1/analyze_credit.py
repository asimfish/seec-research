#!/usr/bin/env python3
"""Fixed descriptive analysis; scores and guards never consult long audit outcomes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SCORES = (
    "source_success",
    "source_progress",
    "feedback_cold_gain",
    "feedback_total_gain",
    "meta_final_gain",
    "adaptation_gain",
)


def ranks(values):
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=float)
    for value in np.unique(values):
        locations = np.flatnonzero(values == value)
        rank = np.count_nonzero(values < value) + (len(locations) + 1) / 2
        result[locations] = rank
    return result


def spearman(a, b):
    a, b = ranks(a), ranks(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def scores(row):
    before, after = row["baseline"]["feedback"], row["short"]["feedback"]
    return {
        "source_success": row["source_before"]["q_total"],
        "source_progress": row["short"]["source"]["q_total"] - row["source_before"]["q_total"],
        "feedback_cold_gain": after["cold_success"] - before["cold_success"],
        "feedback_total_gain": after["q_total"] - before["q_total"],
        "meta_final_gain": after["final_success"] - before["final_success"],
        "adaptation_gain": after["q_adapt"] - before["q_adapt"],
    }


def eligible(row, guard):
    before, after = row["baseline"]["feedback"], row["short"]["feedback"]
    return (
        after["cold_success"] - before["cold_success"] >= -guard - 1e-12
        and after["q_total"] - before["q_total"] >= -guard - 1e-12
    )


def select(rows, score, guard):
    candidates = [r for r in rows if eligible(r, guard)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: (-scores(r)[score], r["candidate"]))[0]["candidate"]


def analyze(summary, plan):
    by_seed, flat = {}, []
    for seed in plan["training_seeds"]:
        rows = [r for r in summary["results"] if r["seed"] == seed]
        assert len(rows) == 8 and len({r["candidate"] for r in rows}) == 8
        target = [r["long"]["audit"]["q_adapt"] - r["baseline"]["audit"]["q_adapt"] for r in rows]
        eligible_rows = [r for r in rows if eligible(r, plan["competence_guard"])]
        uniform = (
            float(
                np.mean(
                    [
                        r["long"]["audit"]["q_adapt"] - r["baseline"]["audit"]["q_adapt"]
                        for r in eligible_rows
                    ]
                )
            )
            if eligible_rows
            else None
        )
        correlations = {name: spearman([scores(r)[name] for r in rows], target) for name in SCORES}
        selections = {}
        for name in SCORES:
            chosen = select(rows, name, plan["competence_guard"])
            row = next((r for r in rows if r["candidate"] == chosen), None)
            selections[name] = (
                None
                if row is None
                else {
                    "candidate": chosen,
                    "long_audit_adaptation_gain": row["long"]["audit"]["q_adapt"]
                    - row["baseline"]["audit"]["q_adapt"],
                    "long_audit_total_gain": row["long"]["audit"]["q_total"]
                    - row["baseline"]["audit"]["q_total"],
                    "long_audit_cold_gain": row["long"]["audit"]["cold_success"]
                    - row["baseline"]["audit"]["cold_success"],
                    "long_competence_guard": eligible(
                        {
                            "baseline": {"feedback": row["baseline"]["audit"]},
                            "short": {"feedback": row["long"]["audit"]},
                        },
                        plan["competence_guard"],
                    ),
                }
            )
        by_seed[str(seed)] = {
            "rank_correlations": correlations,
            "selections": selections,
            "eligible_candidates": [r["candidate"] for r in eligible_rows],
            "uniform_adaptation_gain": uniform,
        }
        for r, gain in zip(rows, target):
            flat.append(
                {
                    "seed": seed,
                    "candidate": r["candidate"],
                    "recipe": r["recipe"],
                    **scores(r),
                    "eligible": eligible(r, plan["competence_guard"]),
                    "long_audit_adaptation_gain": gain,
                    "long_audit_total_gain": r["long"]["audit"]["q_total"]
                    - r["baseline"]["audit"]["q_total"],
                    "long_audit_cold_gain": r["long"]["audit"]["cold_success"]
                    - r["baseline"]["audit"]["cold_success"],
                    "candidate_actions": r["source_before"]["actual_actions"],
                }
            )
    mean_rho, mean_gain = {}, {}
    for name in SCORES:
        rho = [r["rank_correlations"][name] for r in by_seed.values()]
        gain = [r["selections"][name] for r in by_seed.values()]
        mean_rho[name] = None if any(v is None for v in rho) else float(np.mean(rho))
        mean_gain[name] = (
            None
            if any(v is None for v in gain)
            else float(np.mean([v["long_audit_adaptation_gain"] for v in gain]))
        )
    others = [name for name in SCORES if name != "adaptation_gain"]
    comparable_rho = [mean_rho[name] for name in others if mean_rho[name] is not None]
    comparable_gain = [mean_gain[name] for name in others if mean_gain[name] is not None]
    uniform_values = [r["uniform_adaptation_gain"] for r in by_seed.values()]
    uniform_mean = (
        None if any(v is None for v in uniform_values) else float(np.mean(uniform_values))
    )
    criteria = plan["primary_decision"]
    defined = (
        mean_rho["adaptation_gain"] is not None
        and bool(comparable_rho)
        and mean_gain["adaptation_gain"] is not None
        and uniform_mean is not None
    )
    checks = {
        "adaptation_rank_at_least_0_3": defined
        and mean_rho["adaptation_gain"] >= criteria["adaptation_mean_spearman_min"],
        "rank_exceeds_best_other_by_0_1": defined
        and mean_rho["adaptation_gain"] - max(comparable_rho)
        >= criteria["rank_margin_over_best_other"],
        "positive_rank_all_seeds": all(
            r["rank_correlations"]["adaptation_gain"] is not None
            and r["rank_correlations"]["adaptation_gain"] > 0
            for r in by_seed.values()
        ),
        "selected_gain_exceeds_uniform_by_0_02": defined
        and mean_gain["adaptation_gain"] - uniform_mean
        >= criteria["selected_gain_over_uniform_min"],
        "selected_gain_exceeds_best_other_by_0_02": defined
        and bool(comparable_gain)
        and mean_gain["adaptation_gain"] - max(comparable_gain)
        >= criteria["selected_gain_over_best_other_min"],
    }
    selected_safe = all(
        r["selections"]["adaptation_gain"] is not None
        and r["selections"]["adaptation_gain"]["long_competence_guard"]
        for r in by_seed.values()
    )
    checks["long_selected_competence_retained"] = selected_safe
    return {
        "schema": "experience-credit-ranking-analysis-v1",
        "plan_id": plan["plan_id"],
        "by_seed": by_seed,
        "mean_rank_correlations": mean_rho,
        "mean_selected_adaptation_gain": mean_gain,
        "mean_uniform_adaptation_gain": uniform_mean,
        "registered_checks": checks,
        "credit_supported": bool(all(checks.values())),
        "decision": "CONTINUE_TO_CLOSED_LOOP_DESIGN"
        if all(checks.values())
        else "NO_NEW_CREDIT_SUPPORT_RETAIN_SIMPLER_EXPLANATIONS",
        "constant_correlations": "UNDEFINED; no fabricated zero/p-value",
        "candidates": flat,
        "cost": summary["cost"],
        "scope": summary["scope"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    plan = json.loads((args.run / "plan.json").read_text())
    result = analyze(summary, plan)
    (args.run / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.run / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["candidates"][0]))
        writer.writeheader()
        writer.writerows(result["candidates"])
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"candidates", "by_seed"}}, indent=2
        )
    )
