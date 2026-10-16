#!/usr/bin/env python3
"""Registered seed-level analysis; incomplete cells produce no effect conclusion."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from robocore.adaptation.robomme_absorption import paired_contrasts as validate_paired_contrasts
from robocore.adaptation.robomme_continuation import file_sha

VIEWS = {"full_history_intervention_bc": "full", "recent_history_intervention_bc": "recent"}


def paired_contrasts(records):
    # Registered cells each contain two tasks x 50 cases. Calculate gains and
    # interactions on integer success counts before converting to a rate;
    # float subtraction must never turn a mathematical tie into a positive seed.
    rates = validate_paired_contrasts(records)
    successes = Counter()
    for row in records:
        successes[row["seed"], row["view"], row["phase"]] += int(row["success"])
    for result in rates:
        seed = result["seed"]
        full = successes[seed, "full", "after"] - successes[seed, "full", "before"]
        recent = successes[seed, "recent", "after"] - successes[seed, "recent", "before"]
        result["full_absorption_gain"] = full / 100
        result["recent_absorption_gain"] = recent / 100
        result["history_interaction"] = (full - recent) / 100
    return rates


def summarize(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(131)
    means = values[rng.integers(len(values), size=(100000, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    positive, negative = int((values > 0).sum()), int((values < 0).sum())
    n = positive + negative
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n)
    return dict(
        mean=float(values.mean()),
        ci95_seed_bootstrap=[float(low), float(high)],
        exact_two_sided_sign_p=p,
        nonzero_seeds=n,
        ties=len(values) - n,
    )


def analyze(plan, plan_sha, results):
    expected_cases = {(r["task"], r["episode"]) for r in plan["cases"]}
    cells = {}
    for result in results:
        phase, view, seed = result["phase"], result["view"], result["training_seed"]
        key = phase, view, seed
        if key in cells or view not in VIEWS or phase not in ("before", "after"):
            raise ValueError("Duplicated or unregistered model cell")
        if (phase == "before" and seed is not None) or (
            phase == "after" and seed not in plan["training_seeds"]
        ):
            raise ValueError("Unregistered training seed")
        if result["plan_id"] != plan["plan_id"] or result["plan_sha256"] != plan_sha:
            raise ValueError("Evaluation plan binding differs")
        if result["status"] != "COMPLETE" or result["completed"] != 100:
            raise ValueError("Incomplete evaluation model")
        rows = result["cases"]
        keys = [(r["task"], r["episode"]) for r in rows]
        if len(keys) != 100 or set(keys) != expected_cases:
            raise ValueError("Missing or duplicate paired benchmark cases")
        if any(r["status"] != "COMPLETED" or not isinstance(r["success"], bool) for r in rows):
            raise ValueError("Incomplete terminal outcomes")
        cells[key] = rows
    required = {("before", view, None) for view in VIEWS}
    required |= {("after", view, seed) for view in VIEWS for seed in plan["training_seeds"]}
    if cells.keys() != required:
        raise ValueError(
            "All six training seeds, both views and both initial references are mandatory"
        )
    records = []
    for seed in plan["training_seeds"]:
        for view, short in VIEWS.items():
            for phase in ("before", "after"):
                for row in cells[(phase, view, None if phase == "before" else seed)]:
                    records.append({**row, "seed": seed, "view": short, "phase": phase})
    contrasts = paired_contrasts(records)
    primary = summarize([r["full_absorption_gain"] for r in contrasts])
    recent = summarize([r["recent_absorption_gain"] for r in contrasts])
    interaction = summarize([r["history_interaction"] for r in contrasts])
    primary_pass = (
        primary["mean"] >= 0.05
        and primary["ci95_seed_bootstrap"][0] > 0
        and primary["exact_two_sided_sign_p"] <= 0.05
    )
    all_zero = all(r["full_absorption_gain"] == 0 for r in contrasts)
    all_floor = all(r["full_after"] == r["recent_after"] == 0 for r in contrasts)
    ceiling = all(r["full_before"] == 1 for r in contrasts)
    if all_zero or all_floor or ceiling:
        decision = "UNINFORMATIVE_OR_FAILED_FEASIBILITY"
    elif primary_pass:
        decision = "REGISTERED_TWO_TASK_ABSORPTION_GAIN_SUPPORTED"
    elif primary["ci95_seed_bootstrap"][1] < 0.05:
        decision = "PRACTICAL_GAIN_NOT_SUPPORTED_IN_REGISTERED_CONFIGURATION"
    else:
        decision = "INCONCLUSIVE"
    return dict(
        status="COMPLETE_REGISTERED_ANALYSIS",
        decision=decision,
        per_seed=contrasts,
        primary=primary,
        recent_absorption=recent,
        history_interaction=interaction,
        history_interaction_inference_allowed=primary_pass,
        training_seeds=len(contrasts),
        independent_initial_reference_runs=2,
        before_references_shared_not_replicates=True,
        evaluation_episodes=1400,
        conditioning="native initial demonstration/video retained",
        interval_scope="seed variation conditional on these fixed two tasks and cases",
        algorithmic_novelty="NOT_ESTABLISHED; full-history arm aliases one-round DAgger",
        full_suite_generalization="NOT_EVALUATED",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True, action="append")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    result = analyze(
        plan, file_sha(args.plan), [json.loads(path.read_text()) for path in args.result]
    )
    result.update(
        plan_id=plan["plan_id"],
        plan_sha256=file_sha(args.plan),
        result_files=[{"sha256": file_sha(path), "path": str(path)} for path in args.result],
        analyzer_sha256=file_sha(Path(__file__)),
        arithmetic_version="v1.1 integer counts before rate differences",
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
