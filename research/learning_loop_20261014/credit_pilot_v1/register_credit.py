#!/usr/bin/env python3
"""Write a complete immutable plan before collecting/training the credit pilot."""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import run_credit as run


def specs(ids):
    all_tasks = run.pilot.tasks_for(ids)
    recipes = [
        ("one-repeat", 1, True),
        ("two-repeat", 2, True),
        ("two-mixed", 2, False),
        ("three-repeat", 3, True),
        ("three-mixed", 3, False),
        ("all-repeat", None, True),
        ("all-mixed", None, False),
        ("all-goals", None, None),
    ]
    result = []
    for index, (name, unique, repeat) in enumerate(recipes):
        pool = [
            t
            for t in all_tasks
            if (
                unique is None
                or len({t["environment_id"] // (4**j) % 4 for j in range(3)}) == unique
            )
            and (repeat is None or (t["goal"][0] == t["goal"][1]) == repeat)
        ]
        assert pool
        rng = np.random.RandomState(17014 + index)
        chosen = [pool[int(i)] for i in rng.randint(0, len(pool), size=24)]
        result.append(
            {
                "id": f"batch-{index:02d}",
                "recipe": name,
                "task_pool_size": len(pool),
                "sampling": "With replacement; identical copies are weights, not independent data",
                "tasks": chosen,
            }
        )
    return result


def plan(folder):
    parent = json.loads((run.LEGACY / "learner_pilot_v1" / "plan.json").read_text())
    source, feedback, audit = (
        parent["train_environment_ids"],
        parent["development_environment_ids"],
        parent["evaluation_environment_ids"],
    )
    assert all(
        set(a).isdisjoint(b) for a, b in [(source, feedback), (source, audit), (feedback, audit)]
    )
    nf, na = len(run.task_panel(feedback)), len(run.task_panel(audit))
    base_n = len(run.pilot.tasks_for(source)) * 3 * 2
    histories = 3 * (nf + na) + 24 * (24 + nf + na) + 24 * na
    code_names = [
        "run_credit.py",
        "register_credit.py",
        "verify_credit.py",
        "analyze_credit.py",
        "test_credit.py",
        "PROTOCOL.md",
        "../../interactive_learning_20261013/student_coverage_v1/run_coverage.py",
        "../../interactive_learning_20261013/learner_pilot_v1/run_learning_pilot.py",
        "../../interactive_learning_20261013/run_cooking_diagnostic.py",
    ]
    p = {
        "schema": "experience-credit-ranking-plan-v1",
        "registered_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": (
            "Cooking training-side diagnostic; no sealed test, full method "
            "reproduction, robot or recursive efficacy claim"
        ),
        "upstream_revision": run.pilot.UPSTREAM_REVISION
        if hasattr(run.pilot, "UPSTREAM_REVISION")
        else "d52204c94067641df6e6649b19abb359b87ff028",
        "code_hashes": {name: run.pilot.sha(folder / name) for name in code_names},
        "source_ids": source,
        "feedback_ids": feedback,
        "audit_ids": audit,
        "feedback_cases": nf,
        "audit_cases": na,
        "candidate_specs": specs(source),
        "training_seeds": [101, 103, 107],
        "initial_probe_budgets": [0, 8, 20],
        "initial_updates": 800,
        "short_updates": 64,
        "long_updates": 300,
        "batch_sequences": 24,
        "branch_mixture": "12 base histories + 12 candidate histories per update",
        "hidden_size": 96,
        "learning_rate": 0.003,
        "gradient_clip": 1.0,
        "cpu_threads": 2,
        "base_sequences": base_n,
        "candidate_sequences_per_seed": 192,
        "evaluation_histories": histories,
        "evaluation_episodes": histories * 6,
        "total_updates": 3 * 800 + 24 * (64 + 300),
        "max_source_actions": base_n * 140 + 3 * 192 * 120,
        "max_evaluation_actions": histories * 120,
        "max_wall_seconds": 5400,
        "adaptation_budgets": [0, 1, 3, 5],
        "evaluation": (
            "Six own-policy task attempts; no initial probe, teacher, hidden rule or weight updates"
        ),
        "metrics": (
            "Macro average over rule identity, then goals; q_total mean of "
            "R0,R1,R3,R5; q_adapt mean R1,R3,R5 minus R0"
        ),
        "scores": [
            "source_success",
            "source_progress",
            "feedback_cold_gain",
            "feedback_total_gain",
            "meta_final_gain",
            "adaptation_gain",
        ],
        "selection": (
            "All scores use the same short-branch competence-eligible set; "
            "ties use lexical batch ID; uniform is the mean over this set"
        ),
        "competence_guard": 0.05,
        "primary_decision": {
            "adaptation_mean_spearman_min": 0.3,
            "rank_margin_over_best_other": 0.1,
            "selected_gain_over_uniform_min": 0.02,
            "selected_gain_over_best_other_min": 0.02,
            "adaptation_spearman_positive_all_seeds": True,
        },
        "statistical_unit": (
            "Three independent training seeds; descriptive development "
            "panel, no confirmatory p-values"
        ),
        "limits": [
            (
                "Same finite rule family and split previously used by other "
                "pilots; fresh model seeds do not make a new confirmatory "
                "benchmark"
            ),
            "No candidate changes model, teacher label boundary, loss or collector policy",
            (
                "Sequence/update counts match; valid target counts and action "
                "costs differ and are reported"
            ),
            (
                "Fixed candidate bank; this pilot does not run an evolving "
                "producer or compare full SOAR/Meta-Teaching implementations"
            ),
            (
                "Short and long branches independently restart the same weights"
                " AND Adam state; long branch is not a continuation of short"
            ),
            (
                "All audit cases are training-side analysis only and never "
                "influence model/credit/checkpoint selection during this run"
            ),
        ],
        "stop": (
            "Fixed 51 models and 11136 updates; no score, budget, "
            "threshold, data or hyperparameter revision after registered "
            "run; infra failures get explicit invalid-run evidence"
        ),
    }
    p["semantic_sha256"] = run.pilot.digest(run.semantic_binding(p))
    p["plan_id"] = "cooking-credit-" + p["semantic_sha256"][:16]
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    value = plan(Path(__file__).resolve().parent)
    run.pilot.write(args.output, value)
    print(
        json.dumps(
            {
                "plan_id": value["plan_id"],
                "semantic_sha256": value["semantic_sha256"],
                "models": 51,
                "updates": value["total_updates"],
                "evaluation_histories": value["evaluation_histories"],
            }
        )
    )
