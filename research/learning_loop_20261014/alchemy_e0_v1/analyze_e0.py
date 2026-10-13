"""Fixed E0 domain qualification, not a learned-method hypothesis test."""

import argparse
import json
from pathlib import Path

import numpy as np
from run_e0 import checked_plan, sha


def analyze(folder, plan_folder):
    plan = checked_plan(plan_folder)
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["plan_id"] == plan["plan_id"]
    assert summary["plan_sha256"] == sha(plan_folder / "plan.json")
    assert summary["coverage"]["missing"] == summary["coverage"]["aborted"] == 0
    by_arm = {}
    for arm in plan["arms"]:
        rows = [r for r in summary["episodes"] if r["arm"] == arm]
        assert [r["seed"] for r in rows] == plan["environment_seeds"]
        matrix = np.array([r["trial_rewards"] for r in rows])
        by_arm[arm] = {
            "trial_reward_means": matrix.mean(axis=0).tolist(),
            "mean_trial_reward": float(matrix.mean()),
            "mean_last_five_reward": float(matrix[:, 5:].mean()),
            "total_reward_by_seed": matrix.sum(axis=1).tolist(),
            "last_five_mean_by_seed": matrix[:, 5:].mean(axis=1).tolist(),
        }
    differences = np.array(by_arm["bayes_carry"]["last_five_mean_by_seed"]) - np.array(
        by_arm["bayes_reset"]["last_five_mean_by_seed"]
    )
    checks = {
        "paired_later_gap_at_least_two_reward_units": bool(differences.mean() >= 2),
        "gap_positive_at_least_six_of_eight": bool((differences > 0).sum() >= 6),
        "neither_bayes_arm_all_floor": bool(
            by_arm["bayes_carry"]["mean_trial_reward"] > 0
            and by_arm["bayes_reset"]["mean_trial_reward"] > 0
        ),
        "carry_not_all_at_theoretical_maximum": bool(
            by_arm["bayes_carry"]["mean_trial_reward"] < 45
        ),
    }
    report = {
        "schema": "alchemy-e0-fixed-analysis-v1",
        "plan_id": plan["plan_id"],
        "summary_sha256": sha(folder / "summary.json"),
        "by_arm": by_arm,
        "paired_carry_minus_reset_last_five": {
            "by_seed": differences.tolist(),
            "mean": float(differences.mean()),
            "positive_seeds": int((differences > 0).sum()),
        },
        "registered_checks": checks,
        "domain_qualified_for_next_pilot": all(checks.values()),
        "decision": "QUALIFIED_FOR_NEW_REGISTRATION"
        if all(checks.values())
        else "E0_GAP_NOT_ESTABLISHED",
        "statistical_unit": "Eight chemistry/environment seeds, not eighty independent trials",
        "scope": (
            "Descriptive hand-coded Bayesian-myopic qualification; no gradient training, "
            "full ideal-observer score, recursive learner or robot result"
        ),
        "cost": summary["cost"],
    }
    (folder / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan-folder", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    analyze(args.run, args.plan_folder)
