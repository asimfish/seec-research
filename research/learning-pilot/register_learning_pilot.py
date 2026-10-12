#!/usr/bin/env python3
"""Write a complete immutable internal registration before dataset/training."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent


def register():
    target = ROOT / "plan.json"
    assert not target.exists(), "Do not revise a registered plan in place"
    old = json.loads((PARENT / "plan.json").read_text())
    names = ["run_learning_pilot.py", "register_learning_pilot.py", "test_learning_pilot.py",
             "verify_learning_pilot.py", "PROTOCOL.md", "../run_cooking_diagnostic.py"]
    code = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    plan = {
        "schema": "cooking-learned-pilot-plan-v1",
        "registered_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_only_correction": "run1 invalid: add actual terminal feedback token before episode reset. No task, horizon, split, seed or training-budget change.",
        "previous_invalid_plan": json.loads((ROOT / "plan_run1_invalid.json").read_text())["semantic_sha256"],
        "upstream_revision": old["upstream_revision"], "code_hashes": code,
        "previous_diagnostic_plan": old["semantic_sha256"],
        "train_environment_ids": old["train_environment_ids"],
        "development_environment_ids": old["development_environment_ids"],
        "evaluation_environment_ids": old["evaluation_environment_ids"],
        "training_seeds": [7, 19, 41], "behavior_seeds": [11, 23, 37],
        "behavior_epsilons": [0.0, 0.1], "trials": 5, "exploration_horizon": 20,
        "execution_horizon": 20, "updates": 800, "batch_sequences": 24,
        "hidden_size": 96, "learning_rate": 0.003, "auxiliary_weight": 0.2,
        "gradient_clip": 1.0, "cpu_threads": 2, "optimizer": "Adam",
        "arms": ["action_only", "current_state_aux", "observed_facts_aux"],
        "input": "One-hot current raw observation and goal, previous actual action/reward, episode-start, phase, local time; causal GRU across attempts. Every terminal response is ingested as a feedback-only token before reset, with no action loss or extra simulator action. No env_id, hidden mapping, fact labels or symbolic teacher at inference.",
        "labels": {
            "action": "Observable-memory teacher action at current history, even if stochastic collection executed another action.",
            "current_state_aux": "Inventory and first two bowl slots at the current observation.",
            "observed_facts_aux": "Three drawer facts reconstructed only from prior actual pickups and observed inventory; unobserved=unknown.",
        },
        "budget": "Same fixed source, initial weights, per-seed batch indices, 800 updates and model parameter count. All learned policies act from step 0; no teacher prefix. Both exploration/execution costs counted.",
        "checkpoint": "Fixed final update 800; no tuning or checkpoint selection on development/evaluation.",
        "primary_metric": "Mean success rate over five attempts on the 24-map panel; paired differences at training-seed level. 3 seeds: descriptive pilot only, no confirmatory p-values.",
        "secondary_metrics": ["Success by attempt", "Failures charged horizon 20", "Including exploration cost", "Teacher-forced source accuracy", "Reset-between-episodes memory ablation"],
        "hypotheses": {
            "H0_sufficient_history": "Full-history action supervision already sufficient; grounded auxiliary improvement <0.05 or inconsistent across seeds.",
            "H1_grounded_update": "Observed-fact auxiliary success exceeds BOTH action_only and current_state_aux by >=0.05, each seed's difference positive; within-task memory reset lowers success >=0.05.",
            "H2_generic_auxiliary": "Current-state and observed-fact auxiliary supervision improve similarly; result explained by extra supervision rather than evidence updating.",
            "H3_skill_floor": "All learned arms have mean success <0.20. Investigate policy fitting/distribution shift; do not infer no value of task-rule learning.",
        },
        "predictive_check": "Symbolic full-history reference should remain 100% successful; known-fact target must be unknown before its first observed pickup. A learned floor is possible and will be reported, not fixed by changing the evaluation horizon.",
        "stop": "One fixed 9-training-run panel. Stop for registered budget or discrimination exhaustion; no hyperparameter/horizon changes on this panel. Infrastructure bugs may be corrected with an explicit invalid-run record and new plan.",
        "limitations": [
            "Internal timestamp/hash registration, not an independent preregistration service.",
            "Reuses the prior symbolic diagnostic's task split; not a new independent confirmatory test set.",
            "Fixed teacher/perturbed histories: offline BC, not OPD or learning-credit weighting.",
            "Auxiliary known-fact labels may improve representation; not by itself a novel algorithm or proof of task-learning-program absorption.",
            "Public DREAM environment, modified 32/8/24 evaluation protocol, not reproduction of the author's trained method.",
            "CPU because CUDA initialization currently fails; no driver or shared-GPU changes.",
        ],
    }
    body = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    plan["semantic_sha256"] = hashlib.sha256(body).hexdigest()
    plan["plan_id"] = "cooking-learned-v1-" + plan["semantic_sha256"][:16]
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"plan_id": plan["plan_id"], "semantic_sha256": plan["semantic_sha256"], "registered_at_utc": plan["registered_at_utc"]}))


if __name__ == "__main__":
    register()
