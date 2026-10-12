#!/usr/bin/env python3
"""Post-pilot registered mechanism probe: shared actual exploration histories.

This is a controlled evaluation with an externally supplied prefix, not the
teacher-free primary result. No checkpoint, budget or model is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import run_learning_pilot as pilot

CONDITIONS = ("shared_actual_feedback", "shared_inventory_redacted")


def prefix_tokens(events, goal, redacted):
    tokens = []
    previous_action, previous_reward = 9, 0.0
    for t, event in enumerate(events):
        observation = list(event["observation"])
        if redacted:
            observation[3] = 4  # declared redaction intervention, not raw obs
        tokens.append(pilot.encode(observation, goal, previous_action, previous_reward,
                                   t == 0, True, t))
        previous_action, previous_reward = event["executed_action"], event["reward"]
    final = list(events[-1]["next_observation"])
    if redacted:
        final[3] = 4
    tokens.append(pilot.terminal_feedback(final, goal, previous_action, previous_reward, True, len(events)))
    return np.asarray(tokens)


def matched_rollout(Env, Wrapper, model, task, prefix, condition, trials):
    hidden = None
    for token in prefix_tokens(prefix, task["goal"], condition == CONDITIONS[1]):
        _, hidden, _ = pilot.predict(model, token, hidden)
    rows = []
    for trial in range(1, trials + 1):
        env = pilot.environment(Env, Wrapper, task, False)
        state = env.reset()
        previous_action, previous_reward = 9, 0.0
        events = []
        for t in range(20):
            before = state.observation.tolist()
            action, hidden, _ = pilot.predict(model, pilot.encode(
                before, task["goal"], previous_action, previous_reward,
                t == 0, False, t), hidden)
            state, reward, done, _ = env.step(action)
            after = state.observation.tolist()
            events.append({"time": t, "observation": before, "executed_action": action,
                           "next_observation": after, "reward": float(reward)})
            previous_action, previous_reward = action, reward
            if done:
                break
        success = bool(np.array_equal(state.observation[4:6], task["goal"]))
        _, hidden, _ = pilot.predict(model, pilot.terminal_feedback(
            after, task["goal"], previous_action, previous_reward, False, len(events)), hidden)
        rows.append({"trial": trial, "success": success, "actual_steps": len(events),
                     "cost_steps": len(events) if success else 20, "events": events})
    return rows


def register(parent, target):
    assert not target.exists()
    trained = json.loads((parent / "summary.json").read_text())
    plan = {
        "schema": "matched-history-mechanism-plan-v1",
        "parent_summary_sha256": pilot.sha(parent / "summary.json"),
        "parent_plan_sha256": trained["plan_sha256"],
        "runner_sha256": pilot.sha(__file__),
        "registered_parent_verified": json.loads((parent / "verification.json").read_text())["status"],
        "conditions": list(CONDITIONS), "prefix_seed": 11,
        "checkpoint_sha256": {f"{m['arm']}-{m['seed']}": m["checkpoint_sha256"] for m in trained["models"]},
        "input_control": "All nine frozen checkpoints consume the same real 20-action exploration prefix for each task. The redacted condition replaces only prefix inventory responses by EMPTY; execution observations remain original. Terminal feedback is included.",
        "metric": "Five-attempt success, paired by training seed; additional first-attempt success isolates prefix information before new execution evidence.",
        "hypotheses": {
            "H_information_use": "Observed-fact model exceeds both controls by >=0.05 under the common real prefix. Removing identifying inventory responses reduces its success by >=0.05; its full-minus-redacted information effect exceeds each control's by >=0.05.",
            "H_exploration_only": "Auxiliary advantage falls below 0.05 with a common informative prefix; original gain explained by different learned exploration rather than use of common evidence.",
        },
        "limits": ["Designed after primary pilot results; exploratory mechanism probe, not confirmatory inference.",
                   "External symbolic prefix is a shared diagnostic input, not teacher-free autonomous exploration.",
                   "EMPTY redaction is an input intervention, not a claim that the simulator returned that observation.",
                   "No new gradients, checkpoint selection, task/horizon changes or extra information for one arm."]
    }
    import datetime
    plan["registered_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan["semantic_sha256"] = pilot.digest(plan)
    plan["plan_id"] = "matched-history-" + plan["semantic_sha256"][:16]
    pilot.write(target, plan)
    print(json.dumps({"plan_id": plan["plan_id"], "sha256": plan["semantic_sha256"]}))


def run(args):
    parent = args.parent
    plan = json.loads(args.plan.read_text())
    assert plan["runner_sha256"] == pilot.sha(__file__)
    assert pilot.digest({k: v for k, v in plan.items() if k not in {"semantic_sha256", "plan_id"}}) == plan["semantic_sha256"]
    assert plan["parent_summary_sha256"] == pilot.sha(parent / "summary.json")
    assert not args.output.exists()
    args.output.mkdir()
    (args.output / "plan.json").write_bytes(args.plan.read_bytes())
    parent_plan = json.loads((parent / "plan.json").read_text())
    Env, Wrapper = pilot.load_environment(args.upstream)
    torch.set_num_threads(2)
    prefixes, masked_ids = {}, set()
    with (args.output / "prefixes.jsonl").open("w") as f:
        for task in pilot.tasks_for(parent_plan["evaluation_environment_ids"]):
            sequence = pilot.collect_sequence(Env, Wrapper, task, plan["prefix_seed"], 0.0, 0)
            prefixes[task["task_id"]] = sequence["events"]
            f.write(json.dumps({"task": task, "events": sequence["events"]}, sort_keys=True) + "\n")
            masked_ids.add(hashlib.sha256(prefix_tokens(sequence["events"], task["goal"], True).tobytes()).hexdigest())
    assert len(masked_ids) == 1, "Redacted prefix must remove all mapping-identifying observation differences"
    grouped = {}
    chain, count, steps = "0" * 64, 0, 0
    with (args.output / "evaluation.jsonl").open("w") as f:
        for seed in parent_plan["training_seeds"]:
            for arm in pilot.ARMS:
                model = pilot.HistoryPolicy(parent_plan["hidden_size"]).eval()
                checkpoint = parent / "checkpoints" / f"{arm}-{seed}" / "policy.pt"
                assert pilot.sha(checkpoint) == plan["checkpoint_sha256"][f"{arm}-{seed}"]
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                for condition in CONDITIONS:
                    outcomes = []
                    for task in pilot.tasks_for(parent_plan["evaluation_environment_ids"]):
                        rows = matched_rollout(Env, Wrapper, model, task, prefixes[task["task_id"]], condition, 5)
                        for result in rows:
                            record = {**task, **result, "arm": arm, "seed": seed, "condition": condition,
                                      "plan_sha256": plan["semantic_sha256"], "previous_hash": chain}
                            chain = pilot.digest(record)
                            record["record_hash"] = chain
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                            count += 1
                            steps += record["actual_steps"]
                            outcomes.append(result)
                    key = f"{arm}/{seed}/{condition}"
                    grouped[key] = pilot.summarize(outcomes)
                    print(json.dumps({"configuration": key, **grouped[key]}), flush=True)
    n = len(prefixes) * 9 * 2 * 5
    assert count == n
    summary = {"schema": "matched-history-mechanism-summary-v1", "plan_id": plan["plan_id"],
               "plan_sha256": plan["semantic_sha256"], "evaluations": grouped,
               "coverage": {"planned_records": n, "completed_records": n, "missing": 0, "aborted": 0},
               "checkpoint_changes": 0, "optimizer_updates": 0, "chain_head": chain,
               "cost": {"prefix_collection_actions": len(prefixes) * 20, "prefix_collection_resets": len(prefixes),
                        "execution_actions": steps, "execution_resets": n},
               "redacted_prefix_unique_count": len(masked_ids),
               "artifacts": {name: pilot.sha(args.output / name) for name in ["prefixes.jsonl", "evaluation.jsonl", "plan.json"]},
               "scope": "Post-pilot shared-prefix/redaction mechanism diagnostic. External prefix; not teacher-free primary result."}
    pilot.write(args.output / "summary.json", summary)
    print(json.dumps({"status": "COMPLETED", "coverage": summary["coverage"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.register:
        register(args.parent, args.plan)
    else:
        run(args)
