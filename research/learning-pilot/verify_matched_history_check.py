#!/usr/bin/env python3
"""Replay every shared-prefix intervention and checkpoint decision."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_learning_pilot as pilot
from run_matched_history_check import CONDITIONS, prefix_tokens
from verify_learning_pilot import aggregate


def verify(parent, folder, upstream):
    summary = json.loads((folder / "summary.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    original = json.loads((parent / "plan.json").read_text())
    assert plan["parent_summary_sha256"] == pilot.sha(parent / "summary.json")
    assert pilot.digest({k: v for k, v in plan.items() if k not in {"plan_id", "semantic_sha256"}}) == plan["semantic_sha256"]
    for name, expected in summary["artifacts"].items():
        assert pilot.sha(folder / name) == expected
    Env, Wrapper = pilot.load_environment(upstream)
    torch.set_num_threads(2)
    prefixes, prefix_steps = {}, 0
    for line in (folder / "prefixes.jsonl").open():
        record = json.loads(line)
        task = record["task"]
        assert task["environment_id"] in original["evaluation_environment_ids"]
        assert task["task_id"] not in prefixes
        env = pilot.environment(Env, Wrapper, task, True)
        state = env.reset()
        for t, event in enumerate(record["events"]):
            assert state.observation.tolist() == event["observation"] and event["time"] == t
            state, reward, done, _ = env.step(event["executed_action"])
            assert state.observation.tolist() == event["next_observation"]
            assert float(reward) == event["reward"]
            prefix_steps += 1
        assert done and len(record["events"]) == 20
        prefixes[task["task_id"]] = record
    assert set(prefixes) == {t["task_id"] for t in pilot.tasks_for(original["evaluation_environment_ids"])}
    masked_ids = {hashlib.sha256(prefix_tokens(p["events"], p["task"]["goal"], True).tobytes()).hexdigest()
                  for p in prefixes.values()}
    assert len(masked_ids) == summary["redacted_prefix_unique_count"] == 1
    prefix_comparisons, identical_prefixes = 0, 0
    fields = ["observation", "executed_action", "next_observation", "reward"]
    for line in (parent / "evaluation.jsonl").open():
        row = json.loads(line)
        if row["split"] == "evaluation" and row["condition"] == "full_history" and row["trial"] == 0 and row["arm"] in pilot.ARMS:
            prefix_comparisons += 1
            actual = [{k: e[k] for k in fields} for e in row["events"]]
            shared = [{k: e[k] for k in fields} for e in prefixes[row["task_id"]]["events"]]
            identical_prefixes += actual == shared
    models = {}
    for seed in original["training_seeds"]:
        for arm in pilot.ARMS:
            file = parent / "checkpoints" / f"{arm}-{seed}" / "policy.pt"
            assert pilot.sha(file) == plan["checkpoint_sha256"][f"{arm}-{seed}"]
            model = pilot.HistoryPolicy(original["hidden_size"]).eval()
            model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
            models[(arm, seed)] = model
    expected = {(arm, seed, condition, task["task_id"], trial)
                for arm in pilot.ARMS for seed in original["training_seeds"] for condition in CONDITIONS
                for task in pilot.tasks_for(original["evaluation_environment_ids"]) for trial in range(1, 6)}
    seen, hidden_states, grouped, chain, steps = set(), {}, defaultdict(list), "0" * 64, 0
    for line in (folder / "evaluation.jsonl").open():
        row = json.loads(line)
        key = (row["arm"], row["seed"], row["condition"], row["task_id"], row["trial"])
        assert key in expected and key not in seen
        seen.add(key)
        assert row["previous_hash"] == chain
        chain = pilot.digest({k: v for k, v in row.items() if k != "record_hash"})
        assert chain == row["record_hash"]
        assert row["plan_sha256"] == plan["semantic_sha256"]
        model = models[key[:2]]
        context_key = key[:4]
        hidden = hidden_states.get(context_key)
        if row["trial"] == 1:
            assert hidden is None
            prefix = prefixes[row["task_id"]]
            # Only transformed observed responses enter the learner, never
            # stored teacher facts, IDs or action targets.
            for token in prefix_tokens(prefix["events"], row["goal"], row["condition"] == CONDITIONS[1]):
                _, hidden, _ = pilot.predict(model, token, hidden)
        env = pilot.environment(Env, Wrapper, row, False)
        state = env.reset()
        previous_action, previous_reward = 9, 0.0
        for t, event in enumerate(row["events"]):
            assert state.observation.tolist() == event["observation"] and event["time"] == t
            action, hidden, _ = pilot.predict(model, pilot.encode(event["observation"], row["goal"],
                previous_action, previous_reward, t == 0, False, t), hidden)
            assert action == event["executed_action"]
            state, reward, done, _ = env.step(action)
            assert state.observation.tolist() == event["next_observation"]
            assert float(reward) == event["reward"]
            assert not done or t == len(row["events"]) - 1
            previous_action, previous_reward = action, reward
            steps += 1
        assert done and row["actual_steps"] == len(row["events"])
        success = bool(np.array_equal(state.observation[4:6], row["goal"]))
        assert success == row["success"]
        assert row["cost_steps"] == (row["actual_steps"] if success else 20)
        _, hidden, _ = pilot.predict(model, pilot.terminal_feedback(state.observation, row["goal"],
            previous_action, previous_reward, False, row["actual_steps"]), hidden)
        hidden_states[context_key] = hidden
        grouped["/".join(map(str, key[:3]))].append(row)
    assert seen == expected
    assert chain == summary["chain_head"]
    assert steps == summary["cost"]["execution_actions"]
    assert prefix_steps == summary["cost"]["prefix_collection_actions"]
    for key, rows in grouped.items():
        assert aggregate(rows) == summary["evaluations"][key]
    checks = {}
    for arm in pilot.ARMS:
        real = [summary["evaluations"][f"{arm}/{s}/{CONDITIONS[0]}"]["success_rate"] for s in original["training_seeds"]]
        mask = [summary["evaluations"][f"{arm}/{s}/{CONDITIONS[1]}"]["success_rate"] for s in original["training_seeds"]]
        checks[arm] = {"shared_success_mean": float(np.mean(real)), "redacted_success_mean": float(np.mean(mask)),
                       "shared_seed_values": real, "redacted_seed_values": mask,
                       "information_effect": float(np.mean(real) - np.mean(mask))}
    candidate = checks["observed_facts_aux"]
    h1 = all(candidate["shared_success_mean"] - checks[a]["shared_success_mean"] >= .05 and
             candidate["information_effect"] - checks[a]["information_effect"] >= .05
             for a in ["action_only", "current_state_aux"]) and candidate["information_effect"] >= .05
    analysis = {"schema": "matched-history-analysis-v1", "arms": checks,
                "H_information_use_registered_check": h1, "optimizer_updates": 0,
                "scope": "External shared prefix and input redaction; exploratory mechanism probe, not primary teacher-free success."}
    pilot.write(folder / "analysis.json", analysis)
    report = {"status": "PASS", "summary_sha256": pilot.sha(folder / "summary.json"),
              "verifier_sha256": pilot.sha(__file__), "source_prefixes": len(prefixes), "prefix_steps": prefix_steps,
              "recomputed_policy_decisions": steps, "replayed_records": len(seen),
              "raw_observation_max_difference": 0, "all_checkpoint_identities_unchanged": True,
              "autonomous_prefix_comparisons": prefix_comparisons,
              "autonomous_prefixes_identical_to_shared": identical_prefixes,
              "redacted_prefix_unique_count": len(masked_ids),
              "checks": ["Same real prefix for all models", "Redaction removes mapping observations, preserves raw log",
                         "Complete unique configurations", "All actions/outcomes/rewards independently replayed",
                         "Summary metrics recomputed", "All checkpoint hashes unchanged", "Full hash chain"]}
    pilot.write(folder / "verification.json", report)
    print(json.dumps({"verification": report, "analysis": analysis}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    a = p.parse_args()
    verify(a.parent, a.folder, a.upstream)
