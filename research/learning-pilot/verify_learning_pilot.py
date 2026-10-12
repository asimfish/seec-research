#!/usr/bin/env python3
"""Independently replay stored transitions and recompute learned-pilot counts."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_learning_pilot as pilot


def facts_from_event(memory, before, action, after):
    drawers = [(2, 0), (2, 1), (2, 2)]
    position = tuple(before[:2])
    if action == 7 and position in drawers and int(after[3]) in range(4):
        memory[drawers.index(position)] = int(after[3])


def labels(memory):
    return [memory.get(i, 4) for i in range(3)]


def aggregate(rows):
    execution = [r for r in rows if r["trial"] > 0]
    n = len(execution)
    return {"episodes": n, "successes": sum(int(r["success"]) for r in execution),
            "success_rate": sum(int(r["success"]) for r in execution) / n,
            "mean_cost_steps": sum(r["cost_steps"] for r in execution) / n,
            "by_trial": [{"trial": trial,
                          "success_rate": sum(int(r["success"]) for r in execution if r["trial"] == trial) / sum(r["trial"] == trial for r in execution),
                          "mean_cost_steps": sum(r["cost_steps"] for r in execution if r["trial"] == trial) / sum(r["trial"] == trial for r in execution)}
                         for trial in range(1, 6)]}


def verify(args):
    root = args.folder
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert pilot.digest({k: v for k, v in plan.items() if k not in {"semantic_sha256", "plan_id"}}) == plan["semantic_sha256"]
    for name, expected in summary["artifacts"].items():
        assert pilot.sha(root / name) == expected
    Env, Wrapper = pilot.load_environment(args.upstream)
    torch.set_num_threads(plan["cpu_threads"])
    source_steps, source_sequences, source_feedback, source_keys = 0, 0, 0, set()
    for line in (root / "source.jsonl").open():
        record = json.loads(line)
        task = record["task"]
        assert task["environment_id"] in plan["train_environment_ids"]
        assert task["environment_id"] not in plan["development_environment_ids"] + plan["evaluation_environment_ids"]
        source_key = (task["task_id"], record["behavior_seed"], record["epsilon"])
        assert source_key not in source_keys
        source_keys.add(source_key)
        memory = {}
        for trial in range(6):
            env = pilot.environment(Env, Wrapper, task, trial == 0)
            state = env.reset()
            events = [e for e in record["events"] if e["trial"] == trial]
            previous_action, previous_reward = 9, 0.0
            for t, event in enumerate(events):
                assert event["time"] == t
                before = state.observation.tolist()
                assert before == event["observation"]
                # The target is reconstructable without the hidden mapping.
                assert event["facts_before"] == labels(memory)
                assert pilot.encode(before, task["goal"], previous_action, previous_reward, t == 0, trial == 0, t).shape == (55,)
                state, reward, done, _ = env.step(event["executed_action"])
                after = state.observation.tolist()
                assert after == event["next_observation"]
                assert float(reward) == event["reward"]
                facts_from_event(memory, before, event["executed_action"], after)
                previous_action, previous_reward = event["executed_action"], reward
                assert not done or t == len(events) - 1
            assert done
            assert pilot.terminal_feedback(after, task["goal"], previous_action, previous_reward, trial == 0, len(events)).shape == (55,)
            source_steps += len(events)
            source_feedback += 1
        source_sequences += 1
    expected_source = {(task["task_id"], seed, epsilon)
                       for task in pilot.tasks_for(plan["train_environment_ids"])
                       for seed in plan["behavior_seeds"] for epsilon in plan["behavior_epsilons"]}
    assert source_keys == expected_source
    assert summary["dataset"]["physical_steps"] == source_steps
    assert summary["dataset"]["tokens"] == source_steps + source_feedback
    assert summary["dataset"]["terminal_feedback_tokens"] == source_feedback
    checkpoints = {}
    for record in summary["models"]:
        assert record["updates"] == 800 and record["batch_sequences"] == 24
        path = root / "checkpoints" / f"{record['arm']}-{record['seed']}" / "policy.pt"
        assert pilot.sha(path) == record["checkpoint_sha256"]
        model = pilot.HistoryPolicy(plan["hidden_size"]).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        assert pilot.weights_digest(model) == record["final_weights_sha256"]
        checkpoints[(record["arm"], record["seed"])] = model
    for seed in plan["training_seeds"]:
        same_seed = [r for r in summary["models"] if r["seed"] == seed]
        assert len(same_seed) == 3
        assert len({r["initial_weights_sha256"] for r in same_seed}) == 1
        assert len({r["sampling_sha256"] for r in same_seed}) == 1
        assert len({r["parameter_count"] for r in same_seed}) == 1
    expected_keys = set()
    for split, ids in [("development", plan["development_environment_ids"]), ("evaluation", plan["evaluation_environment_ids"])]:
        configs = [(arm, seed, "full_history") for seed in plan["training_seeds"] for arm in pilot.ARMS]
        if split == "evaluation":
            configs += [(arm, seed, "reset_each_episode") for seed in plan["training_seeds"] for arm in pilot.ARMS]
            configs += [("symbolic_full_history", seed, "full_history") for seed in plan["training_seeds"]]
        for arm, seed, condition in configs:
            for task in pilot.tasks_for(ids):
                for trial in range(6):
                    expected_keys.add((split, arm, seed, condition, task["task_id"], trial))
    keys, grouped, states, fact_memory = set(), defaultdict(list), {}, {}
    chain, steps, predictions, auxiliary_correct, auxiliary_total = "0" * 64, 0, 0, 0, 0
    for line in (root / "evaluation.jsonl").open():
        row = json.loads(line)
        key = (row["split"], row["arm"], row["seed"], row["condition"], row["task_id"], row["trial"])
        assert key in expected_keys and key not in keys
        keys.add(key)
        assert row["previous_hash"] == chain
        body = {k: v for k, v in row.items() if k != "record_hash"}
        chain = pilot.digest(body)
        assert chain == row["record_hash"]
        assert row["plan_sha256"] == plan["semantic_sha256"]
        task = {k: row[k] for k in ["environment_id", "goal", "task_id"]}
        env = pilot.environment(Env, Wrapper, task, row["trial"] == 0)
        state = env.reset()
        task_key = key[:5]
        hidden = states.get(task_key)
        memory = fact_memory.setdefault(task_key, {})
        if row["condition"] == "reset_each_episode":
            hidden = None
        previous_action, previous_reward = 9, 0.0
        for t, event in enumerate(row["events"]):
            before = state.observation.tolist()
            assert before == event["observation"] and event["time"] == t
            if row["arm"] != "symbolic_full_history":
                model = checkpoints[(row["arm"], row["seed"])]
                action, hidden, inferred = pilot.predict(model, pilot.encode(
                    before, row["goal"], previous_action, previous_reward,
                    t == 0, row["trial"] == 0, t), hidden)
                assert action == event["executed_action"]
                assert inferred == event["predicted_facts"]
                predictions += 1
                if row["arm"] == "observed_facts_aux" and row["condition"] == "full_history" and row["split"] == "evaluation":
                    auxiliary_correct += sum(a == b for a, b in zip(inferred, labels(memory)))
                    auxiliary_total += 3
            state, reward, done, _ = env.step(event["executed_action"])
            after = state.observation.tolist()
            assert after == event["next_observation"]
            assert float(reward) == event["reward"]
            facts_from_event(memory, before, event["executed_action"], after)
            previous_action, previous_reward = event["executed_action"], reward
            assert not done or t == len(row["events"]) - 1
            steps += 1
        assert done
        assert len(row["events"]) == row["actual_steps"]
        success = bool(np.array_equal(state.observation[4:6], row["goal"])) if row["trial"] else None
        assert success == row["success"]
        cost = row["actual_steps"] if row["trial"] == 0 or success else 20
        assert cost == row["cost_steps"]
        if row["arm"] != "symbolic_full_history":
            _, hidden, _ = pilot.predict(model, pilot.terminal_feedback(after, row["goal"], previous_action,
                                            previous_reward, row["trial"] == 0, row["actual_steps"]), hidden)
            states[task_key] = hidden
        group = "/".join(map(str, key[:4]))
        grouped[group].append(row)
    assert keys == expected_keys
    assert chain == summary["chain_head"]
    assert steps == summary["cost"]["evaluation_steps"]
    assert len(keys) == summary["coverage"]["planned_records"] == summary["coverage"]["completed_records"]
    for key, rows in grouped.items():
        assert aggregate(rows) == summary["evaluations"][key], key
    for arm, records in summary["arms"].items():
        for record in records:
            expected = summary["evaluations"][f"evaluation/{arm}/{record['seed']}/full_history"]
            assert {k: v for k, v in record.items() if k != "seed"} == expected
    report = {"status": "PASS", "summary_sha256": pilot.sha(root / "summary.json"),
              "verifier_sha256": pilot.sha(__file__), "plan_sha256": plan["semantic_sha256"],
              "checks": ["Complete expected unique source/evaluation records", "Source is train split only",
                         "Known-fact labels derived from actual prior observations", "Terminal feedback reconstructed before reset",
                         "Equal initial weights, sample order, model size and optimizer budget",
                         "All 9 final checkpoint identities", "Every stored observation/reward exactly replayed",
                         "All learned executed actions and auxiliary outputs recomputed from checkpoints",
                         "All terminal outcomes and cost penalties recomputed", "All summary means/counts recomputed",
                         "Full evaluation hash chain"],
              "source_sequences": source_sequences, "source_steps": source_steps,
              "evaluation_records": len(keys), "evaluation_steps": steps,
              "recomputed_policy_decisions": predictions, "raw_observation_max_difference": 0,
              "observed_fact_accuracy_on_student_histories": auxiliary_correct / auxiliary_total,
              "scope": "Reset-and-action replay, not simulator snapshot E4, method efficacy, or training reproducibility."}
    pilot.write(root / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    verify(parser.parse_args())
