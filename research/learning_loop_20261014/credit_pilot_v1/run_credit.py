#!/usr/bin/env python3
"""Bounded experience-credit ranking diagnostic. Not a closed-loop efficacy trial."""

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

LEGACY = Path(__file__).resolve().parents[2] / "interactive_learning_20261013"
sys.path.insert(0, str(LEGACY / "student_coverage_v1"))
import run_coverage as coverage  # noqa: E402

pilot = coverage.pilot
BUDGETS = (0, 1, 3, 5)


def semantic_binding(plan):
    return {
        k: v
        for k, v in plan.items()
        if k not in {"plan_id", "semantic_sha256", "registered_at_utc"}
    }


class Ledger:
    def __init__(self, path, plan_hash, plan_file_hash):
        self.handle = path.open("w")
        self.plan_hash = plan_hash
        self.plan_file_hash = plan_file_hash
        self.chain = "0" * 64
        self.count = self.actions = self.resets = 0
        self.max_actions = self.max_records = None

    def add(self, record):
        self.actions += len(record["events"])
        self.resets += len(record["outcomes"])
        if self.max_actions is not None and self.actions > self.max_actions:
            raise RuntimeError("Registered action budget exceeded")
        if self.max_records is not None and self.count >= self.max_records:
            raise RuntimeError("Registered record budget exceeded")
        row = {
            **record,
            "record_index": self.count,
            "plan_sha256": self.plan_hash,
            "plan_file_sha256": self.plan_file_hash,
            "previous_hash": self.chain,
        }
        self.chain = pilot.digest(row)
        row["record_hash"] = self.chain
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.count += 1

    def close(self):
        self.handle.close()


def task_panel(ids):
    """At most two public-valid goals per rule; primary metrics macro-average rules."""
    tasks = []
    for identity in ids:
        available = sorted({identity // (4**j) % 4 for j in range(3)})
        goals = [[available[0], available[0]]]
        if len(available) > 1:
            goals.append([available[0], available[-1]])
        for goal in goals:
            tasks.append(
                {
                    "environment_id": identity,
                    "goal": goal,
                    "task_id": f"credit-{identity:02d}-{goal[0]}{goal[1]}",
                }
            )
    return tasks


def history(
    env_type,
    wrapper_type,
    task,
    model=None,
    *,
    probe_steps=0,
    attempts=6,
    behavior_seed=0,
    epsilon=0.0,
    reset_memory=False,
    annotate=True,
):
    """No hidden identity reaches model/teacher; b=0 has no exploration prefix."""
    order = list(np.random.RandomState(behavior_seed).permutation(3))
    rng = np.random.RandomState(behavior_seed)
    memory, hidden, events, outcomes = {}, None, [], []
    phases = [(-1, True, probe_steps)] if probe_steps else []
    phases += [(k, False, 20) for k in range(attempts)]
    for trial, exploration, horizon in phases:
        if reset_memory:
            hidden = None
        env = pilot.environment(env_type, wrapper_type, task, exploration)
        state = env.reset()
        previous_action, previous_reward = 9, 0.0
        success = False
        for step in range(horizon):
            before = np.array(state.observation, copy=True)
            label = (
                coverage.target(before, task["goal"], memory, order, exploration)
                if annotate or model is None
                else None
            )
            if model is None:
                action = label if label >= 0 else 4
                if rng.rand() < epsilon:
                    action = int(rng.choice(pilot.BEHAVIOR_ACTIONS))
            else:
                token = pilot.encode(
                    before,
                    task["goal"],
                    previous_action,
                    previous_reward,
                    step == 0,
                    exploration,
                    step,
                )
                action, hidden, _ = pilot.predict(model, token, hidden)
            state, reward, done, _ = env.step(action)
            after = np.array(state.observation, copy=True)
            events.append(
                {
                    "trial": trial,
                    "time": step,
                    "exploration": exploration,
                    "observation": before.tolist(),
                    "executed_action": int(action),
                    "next_observation": after.tolist(),
                    "reward": float(reward),
                    "teacher_action": int(label) if label is not None else None,
                    "env_done": bool(done),
                }
            )
            if annotate or model is None:
                pilot.absorb(memory, before, action, after)
            previous_action, previous_reward = action, reward
            success = bool(not exploration and np.array_equal(after[4:6], task["goal"]))
            if done:
                break
        outcomes.append(
            {
                "trial": trial,
                "exploration": exploration,
                "actual_steps": step + 1,
                "success": None if exploration else success,
                "terminated": bool(done),
                "budget_truncated": not done,
            }
        )
        if model is not None:
            _, hidden, _ = pilot.predict(
                model,
                pilot.terminal_feedback(
                    after, task["goal"], previous_action, previous_reward, exploration, step + 1
                ),
                hidden,
            )
    return {
        "task": task,
        "behavior_seed": behavior_seed,
        "epsilon": epsilon,
        "probe_steps": probe_steps,
        "attempts": attempts,
        "annotated": annotate,
        "events": events,
        "outcomes": outcomes,
    }


def reconstruct(record):
    """Only actual causal events define inputs and labels; task ID is not consulted."""
    memory = {}
    order = list(np.random.RandomState(record["behavior_seed"]).permutation(3))
    inputs, labels = [], []
    for outcome in record["outcomes"]:
        events = [e for e in record["events"] if e["trial"] == outcome["trial"]]
        if not events or len(events) != outcome["actual_steps"]:
            raise ValueError("Missing or mismatched episode events")
        previous_action, previous_reward = 9, 0.0
        for step, event in enumerate(events):
            if event["time"] != step or event["exploration"] != outcome["exploration"]:
                raise ValueError("Noncontiguous causal history")
            before = event["observation"]
            inputs.append(
                pilot.encode(
                    before,
                    record["task"]["goal"],
                    previous_action,
                    previous_reward,
                    step == 0,
                    event["exploration"],
                    step,
                )
            )
            label = coverage.target(
                before, record["task"]["goal"], memory, order, event["exploration"]
            )
            if label != event["teacher_action"]:
                raise ValueError("Observable teacher label mismatch")
            labels.append(label)
            pilot.absorb(memory, before, event["executed_action"], event["next_observation"])
            previous_action, previous_reward = event["executed_action"], event["reward"]
        inputs.append(
            pilot.terminal_feedback(
                events[-1]["next_observation"],
                record["task"]["goal"],
                previous_action,
                previous_reward,
                outcome["exploration"],
                len(events),
            )
        )
        labels.append(-100)
    return {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "actions": np.asarray(labels, dtype=np.int64),
    }


def pack(items):
    width = max(len(item["inputs"]) for item in items)
    x = np.zeros((len(items), width, pilot.INPUT_DIM), dtype=np.float32)
    y = np.full((len(items), width), -100, dtype=np.int64)
    for i, item in enumerate(items):
        n = len(item["inputs"])
        x[i, :n], y[i, :n] = item["inputs"], item["actions"]
    return torch.from_numpy(x), torch.from_numpy(y)


def optimizer_digest(optimizer):
    state = optimizer.state_dict()
    h = hashlib.sha256()
    h.update(json.dumps(state["param_groups"], sort_keys=True).encode())
    for index, values in sorted(state["state"].items()):
        for name, value in sorted(values.items()):
            h.update(f"{index}/{name}".encode())
            if isinstance(value, torch.Tensor):
                h.update(str(value.dtype).encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def train(model, optimizer, base, candidate, updates, plan, seed):
    rng = np.random.RandomState(seed + 900000)
    sampling, valid_targets = hashlib.sha256(), 0
    x, y = pack(base + (candidate or []))
    base_n = len(base)
    model.train()
    for step in range(updates):
        if candidate:
            a = rng.randint(0, base_n, size=plan["batch_sequences"] // 2)
            b = base_n + rng.randint(0, len(candidate), size=plan["batch_sequences"] // 2)
            indices = np.concatenate([a, b])
        else:
            indices = rng.randint(0, base_n, size=plan["batch_sequences"])
        sampling.update(indices.astype(np.int64).tobytes())
        predictions, _, _ = model(x[indices])
        targets = y[indices]
        valid_targets += int((targets != -100).sum())
        loss = functional.cross_entropy(predictions.reshape(-1, 9), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan["gradient_clip"])
        optimizer.step()
    model.eval()
    return {
        "updates": updates,
        "sequences_sampled": updates * plan["batch_sequences"],
        "valid_action_targets_sampled": valid_targets,
        "sampling_sha256": sampling.hexdigest(),
        "final_weights_sha256": pilot.weights_digest(model),
        "final_optimizer_sha256": optimizer_digest(optimizer),
    }


def metrics(records):
    by_rule = {}
    for record in records:
        execution = [o for o in record["outcomes"] if not o["exploration"]]
        if len(execution) != 6:
            raise ValueError("Six attempts required for registered adaptation budgets")
        by_rule.setdefault(record["task"]["environment_id"], []).append(
            [int(o["success"]) for o in execution]
        )
    curve = np.mean([np.mean(values, axis=0) for values in by_rule.values()], axis=0)
    total = float(curve[list(BUDGETS)].mean())
    adapt = float(curve[[1, 3, 5]].mean() - curve[0])
    return {
        "rules": len(by_rule),
        "histories": len(records),
        "success_by_attempt": curve.tolist(),
        "cold_success": float(curve[0]),
        "final_success": float(curve[5]),
        "q_total": total,
        "q_adapt": adapt,
        "actual_actions": sum(len(r["events"]) for r in records),
        "resets": sum(len(r["outcomes"]) for r in records),
    }


def evaluate(env_type, wrapper_type, model, tasks, context, ledger):
    rows = []
    for index, task in enumerate(tasks):
        record = history(env_type, wrapper_type, task, model, behavior_seed=0, annotate=False)
        record.update(context)
        record["case_index"] = index
        ledger.add(record)
        rows.append(record)
    return metrics(rows)


def run(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    assert pilot.digest(semantic_binding(plan)) == plan["semantic_sha256"]
    for name, expected in plan["code_hashes"].items():
        assert pilot.sha(args.plan.parent / name) == expected, name
    assert not args.output.exists(), "Never overwrite an experiment"
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_bytes(args.plan.read_bytes())
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    snapshot_names = {}
    for name in plan["code_hashes"]:
        target = snapshot / Path(name).name
        assert not target.exists(), "Snapshot basename collision"
        target.write_bytes((args.plan.parent / name).read_bytes())
        snapshot_names[name] = {
            "path": str(target.relative_to(args.output)),
            "sha256": pilot.sha(target),
        }
    pilot.write(snapshot / "manifest.json", snapshot_names)
    torch.set_num_threads(plan["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    env_type, wrapper_type = pilot.load_environment(args.upstream)
    plan_file_hash = pilot.sha(args.plan)
    source = Ledger(args.output / "source.jsonl", plan["semantic_sha256"], plan_file_hash)
    source.max_actions, source.max_records = (
        plan["max_source_actions"],
        plan["base_sequences"] + 3 * 8 * 24,
    )
    evaluation = Ledger(args.output / "evaluation.jsonl", plan["semantic_sha256"], plan_file_hash)
    evaluation.max_actions, evaluation.max_records = (
        plan["max_evaluation_actions"],
        plan["evaluation_histories"],
    )
    base = []
    for task in pilot.tasks_for(plan["source_ids"]):
        for probe_steps in plan["initial_probe_budgets"]:
            for epsilon in [0.0, 0.1]:
                record = history(
                    env_type,
                    wrapper_type,
                    task,
                    probe_steps=probe_steps,
                    behavior_seed=11 + probe_steps,
                    epsilon=epsilon,
                )
                record.update({"kind": "base_source", "split": "source"})
                source.add(record)
                base.append(reconstruct(record))
    assert len(base) == plan["base_sequences"]
    feedback, audit = task_panel(plan["feedback_ids"]), task_panel(plan["audit_ids"])
    results, training = [], []
    for seed in plan["training_seeds"]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = pilot.HistoryPolicy(plan["hidden_size"])
        optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
        initial_record = train(model, optimizer, base, None, plan["initial_updates"], plan, seed)
        g0 = args.output / "checkpoints" / f"seed-{seed}" / "initial.pt"
        g0.parent.mkdir(parents=True)
        torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict()}, g0)
        initial_record.update(
            {
                "seed": seed,
                "stage": "initial",
                "checkpoint": str(g0.relative_to(args.output)),
                "checkpoint_sha256": pilot.sha(g0),
            }
        )
        training.append(initial_record)
        pilot.write(args.output / "training.json", training)
        baseline = {}
        for split, tasks in [("feedback", feedback), ("audit", audit)]:
            baseline[split] = evaluate(
                env_type,
                wrapper_type,
                model,
                tasks,
                {
                    "kind": "evaluation",
                    "seed": seed,
                    "candidate": None,
                    "stage": "initial",
                    "split": split,
                },
                evaluation,
            )
        pilot.write(g0.with_suffix(".json"), {"training": initial_record, "baseline": baseline})
        initial_state = copy.deepcopy(model.state_dict())
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        initial_digest = pilot.weights_digest(model)
        initial_opt_digest = optimizer_digest(optimizer)
        for spec in plan["candidate_specs"]:
            records, data = [], []
            for index, task in enumerate(spec["tasks"]):
                record = history(env_type, wrapper_type, task, model, behavior_seed=0)
                record.update(
                    {
                        "kind": "candidate_source",
                        "split": "source",
                        "seed": seed,
                        "candidate": spec["id"],
                        "case_index": index,
                        "collector_weights_sha256": initial_digest,
                    }
                )
                source.add(record)
                records.append(record)
                data.append(reconstruct(record))
            source_before = metrics(records)
            item = {
                "seed": seed,
                "candidate": spec["id"],
                "recipe": spec["recipe"],
                "baseline": baseline,
                "source_before": source_before,
            }
            for stage, updates in [
                ("short", plan["short_updates"]),
                ("long", plan["long_updates"]),
            ]:
                branch = pilot.HistoryPolicy(plan["hidden_size"])
                branch.load_state_dict(copy.deepcopy(initial_state))
                opt = torch.optim.Adam(branch.parameters(), lr=plan["learning_rate"])
                opt.load_state_dict(copy.deepcopy(initial_optimizer))
                assert pilot.weights_digest(branch) == initial_digest
                assert optimizer_digest(opt) == initial_opt_digest
                trained = train(branch, opt, base, data, updates, plan, seed)
                checkpoint = g0.parent / f"{spec['id']}-{stage}.pt"
                torch.save(
                    {"weights": branch.state_dict(), "optimizer": opt.state_dict()}, checkpoint
                )
                trained.update(
                    {
                        "seed": seed,
                        "candidate": spec["id"],
                        "stage": stage,
                        "initial_weights_sha256": initial_digest,
                        "initial_optimizer_sha256": initial_opt_digest,
                        "checkpoint": str(checkpoint.relative_to(args.output)),
                        "checkpoint_sha256": pilot.sha(checkpoint),
                    }
                )
                training.append(trained)
                pilot.write(args.output / "training.json", training)
                panels = (
                    [("source", spec["tasks"]), ("feedback", feedback), ("audit", audit)]
                    if stage == "short"
                    else [("audit", audit)]
                )
                item[stage] = {}
                for split, tasks in panels:
                    item[stage][split] = evaluate(
                        env_type,
                        wrapper_type,
                        branch,
                        tasks,
                        {
                            "kind": "evaluation",
                            "seed": seed,
                            "candidate": spec["id"],
                            "stage": stage,
                            "split": split,
                        },
                        evaluation,
                    )
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "candidate": spec["id"],
                            "stage": stage,
                            "audit": item[stage]["audit"]["q_adapt"],
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            results.append(item)
            if time.monotonic() - started > plan["max_wall_seconds"]:
                raise RuntimeError("Registered wall-time budget exceeded")
    source.close()
    evaluation.close()
    assert source.count == plan["base_sequences"] + 3 * 8 * 24
    assert evaluation.count == plan["evaluation_histories"]
    assert sum(r["updates"] for r in training) == plan["total_updates"]
    assert len(training) == 51 and len(results) == 24
    for seed in plan["training_seeds"]:
        for stage in ["short", "long"]:
            values = [
                r["sampling_sha256"] for r in training if r["seed"] == seed and r["stage"] == stage
            ]
            assert len(set(values)) == 1
    pilot.write(args.output / "training.json", training)
    summary = {
        "schema": "experience-credit-ranking-v1",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["semantic_sha256"],
        "results": results,
        "source_chain_head": source.chain,
        "evaluation_chain_head": evaluation.chain,
        "coverage": {
            "source_histories": source.count,
            "evaluation_histories": evaluation.count,
            "missing": 0,
            "aborted": 0,
        },
        "cost": {
            "initial_updates": 3 * plan["initial_updates"],
            "short_branch_updates": 24 * plan["short_updates"],
            "long_branch_updates": 24 * plan["long_updates"],
            "total_updates": plan["total_updates"],
            "source_actions": source.actions,
            "source_resets": source.resets,
            "unique_annotation_positions": source.actions,
            "teacher_label_function_calls": 2 * source.actions,
            "evaluation_actions": evaluation.actions,
            "evaluation_resets": evaluation.resets,
            "sampled_valid_targets": sum(t["valid_action_targets_sampled"] for t in training),
            "wall_seconds": time.monotonic() - started,
        },
        "scope": (
            "Training-side Cooking credit-ranking development pilot; reused"
            " finite family, no sealed test or robot/RSI efficacy."
        ),
    }
    summary["artifacts"] = {
        n: pilot.sha(args.output / n)
        for n in ["plan.json", "source.jsonl", "evaluation.jsonl", "training.json"]
    }
    pilot.write(args.output / "summary.json", summary)
    print(json.dumps({"status": "COMPLETED", "cost": summary["cost"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "plan", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
