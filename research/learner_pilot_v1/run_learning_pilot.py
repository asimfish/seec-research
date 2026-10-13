#!/usr/bin/env python3
"""Registered CPU learning diagnostic on unmodified DREAM Cooking transitions.

This is recurrent behavioral cloning with an auxiliary-label ablation, not a
reproduction of DREAM/AD/RL2 and not on-policy distillation. Task identity never
enters encode(). Auxiliary facts come only from actual observed pickup events.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_cooking_diagnostic import (  # noqa: E402
    absorb, digest, explore_action, load_environment, solve_action,
)

ARMS = ("action_only", "current_state_aux", "observed_facts_aux")
OBS_CARDINALITIES = (3, 3, 4, 5, 5, 5, 5)
INPUT_DIM = sum(OBS_CARDINALITIES) + 10 + 10 + 1 + 1 + 2 + 1
BEHAVIOR_ACTIONS = (0, 1, 2, 3, 4, 7, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def encode(observation, goal, previous_action, previous_reward, episode_start,
           exploration, local_step):
    """The entire student boundary. No environment, identity, or fact argument."""
    values = list(map(int, observation))
    assert len(values) == 7
    result = []
    for value, size in zip(values, OBS_CARDINALITIES):
        assert 0 <= value < size
        result.extend(float(i == value) for i in range(size))
    for value in ([4, 4] if exploration else list(map(int, goal))):
        assert 0 <= value < 5
        result.extend(float(i == value) for i in range(5))
    assert 0 <= previous_action <= 9
    result.extend(float(i == previous_action) for i in range(10))
    result.extend([float(previous_reward), float(episode_start),
                   float(exploration), float(not exploration), local_step / 20.0])
    assert len(result) == INPUT_DIM
    return np.asarray(result, dtype=np.float32)


def fact_targets(memory):
    return np.asarray([memory.get(i, 4) for i in range(3)], dtype=np.int64)


def state_targets(observation):
    return np.asarray(observation[3:6], dtype=np.int64)


def terminal_feedback(observation, goal, last_action, last_reward, exploration, steps):
    """Consume a real terminal response without requesting another env action."""
    return encode(observation, goal, last_action, last_reward, False, exploration, steps)


class HistoryPolicy(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding = nn.Linear(INPUT_DIM, hidden_size)
        self.history = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actions = nn.Linear(hidden_size, 9)
        self.auxiliary = nn.Linear(hidden_size, 15)

    def forward(self, inputs, hidden=None):
        output, hidden = self.history(torch.tanh(self.embedding(inputs)), hidden)
        return self.actions(output), self.auxiliary(output).reshape(
            *output.shape[:2], 3, 5), hidden


def weights_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tasks_for(ids):
    tasks = []
    for env_id in ids:
        # Evaluator constructs a valid instruction, as upstream does. This
        # hidden assignment is never passed to a policy or auxiliary target.
        available = sorted({(env_id // (4 ** i)) % 4 for i in range(3)})
        for first in available:
            for second in available:
                tasks.append({"environment_id": env_id, "goal": [first, second],
                              "task_id": f"cooking-{env_id:02d}-{first}{second}"})
    return tasks


def environment(Env, Wrapper, task, exploration):
    base = Env(np.asarray([task["environment_id"]]), lambda state: state,
               max_steps=20)
    if exploration:
        return base
    env = Wrapper(base, [], seed=0, fixed_instructions=True)
    env.registered_goal = task["goal"]
    return env


def collect_sequence(Env, Wrapper, task, behavior_seed, epsilon, trials):
    rng = np.random.RandomState(behavior_seed)
    tie_order = list(rng.permutation(3))
    memory, tokens, labels, facts, states, events = {}, [], [], [], [], []
    for trial in range(trials + 1):
        exploration = trial == 0
        env = environment(Env, Wrapper, task, exploration)
        state = env.reset()
        previous_action, previous_reward = 9, 0.0
        for t in range(20):
            before = np.array(state.observation, copy=True)
            target = (explore_action(before, memory, tie_order) if exploration
                      else solve_action(before, task["goal"], memory, tie_order))
            tokens.append(encode(before, task["goal"], previous_action,
                                 previous_reward, t == 0, exploration, t))
            labels.append(target)
            facts.append(fact_targets(memory))
            states.append(state_targets(before))
            executed = int(rng.choice(BEHAVIOR_ACTIONS)) if rng.rand() < epsilon else target
            state, reward, done, _ = env.step(executed)
            after = np.array(state.observation, copy=True)
            events.append({"trial": trial, "time": t, "observation": before.tolist(),
                           "teacher_action": target, "executed_action": executed,
                           "next_observation": after.tolist(), "reward": float(reward),
                           "facts_before": facts[-1].tolist()})
            absorb(memory, before, executed, after)
            previous_action, previous_reward = executed, reward
            if done:
                break
        # A last-step pickup is still part of the visible interaction history.
        # This token has no action target and spends no simulator action.
        tokens.append(terminal_feedback(after, task["goal"], previous_action,
                                        previous_reward, exploration, t + 1))
        labels.append(-100)
        facts.append(fact_targets(memory))
        states.append(state_targets(after))
    return {"inputs": np.asarray(tokens), "actions": np.asarray(labels),
            "facts": np.asarray(facts), "states": np.asarray(states),
            "events": events, "task": task, "behavior_seed": behavior_seed,
            "epsilon": epsilon}


def make_dataset(Env, Wrapper, plan, output):
    sequences = []
    with (output / "source.jsonl").open("w") as handle:
        for task in tasks_for(plan["train_environment_ids"]):
            for behavior_seed in plan["behavior_seeds"]:
                for epsilon in plan["behavior_epsilons"]:
                    item = collect_sequence(Env, Wrapper, task, behavior_seed,
                                            epsilon, plan["trials"])
                    sequences.append(item)
                    handle.write(json.dumps({k: v for k, v in item.items()
                                             if k not in {"inputs", "actions", "facts", "states"}},
                                            sort_keys=True) + "\n")
    lengths = [len(item["inputs"]) for item in sequences]
    count, width = len(sequences), max(lengths)
    inputs = np.zeros((count, width, INPUT_DIM), dtype=np.float32)
    actions = np.full((count, width), -100, dtype=np.int64)
    facts = np.full((count, width, 3), -100, dtype=np.int64)
    states = np.full_like(facts, -100)
    for i, item in enumerate(sequences):
        n = lengths[i]
        inputs[i, :n], actions[i, :n] = item["inputs"], item["actions"]
        facts[i, :n], states[i, :n] = item["facts"], item["states"]
    metadata = {"sequences": count, "tokens": sum(lengths), "max_length": width,
                "physical_steps": sum(len(item["events"]) for item in sequences),
                "terminal_feedback_tokens": count * 6,
                "source_sha256": sha(output / "source.jsonl"),
                "environment_ids": plan["train_environment_ids"],
                "source": "Common deterministic and epsilon-perturbed observable-memory teacher histories; not student on-policy data."}
    write(output / "dataset.json", metadata)
    return tuple(torch.from_numpy(a) for a in (inputs, actions, facts, states)), metadata


def train(dataset, plan, arm, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = HistoryPolicy(plan["hidden_size"])
    initial_digest = weights_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    sampler = np.random.RandomState(seed + 100000)
    inputs, actions, facts, states = dataset
    sampling = hashlib.sha256()
    loss_records = []
    started = time.time()
    for update in range(1, plan["updates"] + 1):
        indices = sampler.randint(0, len(inputs), size=plan["batch_sequences"])
        sampling.update(indices.astype(np.int64).tobytes())
        x, y = inputs[indices], actions[indices]
        predictions, auxiliary, _ = model(x)
        action_loss = F.cross_entropy(predictions.reshape(-1, 9), y.reshape(-1))
        if arm == "action_only":
            auxiliary_loss = action_loss.new_zeros(())
        else:
            targets = facts[indices] if arm == "observed_facts_aux" else states[indices]
            auxiliary_loss = F.cross_entropy(auxiliary.reshape(-1, 5), targets.reshape(-1))
        loss = action_loss + plan["auxiliary_weight"] * auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), plan["gradient_clip"])
        optimizer.step()
        if update == 1 or update % 100 == 0:
            valid = y != -100
            accuracy = ((predictions.argmax(-1) == y) & valid).sum().item() / valid.sum().item()
            row = {"update": update, "action_loss": action_loss.item(),
                   "auxiliary_loss": auxiliary_loss.item(), "teacher_forced_accuracy": accuracy}
            loss_records.append(row)
            print(json.dumps({"arm": arm, "seed": seed, **row}), flush=True)
    model.eval()
    folder = output / "checkpoints" / f"{arm}-{seed}"
    folder.mkdir(parents=True)
    checkpoint = folder / "policy.pt"
    torch.save(model.state_dict(), checkpoint)
    metadata = {"arm": arm, "seed": seed, "initial_weights_sha256": initial_digest,
                "final_weights_sha256": weights_digest(model), "checkpoint_sha256": sha(checkpoint),
                "sampling_sha256": sampling.hexdigest(), "plan_sha256": plan["semantic_sha256"],
                "source_sha256": sha(output / "source.jsonl"), "updates": plan["updates"],
                "batch_sequences": plan["batch_sequences"], "parameter_count": sum(p.numel() for p in model.parameters()),
                "input_dim": INPUT_DIM, "hidden_size": plan["hidden_size"],
                "runtime_seconds": time.time() - started, "training": loss_records,
                "checkpoint_rule": "Fixed final update; no development/test checkpoint selection."}
    write(folder / "metadata.json", metadata)
    return model, metadata


def predict(model, token, hidden):
    with torch.inference_mode():
        logits, auxiliary, hidden = model(torch.from_numpy(token).reshape(1, 1, -1), hidden)
    assert logits.shape == (1, 1, 9)
    return int(logits[0, 0].argmax()), hidden, auxiliary[0, 0].argmax(-1).tolist()


def rollout(Env, Wrapper, model, task, trials, reset_memory=False, symbolic_seed=None):
    hidden, memory, events, outcomes = None, {}, [], []
    tie_order = list(np.random.RandomState(symbolic_seed or 0).permutation(3))
    for trial in range(trials + 1):
        exploration = trial == 0
        if reset_memory:
            hidden = None
        env = environment(Env, Wrapper, task, exploration)
        state = env.reset()
        previous_action, previous_reward = 9, 0.0
        success = False
        for t in range(20):
            before = np.array(state.observation, copy=True)
            token = encode(before, task["goal"], previous_action, previous_reward,
                           t == 0, exploration, t)
            if model is None:
                action = (explore_action(before, memory, tie_order) if exploration
                          else solve_action(before, task["goal"], memory, tie_order))
                predicted_facts = None
            else:
                action, hidden, predicted_facts = predict(model, token, hidden)
            state, reward, done, _ = env.step(action)
            after = np.array(state.observation, copy=True)
            events.append({"trial": trial, "time": t, "observation": before.tolist(),
                           "executed_action": action, "next_observation": after.tolist(),
                           "reward": float(reward), "predicted_facts": predicted_facts})
            if model is None:
                absorb(memory, before, action, after)
            success = bool(not exploration and np.array_equal(after[4:6], task["goal"]))
            previous_action, previous_reward = action, reward
            if done:
                break
        outcomes.append({"trial": trial, "success": success if not exploration else None,
                         "actual_steps": t + 1,
                         "cost_steps": t + 1 if success or exploration else 20})
        if model is not None:
            # No action is executed from this output. Only ingest the real
            # final observation/action/reward before the next episode reset.
            _, hidden, _ = predict(model, terminal_feedback(
                after, task["goal"], previous_action, previous_reward,
                exploration, t + 1), hidden)
    return outcomes, events


def summarize(records):
    execution = [r for r in records if r["trial"] > 0]
    return {"episodes": len(execution), "successes": sum(r["success"] for r in execution),
            "success_rate": float(np.mean([r["success"] for r in execution])),
            "mean_cost_steps": float(np.mean([r["cost_steps"] for r in execution])),
            "by_trial": [{"trial": trial,
                          "success_rate": float(np.mean([r["success"] for r in execution if r["trial"] == trial])),
                          "mean_cost_steps": float(np.mean([r["cost_steps"] for r in execution if r["trial"] == trial]))}
                         for trial in sorted({r["trial"] for r in execution})]}


def run(args):
    plan = json.loads(args.plan.read_text())
    binding = {k: v for k, v in plan.items() if k not in {"semantic_sha256", "plan_id"}}
    assert digest(binding) == plan["semantic_sha256"]
    for name, expected in plan["code_hashes"].items():
        assert sha(args.plan.parent / name) == expected, name
    assert not args.output.exists(), "Never overwrite a run"
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_bytes(args.plan.read_bytes())
    torch.set_num_threads(plan["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    Env, Wrapper = load_environment(args.upstream)
    dataset, dataset_meta = make_dataset(Env, Wrapper, plan, args.output)
    model_records = []
    models = {}
    for seed in plan["training_seeds"]:
        for arm in ARMS:
            model, metadata = train(dataset, plan, arm, seed, args.output)
            models[(arm, seed)] = model
            model_records.append(metadata)
        same_seed = [r for r in model_records if r["seed"] == seed]
        assert len({r["initial_weights_sha256"] for r in same_seed}) == 1
        assert len({r["sampling_sha256"] for r in same_seed}) == 1
    # All checkpoint bytes are finalized before any test evaluation.
    write(args.output / "training_manifest.json", model_records)
    summary = {"schema": "cooking-learned-pilot-v1", "plan_sha256": plan["semantic_sha256"],
               "plan_id": plan["plan_id"], "dataset": dataset_meta,
               "models": model_records, "arms": {}, "evaluations": {},
               "scope": "Three-seed recurrent BC auxiliary-supervision diagnostic. No DREAM/AD reproduction, OPD, robot result or causal-credit efficacy."}
    chain, record_count, actual_steps, resets = "0" * 64, 0, 0, 0
    log = (args.output / "evaluation.jsonl").open("w")
    for split, ids in [("development", plan["development_environment_ids"]),
                       ("evaluation", plan["evaluation_environment_ids"])]:
        configurations = [(arm, seed, "full_history", models[(arm, seed)])
                          for seed in plan["training_seeds"] for arm in ARMS]
        if split == "evaluation":
            configurations += [(arm, seed, "reset_each_episode", models[(arm, seed)])
                               for seed in plan["training_seeds"] for arm in ARMS]
            configurations += [("symbolic_full_history", seed, "full_history", None)
                               for seed in plan["training_seeds"]]
        for arm, seed, condition, model in configurations:
            records = []
            for task in tasks_for(ids):
                outcomes, events = rollout(Env, Wrapper, model, task, plan["trials"],
                                           reset_memory=condition == "reset_each_episode",
                                           symbolic_seed=seed)
                for outcome in outcomes:
                    row = {"arm": arm, "seed": seed, "condition": condition,
                           "split": split, **task, **outcome,
                           "events": [e for e in events if e["trial"] == outcome["trial"]],
                           "plan_sha256": plan["semantic_sha256"], "previous_hash": chain}
                    chain = digest(row)
                    row["record_hash"] = chain
                    log.write(json.dumps(row, sort_keys=True) + "\n")
                    record_count += 1
                    actual_steps += row["actual_steps"]
                    resets += 1
                    records.append({k: v for k, v in row.items() if k != "events"})
            key = f"{split}/{arm}/{seed}/{condition}"
            summary["evaluations"][key] = summarize(records)
            if split == "evaluation" and condition == "full_history":
                summary["arms"].setdefault(arm, []).append({"seed": seed, **summarize(records)})
            print(json.dumps({"evaluation": key, **summarize(records)}), flush=True)
    log.close()
    expected = (len(tasks_for(plan["development_environment_ids"])) * 9 +
                len(tasks_for(plan["evaluation_environment_ids"])) * 21) * 6
    assert record_count == expected
    summary["coverage"] = {"planned_records": expected, "completed_records": record_count,
                           "missing": 0, "aborted": 0}
    summary["chain_head"] = chain
    summary["cost"] = {"evaluation_steps": actual_steps, "evaluation_resets": resets,
                       "optimizer_updates": sum(m["updates"] for m in model_records),
                       "source_steps": dataset_meta["physical_steps"],
                       "source_resets": dataset_meta["sequences"] * 6}
    summary["artifacts"] = {name: sha(args.output / name) for name in
                            ["source.jsonl", "dataset.json", "evaluation.jsonl", "training_manifest.json", "plan.json"]}
    write(args.output / "summary.json", summary)
    print(json.dumps({"status": "COMPLETED", "coverage": summary["coverage"], "arms": summary["arms"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
