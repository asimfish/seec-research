#!/usr/bin/env python3
"""Independent raw-action and policy replay plus a fixed branch training repeat."""

import argparse
import json
from pathlib import Path

import analyze_credit as analysis
import numpy as np
import run_credit as run
import torch


def replay(record, env_type, wrapper_type, model):
    hidden, memory = None, {}
    rng = np.random.RandomState(record["behavior_seed"])
    order = list(np.random.RandomState(record["behavior_seed"]).permutation(3))
    checked_actions = checked_policy = 0
    for outcome in record["outcomes"]:
        env = run.pilot.environment(env_type, wrapper_type, record["task"], outcome["exploration"])
        state = env.reset()
        previous_action, previous_reward = 9, 0.0
        events = [e for e in record["events"] if e["trial"] == outcome["trial"]]
        assert len(events) == outcome["actual_steps"] and events
        success = False
        for t, event in enumerate(events):
            before = np.asarray(state.observation)
            assert np.array_equal(before, event["observation"])
            assert event["time"] == t and event["exploration"] == outcome["exploration"]
            if record["annotated"]:
                label = run.coverage.target(
                    before, record["task"]["goal"], memory, order, outcome["exploration"]
                )
                assert label == event["teacher_action"]
            else:
                assert event["teacher_action"] is None
                label = None
            if model is not None:
                token = run.pilot.encode(
                    before,
                    record["task"]["goal"],
                    previous_action,
                    previous_reward,
                    t == 0,
                    outcome["exploration"],
                    t,
                )
                action, hidden, _ = run.pilot.predict(model, token, hidden)
                checked_policy += 1
            else:
                action = label if label >= 0 else 4
                if rng.rand() < record["epsilon"]:
                    action = int(rng.choice(run.pilot.BEHAVIOR_ACTIONS))
            assert action == event["executed_action"]
            state, reward, done, _ = env.step(action)
            after = np.asarray(state.observation)
            assert np.array_equal(after, event["next_observation"])
            assert float(reward) == event["reward"] and bool(done) == event["env_done"]
            if record["annotated"]:
                run.pilot.absorb(memory, before, action, after)
            previous_action, previous_reward = action, reward
            success = (
                None
                if outcome["exploration"]
                else bool(np.array_equal(after[4:6], record["task"]["goal"]))
            )
            checked_actions += 1
        assert outcome["success"] == success
        assert outcome["terminated"] == bool(done) and outcome["budget_truncated"] == (not done)
        if model is not None:
            _, hidden, _ = run.pilot.predict(
                model,
                run.pilot.terminal_feedback(
                    after,
                    record["task"]["goal"],
                    previous_action,
                    previous_reward,
                    outcome["exploration"],
                    len(events),
                ),
                hidden,
            )
    return checked_actions, checked_policy


def verify(args):
    root = args.run
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    assert run.pilot.digest(run.semantic_binding(plan)) == plan["semantic_sha256"]
    assert summary["plan_sha256"] == plan["semantic_sha256"]
    for name, expected in plan["code_hashes"].items():
        assert run.pilot.sha(args.plan_folder / name) == expected
    snapshot = json.loads((root / "source_snapshot/manifest.json").read_text())
    assert set(snapshot) == set(plan["code_hashes"])
    for name, entry in snapshot.items():
        assert run.pilot.sha(root / entry["path"]) == entry["sha256"] == plan["code_hashes"][name]
    for name, expected in summary["artifacts"].items():
        assert run.pilot.sha(root / name) == expected
    sets = [set(plan[k]) for k in ["source_ids", "feedback_ids", "audit_ids"]]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
    torch.set_num_threads(plan["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    env_type, wrapper_type = run.pilot.load_environment(args.upstream)
    training = json.loads((root / "training.json").read_text())
    checkpoints, infos = {}, {}
    for info in training:
        key = (info["seed"], info.get("candidate"), info["stage"])
        assert key not in checkpoints
        checkpoint = root / info["checkpoint"]
        assert run.pilot.sha(checkpoint) == info["checkpoint_sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = run.pilot.HistoryPolicy(plan["hidden_size"])
        model.load_state_dict(state["weights"])
        model.eval()
        assert run.pilot.weights_digest(model) == info["final_weights_sha256"]
        opt = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
        opt.load_state_dict(state["optimizer"])
        assert run.optimizer_digest(opt) == info["final_optimizer_sha256"]
        checkpoints[key], infos[key] = model, info
    assert len(checkpoints) == 51 and sum(r["updates"] for r in training) == plan["total_updates"]
    specs = {s["id"]: s for s in plan["candidate_specs"]}
    source_seen, evaluation_seen = set(), set()
    expected_base = {
        (t["task_id"], b, e)
        for t in run.pilot.tasks_for(plan["source_ids"])
        for b in plan["initial_probe_budgets"]
        for e in [0.0, 0.1]
    }
    expected_candidates = {
        (s, c, i) for s in plan["training_seeds"] for c in specs for i in range(24)
    }
    expected_eval = {}
    for seed in plan["training_seeds"]:
        for stage, candidates in [
            ("initial", [None]),
            ("short", list(specs)),
            ("long", list(specs)),
        ]:
            for candidate in candidates:
                panels = [
                    ("feedback", run.task_panel(plan["feedback_ids"])),
                    ("audit", run.task_panel(plan["audit_ids"])),
                ]
                if stage == "short":
                    panels.insert(0, ("source", specs[candidate]["tasks"]))
                if stage == "long":
                    panels = panels[-1:]
                for split, tasks in panels:
                    for i, task in enumerate(tasks):
                        expected_eval[(seed, candidate, stage, split, i)] = task
    assert len(expected_eval) == plan["evaluation_histories"]
    groups, source_groups, base, candidate_data = {}, {}, [], {}
    counts = {
        "source_actions": 0,
        "evaluation_actions": 0,
        "source_policy_actions": 0,
        "evaluation_policy_actions": 0,
        "source_resets": 0,
        "evaluation_resets": 0,
    }
    for filename, expected_head in [
        ("source.jsonl", summary["source_chain_head"]),
        ("evaluation.jsonl", summary["evaluation_chain_head"]),
    ]:
        chain, record_count = "0" * 64, 0
        for line in (root / filename).open():
            record = json.loads(line)
            claimed = record.pop("record_hash")
            assert record["previous_hash"] == chain and run.pilot.digest(record) == claimed
            assert (
                record["record_index"] == record_count
                and record["plan_sha256"] == plan["semantic_sha256"]
                and record["plan_file_sha256"] == run.pilot.sha(root / "plan.json")
            )
            chain, record_count = claimed, record_count + 1
            if filename == "source.jsonl":
                assert (
                    record["split"] == "source"
                    and record["task"]["environment_id"] in sets[0]
                    and record["annotated"]
                )
                if record["kind"] == "base_source":
                    key = (record["task"]["task_id"], record["probe_steps"], record["epsilon"])
                    assert key in expected_base and key not in source_seen
                    source_seen.add(key)
                    model = None
                    base.append(run.reconstruct(record))
                else:
                    key = (record["seed"], record["candidate"], record["case_index"])
                    assert key in expected_candidates and key not in source_seen
                    assert (
                        record["task"] == specs[record["candidate"]]["tasks"][record["case_index"]]
                    )
                    source_seen.add(key)
                    model = checkpoints[(record["seed"], None, "initial")]
                    assert run.pilot.weights_digest(model) == record["collector_weights_sha256"]
                    group = (record["seed"], record["candidate"])
                    source_groups.setdefault(group, []).append(record)
                    candidate_data.setdefault(group, []).append(run.reconstruct(record))
                prefix = "source"
            else:
                key = (
                    record["seed"],
                    record["candidate"],
                    record["stage"],
                    record["split"],
                    record["case_index"],
                )
                assert key in expected_eval and key not in evaluation_seen
                assert (
                    record["task"] == expected_eval[key]
                    and not record["annotated"]
                    and record["probe_steps"] == 0
                )
                evaluation_seen.add(key)
                model = checkpoints[key[:3]]
                groups.setdefault(key[:4], []).append(record)
                prefix = "evaluation"
            actions, decisions = replay(record, env_type, wrapper_type, model)
            counts[prefix + "_actions"] += actions
            counts[prefix + "_policy_actions"] += decisions
            counts[prefix + "_resets"] += len(record["outcomes"])
        assert chain == expected_head
    assert source_seen == expected_base | expected_candidates and evaluation_seen == set(
        expected_eval
    )
    assert len(base) == plan["base_sequences"]
    for row in summary["results"]:
        seed, candidate = row["seed"], row["candidate"]
        assert row["source_before"] == run.metrics(source_groups[(seed, candidate)])
        for split in ["feedback", "audit"]:
            assert row["baseline"][split] == run.metrics(groups[(seed, None, "initial", split)])
        for stage in ["short", "long"]:
            for split in row[stage]:
                assert row[stage][split] == run.metrics(groups[(seed, candidate, stage, split)])
            info = infos[(seed, candidate, stage)]
            origin = infos[(seed, None, "initial")]
            assert info["initial_weights_sha256"] == origin["final_weights_sha256"]
            assert info["initial_optimizer_sha256"] == origin["final_optimizer_sha256"]
    for prefix in ["source", "evaluation"]:
        assert counts[prefix + "_actions"] == summary["cost"][prefix + "_actions"]
        assert counts[prefix + "_resets"] == summary["cost"][prefix + "_resets"]
    assert (
        counts["source_actions"] <= plan["max_source_actions"]
        and counts["evaluation_actions"] <= plan["max_evaluation_actions"]
    )
    # Fixed first branch, not a branch selected for favorable scientific scores.
    seed, candidate = plan["training_seeds"][0], plan["candidate_specs"][0]["id"]
    state = torch.load(
        root / infos[(seed, None, "initial")]["checkpoint"], map_location="cpu", weights_only=True
    )
    model = run.pilot.HistoryPolicy(plan["hidden_size"])
    model.load_state_dict(state["weights"])
    opt = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    opt.load_state_dict(state["optimizer"])
    repeated = run.train(
        model, opt, base, candidate_data[(seed, candidate)], plan["short_updates"], plan, seed
    )
    for k, value in repeated.items():
        assert value == infos[(seed, candidate, "short")][k], k
    if (root / "analysis.json").exists():
        assert json.loads((root / "analysis.json").read_text()) == analysis.analyze(summary, plan)
    result = {
        "status": "PASS",
        "plan_id": plan["plan_id"],
        "summary_sha256": run.pilot.sha(root / "summary.json"),
        "source_histories": len(source_seen),
        "evaluation_histories": len(evaluation_seen),
        **counts,
        "raw_observation_max_difference": 0,
        "checkpoints_bound": 51,
        "fixed_short_branch_training_repeat": "EXACT",
        "extra_verification_updates": plan["short_updates"],
        "extra_verification_actions": counts["source_actions"] + counts["evaluation_actions"],
        "extra_verification_resets": counts["source_resets"] + counts["evaluation_resets"],
        "extra_verification_teacher_label_calls": 2 * counts["source_actions"],
        "scope": (
            "All action/label/policy replay, bound checkpoints and one "
            "repeated short training; not snapshot E4 or 51 repeated "
            "trainings"
        ),
    }
    run.pilot.write(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "upstream", "plan-folder"]:
        parser.add_argument("--" + name, type=Path, required=True)
    verify(parser.parse_args())
