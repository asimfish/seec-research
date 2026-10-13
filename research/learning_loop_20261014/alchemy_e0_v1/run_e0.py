"""Registered zero-training qualification on unmodified Symbolic Alchemy."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from dm_alchemy import symbolic_alchemy as sa
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs
from policies import VisibleBayesGreedy, VisibleRandom, privileged_greedy_action


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=numpy_scalar
    ).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize_observation(obs):
    assert set(obs) == {"symbolic_obs"}
    return {"symbolic_obs": np.asarray(obs["symbolic_obs"], dtype=np.float32).tolist()}


def chemistry_digest(env):
    chem = env._chemistry
    data = {
        "potion_dim": list(chem.potion_map.dim_map),
        "potion_dir": list(chem.potion_map.dir_map),
        "stone_dir": list(chem.stone_map.latent_pos_dir),
        "rotation": chem.rotation.tolist(),
        "graph": graphs.convert_graph_to_adj_mat(chem.graph).tolist(),
    }
    return hashlib.sha256(canonical(data)).hexdigest()


def make_policy(arm, seed, precomputed):
    if arm == "visible_random":
        return VisibleRandom(seed + 100000)
    if arm in {"bayes_reset", "bayes_carry"}:
        return VisibleBayesGreedy(precomputed, carry=arm == "bayes_carry")
    assert arm == "privileged_greedy"
    return None


def checked_plan(folder):
    plan = json.loads((folder / "plan.json").read_text())
    for name, digest in plan["code_hashes"].items():
        assert sha(folder / name) == digest, name
    identity = {k: v for k, v in plan.items() if k not in {"plan_id", "registered_at_utc"}}
    assert plan["plan_id"] == "alchemy-e0-" + hashlib.sha256(canonical(identity)).hexdigest()[:16]
    runtime = json.loads((folder / "RUNTIME.json").read_text())
    assert np.__version__ == runtime["versions"]["numpy"]
    assert platform.python_version() == runtime["versions"]["python"]
    root = Path(sa.__file__).resolve().parents[1]
    for name, digest in runtime["upstream_source_file_hashes"].items():
        assert sha(root / name) == digest, name
    for name, digest in runtime["generated_binding_hashes"].items():
        assert sha(root / name) == digest, name
    return plan


def run(folder, output):
    plan = checked_plan(folder)
    plan_sha = sha(folder / "plan.json")
    if output.exists():
        raise FileExistsError("Never overwrite a scientific run")
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes((folder / "plan.json").read_bytes())
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in plan["code_hashes"]:
        (snapshot / name).write_bytes((folder / name).read_bytes())
    precomputed = precomputed_maps.load_from_level_name(plan["level"])
    assert precomputed is not None
    started = time.monotonic()
    chain = "0" * 64
    episodes, records, steps = [], 0, 0
    try:
        with (output / "events.jsonl").open("w") as log:
            for seed in plan["environment_seeds"]:
                for arm in plan["arms"]:
                    env = sa.get_symbolic_alchemy_level(plan["level"], seed=seed)
                    assert env.max_steps_per_trial == 20 and env._num_trials == 10
                    assert not env.see_chemistries and not env.trackers
                    policy = make_policy(arm, seed, precomputed)
                    timestep = env.reset()
                    chem = chemistry_digest(env)
                    rewards, initial_keys, worlds, updates = [], [], [], []
                    for trial in range(10):
                        assert env.trial_number == trial
                        obs = timestep.observation
                        initial_keys.append(
                            hashlib.sha256(canonical(serialize_observation(obs))).hexdigest()
                        )
                        if policy is not None:
                            policy.begin_trial(obs)
                        if isinstance(policy, VisibleBayesGreedy):
                            worlds.append(policy.belief.num_world_states)
                        reward = 0.0
                        for step in range(20):
                            if time.monotonic() - started > plan["max_wall_seconds"]:
                                raise TimeoutError("Registered wall budget exceeded")
                            before = timestep.observation
                            action = (
                                privileged_greedy_action(env, before, step)
                                if policy is None
                                else policy.act(before, step)
                            )
                            env.action_spec().validate(np.int64(action))
                            timestep = env.step(action)
                            changed_trial = env.trial_number != trial
                            if policy is not None:
                                policy.observe(before, action, timestep.observation, changed_trial)
                            reward += float(timestep.reward or 0)
                            row = {
                                "plan_id": plan["plan_id"],
                                "plan_sha256": plan_sha,
                                "arm": arm,
                                "environment_seed": seed,
                                "trial": trial,
                                "step": step,
                                "observation": serialize_observation(before),
                                "action": int(action),
                                "next_observation": serialize_observation(timestep.observation),
                                "reward": float(timestep.reward or 0),
                                "last": bool(timestep.last()),
                                "changed_trial": changed_trial,
                                "previous_hash": chain,
                            }
                            chain = hashlib.sha256(canonical(row)).hexdigest()
                            row["record_hash"] = chain
                            log.write(json.dumps(row, sort_keys=True) + "\n")
                            records += 1
                            steps += 1
                            if changed_trial:
                                assert step == 19
                            assert chemistry_digest(env) == chem
                        rewards.append(reward)
                        if isinstance(policy, VisibleBayesGreedy):
                            updates.append(policy.outcome_updates)
                    assert timestep.last()
                    episode = {
                        "seed": seed,
                        "arm": arm,
                        "trial_rewards": rewards,
                        "total_reward": sum(rewards),
                        "actions": 200,
                        "resets": 1,
                        "chemistry_digest": chem,
                        "trial_initial_observation_keys": initial_keys,
                        "belief_world_states_at_trial_start": worlds,
                        "cumulative_outcome_updates": updates,
                        "boundary_potions_without_old_observation": getattr(
                            policy, "skipped_boundary_potions", 0
                        ),
                    }
                    episodes.append(episode)
                    print(
                        json.dumps({"seed": seed, "arm": arm, "completed_trials": 10}), flush=True
                    )
        assert records == plan["expected_actions"]
        for seed in plan["environment_seeds"]:
            rows = [r for r in episodes if r["seed"] == seed]
            assert len({r["chemistry_digest"] for r in rows}) == 1
            assert all(
                r["trial_initial_observation_keys"] == rows[0]["trial_initial_observation_keys"]
                for r in rows
            )
        summary = {
            "schema": "alchemy-e0-qualification-v1",
            "plan_id": plan["plan_id"],
            "plan_sha256": sha(folder / "plan.json"),
            "status": "COMPLETE_NOT_METHOD_EFFICACY",
            "episodes": episodes,
            "ledger_chain_head": chain,
            "ledger_sha256": sha(output / "events.jsonl"),
            "coverage": {
                "planned_episodes": len(plan["arms"]) * len(plan["environment_seeds"]),
                "completed_episodes": len(episodes),
                "missing": 0,
                "aborted": 0,
            },
            "cost": {
                "environment_actions": steps,
                "resets": len(episodes),
                "gradient_updates": 0,
                "wall_seconds": time.monotonic() - started,
            },
            "scope": (
                "Zero-training symbolic-domain qualification; Bayesian myopic adapters, "
                "not the full ideal observer or original method reproduction"
            ),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as exc:
        (output / "incomplete.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "reason": str(exc),
                    "completed_episodes": len(episodes),
                    "completed_steps": steps,
                    "gradient_updates": 0,
                    "wall_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-folder", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan_folder, args.output)
