"""Independent simulator/policy replay of every registered E0 ledger record."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dm_alchemy import symbolic_alchemy as sa
from dm_alchemy.ideal_observer import precomputed_maps
from policies import VisibleBayesGreedy, privileged_greedy_action
from run_e0 import canonical, checked_plan, chemistry_digest, make_policy, sha


def replay(folder, plan_folder):
    plan = checked_plan(plan_folder)
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["plan_sha256"] == sha(plan_folder / "plan.json")
    assert summary["ledger_sha256"] == sha(folder / "events.jsonl")
    precomputed = precomputed_maps.load_from_level_name(plan["level"])
    chain = "0" * 64
    count, env, policy, key = 0, None, None, None
    timestep = None
    replayed_rewards = {}
    posterior_matches, posterior_comparisons = 0, 0
    with (folder / "events.jsonl").open() as log:
        for line in log:
            row = json.loads(line)
            digest = row.pop("record_hash")
            assert row["previous_hash"] == chain
            assert hashlib.sha256(canonical(row)).hexdigest() == digest
            chain = digest
            assert (
                row["plan_id"] == plan["plan_id"] and row["plan_sha256"] == summary["plan_sha256"]
            )
            new_key = (row["environment_seed"], row["arm"])
            if new_key != key:
                if env is not None:
                    assert timestep.last()
                key = new_key
                env = sa.get_symbolic_alchemy_level(plan["level"], seed=key[0])
                policy = make_policy(key[1], key[0], precomputed)
                timestep = env.reset()
                expected = next(r for r in summary["episodes"] if (r["seed"], r["arm"]) == key)
                assert chemistry_digest(env) == expected["chemistry_digest"]
            trial, step = divmod(count % 200, 20)
            assert trial == row["trial"] and step == row["step"]
            if step == 0 and policy is not None:
                policy.begin_trial(timestep.observation)
            if step == 0 and isinstance(policy, VisibleBayesGreedy):
                posterior_comparisons += 1
                assert (
                    policy.belief.num_world_states
                    == expected["belief_world_states_at_trial_start"][trial]
                )
                posterior_matches += 1
            np.testing.assert_array_equal(
                timestep.observation["symbolic_obs"],
                np.array(row["observation"]["symbolic_obs"], dtype=np.float32),
            )
            action = (
                privileged_greedy_action(env, timestep.observation, step)
                if policy is None
                else policy.act(timestep.observation, step)
            )
            assert int(action) == row["action"]
            before = timestep.observation
            timestep = env.step(action)
            changed = env.trial_number != trial
            if policy is not None:
                policy.observe(before, action, timestep.observation, changed)
            np.testing.assert_array_equal(
                timestep.observation["symbolic_obs"],
                np.array(row["next_observation"]["symbolic_obs"], dtype=np.float32),
            )
            assert bool(timestep.last()) == row["last"] and changed == row["changed_trial"]
            reward = float(timestep.reward or 0)
            assert reward == row["reward"]
            reward_key = (*key, trial)
            replayed_rewards[reward_key] = replayed_rewards.get(reward_key, 0.0) + reward
            count += 1
    assert timestep.last() and chain == summary["ledger_chain_head"]
    assert count == plan["expected_actions"]
    for r in summary["episodes"]:
        assert r["trial_rewards"] == [replayed_rewards[(r["seed"], r["arm"], j)] for j in range(10)]
    report = {
        "status": "PASS",
        "plan_id": plan["plan_id"],
        "summary_sha256": sha(folder / "summary.json"),
        "records": count,
        "raw_observation_max_difference": 0,
        "recomputed_policy_actions": count,
        "trial_posterior_counts_compared": posterior_comparisons,
        "trial_posterior_counts_equal": posterior_matches,
        "extra_verification_actions": count,
        "extra_verification_resets": len(summary["episodes"]),
        "gradient_updates": 0,
        "scope": (
            "All actions, observations, rewards, posterior counts and plan/ledger bindings; "
            "not physical snapshot E4"
        ),
    }
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan-folder", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    replay(args.run, args.plan_folder)
