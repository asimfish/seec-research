#!/usr/bin/env python3
"""Controlled diagnostic on the pinned, unmodified DREAM Cooking environment.

Not a DREAM reproduction or a learned robot result. Policies receive observations,
instructions, and their own events, never the environment's privileged env_id.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM_REVISION = "d52204c94067641df6e6649b19abb359b87ff028"
DRAWERS = ((2, 0), (2, 1), (2, 2))
BOWL = (1, 1)
EMPTY = 4
ARMS = ("reset_history", "terminal_success_only", "full_history",
        "utility_memory", "random_exploration", "privileged_oracle")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def move(position, destination):
    if position[0] != destination[0]:
        return 2 if position[0] < destination[0] else 0
    if position[1] != destination[1]:
        return 1 if position[1] < destination[1] else 3
    return None


def absorb(memory, before, action, after):
    """Only an actually observed pickup response certifies a drawer fact."""
    position = tuple(map(int, before[:2]))
    if action == 7 and position in DRAWERS and 0 <= int(after[3]) < EMPTY:
        memory[DRAWERS.index(position)] = int(after[3])


def explore_action(obs, memory, tie_order):
    if int(obs[3]) != EMPTY:
        return 8  # discard at a drawer, never put unknown food in the bowl
    unknown = [i for i in tie_order if i not in memory]
    if not unknown:
        return 4
    index = min(unknown, key=lambda i: (sum(abs(int(obs[j])-DRAWERS[i][j]) for j in (0, 1)), tie_order.index(i)))
    return move(obs[:2], DRAWERS[index]) if tuple(obs[:2]) != DRAWERS[index] else 7


def solve_action(obs, goal, memory, tie_order):
    bowl = list(map(int, obs[4:7]))
    phase = next((i for i in range(2) if bowl[i] == EMPTY), 2)
    if phase == 2:
        return 4
    wanted = goal[phase]
    if int(obs[3]) == wanted:
        action = move(obs[:2], BOWL)
        return 8 if action is None else action
    if int(obs[3]) != EMPTY:
        # Wrong food may be discarded off the bowl; dropping at the bowl would
        # irreversibly contaminate it.
        action = move(obs[:2], DRAWERS[0]) if tuple(obs[:2]) == BOWL else None
        return action if action is not None else 8
    matches = [i for i in tie_order if memory.get(i) == wanted]
    candidates = matches or [i for i in tie_order if i not in memory]
    if not candidates:
        return 4
    index = min(candidates, key=lambda i: (sum(abs(int(obs[j])-DRAWERS[i][j]) for j in (0, 1)), tie_order.index(i)))
    action = move(obs[:2], DRAWERS[index])
    return 7 if action is None else action


def load_environment(upstream):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    assert revision == UPSTREAM_REVISION
    assert not subprocess.check_output(["git", "diff", "--name-only"], cwd=upstream, text=True).strip()
    # The only compatibility shim restores NumPy's removed integer dtype alias;
    # upstream transition, observation, reward and termination code is untouched.
    if "int" not in np.__dict__:
        np.int = int
    sys.path.insert(0, str(upstream))
    from envs.cooking import CookingGridEnv, InstructionWrapper
    class FixedInstructionWrapper(InstructionWrapper):
        def _generate_instructions(self, test=False):
            return list(self.registered_goal)
    return CookingGridEnv, FixedInstructionWrapper


def experiment(args):
    plan = json.loads(args.plan.read_text())
    assert plan["runner_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert plan["upstream_revision"] == UPSTREAM_REVISION
    assert not args.output.exists(), "Never overwrite an existing experiment"
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_bytes(args.plan.read_bytes())
    Env, Wrapper = load_environment(args.upstream)
    rows, chain, steps, resets = [], "0"*64, 0, 0
    started = time.time()
    log = (args.output / "episodes.jsonl").open("w")
    events = (args.output / "events.jsonl").open("w")

    def record(value):
        nonlocal chain
        value["previous_hash"] = chain
        chain = digest(value)
        value["record_hash"] = chain
        log.write(json.dumps(value, sort_keys=True)+"\n")

    tasks = plan["evaluation_tasks"]
    for seed in plan["seeds"]:
        for task in tasks:
            task_id, env_id, goal = task["task_id"], task["environment_id"], task["goal"]
            tie_order = list(np.random.RandomState(seed).permutation(3))
            # Privileged assignment exists only in this evaluator scope.
            truth = {i: (env_id // (4**i)) % 4 for i in range(3)}
            for arm in ARMS:
                base = Env(np.array([env_id]), lambda state: state, max_steps=20)
                state = base.reset(); resets += 1
                memory, exploration = {}, []
                rng = np.random.RandomState(seed * 1000 + env_id)
                for t in range(20):
                    before = np.array(state.observation, copy=True)
                    action = int(rng.choice([0,1,2,3,7,8])) if arm == "random_exploration" else explore_action(before, memory, tie_order)
                    state, reward, done, _ = base.step(action); steps += 1
                    after = np.array(state.observation, copy=True)
                    absorb(memory, before, action, after)
                    exploration.append({"time": t, "observation": before.tolist(), "executed_action": action, "next_observation": after.tolist(), "reward": reward})
                record({"phase": "exploration", "task_id": task_id, "arm": arm, "seed": seed,
                        "steps": 20, "return": 0, "task_success": None, "facts": len(memory), "plan_sha256": plan["semantic_sha256"]})
                events.write(json.dumps({"task_id":task_id,"arm":arm,"seed":seed,"events":exploration})+"\n")
                if arm in {"reset_history", "terminal_success_only"}:
                    memory = {}
                elif arm == "privileged_oracle":
                    memory = dict(truth)
                # Both full_history and utility_memory preserve exactly the
                # same sufficient facts. We do not invent an advantage here.
                for trial in range(1, plan["trials"]+1):
                    base = Env(np.array([env_id]), lambda state: state, max_steps=20)
                    env = Wrapper(base, [], seed=seed, fixed_instructions=True)
                    env.registered_goal = goal
                    state = env.reset(); resets += 1
                    if arm == "reset_history":
                        memory = {}
                    starting_memory = dict(memory)
                    local = dict(memory)
                    trajectory, success, total_reward = [], False, 0.0
                    for t in range(20):
                        before = np.array(state.observation, copy=True)
                        action = solve_action(before, list(map(int,state.instructions)), local, tie_order)
                        state, reward, done, _ = env.step(action); steps += 1
                        after = np.array(state.observation, copy=True)
                        absorb(local, before, action, after)
                        trajectory.append({"time":t,"observation":before.tolist(),"executed_action":action,"next_observation":after.tolist(),"reward":float(reward)})
                        total_reward += reward
                        success = bool(np.array_equal(after[4:6], goal))
                        if done:
                            break
                    if arm != "terminal_success_only" or success:
                        memory = local
                    else:
                        memory = starting_memory
                    row = {"phase":"exploitation","task_id":task_id,"arm":arm,"seed":seed,
                           "trial":trial,"steps":len(trajectory),"success":success,
                           "return":round(total_reward,8),"initial_facts":len(starting_memory),
                           "final_facts":len(local),"plan_sha256":plan["semantic_sha256"]}
                    rows.append(row); record(row)
                    events.write(json.dumps({"task_id":task_id,"arm":arm,"seed":seed,"trial":trial,"events":trajectory})+"\n")
    log.close(); events.close()
    summary = {"schema":"interactive-cooking-diagnostic-v1", "plan_sha256":plan["semantic_sha256"],
               "upstream_revision":UPSTREAM_REVISION,"runner_sha256":plan["runner_sha256"],
               "source_revision":plan["source_revision"],"task_count":len(tasks),"seeds":plan["seeds"],
               "trials":plan["trials"],"gradient_updates":0,"teacher_model_calls":0,
               "steps":steps,"resets":resets,"chain_head":chain,"arms":{},
               "scope":"Modified evaluation protocol on original DREAM Cooking transitions. Hand-coded skill-level memory policies; not trained DREAM, AD, RL2, VLA, or a robot efficacy result.",
               "runtime_seconds":round(time.time()-started,3)}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        seed_means = [float(np.mean([r["steps"] for r in arm_rows if r["seed"]==s])) for s in plan["seeds"]]
        summary["arms"][arm] = {"episodes":len(arm_rows),"successes":sum(r["success"] for r in arm_rows),
             "mean_steps":float(np.mean([r["steps"] for r in arm_rows])),"seed_mean_steps":seed_means,
             "by_trial":[{"trial":k,"successes":sum(r["success"] for r in arm_rows if r["trial"]==k),
               "total":sum(r["trial"]==k for r in arm_rows),
               "mean_steps":float(np.mean([r["steps"] for r in arm_rows if r["trial"]==k]))} for k in range(1,plan["trials"]+1)]}
    expected = len(tasks)*len(plan["seeds"])*len(ARMS)*(plan["trials"]+1)
    summary["coverage"]={"planned_records":expected,"completed_records":len(rows)+len(tasks)*len(plan["seeds"])*len(ARMS),"missing":0,"aborted":0}
    summary["artifacts"]={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ["episodes.jsonl","events.jsonl","plan.json"]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"coverage":summary["coverage"],"task_count":len(tasks),"arms":summary["arms"],"steps":steps},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    experiment(parser.parse_args())
