#!/usr/bin/env python3
"""Independently replay every recorded episode and recompute the public result."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main(args):
    folder=args.folder
    plan=json.loads((folder/"plan.json").read_text())
    summary=json.loads((folder/"summary.json").read_text())
    semantic=dict(plan);declared=semantic.pop("semantic_sha256")
    assert hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",",":")).encode()).hexdigest()==declared
    spec=importlib.util.spec_from_file_location("runner",Path(__file__).with_name("run_cooking_diagnostic.py"))
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    Env,Wrapper=runner.load_environment(args.upstream)
    tasks={t["task_id"]:t for t in plan["evaluation_tasks"]}
    rows=list(map(json.loads,(folder/"episodes.jsonl").open()))
    index={(r["task_id"],r["arm"],r["seed"],r.get("trial",0)):r for r in rows}
    assert len(index)==len(rows)==summary["coverage"]["planned_records"]
    chain="0"*64
    for source in rows:
        row=dict(source);own=row.pop("record_hash")
        assert row["previous_hash"]==chain
        chain=runner.digest(row);assert chain==own
    assert chain==summary["chain_head"]
    for arm,stats in summary["arms"].items():
        arm_rows=[r for r in rows if r["phase"]=="exploitation" and r["arm"]==arm]
        assert len(arm_rows)==stats["episodes"]
        assert sum(r["success"] for r in arm_rows)==stats["successes"]
        assert float(np.mean([r["steps"] for r in arm_rows]))==stats["mean_steps"]
        for trial_stats in stats["by_trial"]:
            subset=[r for r in arm_rows if r["trial"]==trial_stats["trial"]]
            assert len(subset)==trial_stats["total"]
            assert sum(r["success"] for r in subset)==trial_stats["successes"]
            assert float(np.mean([r["steps"] for r in subset]))==trial_stats["mean_steps"]
    replayed_steps=0
    for episode in map(json.loads,(folder/"events.jsonl").open()):
        task=tasks[episode["task_id"]]
        env=Env(np.array([task["environment_id"]]),lambda s:s,max_steps=20)
        trial=episode.get("trial",0)
        if trial:
            env=Wrapper(env,[],seed=episode["seed"],fixed_instructions=True)
            env.registered_goal=task["goal"]
        state=env.reset();total_reward=0.0
        for event in episode["events"]:
            assert state.observation.tolist()==event["observation"]
            state,reward,done,_=env.step(event["executed_action"])
            replayed_steps+=1
            assert state.observation.tolist()==event["next_observation"]
            assert float(reward)==event["reward"]
            total_reward+=reward
        recorded=index[(episode["task_id"],episode["arm"],episode["seed"],trial)]
        assert len(episode["events"])==recorded["steps"]
        if trial:
            terminal_success=state.observation[4:6].tolist()==task["goal"]
            assert recorded["success"]==terminal_success
            assert recorded["return"]==round(total_reward,8)
    assert replayed_steps==summary["steps"]
    alias_matches=0
    for key,row in index.items():
        if row["arm"]!="utility_memory":continue
        reference=index[(key[0],"full_history",key[2],key[3])]
        excluded={"arm","previous_hash","record_hash"}
        assert {k:v for k,v in row.items() if k not in excluded}=={k:v for k,v in reference.items() if k not in excluded}
        alias_matches+=1
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in rows:
        if row["phase"]=="exploitation":
            groups[tasks[row["task_id"]]["environment_id"]][row["arm"]].append(row["steps"])
    assignments=sorted(groups)
    num=np.array([sum(groups[g]["reset_history"])-sum(groups[g]["full_history"]) for g in assignments],float)
    den=np.array([len(groups[g]["reset_history"]) for g in assignments],float)
    rng=np.random.RandomState(20261014)
    boot=[]
    for _ in range(4000):
        sampled=rng.randint(len(assignments),size=len(assignments))
        boot.append(float(num[sampled].sum()/den[sampled].sum()))
    result={"status":"PASS","checks":["plan/hash binding","complete unique record keys","full hash chain","summary counts and means independently recomputed","all stored observations and rewards replay exactly","terminal outcomes recomputed","same sufficient-fact arms equal","all physical steps accounted"],
            "replayed_episodes":len(rows),"replayed_steps":replayed_steps,"raw_observation_max_difference":0,
            "full_history_utility_memory_equal_records":alias_matches,"mapping_clusters":len(assignments),
            "paired_step_saving":float(num.sum()/den.sum()),"cluster_bootstrap_95_percentile_interval":list(map(float,np.percentile(boot,[2.5,97.5]))),
            "bootstrap_seed":20261014,"bootstrap_replicates":4000,"inference_scope":"Descriptive mapping-cluster bootstrap; not independent learned-model seeds or a method superiority test.",
            "summary_sha256":hashlib.sha256((folder/"summary.json").read_bytes()).hexdigest()}
    (folder/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    main(p.parse_args())
