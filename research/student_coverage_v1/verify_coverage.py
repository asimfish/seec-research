#!/usr/bin/env python3
"""Independent replay, prefix-grounded labels and all-checkpoint verification."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import run_coverage as experiment

pilot=experiment.pilot


def observed_fact_update(memory,before,action,after):
    drawers=[(2,0),(2,1),(2,2)]
    if action==7 and tuple(before[:2]) in drawers and after[3] in range(4):
        memory[drawers.index(tuple(before[:2]))]=after[3]


def aggregate(rows):
    executions=[r for r in rows if r['trial']>0]
    def one(items):
        return {'success_rate':sum(r['success'] for r in items)/len(items),
                'mean_cost_steps':sum(r['cost_steps'] for r in items)/len(items)}
    return {'episodes':len(executions),'successes':sum(r['success'] for r in executions),**one(executions),
            'by_trial':[{'trial':t,**one([r for r in executions if r['trial']==t])} for t in range(1,6)]}


def load_model(path,expected,plan):
    assert pilot.sha(path)==expected
    model=pilot.HistoryPolicy(plan['hidden_size']).eval()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return model


def verify(args):
    root=args.folder;plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert pilot.digest({k:v for k,v in plan.items() if k not in {'semantic_sha256','plan_id'}})==plan['semantic_sha256']
    for name,expected in plan['code_hashes'].items():assert pilot.sha(root.parent/name)==expected,name
    for name,expected in summary['artifacts'].items():assert pilot.sha(root/name)==expected,name
    assert pilot.sha(args.parent/'summary.json')==plan['parent_summary_sha256']
    assert pilot.sha(args.parent/'source.jsonl')==plan['base_source_sha256']
    torch.set_num_threads(plan['cpu_threads']);Env,Wrapper=pilot.load_environment(args.upstream)
    collectors={};models={}
    for m in summary['models']:
        key=(m['arm'],m['seed']);models[key]=load_model(root/'checkpoints'/f'{m["arm"]}-{m["seed"]}'/'policy.pt',m['checkpoint_sha256'],plan)
        assert pilot.weights_digest(models[key])==m['final_weights_sha256']
        assert m['updates']==600 and m['parameter_count']==63576
        initial=args.parent/'checkpoints'/f'action_only-{m["seed"]}'/'policy.pt'
        assert pilot.sha(initial)==m['initial_checkpoint_sha256']==plan['initial_checkpoint_sha256'][str(m['seed'])]
        init_model=load_model(initial,m['initial_checkpoint_sha256'],plan);assert pilot.weights_digest(init_model)==m['initial_weights_sha256']
        for r in m['rounds']:
            path=root/'collectors'/f'{m["arm"]}-{m["seed"]}'/f'round-{r["round"]}.pt'
            collector=load_model(path,r['collector_checkpoint_sha256'],plan)
            assert pilot.weights_digest(collector)==r['collector_weights_sha256']
            collectors[(*key,r['round'])]=collector
            assert r['dataset_sequences']==1062+177*r['round'] and r['updates']==300
    for seed in plan['training_seeds']:
        rows=[r for r in summary['models'] if r['seed']==seed];assert len(rows)==4
        assert len({r['initial_weights_sha256'] for r in rows})==1
        for index in [0,1]:assert len({r['rounds'][index]['sampling_sha256'] for r in rows})==1
    expected={(arm,seed,r,t['task_id']) for arm in experiment.ARMS for seed in plan['training_seeds'] for r in [1,2] for t in pilot.tasks_for(plan['train_environment_ids'])}
    seen=set();chain='0'*64;source_steps=0;source_predictions=0;labels=0;ignored=0;grouped=defaultdict(lambda:{'steps':0,'sequences':0})
    for line in (root/'collected.jsonl').open():
        row=json.loads(line);key=(row['arm'],row['seed'],row['round'],row['task']['task_id'])
        assert key in expected and key not in seen;seen.add(key)
        assert row['split']=='train' and row['task']['environment_id'] in plan['train_environment_ids']
        assert row['previous_hash']==chain;chain=pilot.digest({k:v for k,v in row.items() if k!='record_hash'});assert chain==row['record_hash']
        assert row['plan_sha256']==plan['semantic_sha256']
        model=collectors[key[:3]];assert pilot.weights_digest(model)==row['collector_weights_sha256']
        sequence=experiment.from_events(row);cursor=0;hidden=None;memory={};rng=np.random.RandomState(row['behavior_seed']);order=list(rng.permutation(3))
        for trial in range(6):
            env=pilot.environment(Env,Wrapper,row['task'],trial==0);state=env.reset();previous_action,previous_reward=9,0.0
            events=[e for e in row['events'] if e['trial']==trial]
            for t,event in enumerate(events):
                before=state.observation.tolist();assert before==event['observation'] and event['time']==t
                token=pilot.encode(before,row['task']['goal'],previous_action,previous_reward,t==0,trial==0,t)
                assert np.array_equal(token,sequence['inputs'][cursor])
                facts=[memory.get(i,4) for i in range(3)];assert facts==sequence['facts'][cursor].tolist()
                bad=bool(trial>0 and any(v!=4 and v!=wanted for v,wanted in zip(before[4:6],row['task']['goal'])))
                reference=pilot.explore_action(before,memory,order) if trial==0 else pilot.solve_action(before,row['task']['goal'],memory,order)
                assert sequence['actions'][cursor]==(-100 if bad else reference)
                ignored+=bad;labels+=not bad
                if row['arm'].startswith('student'):
                    predicted,hidden,_=pilot.predict(model,token,hidden);assert predicted==event['executed_action'];source_predictions+=1
                else:
                    executed=int(rng.choice(pilot.BEHAVIOR_ACTIONS)) if rng.rand()<.1 else reference
                    assert executed==event['executed_action']
                state,reward,done,_=env.step(event['executed_action'])
                assert state.observation.tolist()==event['next_observation'] and float(reward)==event['reward']
                observed_fact_update(memory,before,event['executed_action'],state.observation.tolist())
                previous_action,previous_reward=event['executed_action'],reward;cursor+=1;source_steps+=1
                assert not done or t==len(events)-1
            assert done and len(events)<=20
            final=pilot.terminal_feedback(state.observation,row['task']['goal'],previous_action,previous_reward,trial==0,len(events))
            assert np.array_equal(final,sequence['inputs'][cursor]) and sequence['actions'][cursor]==-100
            assert sequence['facts'][cursor].tolist()==[memory.get(i,4) for i in range(3)]
            if row['arm'].startswith('student'):_,hidden,_=pilot.predict(model,final,hidden)
            cursor+=1
        assert cursor==len(sequence['inputs']);grouped[key[:3]]['steps']+=len(row['events']);grouped[key[:3]]['sequences']+=1
    assert seen==expected and chain==summary['source_chain_head']
    source_count=len(seen)
    for r in summary['collection']:
        group=grouped[(r['arm'],r['seed'],r['round'])];assert group['steps']==r['collection_steps'] and group['sequences']==r['new_sequences']==177
    assert source_steps==summary['cost']['collection_actions']
    expected={(arm,seed,condition,t['task_id'],trial) for arm in experiment.ARMS for seed in plan['training_seeds'] for condition in ['full_history','reset_each_episode'] for t in pilot.tasks_for(plan['evaluation_environment_ids']) for trial in range(6)}
    seen=set();hidden_states={};grouped=defaultdict(list);chain='0'*64;eval_steps=0;predictions=0
    for line in (root/'evaluation.jsonl').open():
        row=json.loads(line);key=(row['arm'],row['seed'],row['condition'],row['task_id'],row['trial'])
        assert key in expected and key not in seen;seen.add(key)
        assert row['split']=='evaluation' and row['environment_id'] in plan['evaluation_environment_ids']
        assert row['plan_sha256']==plan['semantic_sha256']
        assert row['previous_hash']==chain;chain=pilot.digest({k:v for k,v in row.items() if k!='record_hash'});assert chain==row['record_hash']
        env=pilot.environment(Env,Wrapper,row,row['trial']==0);state=env.reset();model=models[key[:2]]
        hidden=None if row['condition']=='reset_each_episode' else hidden_states.get(key[:4]);previous_action,previous_reward=9,0.0
        for t,event in enumerate(row['events']):
            before=state.observation.tolist();assert before==event['observation'] and event['time']==t
            action,hidden,facts=pilot.predict(model,pilot.encode(before,row['goal'],previous_action,previous_reward,t==0,row['trial']==0,t),hidden)
            assert action==event['executed_action'] and facts==event['predicted_facts'];predictions+=1
            state,reward,done,_=env.step(action);assert state.observation.tolist()==event['next_observation'] and float(reward)==event['reward']
            previous_action,previous_reward=action,reward;eval_steps+=1;assert not done or t==len(row['events'])-1
        assert done and len(row['events'])==row['actual_steps']
        success=bool(np.array_equal(state.observation[4:6],row['goal'])) if row['trial'] else None;assert success==row['success']
        assert row['cost_steps']==(row['actual_steps'] if success or row['trial']==0 else 20)
        _,hidden,_=pilot.predict(model,pilot.terminal_feedback(state.observation,row['goal'],previous_action,previous_reward,row['trial']==0,len(row['events'])),hidden)
        hidden_states[key[:4]]=hidden;grouped['/'.join(map(str,key[:3]))].append(row)
    assert seen==expected and chain==summary['chain_head'] and eval_steps==summary['cost']['evaluation_actions']
    for key,rows in grouped.items():assert aggregate(rows)==summary['evaluations'][key]
    for arm,rows in summary['arms'].items():
        for row in rows:assert {k:v for k,v in row.items() if k!='seed'}==summary['evaluations'][f'{arm}/{row["seed"]}/full_history']
    assert len(seen)==summary['coverage']['planned_records']==summary['coverage']['completed_records']
    report={'status':'PASS','summary_sha256':pilot.sha(root/'summary.json'),'verifier_sha256':pilot.sha(__file__),
            'source_sequences':source_count,
            'source_actions':source_steps,'source_policy_decisions_recomputed':source_predictions,
            'valid_action_labels':labels,'corrupted_action_targets_ignored':ignored,
            'evaluation_records':len(seen),'evaluation_actions':eval_steps,'evaluation_policy_decisions_recomputed':predictions,
            'raw_observation_max_difference':0,'checks':['Source strictly training tasks','Known facts from actual preceding pickups only','Student collection actions from recorded pre-update checkpoints',
             'Offline epsilon actions from registered RNG and observable teacher','No action-positive target after irreversible bowl corruption',
             'All final checkpoints and starting checkpoint bindings','Same batch indices, parameter count and update count',
             'Every terminal feedback included','All transitions/rewards, action outputs, outcomes and aggregates independently reconstructed','Full source and evaluation chains'],
            'scope':'Independent reset/action and checkpoint replay; not optimizer reproducibility or new-method efficacy'}
    pilot.write(root/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['folder','parent','upstream']:p.add_argument('--'+name,type=Path,required=True)
    verify(p.parse_args())
