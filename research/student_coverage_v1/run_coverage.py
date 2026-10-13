#!/usr/bin/env python3
"""Matched 2x2 history-collection / auxiliary-learning continuation experiment.

This implements a DAgger-style baseline, not a new method. The expert only
uses current visible observation and actual past pickups. No teacher prefix
or hidden fact vector enters learned inference.
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'learner_pilot_v1'))
import run_learning_pilot as pilot

ARMS=('offline_action','offline_facts','student_action','student_facts')


def corrupt(obs,goal):
    return any(v!=4 and v!=wanted for v,wanted in zip(obs[4:6],goal))


def target(obs,goal,memory,order,exploration):
    # A wrong bowl cannot be repaired. Do not pretend a corrective action
    # here completes the task; action learning ignores this absorbing failure.
    if not exploration and corrupt(obs,goal):
        return -100
    return pilot.explore_action(obs,memory,order) if exploration else pilot.solve_action(obs,goal,memory,order)


def from_events(record):
    """Reconstruct all targets from actual prefixes, not recorded fact copies."""
    task=record['task'];memory={};inputs=[];actions=[];facts=[];states=[]
    order=list(np.random.RandomState(record.get('behavior_seed',11)).permutation(3))
    for trial in range(6):
        events=[e for e in record['events'] if e['trial']==trial]
        previous_action,previous_reward=9,0.0
        assert events
        for t,e in enumerate(events):
            assert e['time']==t
            obs=e['observation']
            inputs.append(pilot.encode(obs,task['goal'],previous_action,previous_reward,t==0,trial==0,t))
            actions.append(target(obs,task['goal'],memory,order,trial==0))
            facts.append(pilot.fact_targets(memory));states.append(pilot.state_targets(obs))
            pilot.absorb(memory,obs,e['executed_action'],e['next_observation'])
            previous_action,previous_reward=e['executed_action'],e['reward']
        inputs.append(pilot.terminal_feedback(e['next_observation'],task['goal'],previous_action,previous_reward,trial==0,len(events)))
        actions.append(-100);facts.append(pilot.fact_targets(memory));states.append(pilot.state_targets(e['next_observation']))
    return {'inputs':np.asarray(inputs,dtype=np.float32),'actions':np.asarray(actions,dtype=np.int64),
            'facts':np.asarray(facts,dtype=np.int64),'states':np.asarray(states,dtype=np.int64),
            'events':record['events'],'task':task}


def collect(Env,Wrapper,model,task,arm,seed,round_index):
    behavior_seed=101+round_index*100+seed
    if arm.startswith('offline'):
        original=pilot.collect_sequence(Env,Wrapper,task,behavior_seed,.1,5)
        record={k:v for k,v in original.items() if k not in {'inputs','actions','facts','states'}}
        record['behavior']='observable_teacher_epsilon_0.1'
    else:
        _,events=pilot.rollout(Env,Wrapper,model,task,5)
        record={'task':task,'behavior_seed':behavior_seed,'epsilon':0.0,'events':events,'behavior':'student_beta_0'}
    record.update({'arm':arm,'seed':seed,'round':round_index,'split':'train'})
    return from_events(record),record


def pack(items):
    width=max(len(item['inputs']) for item in items);n=len(items)
    x=np.zeros((n,width,pilot.INPUT_DIM),dtype=np.float32)
    y=np.full((n,width),-100,dtype=np.int64);f=np.full((n,width,3),-100,dtype=np.int64)
    for i,item in enumerate(items):
        k=len(item['inputs']);x[i,:k]=item['inputs'];y[i,:k]=item['actions'];f[i,:k]=item['facts']
    return tuple(torch.from_numpy(v) for v in (x,y,f))


def update(model,optimizer,data,plan,arm,seed,round_index):
    rng=np.random.RandomState(seed+100000+round_index*10000)
    sample_hash=hashlib.sha256();records=[];model.train()
    x,y,f=data
    for step in range(1,plan['updates_per_round']+1):
        indices=rng.randint(0,len(x),size=plan['batch_sequences'])
        sample_hash.update(indices.astype(np.int64).tobytes())
        predictions,auxiliary,_=model(x[indices])
        action_loss=F.cross_entropy(predictions.reshape(-1,9),y[indices].reshape(-1))
        aux_loss=F.cross_entropy(auxiliary.reshape(-1,5),f[indices].reshape(-1)) if arm.endswith('facts') else action_loss.new_zeros(())
        optimizer.zero_grad(set_to_none=True)
        (action_loss+plan['auxiliary_weight']*aux_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip']);optimizer.step()
        if step==1 or step%100==0:
            row={'update':step,'action_loss':action_loss.item(),'auxiliary_loss':aux_loss.item()};records.append(row)
            print(json.dumps({'arm':arm,'seed':seed,'round':round_index,**row}),flush=True)
    model.eval()
    return {'sampling_sha256':sample_hash.hexdigest(),'training':records,
            'updates':plan['updates_per_round'],'dataset_sequences':len(x),
            'action_targets':int((y!=-100).sum()),'fact_targets':int((f!=-100).sum())}


def run(args):
    plan=json.loads(args.plan.read_text());assert pilot.digest({k:v for k,v in plan.items() if k not in {'semantic_sha256','plan_id'}})==plan['semantic_sha256']
    for name,digest in plan['code_hashes'].items():assert pilot.sha(args.plan.parent/name)==digest,name
    assert pilot.sha(args.parent/'summary.json')==plan['parent_summary_sha256']
    assert pilot.sha(args.parent/'source.jsonl')==plan['base_source_sha256']
    assert not args.output.exists();args.output.mkdir(parents=True);(args.output/'plan.json').write_bytes(args.plan.read_bytes())
    torch.set_num_threads(plan['cpu_threads']);torch.use_deterministic_algorithms(True)
    Env,Wrapper=pilot.load_environment(args.upstream)
    base=[from_events(json.loads(line)) for line in (args.parent/'source.jsonl').open()]
    assert len(base)==1062
    models={};metadata=[];collection=[];chain='0'*64
    source_log=(args.output/'collected.jsonl').open('w')
    for seed in plan['training_seeds']:
        initial=args.parent/'checkpoints'/f'action_only-{seed}'/'policy.pt'
        assert pilot.sha(initial)==plan['initial_checkpoint_sha256'][str(seed)]
        for arm in ARMS:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=pilot.HistoryPolicy(plan['hidden_size']);model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True));model.eval()
            first=pilot.weights_digest(model);optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate']);pool=list(base)
            rounds=[];started=time.time()
            for round_index in range(1,plan['rounds']+1):
                behavior_weights=pilot.weights_digest(model);before=len(pool);physical_steps=0;corrupted_steps=0
                collector_dir=args.output/'collectors'/f'{arm}-{seed}';collector_dir.mkdir(parents=True,exist_ok=True)
                torch.save(model.state_dict(),collector_dir/f'round-{round_index}.pt')
                collector_sha=pilot.sha(collector_dir/f'round-{round_index}.pt')
                for task in pilot.tasks_for(plan['train_environment_ids']):
                    item,record=collect(Env,Wrapper,model,task,arm,seed,round_index)
                    record['collector_weights_sha256']=behavior_weights;record['collector_checkpoint_sha256']=collector_sha;record['plan_sha256']=plan['semantic_sha256']
                    record['previous_hash']=chain;chain=pilot.digest(record);record['record_hash']=chain
                    source_log.write(json.dumps(record,sort_keys=True)+'\n');pool.append(item)
                    physical_steps+=len(item['events'])
                    corrupted_steps+=sum(e['trial']>0 and corrupt(e['observation'],task['goal']) for e in item['events'])
                assert len(pool)-before==177
                rows=update(model,optimizer,pack(pool),plan,arm,seed,round_index)
                rows.update({'round':round_index,'collector_weights_sha256':behavior_weights,'collector_checkpoint_sha256':collector_sha,
                             'new_sequences':177,'collection_steps':physical_steps,'corrupted_action_targets_ignored':corrupted_steps})
                rounds.append(rows);collection.append({'arm':arm,'seed':seed,**{k:v for k,v in rows.items() if k!='training'}})
            folder=args.output/'checkpoints'/f'{arm}-{seed}';folder.mkdir(parents=True)
            torch.save(model.state_dict(),folder/'policy.pt')
            record={'arm':arm,'seed':seed,'initial_weights_sha256':first,'initial_checkpoint_sha256':pilot.sha(initial),
                    'final_weights_sha256':pilot.weights_digest(model),'checkpoint_sha256':pilot.sha(folder/'policy.pt'),
                    'plan_sha256':plan['semantic_sha256'],'parameter_count':sum(p.numel() for p in model.parameters()),
                    'updates':plan['updates_per_round']*plan['rounds'],'rounds':rounds,'runtime_seconds':time.time()-started}
            pilot.write(folder/'metadata.json',record);metadata.append(record);models[(arm,seed)]=model
        same=[r for r in metadata if r['seed']==seed]
        assert len({r['initial_weights_sha256'] for r in same})==1
        for k in range(plan['rounds']):assert len({r['rounds'][k]['sampling_sha256'] for r in same})==1
    source_log.close();pilot.write(args.output/'training_manifest.json',metadata)
    summary={'schema':'cooking-student-coverage-2x2-v1','plan_id':plan['plan_id'],'plan_sha256':plan['semantic_sha256'],
             'parent_summary_sha256':plan['parent_summary_sha256'],'models':metadata,'collection':collection,
             'source_chain_head':chain,'arms':{},'evaluations':{},'cost':{'additional_optimizer_updates':sum(r['updates'] for r in metadata),
             'collection_actions':sum(r['collection_steps'] for r in collection),'collection_resets':177*6*len(collection),
             'expert_action_queries':sum(r['collection_steps'] for r in collection)},
             'scope':'Two-round DAgger-style collection x observed-fact auxiliary; method-comparator diagnostic, not novel OPD algorithm or robot efficacy.'}
    chain='0'*64;count=0;actions=0
    with (args.output/'evaluation.jsonl').open('w') as handle:
        for seed in plan['training_seeds']:
            for arm in ARMS:
                for condition in ['full_history','reset_each_episode']:
                    records=[]
                    for task in pilot.tasks_for(plan['evaluation_environment_ids']):
                        outcomes,events=pilot.rollout(Env,Wrapper,models[(arm,seed)],task,5,condition=='reset_each_episode')
                        for outcome in outcomes:
                            row={'arm':arm,'seed':seed,'condition':condition,'split':'evaluation',**task,**outcome,
                                 'events':[e for e in events if e['trial']==outcome['trial']],
                                 'plan_sha256':plan['semantic_sha256'],'previous_hash':chain}
                            chain=pilot.digest(row);row['record_hash']=chain;handle.write(json.dumps(row,sort_keys=True)+'\n')
                            records.append(row);count+=1;actions+=row['actual_steps']
                    summary['evaluations'][f'{arm}/{seed}/{condition}']=pilot.summarize(records)
                    if condition=='full_history':summary['arms'].setdefault(arm,[]).append({'seed':seed,**pilot.summarize(records)})
                    print(json.dumps({'evaluation':f'{arm}/{seed}/{condition}',**pilot.summarize(records)}),flush=True)
    assert count==len(pilot.tasks_for(plan['evaluation_environment_ids']))*6*12*2
    summary['coverage']={'planned_records':count,'completed_records':count,'missing':0,'aborted':0};summary['chain_head']=chain
    summary['cost'].update({'evaluation_actions':actions,'evaluation_resets':count,'base_source_actions_reused':79904,
                            'base_training_updates_per_initial_model':800})
    summary['artifacts']={name:pilot.sha(args.output/name) for name in ['plan.json','collected.jsonl','training_manifest.json','evaluation.jsonl']}
    pilot.write(args.output/'summary.json',summary);print(json.dumps({'status':'COMPLETED','arms':summary['arms']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','upstream','plan','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
