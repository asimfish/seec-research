#!/usr/bin/env python3
"""Trace-level parent diagnosis; no fitting, environment actions or causal claim."""
import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'learner_pilot_v1'))
import run_learning_pilot as pilot


def compatible_actions(observation, goal, memory, exploration=False):
    fn = pilot.explore_action if exploration else pilot.solve_action
    return {fn(observation, memory, list(order)) if exploration else
            fn(observation, goal, memory, list(order))
            for order in itertools.permutations(range(3))}


def bowl_corrupted(observation, goal):
    return any(int(value) != 4 and int(value) != int(wanted)
               for value, wanted in zip(observation[4:6], goal))


def analyze(parent, output):
    assert not output.exists(), 'Do not overwrite an audit'
    verified = json.loads((parent/'verification.json').read_text())
    summary = json.loads((parent/'summary.json').read_text())
    assert verified['status']=='PASS' and verified['summary_sha256']==pilot.sha(parent/'summary.json')
    grouped, memories, task_counts = defaultdict(Counter), {}, defaultdict(Counter)
    chain, total = '0'*64, 0
    for line in (parent/'evaluation.jsonl').open():
        row=json.loads(line)
        assert row['previous_hash']==chain
        chain=pilot.digest({k:v for k,v in row.items() if k!='record_hash'})
        assert chain==row['record_hash']
        total+=1
        if row['split']!='evaluation' or row['condition']!='full_history' or row['arm'] not in pilot.ARMS:
            continue
        key=(row['arm'],row['seed']); counts=grouped[key]
        memory=memories.setdefault((*key,row['task_id']),{})
        if row['trial']>0:
            counts['episodes']+=1;counts['successes']+=row['success']
            if all(food in memory.values() for food in row['goal']):
                counts['goal_known_episodes']+=1;counts['goal_known_successes']+=row['success']
        corrupted=False
        for event in row['events']:
            before,after=event['observation'],event['next_observation']
            action=event['executed_action']
            if row['trial']>0:
                counts['execution_decisions']+=1
                agrees=action in compatible_actions(before,row['goal'],memory)
                counts['reference_compatible_decisions']+=agrees
                if row['arm']=='observed_facts_aux':
                    correct=event['predicted_facts']==pilot.fact_targets(memory).tolist()
                    counts['all_facts_correct_decisions']+=correct
                    counts['correct_facts_reference_compatible']+=bool(correct and agrees)
                    counts['incorrect_facts_reference_compatible']+=bool(not correct and agrees)
                if action==6:counts['explicit_end_decisions']+=1
                if bowl_corrupted(after,row['goal']):corrupted=True
            pilot.absorb(memory,before,action,after)
        if row['trial']>0 and not row['success']:
            counts['failed_episodes']+=1
            counts['failed_with_corrupted_bowl']+=corrupted
            counts['failed_with_explicit_early_end']+=bool(row['events'][-1]['executed_action']==6 and row['actual_steps']<20)
    assert chain==summary['chain_head'] and total==summary['coverage']['completed_records']
    report={}
    for arm in pilot.ARMS:
        seeds=[]; pooled=Counter()
        for seed in [7,19,41]:
            c=grouped[(arm,seed)];pooled.update(c);seeds.append({'seed':seed,**dict(c)})
        r={'pooled_counts':dict(pooled),'seeds':seeds,
           'goal_known_success_rate':pooled['goal_known_successes']/pooled['goal_known_episodes'],
           'reference_compatible_rate':pooled['reference_compatible_decisions']/pooled['execution_decisions'],
           'failed_corrupted_bowl_rate':pooled['failed_with_corrupted_bowl']/pooled['failed_episodes']}
        if arm=='observed_facts_aux':
            n=pooled['all_facts_correct_decisions'];other=pooled['execution_decisions']-n
            r.update({'all_facts_correct_rate':n/pooled['execution_decisions'],
                      'reference_compatible_given_facts_correct':pooled['correct_facts_reference_compatible']/n,
                      'reference_compatible_given_facts_incorrect':pooled['incorrect_facts_reference_compatible']/other})
        report[arm]=r
    result={'status':'COMPLETED_TRACE_DIAGNOSIS','parent_summary_sha256':pilot.sha(parent/'summary.json'),
            'audit_sha256':pilot.sha(__file__),'records_checked':total,'arms':report,
            'definition':'Reference-compatible means at least one of the six registered symbolic tie orders agrees at the exact observed state with only actual prior pickup facts. Disagreement is not by itself proof of a bad action.',
            'limits':['Post-primary trace audit; no hypothesis significance test',
                      'Fact correctness is measured at actual visited states; conditions are not randomized',
                      'A corrupted bowl is an irreversible observed error under the original task, not a root-cause inference'],
            'additional_environment_steps':0,'optimizer_updates':0}
    pilot.write(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.parent,a.output)
