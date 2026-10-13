import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_coverage as run


def fixture():
    Env,Wrapper=run.pilot.load_environment(Path('/tmp/robot-task-dream-20261013'))
    task={'environment_id':27,'goal':[3,3],'task_id':'unit-training-only'}
    raw=run.pilot.collect_sequence(Env,Wrapper,task,11,.1,5)
    record={'task':task,'behavior_seed':11,'events':raw['events']}
    return Env,Wrapper,task,raw,record


def test_reconstruction_preserves_all_observation_tokens_and_facts():
    _,_,_,old,record=fixture();new=run.from_events(record)
    assert np.array_equal(new['inputs'],old['inputs'])
    assert np.array_equal(new['facts'],old['facts'])
    assert len(new['actions'])==len(old['actions'])


def test_future_feedback_and_fact_copies_do_not_change_current_labels():
    _,_,_,_,record=fixture();before=run.from_events(record)
    record['task']['task_id']='another-name'
    for e in record['events']:e['facts_before']=[99,99,99]
    same=run.from_events(record)
    assert np.array_equal(before['inputs'],same['inputs'])
    assert np.array_equal(before['facts'],same['facts'])
    record['events'][-1]['next_observation'][3]=2
    after=run.from_events(record)
    assert np.array_equal(before['inputs'][:-1],after['inputs'][:-1])
    assert np.array_equal(before['facts'][:-1],after['facts'][:-1])


def test_irreversible_failure_abstains_without_dropping_history():
    obs=[1,1,3,2,2,4,4]
    assert run.corrupt(obs,[3,3])
    assert run.target(obs,[3,3],{0:3},[0,1,2],False)==-100
    assert run.target(obs,[3,3],{0:3},[0,1,2],True)!=-100
    assert not run.corrupt([1,1,3,4,3,4,4],[3,3])


def test_student_history_has_no_expert_prefix_or_teacher_execution():
    Env,Wrapper,task,_,_=fixture();model=run.pilot.HistoryPolicy(96).eval()
    with torch.no_grad():
        model.actions.weight.zero_();model.actions.bias.fill_(-100);model.actions.bias[4]=100
    sequence,record=run.collect(Env,Wrapper,model,task,'student_action',7,1)
    assert len(record['events'])==120
    assert {e['executed_action'] for e in record['events']}=={4}
    assert len(sequence['inputs'])==126
    assert (sequence['facts']==4).all()
    assert sequence['actions'][0]!=4
    assert record['behavior']=='student_beta_0'


def test_offline_controls_have_identical_histories_for_a_paired_seed():
    Env,Wrapper,task,_,_=fixture();model=run.pilot.HistoryPolicy(96)
    a,ra=run.collect(Env,Wrapper,model,task,'offline_action',19,2)
    b,rb=run.collect(Env,Wrapper,model,task,'offline_facts',19,2)
    for key in ['inputs','actions','facts']:assert np.array_equal(a[key],b[key])
    assert ra['events']==rb['events']
