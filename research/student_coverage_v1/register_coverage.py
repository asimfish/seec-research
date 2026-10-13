#!/usr/bin/env python3
import datetime
import json
from pathlib import Path
import run_coverage as experiment

pilot=experiment.pilot
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'learner_pilot_v1'/'run2'


def main():
    assert not (ROOT/'plan.json').exists()
    old=json.loads((PARENT/'plan.json').read_text());summary=json.loads((PARENT/'summary.json').read_text())
    assert json.loads((PARENT/'verification.json').read_text())['status']=='PASS'
    names=['run_coverage.py','verify_coverage.py','register_coverage.py','test_coverage.py','PROTOCOL.md',
           '../learner_pilot_v1/run_learning_pilot.py','../run_cooking_diagnostic.py']
    plan={'schema':'cooking-coverage-plan-v1','registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'parent_summary_sha256':pilot.sha(PARENT/'summary.json'),'base_source_sha256':pilot.sha(PARENT/'source.jsonl'),
          'parent_audit_sha256':pilot.sha(ROOT/'parent_audit.json'),'upstream_revision':old['upstream_revision'],
          'code_hashes':{name:pilot.sha(ROOT/name) for name in names},
          'initial_checkpoint_sha256':{str(m['seed']):m['checkpoint_sha256'] for m in summary['models'] if m['arm']=='action_only'},
          'arms':list(experiment.ARMS),'training_seeds':[7,19,41],'rounds':2,'updates_per_round':300,
          'new_sequences_per_round':177,'base_sequences':1062,'batch_sequences':24,'hidden_size':96,
          'learning_rate':.003,'auxiliary_weight':.2,'gradient_clip':1.,'cpu_threads':2,
          'train_environment_ids':old['train_environment_ids'],'evaluation_environment_ids':old['evaluation_environment_ids'],
          'collection':'177 histories per round, per arm, per seed; offline epsilon .1 vs beta-zero student; same base pool and initial action-only checkpoint',
          'input':'Parent causal encoder and terminal-feedback semantics unchanged; no hidden mapping, label, expert or structured memory at inference',
          'label_guard':'Ignore action target when visible bowl already irreversibly contradicts goal; preserve all actual inputs/fact targets. Common guard applied to all old/new data and all arms.',
          'evaluation':'Fixed final 600 additional updates, all checkpoints before evaluation; full-history plus same-checkpoint reset-memory ablation',
          'predictions':{'coverage_gain':'student_action minus offline_action >=.05, positive in all seeds',
                         'facts_survive_coverage':'student_facts minus student_action >=.05, positive in all seeds',
                         'ordinary_dagger_explanation':'If facts_survive_coverage fails, no claim that factual objective is the essential missing mechanism'},
          'limits':['Prior diagnostic panel and seeds reused','Sequence and optimizer budget matched, not token or interaction budget','New common label-validity guard disclosed','DAgger/GMPS and task-belief supervision are existing mechanisms','No hyperparameter tuning or robot/RSI/new-method claim'],
          'stop':'Registered 12 runs, 7200 additional updates; never tune this panel'}
    plan['semantic_sha256']=pilot.digest(plan);plan['plan_id']='cooking-coverage-'+plan['semantic_sha256'][:16]
    pilot.write(ROOT/'plan.json',plan);print(json.dumps({'plan_id':plan['plan_id'],'semantic_sha256':plan['semantic_sha256']},indent=2))


if __name__=='__main__':main()
