import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('matched_history', ROOT / 'run_matched_history_check.py')
matched = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matched)


def test_redaction_removes_mapping_evidence_and_preserves_raw_prefix():
    Env, Wrapper = matched.pilot.load_environment(Path('/tmp/robot-task-dream-20261013'))
    prefixes = []
    for env_id in [0, 21]:
        task = {'environment_id': env_id, 'goal': [0, 0], 'task_id': f'prefix-only-{env_id}'}
        sequence = matched.pilot.collect_sequence(Env, Wrapper, task, 11, 0.0, 0)
        prefixes.append(sequence['events'])
    original = repr(prefixes)
    real = [matched.prefix_tokens(p, [0, 0], False) for p in prefixes]
    redacted = [matched.prefix_tokens(p, [0, 0], True) for p in prefixes]
    assert not np.array_equal(*real)
    assert np.array_equal(*redacted)
    assert repr(prefixes) == original
    assert redacted[0].shape == (21, 55)


def test_evaluator_metadata_never_enters_shared_student_prefix():
    Env, Wrapper = matched.pilot.load_environment(Path('/tmp/robot-task-dream-20261013'))
    task = {'environment_id': 21, 'goal': [1, 1], 'task_id': 'prefix-only'}
    events = matched.pilot.collect_sequence(Env, Wrapper, task, 11, 0.0, 0)['events']
    before = matched.prefix_tokens(events, [1, 1], False)
    for event in events:
        event['facts_before'] = [99, 99, 99]
        event['environment_id'] = 63
        event['teacher_privileged'] = True
    assert np.array_equal(before, matched.prefix_tokens(events, [1, 1], False))
