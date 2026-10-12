import importlib.util
from pathlib import Path

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("learned_pilot", Path(__file__).with_name("run_learning_pilot.py"))
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def token():
    return pilot.encode([0, 0, 0, 4, 4, 4, 4], [1, 2], 9, 0.0, True, False, 0)


def test_student_boundary_has_no_identity_or_fact_input():
    import inspect
    args = inspect.signature(pilot.encode).parameters
    assert set(args) == {"observation", "goal", "previous_action", "previous_reward",
                         "episode_start", "exploration", "local_step"}
    assert token().shape == (pilot.INPUT_DIM,)
    assert token().dtype == np.float32


def test_unknown_drawers_not_supervised_from_hidden_truth():
    memory = {}
    assert pilot.fact_targets(memory).tolist() == [4, 4, 4]
    before = np.array([2, 0, 3, 4, 4, 4, 4])
    after = before.copy()
    pilot.absorb(memory, before, 7, after)
    assert pilot.fact_targets(memory).tolist() == [4, 4, 4]
    after[3] = 2
    pilot.absorb(memory, before, 7, after)
    assert pilot.fact_targets(memory).tolist() == [2, 4, 4]


def test_terminal_feedback_includes_last_observation_action_reward():
    final = [2, 0, 3, 2, 4, 4, 4]
    feedback = pilot.terminal_feedback(final, [1, 2], 7, 0.25, False, 20)
    ordinary = pilot.encode(final, [1, 2], 7, 0.25, False, False, 20)
    assert np.array_equal(feedback, ordinary)
    # Inventory 2 and previous pickup differ from a reset token. This is the
    # last-step fact that the interrupted run did not expose to its student.
    assert not np.array_equal(feedback, token())


def test_collection_every_terminal_response_has_feedback_only_target():
    Env, Wrapper = pilot.load_environment(Path('/tmp/robot-task-dream-20261013'))
    item = pilot.collect_sequence(Env, Wrapper,
                                  {'environment_id': 21, 'goal': [1, 1], 'task_id': 'train-test'},
                                  11, 0.0, 5)
    assert len(item['inputs']) == len(item['events']) + 6
    assert np.count_nonzero(item['actions'] == -100) == 6
    cursor = 0
    for trial in range(6):
        events = [e for e in item['events'] if e['trial'] == trial]
        last = events[-1]
        index = cursor + len(events)
        expected = pilot.terminal_feedback(last['next_observation'], [1, 1],
                                          last['executed_action'], last['reward'], trial == 0, len(events))
        assert np.array_equal(item['inputs'][index], expected)
        assert item['actions'][index] == -100
        cursor = index + 1


def test_causal_prefix_not_affected_by_future_observations():
    torch.manual_seed(7)
    model = pilot.HistoryPolicy(12).eval()
    x = torch.randn(1, 8, pilot.INPUT_DIM)
    altered = x.clone()
    altered[:, 5:] = torch.randn_like(altered[:, 5:])
    a = model(x)[0]
    b = model(altered)[0]
    assert torch.equal(a[:, :5], b[:, :5])


def test_training_sequence_and_online_stream_agree():
    torch.manual_seed(7)
    model = pilot.HistoryPolicy(12).eval()
    x = torch.randn(1, 6, pilot.INPUT_DIM)
    batched = model(x)[0]
    hidden, pieces = None, []
    for i in range(6):
        logits, _, hidden = model(x[:, i:i+1], hidden)
        pieces.append(logits)
    assert torch.allclose(batched, torch.cat(pieces, dim=1), atol=1e-7, rtol=1e-6)
    assert torch.equal(batched.argmax(-1), torch.cat(pieces, dim=1).argmax(-1))


def test_fresh_task_prediction_has_no_hidden_persistent_state():
    torch.manual_seed(7)
    model = pilot.HistoryPolicy(12).eval()
    first, _, _ = pilot.predict(model, token(), None)
    pilot.predict(model, np.ones(pilot.INPUT_DIM, dtype=np.float32), None)
    third, _, _ = pilot.predict(model, token(), None)
    assert first == third


def test_paired_initial_weights_and_sampling_identical():
    hashes = []
    samplings = []
    for _ in pilot.ARMS:
        torch.manual_seed(19)
        hashes.append(pilot.weights_digest(pilot.HistoryPolicy(12)))
        samplings.append(np.random.RandomState(100019).randint(0, 20, size=(8, 4)).tobytes())
    assert len(set(hashes)) == len(set(samplings)) == 1


def test_same_architecture_for_all_auxiliary_labels():
    model = pilot.HistoryPolicy(12)
    actions, auxiliary, _ = model(torch.from_numpy(token()).reshape(1, 1, -1))
    assert actions.shape == (1, 1, 9)
    assert auxiliary.shape == (1, 1, 3, 5)
    assert pilot.state_targets([0, 0, 0, 2, 1, 4, 4]).tolist() == [2, 1, 4]
