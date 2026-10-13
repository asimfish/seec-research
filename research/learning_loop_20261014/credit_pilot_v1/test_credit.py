import copy
from pathlib import Path

import analyze_credit as analysis
import numpy as np
import pytest
import run_credit as run
import torch


@pytest.fixture
def env():
    return run.pilot.load_environment(Path("/tmp/robot-task-dream-20261013"))


def waiting_model():
    model = run.pilot.HistoryPolicy(96).eval()
    with torch.no_grad():
        model.actions.weight.zero_()
        model.actions.bias.fill_(-100)
        model.actions.bias[4] = 100
    return model


def task():
    return {"environment_id": 21, "goal": [1, 1], "task_id": "source-fixture"}


def test_true_cold_start_no_probe_or_teacher_calls(env, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Teacher must not participate in evaluation")

    monkeypatch.setattr(run.coverage, "target", forbidden)
    monkeypatch.setattr(run.pilot, "absorb", forbidden)
    record = run.history(*env, task(), waiting_model(), annotate=False)
    assert len(record["outcomes"]) == 6
    assert all(not o["exploration"] for o in record["outcomes"])
    assert record["events"][0]["trial"] == record["events"][0]["time"] == 0
    assert {e["executed_action"] for e in record["events"]} == {4}
    assert all(e["teacher_action"] is None for e in record["events"])


def test_visible_targets_ignore_rule_identity_and_future_events(env):
    record = run.history(*env, task(), behavior_seed=11, probe_steps=8)
    before = run.reconstruct(record)
    metadata = copy.deepcopy(record)
    metadata["task"]["environment_id"] = 63
    metadata["task"]["task_id"] = "different-hidden-name"
    unchanged = run.reconstruct(metadata)
    assert np.array_equal(before["inputs"], unchanged["inputs"])
    assert np.array_equal(before["actions"], unchanged["actions"])
    future = copy.deepcopy(record)
    future["events"][-1]["next_observation"][3] = 3
    after = run.reconstruct(future)
    assert np.array_equal(before["inputs"][:-1], after["inputs"][:-1])
    assert np.array_equal(before["actions"], after["actions"])


def test_missing_episode_feedback_rejected_and_terminal_not_a_target(env):
    record = run.history(*env, task(), probe_steps=8)
    sequence = run.reconstruct(record)
    assert len(sequence["inputs"]) == len(record["events"]) + len(record["outcomes"])
    assert np.count_nonzero(sequence["actions"] == -100) >= len(record["outcomes"])
    record["events"].pop()
    with pytest.raises(ValueError, match="Missing or mismatched"):
        run.reconstruct(record)


def test_task_histories_reset_between_calls(env):
    torch.manual_seed(99)
    model = run.pilot.HistoryPolicy(96).eval()
    a = run.history(*env, task(), model, annotate=False)
    run.history(
        *env,
        {"environment_id": 63, "goal": [3, 3], "task_id": "source-other"},
        model,
        annotate=False,
    )
    b = run.history(*env, task(), model, annotate=False)
    assert a["events"] == b["events"] and a["outcomes"] == b["outcomes"]


def test_actual_terminal_response_consumed_before_episode_reset(env, monkeypatch):
    model = waiting_model()
    original = model.forward
    seen = []

    def recording(inputs, hidden=None):
        seen.append(inputs.detach().clone())
        return original(inputs, hidden)

    monkeypatch.setattr(model, "forward", recording)
    record = run.history(*env, task(), model, annotate=False)
    assert len(seen) == len(record["events"]) + len(record["outcomes"]) == 126
    assert seen[20][0, 0, -1].item() == 1.0
    assert seen[21][0, 0, -1].item() == 0.0
    assert seen[20][0, 0, -4].item() == 0.0
    assert seen[21][0, 0, -4].item() == 1.0


def test_branch_training_restarts_same_optimizer_and_sampling(env):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(9)
    data = [run.reconstruct(run.history(*env, task(), probe_steps=0))]
    plan = {"batch_sequences": 2, "gradient_clip": 1.0}
    base = run.pilot.HistoryPolicy(96)
    opt = torch.optim.Adam(base.parameters(), lr=0.003)
    run.train(base, opt, data, None, 2, plan, 9)
    weights, optimizer = copy.deepcopy(base.state_dict()), copy.deepcopy(opt.state_dict())
    digests = []
    for _ in range(2):
        branch = run.pilot.HistoryPolicy(96)
        branch.load_state_dict(weights)
        new_opt = torch.optim.Adam(branch.parameters(), lr=0.003)
        new_opt.load_state_dict(copy.deepcopy(optimizer))
        digests.append(run.train(branch, new_opt, data, data, 3, plan, 9))
    assert digests[0] == digests[1]


def test_metrics_macro_rules_and_real_cold_attempt():
    def rec(rule, curve):
        return {
            "task": {"environment_id": rule},
            "outcomes": [{"exploration": False, "success": x} for x in curve],
            "events": [],
        }

    result = run.metrics(
        [rec(1, [0, 1, 1, 1, 1, 1]), rec(1, [0, 1, 1, 1, 1, 1]), rec(2, [1, 1, 1, 1, 1, 1])]
    )
    assert result["cold_success"] == 0.5
    assert result["q_total"] == 0.875
    assert result["q_adapt"] == 0.5


def test_scores_and_selection_do_not_consult_long_audit():
    baseline = {"cold_success": 0.25, "q_total": 0.5, "q_adapt": 0.25, "final_success": 0.5}
    row = {
        "candidate": "batch-00",
        "source_before": {"q_total": 0.2},
        "baseline": {"feedback": baseline},
        "short": {"source": {"q_total": 0.4}, "feedback": {**baseline, "q_adapt": 0.4}},
        "long": {"audit": {"q_adapt": 0}},
    }
    first = analysis.scores(row)
    row["long"]["audit"]["q_adapt"] = 999
    assert first == analysis.scores(row)
    assert analysis.select([row], "adaptation_gain", 0.05) == "batch-00"
    row["short"]["feedback"]["cold_success"] = 0
    assert analysis.select([row], "adaptation_gain", 0.05) is None


def test_constant_scores_undefined_and_ties_use_average_ranks():
    assert analysis.spearman([1, 1, 1], [1, 2, 3]) is None
    assert np.array_equal(analysis.ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
    assert analysis.spearman([1, 2, 3], [3, 2, 1]) == -1


def test_semantic_plan_identity_excludes_time_but_binds_budget():
    a = {"registered_at_utc": "first", "short_updates": 64}
    b = {"registered_at_utc": "second", "short_updates": 64}
    assert run.pilot.digest(run.semantic_binding(a)) == run.pilot.digest(run.semantic_binding(b))
    b["short_updates"] = 65
    assert run.pilot.digest(run.semantic_binding(a)) != run.pilot.digest(run.semantic_binding(b))
