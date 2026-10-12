# Learned task-history pilot and shared-feedback probe

Read `PROTOCOL.md`, `plan.json`, and `RESULTS.md`. The valid result is `run2`.
`run1` is interrupted and INVALID; its code snapshot, plan and partial source /
training are preserved. The correction only ingests the actual terminal
observation/reward before reset. It changes no task or training parameters.

This is a three-seed, three-arm recurrent BC diagnostic, not a reproduction of
DREAM, AD, RL² or VariBAD, and not OPD. Full-history action-only, current-state
auxiliary and observed-fact auxiliary share source histories, initialization,
minibatch indices, architecture and 800-update budgets. Unknown facts remain
unknown until an actual observed pickup. No fact labels, hidden mapping, expert
or symbolic controller enter neural inference.

The official DREAM checkout is pinned to
`d52204c94067641df6e6649b19abb359b87ff028`. No upstream source is redistributed;
upstream has no explicit repository license. The observation/reward/transition
and 20-step horizon are original; the 32/8/24 split and repeated five-attempt
protocol are ours. Test assignments reuse the prior diagnostic and are not an
independent confirmatory panel. Three optimizer seeds support descriptive
pilot comparisons, not a conference-level performance claim.

The runtime is Python 3.10.12, NumPy 2.2.6, PyTorch 2.13.0+cu130 (CPU), Gym
0.23.1 and gym_notices 0.1.0. Use a compatible isolated environment. CUDA was
unavailable; no shared driver or GPU configuration was changed. The only
upstream compatibility change is the removed `np.int` dtype alias in our loader.

From the local research folder, preserving its sibling diagnostic adapter:

```bash
python run_learning_pilot.py --upstream /path/to/pinned/dream \
  --plan plan.json --output /path/to/fresh-run
python verify_learning_pilot.py --upstream /path/to/pinned/dream \
  --folder /path/to/fresh-run
python analyze_learning_pilot.py --folder /path/to/fresh-run
```

The public copy retains the same sibling import: `learning-pilot/` is next to
`run_cooking_diagnostic.py`. Use the copied immutable plan for reproduction.
The registration script records original research provenance; rerunning it
would generate a new internal registration rather than reproduce its timestamp.
The adapter tests currently expect the pinned checkout at
`/tmp/robot-task-dream-20261013`.

From the public repository root, the full tested boundary suite is:

```bash
python -m pytest -q research/test_cooking_diagnostic.py \
  research/learning-pilot/test_learning_pilot.py \
  research/learning-pilot/test_matched_history_check.py
```

It has 15 tests. Install the stated runtime dependencies and place the pinned
official checkout at the fixture path before invoking them. No upstream code
is included in the public repository.

All nine final checkpoints were completed before development/test evaluation;
development did not select a checkpoint. A separate registered post-pilot probe
(`matched_plan.json`) holds the 20-step prefix constant across checkpoints and
redacts only its inventory feedback. This supplied-prefix experiment is a
mechanism diagnostic, not the primary teacher-free result. Its raw log preserves
real observations, with redaction applied only at learner input. No gradients or
checkpoint changes occur. Run and verify it with `run_matched_history_check.py`
and `verify_matched_history_check.py` after the primary result is verified.

Budget: valid source 79,904 physical actions, 86,276 input tokens including 6,372
feedback-only tokens; nine runs × 800 = 7,200 optimizer updates. The primary and
memory-ablation evaluation spends 322,100 actual actions. The interrupted run
also spent 79,904 source actions and 2,700–2,799 updates; this overhead is recorded
separately in `invalid_run_cost.json`, never hidden or counted as valid training.
Matched-prefix diagnostic cost is separately reported; no budget advantage is
claimed. Raw logs/checkpoints remain in the research workspace; public summaries,
code, plans, verification records and figures are exported without private data.
