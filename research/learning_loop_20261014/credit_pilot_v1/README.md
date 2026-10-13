# Experience-credit ranking diagnostic

Registered plan: `cooking-credit-5d8bf5ccf37f8781`.
Semantic SHA-256: `5d8bf5ccf37f878104b82f81f8800b7cbe82097dddc65e15592ff108a7efa7ea`.
Exact plan-file SHA-256: `f095d7833d751b980a794bc1048886ed83fcf05d4267778acc4bd3060811c19f`.

See `PROTOCOL.md` and `plan.json` before interpreting a number. The three learner
seeds are new; the finite Cooking rule family and split have been used in previous
research. Feedback/audit are training-side diagnostics, not an unseen sealed test.
This study is not a full SOAR, Meta-Teaching, DREAM or AD reproduction, not a
closed-loop experiment and not robot efficacy.
Registration uses local timestamped, content-bound artifacts, not an independent
preregistration service. First attempts have no prior exploration prefix but can
learn through their own actions and responses within the episode; they are not
zero-interaction accuracy.

## Reproduction

Use the original DREAM checkout at
`d52204c94067641df6e6649b19abb359b87ff028`, with no transition changes. The legacy
adapter adds NumPy's removed `np.int` alias and a fixed-instruction wrapper.
Execution environment: Python 3.10.12, NumPy 2.2.6, Torch 2.13.0+cu130, Gym 0.23.1;
all model work is CPU, two threads. Older/newer Torch versions may not yield the
same weights. No shared GPU driver/settings change is required.

The code layout includes frozen adapters in
`research/interactive_learning_20261013/{run_cooking_diagnostic.py,learner_pilot_v1/run_learning_pilot.py,student_coverage_v1/run_coverage.py}`.
Each file is bound by the plan; no edits to those previous experiments.

From the repository root, replace `DREAM_CHECKOUT` and select a fresh output:

```sh
python3 research/learning_loop_20261014/credit_pilot_v1/run_credit.py \
  --upstream DREAM_CHECKOUT \
  --plan research/learning_loop_20261014/credit_pilot_v1/plan.json \
  --output fresh-credit-run
python3 research/learning_loop_20261014/credit_pilot_v1/analyze_credit.py --run fresh-credit-run
python3 research/learning_loop_20261014/credit_pilot_v1/verify_credit.py \
  --run fresh-credit-run --upstream DREAM_CHECKOUT \
  --plan-folder research/learning_loop_20261014/credit_pilot_v1
```

The local run uses a process-local dependency directory for Gym compatibility;
it does not replace the user's global environment. Raw action histories and model
checkpoints stay local. Public outputs contain the exact authored adapters,
registered plan, summaries, analysis, cost and independent verification.

## What the comparison means

Each candidate collects 24 real student histories. Short and long branches restart
the same initial model and Adam state, with 64 versus 300 updates; long does not
continue short. Every branch samples 12 common-base and 12 candidate histories
per update. Different valid label counts/action costs are explicitly reported.

Scores predict longer-training adaptation utility, with selection constrained
only by short feedback competence. Uniform is the average over eligible
candidates: a random candidate-selection reference, not pooled-data training.
Ties/undefined correlations remain visible. No audit result chooses a model.

The independent verifier replays all recorded actions, observations, rewards,
labels and policy decisions, binds all 51 checkpoints, and retrains one fixed
first short branch. It does not repeat all 51 trainings or perform physics
snapshot E4. Verification overhead is reported separately.
