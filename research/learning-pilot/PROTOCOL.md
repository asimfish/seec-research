# Registered learned-history pilot

This follows the completed symbolic Cooking diagnostic. That diagnostic could
not distinguish a new mechanism from a strong full-history policy. The next
cheapest separating test is actual learning with a matched auxiliary objective.

Three hypotheses compete: action-only history learning is sufficient; generic
extra supervision explains any gain; or grounded historical-fact supervision
specifically improves unseen-task action selection. A fourth infrastructure
explanation is a control/behavior-cloning floor. `plan.json` freezes predictions,
code hashes, hyperparameters, input boundary, splits and stop conditions before
the source collection and any optimizer updates. The registration is internal.

The three arms have the same recurrent architecture, initialization, source
histories, per-seed sample order, and optimizer budget. Only auxiliary targets
differ. Grounded fact targets are reconstructed from previously observed actual
pickups, including failed/nonrewarded exploration, not from hidden drawer IDs.
The current-state auxiliary predicts the inventory and two visible bowl slots.
No auxiliary label or structured memory is fed to the policy at evaluation.

The network, including its exploration actions, controls from step zero. It
receives only current visible observation, instructions when revealed, previous
actual action/reward, episode boundary, phase and local time. Its causal GRU
retains state across attempts on one mapping and is cleared for each new task.
The actual last observation/action/reward is ingested as a feedback-only token
before each episode reset; no action is requested and no simulator step is spent.
The reset-between-episodes condition is an evaluation ablation of the same
checkpoint. All arrays are one-hot according to the published observation
schema; no test data is used to fit transformations.

Data includes deterministic observable-memory teacher histories and fixed 10%
action perturbations for some recovery-state coverage. This is offline BC, not
student-distribution OPD. The teacher uses observed facts and a frozen skill
controller; auxiliary targets cannot label unobserved drawers. Source metadata
contains evaluator IDs, but the input encoder does not accept identity.

The public environment's transition, reward, terminal definition and 20-step
horizon stay unchanged. Splits reuse the prior 32/8/24 diagnostic assignments;
these are unseen to the neural optimizer, not an untouched confirmatory panel.
Every valid goal is evaluated for five attempts after up to 20 exploration
actions. All costs include failed actions and exploration. Three optimizer seeds
are a descriptive pilot; episodes are nested under mappings and training seed.

The fixed final checkpoint is evaluated. Development is descriptive only.
Observed-fact supervision must exceed both controls by five percentage points
and improve each seed to support its preregistered pilot prediction. A success
floor in all arms triggers a control/distribution-shift diagnosis, not parameter
tuning on the held-out panel. No method-superiority, robot, RSI acceleration or
author-method reproduction claim follows from auxiliary labels alone.

Run all adapter/boundary tests before `register_learning_pilot.py`, then execute
the newly written immutable plan. Never overwrite a run. Preserve partial runs
and explicitly label an infrastructure failure before a corrected registration.

The first registered run was interrupted and marked INVALID_INPUT_BOUNDARY
during verifier design: its terminal next observation/reward was logged but
not ingested by the GRU. The corrected run adds feedback-only terminal tokens,
preserves the old code/plan/partial source and training, and changes no task,
split, horizon, model size, loss weight or optimizer budget. Its new registration
is made before corrected source collection and training. No evaluation result
from the interrupted run is interpreted.
