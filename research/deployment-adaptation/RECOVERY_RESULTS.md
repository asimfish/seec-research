# RoboMME learner-state expert qualification — 2026-10-16

The fixed released student failed PatternLock train episode 0 at control 68.
Replaying its first 64 commands reproduced all 113 policy-visible frames exactly
(front/wrist RGB and float32 eight-dimensional state). A privileged upstream
planner then completed the task with 26 additional controls, at total control
90. The planner also succeeded from the episode's initial state in 42 controls.

| Registered diagnostic | Student controls replayed | Expert controls | Official outcome |
|---|---:|---:|---|
| Expert from reset | 0 | 42 | success |
| Expert at learner prefix | 64 | 26 | success |

These are two qualification executions on one training instance. The anchor was
chosen retrospectively as the last action-chunk boundary before the observed
failure. This does not validate an autonomous query trigger, a recovery success
rate, a memory failure diagnosis, post-training improvement, or generalization.
The expert reads privileged task state; that state is never a student input.
The prefix is replayed from reset, with no intermediate state restoration.

Plan: `robomme-failure-expert-ad7bdc5fdf8626f7`.
SHA-256: `a3b27e2f8c471ba7af1f1fe0269daaf3869778ec82913312d65e2877500c658c`.
Benchmark: `fae9f7300347f1146c2ae4844faa4eef7e38098e`.
The implementation is `scripts/qualify_robomme_failure_expert.py`, bound by its
file SHA in the plan. Plans, results, unchanged videos, selected raw figure
frames, and training views are retained in `recovery_qualification_20261016/`.
The qualification code and simulator do not use validation/test outcomes for
training. Reset's internal demonstration simulation remains uninstrumented;
it is not zero cost. New explicit simulation: 42 + 64 + 26 = 132 controls;
68 of those are expert controls. The earlier failed student used 68 controls.

## What enters the absorption experiment

The 26 executed correction actions produce seven complete native 20-action
training chunks. Terminal targets are not padded or invented. Sliding chunks
overlap (140 target positions, 26 unique controls), identically in both views.
At every anchor, current front/wrist images, state, targets, and row IDs agree.
Only memory selection differs:

* Recent-history intervention BC: the latest 32 causal frames.
* Full-history intervention BC: the pinned upstream even-sampling rule over the complete
  causal prefix, at most 32 frames; 16 tokens per frame, 512 tokens total.

At the first anchor, frame 112, recent history contains frames 81–112. Full
history retains some of the original demonstration frames 0–48. This is a data
property, not evidence that forgetting caused this particular failure.
The proposed candidate currently aliases full-history intervention BC. Only
the first chunk anchor is student-visited; the other six are visited after
expert takeover. Dense student-state labeling for full-history DAgger/OPD
still requires its own collection experiment. It is not a
separate novel algorithm or an additional comparison arm. No absorption
updates or autonomous post-training evaluation have run.

The matched target-set SHA-256 is
`546b590387b66202fde028d493784619135800bc077f7f899239e0f6f4ee0d70`.
The exporter initially rejected the planner's native seven-dimensional stick
actions. The correction now follows the pinned benchmark's `RecordWrapper`
and `episode_dataset_resolver`: append an unused gripper placeholder -1 for the
eight-dimensional policy layout. `DemonstrationWrapper` passes only the first
seven dimensions to stick physics. The original seven joint targets are kept;
the raw qualification evidence was not changed or recollected.

## Standard reference evaluation

The unchanged 16-task × 50-validation-case plan remains
`robomme-reference-e0ad4d742f53cd15`. The first attempt completed cases 0–26,
then aborted before control on BinFill validation episode 27. An environment
creation/close-only reproduction also failed on creation 28, whereas episode
27 created successfully in a fresh process. This localizes a resource-lifetime
problem to the simulator/render process; its exact internal field is unproven.
Pending cases now use separate simulator processes.

The first isolated continuation aborted at 128 controls because the upstream
server's WebSocket keepalive timed out. It is retained with null success.
The next execution changes only the server heartbeat allowance to 600 seconds,
matching the existing client allowance. It retains both infrastructure attempts
and their cost, the original 27 completed outcomes, case order, model, seed,
horizon, and observation/action boundary. Upstream `Policy.reset` restores the
same seed independently each episode. No completed failure is retried.

Current continuation: `robomme-continuation-e3fe41c45265a43a`, execution snapshot
`f8d4df471e30cb0e532f3453d677efb05a896419`. The original six implementation
files still match snapshot `77b0344b439aa712987427f1dcac40986688f57a`.
The additional worker and transport files are separately hash-bound. The
read-only progress audit checks every closed case's video/trace/program hashes
and every executed action against its saved prediction. Validation does not
store lossless H5 images or independently replay physics.

An audited snapshot has 30 completed cases, 16,857 controls, 1,068 programs and
1,128 hashed artifacts. Both infrastructure aborts are retained, including
128 aborted controls. This is incomplete coverage, not a full-suite success
rate. `validation-progress.json` carries its timestamp; it is not a live feed.
Official test remains untouched. This runtime uses current pinned source,
not the policy repository's older release submodule.

## Next comparison

Expand qualified train-side learner failures and prospective query budgets
before a multi-seed absorption study. Keep full-history DAgger/OPD as a strong
baseline, share targets/initialization/updates and account for all expert and
replay costs. The primary outcome remains autonomous task-balanced success;
also report success versus actual expert-control budget, recovery and retention.
One qualified recovery cannot establish that the next policy needs less help.

The public figure can be regenerated using `plot-expert-recovery.py`,
`assets/robot-reference/PatternLock-recovery-filmstrip.npz`, and
`data/robomme-expert-prefix-result.json`, with `--input`, `--result`, and
`--output-prefix` respectively. It requires NumPy and Matplotlib. These are
exact exported inputs; rendering does not require the simulator or model.
