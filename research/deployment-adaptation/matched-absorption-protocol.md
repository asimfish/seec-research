# Matched corrective-experience absorption — protocol v1

Status: DESIGN FIXED; concrete dataset/runner fingerprints must be committed
before the first non-qualification optimizer update. This is a two-task
RoboMME absorption study, not the full benchmark, new-rule ICL, or a new
algorithm claim. The full-history candidate is exactly the full-history
student-state BC/one-round DAgger baseline and is trained only once.

## Questions and contrasts

Q1: At a fixed compute budget, does absorbing executed oracle corrections
improve autonomous terminal success relative to the same initial policy?
Q2: Does complete causal history improve the absorption gain, beyond the
initial policy's existing advantage from seeing that history?

Four cells: full-history before/after, recent-history before/after. Within
training seed s, record G_full(s)=SR(full,after,s)-SR(full,before),
G_recent(s)=SR(recent,after,s)-SR(recent,before), and
I(s)=G_full(s)-G_recent(s). A full/recent success gap alone is not absorption.
The two before policies have no optimizer updates and are evaluated once per
view with inference seed 7. Their deterministic records are shared references,
not independent repeated measurements for each training seed.

## Common weights and data

Initialize every arm from the unmodified released FrameSamp+Modul checkpoint:
repository Yinpei/perceptual-framesamp-modul, revision
c0f565dd40082f32863b3ee9db99de5ed5d3d4e0, archive SHA-256
2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62.
No qualification-trained checkpoint is a primary-study initial policy.
Native source revision: ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b.

PatternLock and PickHighlight train instances 1–4 supply source trajectories;
instances 5–8 are development diagnostics and supply no labels. Official
validation/test supply no training data. Exclude the previously diagnosed
instance 0. Pretraining exposure of any official instance is unknown; do not
call these rules or tasks unseen by the released model.

The original 36 eligible queries produced 33 full action chunks; 2 terminal
partial chunks and 1 unqualified attempt remain in costs, not training. That
early dataset covers only 33/124 source policy-decision positions. It cannot
be used as the primary absorption dataset. The coverage supplement
robomme-dense-query-e3c7b8af9beaa027 adds all 88 later source policy decisions,
without changing the original source plan or retrying an old failure. It was
designed after the early coverage audit and is disclosed as a supplement.

Final admission is fixed: exact public-prefix replay; QUERY_COMPLETE; finite
20-action chunk. Never pad a partial terminal chunk. Keep every excluded
query and its reason. Require all expected query attempts to be accounted
for, no duplicated (case, anchor), exact replay on every query, and at least
80% complete-chunk coverage separately for each task. This engineering gate
is set during collection, before any absorption outcomes; it is not a
confirmatory scientific threshold. Failure means dataset not qualified,
not evidence that the learning mechanism is ineffective.

Full and recent views have the same current RGB/state, prompt, action labels,
case order and sampling sequence. Only historical frame selection changes:
up to 32 uniformly spaced causal frames versus the most recent 32 causal
frames; 512 memory tokens each. The policy during evaluation uses the same
selection rule as that arm during training. Reset/video demonstration frames
are inputs, never fabricated supervised targets. Privileged oracle state,
planner subgoal and teacher-access labels never enter the student.

## Optimization and checkpoint

Training seeds: 7, 17, 27, 37, 47, 57; six paired seeds. All seeds and both arms
are mandatory. Exactly 10,000 optimizer updates per arm/seed, batch size 1,
uniform sampling over the fixed admitted native action chunks. Pair initial
weights, sampled row IDs and native random seeds. Use the original full model
and image freeze rule, AdamW, 10,000-step warmup (peak learning rate 5e-5),
EMA 0.999. No memory-only or smaller-model substitution. Native FSDP uses two
explicitly allocated A100 80GB devices, allowed pairs 0,1 and 3,7 on mainskill.

Use only the final 10,000-update EMA. Development cannot select checkpoints,
seeds, arms, learning rate or stopping time. Save all update metrics and row
IDs. A sealed XOR export preserves every EMA parameter's exact bf16 inference
bits, matching the released inference loader's precision. It does not claim
fp32 or optimizer-resumable storage. Independently decode and fingerprint the
export, then copy/hash it to durable storage before its RAM holder expires.

## Evaluation

Primary evaluation: both tasks, official val instances 0–49, 100 episodes per
trained model. All six seeds, two trained views, plus two shared initial-view
references: 1,400 evaluation episodes total. Fixed horizon 1300, predict 20
controls and execute 16, policy inference seed 7 reset per episode. Student
controls from the first post-conditioning step; no teacher prefix, extra
expert intervention, oracle input or checkpoint selection on validation.
RoboMME's initial demonstration/video conditioning remains part of its native
input. 'Autonomous' means no additional expert after that conditioning.

Fresh simulator process and independent policy context per case. One model
server may serialize requests from distinct connection-local contexts only
after its native interleaving equality gate passes. All before/after inference
uses the same physical RTX 5090 GPU 1, seed and native precision. Completed task failures are not rerun. Infrastructure
aborts remain explicit and only identical-config retries with reason codes
are allowed. Missing/aborted models or cases invalidate complete-study claims;
do not turn missing into failure or discard seeds. Report per-task terminal
success and their arithmetic mean. The independent original 800-case released
reference is a separate experiment and its outcomes are not merged here.

The eight development instances may exercise the evaluator before primary
validation. They cannot choose a different checkpoint or modify this protocol.
Before primary evaluation, require: model export/reload parity, training/eval
memory-index parity at all boundary lengths, deterministic repeated forecasts,
and no extra teacher calls. The whole-initial-model and per-arm input versions
must match recorded fingerprints. Include the independent terminal outcome;
training loss is not a robot success metric.

## Statistics and decisions

Training seed is the statistical unit. Report all six seed-wise task-balanced
rates and paired gains, mean paired gain, 95% seed bootstrap intervals (100,000
resamples, seed 131), and the exact two-sided sign test over nonzero gains.
Ties reduce the sign-test sample size. Episodes are paired benchmark cases,
not thousands of independent training runs. Intervals are conditional on this
two-task corpus; they do not quantify generalization to all robotics tasks.

Primary practical threshold: mean G_full >= 0.05. Support an absorption gain
only if this threshold holds, its interval excludes zero, and the sign-test
p <= 0.05. Test the history interaction I only after the primary comparison
passes; otherwise report I descriptively. This is hierarchical gatekeeping,
not independent claims from each cell. A significant history interaction does
not establish algorithmic novelty over full-history DAgger.

If the interval includes zero, report INCONCLUSIVE. If its upper bound is below
0.05, report that this registered configuration does not support the practical
gain. All-zero/ceiling initial success or all-floor trained policies is reported
as an uninformative/failed feasibility outcome; never tune until significant.
Method, data, threshold or budget changes require a disclosed new experiment.

## Scope, costs and scientific stop conditions

Report all source episodes, 1,924 source control steps, all query attempts,
actual extra expert controls, replay controls, resets, uninstrumented native
reset-demonstration costs, preprocessing, failed attempts, GPU hours and
inference overhead. The 88-query supplement permits at most 1,760 new expert
controls. All views share these costs. No budget advantage follows.

This first comparison evaluates success at zero additional expert budget.
It cannot establish an entire expert-budget curve, non-regression on the
other 14 tasks, new-rule learning, strict iterative OPD, or multi-generation
RSI. Source states are from one fixed released policy; this is fixed-data
one-round student-state absorption, not current-policy recollection every
update. Retention, broader baselines, active requests and repeated generations
remain later experiments. Do not write conference readiness from this study.
