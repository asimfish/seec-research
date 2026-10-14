# Deployment feedback → autonomous recovery and adaptation

2026-10-14. **Draft experiment design; no new method efficacy result.** Implementation and infrastructure qualification are authorized. A full training run requires a concrete frozen data/learner/budget manifest, not a new approval request.

## Question and outcomes

Can failures and expert corrections collected at deployment improve the next policy's autonomous completion, recovery and use of its own subsequent feedback?

Primary outcome: **task-balanced teacher-free success rate** under the benchmark's fixed rollout horizon. Report each task and each training seed. The primary comparison pairs training seeds and identical task/initial-state cases. Confidence intervals resample training seeds after task aggregation; episode counts are not treated as independent training replicates.

Existing QwenOFT benchmark records are inventoried in `historical_baseline_inventory.json` with original episode counts and failed attempts. They are historical references, not new-module results. Strong baseline ceiling performance may hide a recovery benefit: qualify task difficulty on development data, freeze any hard/recovery cohort before final evaluation, and keep overall success beside subgroup results. Do not select tasks because they show the desired test-set difference.

Secondary outcomes: success on preregistered recovery-start cases; success at attempts 1/2/3/5; cumulative successes under a fixed interaction budget; expert interventions per 1,000 control steps; teacher queries; environment steps and resets; successful-episode completion time. Failed episodes are censored for time-to-success, not counted as fast completions. Report runtime and simulator control time separately.

**Autonomous competence and expert dependence need different evaluations.**
Teacher-free rollouts necessarily have zero expert queries, even for a policy
that fails every task. Those zeroes cannot establish reduced dependence. A
separate assisted-deployment protocol must freeze the teacher, a common public
observation takeover rule, and expert-control budgets before method results.
Report success versus expert-control budget, control-step fraction and query
cost together; a minimum budget at a preregistered completion target is
undefined when that target is never reached. Do not reward refusing assistance
while failing. Do not retune takeover thresholds separately for each arm.
Assisted query records never enter training or checkpoint selection; their
labels and assisted success cannot replace the teacher-free primary outcome.
The current generic runner implements teacher-free evaluation only. The
assisted evaluation surface and its concrete budget grid remain unimplemented
and unregistered; source-collection assistance counts are not deployment
dependence results.

Recovery cases should come from actual student failures. A stored qpos/qvel state alone does not establish exact contact/controller restoration. Prefer a verified replay of the original prefix from an official initial state until state restoration is independently qualified. Count prefix replay cost; report conditional recovery separately from full task rollouts controlled by the policy from step zero.

Do not use later-minus-first improvement as the main outcome: our credit pilot already selected learners whose initial and total competence declined. Absolute curves and retained competence remain visible.

## Benchmarks and splits

1. **Standard robot post-training:** official LIBERO-Spatial/Object/Goal/10 task definitions, initial-state files and success predicates. Qualification used Spatial. The first learned reference uses LIBERO-10 and an existing released QwenOFT checkpoint: archived nominal Spatial records are already near ceiling, while the long-horizon suite offers a more informative failure audit. This reversible ordering choice precedes the new reference results. The intended full study covers the four suites. Full-horizon evaluation and official preprocessing must be matched to the chosen upstream policy recipe before registration. The short qualification does not count as a benchmark score.
2. **Second robot domain:** SimplerEnv after LIBERO reference fidelity is established. Use its declared visual-matching/variant-aggregation protocol; do not merge scores across those settings.
3. **Own-experience adaptation extension:** same stable deployment condition across support attempts, fresh query initial states, weights frozen. Clearly label this as an extension. Do not infer unknown-rule adaptation from ordinary LIBERO success or retries on the same state.

Source trajectories alone enter training. Development selects a checkpoint. Final benchmark cases cannot select methods, checkpoints, budgets or nuisance settings. Use separate generated development starts if reporting the entire official initial-state set at final evaluation; validate byte-level initial-state references for overlap. A proposed split is not accepted merely because its integer seeds differ.

## Baseline ladder

| Arm | Data/learning operation | Code reuse and current fidelity |
|---|---|---|
| Frozen policy | Same starting checkpoint, no absorption | Existing RoboCore/StarVLA policy bridge; actual checkpoint identity required |
| Additional expert BC/SFT | Additional expert demonstrations, matched update budget | Official robomimic BC/BC-RNN or selected VLA trainer; source pinned; training adapter pending |
| DAgger / on-policy distillation | Label states actually visited by student, aggregate and retrain | HumanCompatibleAI/imitation collection semantics; view construction implemented; full upstream execution pending |
| Recovery BC | Student prefix, expert takeover, corrective suffix | PLD-style data-use comparator; not a full PLD residual-RL reproduction |
| History-conditioned corrections | Same corrective targets, full failure history as context | RoboTTT/AD data-use comparator; not a full RoboTTT reproduction; context-policy backend pending |
| Candidate mechanism | Defined only after the above comparisons identify a transfer failure | Reserved interface, not an invented implemented method |

All data-use comparisons share architecture, input capacity, normalization, initialization, optimizer, action head and update count. Plain BC versus a larger memory model cannot isolate history organization. Identical source/target/context views are aliases, not independent trained arms. Use upstream released implementations when available; record adaptations and missing components explicitly.

## Staged execution

### E0: infrastructure qualification

Pin LIBERO source and dependencies. Run actual manipulation environments on explicit official initial states. Check visible image/proprioception shapes, action dimension, reset identity, simulator success reporting, trajectory roundtrip, policy-history boundary and complete cost. Compare explicit reset with the raw upstream state path. No gradients, expert ceiling or learned-policy effect is claimed.

### E1: reference reproduction

Run one official BC/BC-RNN or released VLA checkpoint through its upstream evaluator and the adapter on the same cases. Explain any success mismatch before absorption experiments. A short smoke or compact reimplementation is insufficient.

The October 14 fixed-checkpoint reference completed all 500 LIBERO-10 cases and
uses content-bound archived integration code without Git metadata. Report
that fidelity level explicitly; the exact checkpoint-release source revision
is unavailable. A separate adapter check repeats the fixed first case from
each shard (tasks 0/4/7, initial state 0), comparing image sequences, commands,
rewards and outcomes. Those repeats are engineering checks, not extra seeds.
Standard stabilization actions count in total simulation cost and have no
learning target; expert dependence divides by policy control steps. Reference
failure frames may inform engineering analysis, but cannot be absorbed into a
policy later scored on those same cases. Correction training needs disjoint
source states and a separately frozen query protocol.

Control-interface qualification is separate from teacher strength. A frozen
checkpoint that succeeds when executing eight-action programs is not thereby
qualified as a one-step expert after projecting each new prediction to its
first action. In the registered three-case control check, this projection
timed out at 520 steps on a case where the original program completed in 257;
the other two completed in 333/303 steps versus the original 230/259. These
are control/teacher diagnostics, not learned-method gains or a novelty claim.
Physical control timing, checkpoint and initial states stayed fixed; model
query/replanning cadence changed. Exact causes beyond that intervention still
need trajectory analysis. The old frozen qualifier is retained.

All future data-use arms must share the learner's action horizon and executor.
The source-dependent teacher now rejects silent single-action collection from
a chunked expert and exposes full program labels at actual student anchors.
Those future targets belong to the anchor input and output offsets 0–7,
not the later student states. A program-bound source annotation schema,
collector and boundary mixing/takeover protocol are now implemented; a
qualified trainer and concrete budget/data manifest remain required before
absorption. Program replay matched the fixed reference on all three registered
cases, with zero updates. The full reference scored 476/500 (95.2%), with
complete non-aborted coverage and all trace/failure-camera checksums verified.
This is a released-checkpoint baseline, not an absorption or adaptation gain.
The single-anchor QwenOFT example bridge now reuses the original normalization
and matches its camera pixels/order in an offline qualification. This is data
compatibility, not a qualified training run; history rows are rejected until
an actual context-capable learner is registered. See the final report.
Teacher-assisted evaluation must not count a changed replanning interval as a
learning gain; include a frozen-policy executor control wherever they differ.

### E2: one-round absorption

Use five paired training seeds in the proposed pilot; freeze the exact values, updates, batch size, checkpoint rule and total expert/environment budget before running. Compare expert-only, student-state labels, recovery and history-conditioned correction views. Report complete cost including rejected trajectories and teacher-assisted collection. Pilot is exploratory and does not supply the main-paper inference.

### E3: deployment adaptation

For each policy, test empty versus valid self-history, demonstration context and shuffled/irrelevant feedback under the same public feedback interface and interaction budget. Add a fresh-query transfer test to rule out repeating a memorized trajectory. Test-time parameter fine-tuning is a separate arm and its updates count as cost.

### E4: recursive absorption

Only after E1–E3, run generations 0/1/2/3 against repeated training of a fixed pool, ordinary OPD, and a fixed data engine. Use fresh evaluation cases and report old-skill retention. Producer scoring is a later optional intervention, not the identity of the current method.

## Failure handling and claim boundaries

Missing or aborted evaluations invalidate aggregate efficacy statistics; preserve their records and causes. No teacher may intervene or provide labels during a teacher-free evaluation. Success uses the benchmark predicate, not an agent-written replacement. Infrastructure repair cannot silently change task parameters or test cases. Registered historical failures remain unchanged.

Current completed Cooking/credit/Alchemy evidence remains development-only.
LIBERO E0 is a simulator/adapter check; the 500-case standard fixed-checkpoint
reference is complete at the archived-integration fidelity level above.
The proposed method's absorption, adaptation extension and recursive effects
remain NOT_RUN until their corresponding records exist.

## Sources

- LIBERO environment and evaluation: https://github.com/Lifelong-Robot-Learning/LIBERO
- robomimic BC: https://github.com/ARISE-Initiative/robomimic
- DAgger implementation: https://github.com/HumanCompatibleAI/imitation
- PLD: https://openreview.net/pdf?id=eUGoqrZ6Ea
- RECAP: https://arxiv.org/html/2511.14759v2
- RoboTTT: https://arxiv.org/html/2607.15275v1
- SimplerEnv: https://github.com/simpler-env/SimplerEnv
