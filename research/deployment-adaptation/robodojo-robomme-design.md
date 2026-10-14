# Deployment correction absorption on RoboDojo and RoboMME

2026-10-15. Research design **DRAFT / METHOD EXPERIMENTS NOT_RUN**. This supersedes
the proposed LIBERO/SimplerEnv main-benchmark choice, following the user's
explicit benchmark selection. LIBERO's 500-case reference and eight-update
engineering qualification remain separate historical evidence.

## Story and falsifiable mechanism

A deployed policy can be helped without learning what made the help necessary.
We ask whether retaining the policy's own observations, failed actions and
subsequent expert correction improves the next autonomous deployment, beyond
ordinary expert demonstrations or DAgger state coverage. There are two clocks:
fast, causal episode memory with the policy weights fixed; slow, post-deployment
parameter updates from a source-only experience archive. Neither a memory
module nor OPD is new by itself. The candidate contribution is a measured
interaction between how corrective experience is absorbed and how the next
policy uses its own history. That interaction is an untested hypothesis.

For deployment generation g, h_t is the **public executed** observation/action
history and m_t=f_theta(m_{t-1},o_t,a_{t-1}). The policy emits an anchored action
chunk A_t=pi_theta(o_t,m_t). A correction record stores the complete student
prefix h_t, the actual query anchor, expert chunk A*_t, executed takeover,
public outcomes and provenance. The slow update minimizes the usual masked
action loss on those anchors; failed student commands remain context, not
positive imitation targets. Predicted offsets of a chunk are not relabelled
as teacher actions at future student states. At the next deployment, the
expert, oracle subgoal and correction archive are absent from the policy input.

The first candidate implementation reuses a released memory-policy backbone
and adds **no new teacher, reward or privileged memory input**. Its experiment
variable is the experience view: corrected suffix alone versus the same
corrected anchors paired with their causal executed prefixes. A full-history
DAgger arm is mandatory. If it matches the candidate, the explanation is
ordinary history-aware state coverage; do not claim a new RSI mechanism.
Learned event selection or multi-generation credit is deferred until this
basic contrast separates. We will not add a new selector just to rename OPD.

## Why these two benchmarks

| Benchmark | Main scientific role | Standard protocol | What it cannot establish alone |
|---|---|---|---|
| RoboDojo | Physical execution, failure correction and autonomous recovery; memory/long-horizon categories separately | Official success rate and progress score, native task configurations, same camera/action/cadence per policy | New-rule learning, continuous deployment improvement or hardware transfer |
| RoboMME | Whether absorption preserves and improves causal history use across counting, occlusion, referential and procedural tasks | 16 tasks; official train 100, val 50, test 50 per task; continuous-control evaluation at the released 1300-step limit | Multi-episode meta-learning or unknown dynamics adaptation |

RoboMME reset can reveal a full conditioning video. Keep all its frames,
including chronological order and later step batches, for memory-capable
policies. A current-frame baseline deliberately receives less history; that
is an architecture/information contrast, not evidence for our data mechanism.
Conditioning demonstrations are observation context, not student-executed
actions; retain the distinction when building correction histories. No
`maniskill_obs`, hidden object identity, simulator task progress, or oracle
`simple_subgoal_online`/`grounded_subgoal_online` enters an autonomous policy.
The official privileged oracle is labelled an upper-bound/teacher control.
The `multi_choice` oracle-planner track is reported separately from continuous
robot control; its scores cannot replace VLA success.

At pinned RoboDojo revision 09bd5f5e762b949f5858ece31347ce62057b3a5e,
the source inventory finds 42 canonical tasks plus 12 `_random` layouts,
**54 runnable configurations**. Native `_task.yml` assigns 25 or 50 episodes;
the complete configuration sweep sums to 2100 episodes, while canonical-only
native coverage sums to 1800. Do not describe this as 42 tasks times 50 or
silently mix a historical 42-task protocol with the current CLI. Preserve the
official category aggregation and report standard/random layouts separately.
The metadata audit is not a simulator or policy result.

## Strong references and controlled data-use arms

RoboMME source implementations: official pi0.5 current-observation baseline,
MME-VLA FrameSamp+Modul, RMT/TTT memory variants, MemER, MemoryVLA, SAM2Act+,
Diffusion Policy and OpenVLA-OFT. First reproduce pi0.5 and FrameSamp+Modul with
released checkpoints and source evaluation interfaces. RMT is a controlled
recurrent reference; TTT has fast-weight updates and must not be described as
ordinary fixed-weight context memory. Oracle subgoals are upper bounds. Not
all listed implementations are integrated or reproduced yet.

RoboDojo source references: released StarVLA QwenOFT/QwenPI_v3 and the official
XPolicyLab policy adapters, including a stronger robot-pretrained policy when
its checkpoints are available. A LIBERO checkpoint is not a RoboDojo baseline:
the StarVLA RoboDojo recipe uses head/left-wrist/right-wrist RGB, a 14D ARX X5
state/action interface, 50 predicted actions and 16 executed actions. Match
each release's coordinate, gripper and normalization contract exactly.

Within each benchmark, compare these **data-use** arms with identical initial
weights, memory architecture, optimizer updates, action loss and checkpoint
rule:

1. No update: starting policy and retention control.
2. More offline expert BC/SFT: ordinary demonstration growth.
3. DAgger/OPD: expert labels at actual student-visited anchors; full causal
   history supported by the same network.
4. Corrected suffix BC: same correction targets, only a fixed local history.
5. Complete failure-to-correction history: identical correction targets with
   the complete public prefix. This is the candidate mechanism, not a proven
   improvement over arm 3.

First isolate history with arms 4/5 using the exact same target IDs and update
budget. Then compare against arm 3 to test whether anything exceeds generic
history-aware aggregation. A separate equal-cost study allows BC to spend the
candidate's additional annotation/replay/selection cost on more data. Report
both equal-update and total-cost contrasts; do not claim budget advantage from
matching updates alone. Prior nominal success does not qualify an expert to
recover from student failures: audit actual-source failure takeover first.

## Metrics, interventions and inference

**Primary outcome:** official task-balanced autonomous success rate on the
untouched official test split, per task and training seed. RoboDojo progress
score is co-reported as an official secondary metric, never substituted for
failed completion. Use five paired training seeds for the registered pilot;
episode counts are not independent training replicates. Give seed-level
paired differences and intervals, plus all task outcomes and coverage. Freeze
initialization, seed values, update budget and selected checkpoint rule before
training. Validation can choose a checkpoint; test cannot choose task subsets,
teachers, prompts, memory rules or parameters.

**Expert dependence:** autonomous success plus success at fixed common expert
budgets 0/1/2/5 intervention chunks, with attempted queries, executed expert
steps, fraction of control and wall time. Use the same public takeover trigger
and expert for every arm. No hidden failure predicate triggers assistance.
Do not read zero expert calls during autonomous evaluation as improvement.

**Recovery:** success after a frozen source-independent failure prefix, and
full-from-step-zero success beside it. Use actual learner failures; injected
grasp perturbations are a separate registered stress test. RoboMME already
contains grasp-failure/recovery utilities; these are upstream capabilities,
not our innovation. Keep all registered failures, including unrecoverable
ones. Replaying a prefix is allowed only after initial-state reproduction;
arbitrary contact-state restoration remains unqualified.

**History dependence:** same checkpoint and test cases with valid history,
cleared older history, and unrelated/shuffled history of the same token/frame
budget; retain the current observation in all conditions. Initial conditioning
videos have their own intervention. Report absolute success and paired loss,
not only later-minus-first slopes. Normal memory use is not new-rule learning.
Multi-attempt support/query experiments, if added, are a separately registered
extension; support is counted and fresh query states remain disjoint.

**Absorption and retention:** one-round before/after autonomous success,
assistance curve and old-task success. Generations 0/1/2/3 are a later test
against fixed-data and ordinary OPD loops; no current run supports an RSI
acceleration claim. If all arms are at floor, first diagnose perception/action
interfaces and expert recovery; do not train against an artificially shortened
horizon or silently select the easy tasks to produce a gain.

## Execution sequence and resource constraints

1. Pin official source, audit metadata/task inventory and establish public
   observation/action boundaries. Training-only environment bring-up uses
   PickXtimes, VideoUnmask, PickHighlight, PatternLock (one per suite) selected
   before outcomes; it is not a four-task benchmark score.
2. Qualify simulator rendering, reset, step and actual conditioning-video
   delivery, then released policy loading/cadence and complete validation
   evaluation. A bounded hold-action smoke is engineering only.
3. Acquire source data and a meaningful learner/expert pair; record their
   checkpoint/source/task identities and expert recovery coverage.
4. Freeze one-round paired data-use pilot; collect source failures/corrections,
   train; select only on validation; evaluate all official test tasks.
5. Only after autonomous correction absorption separates: history intervention,
   equal-cost and multi-generation studies. Failed/negative runs stay visible.

30109 has A100 GPUs and mainskill is presently occupied. New packages live in
a task-specific environment; shared drivers and other workloads are untouched.
The initial simulator bring-up inherits Torch 2.6 from a qualified policy env,
whereas the RoboMME lock requests Torch 2.9.1: mark that run as a compatibility
probe, **not exact released-build reproduction**. The ManiSkill fork remains
at the official 07be6fbc66350ddca200abfb0a11b692f078f7fd. Exact-release policy
evaluation requires its own runtime qualification.

The current shared filesystem has about 33 GB free, with an 8 GB reserve.
Official RoboMME HDF5 downloads sum to 56.44 GB compressed; the released
FrameSamp+Modul archive is 11.88 GB before extraction. Do not download the
whole corpus or create both compressed and unpacked checkpoints blindly.
Use existing assets read-only where source identities match, budget subset
data, and register a larger storage target before full-suite training. These
are access/runtime constraints, not measured method failures.

## Primary sources

- [RoboDojo source](https://github.com/RoboDojo-Benchmark/RoboDojo), pinned above;
  [documentation](https://robodojo-benchmark.com/doc/),
  [paper](https://arxiv.org/abs/2607.04434).
- [RoboMME source](https://github.com/RoboMME/robomme_benchmark), pinned
  fae9f7300347f1146c2ae4844faa4eef7e38098e;
  [paper](https://arxiv.org/abs/2603.04639).
- [Official MME-VLA implementations](https://github.com/RoboMME/robomme_policy_learning),
  pinned ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b.
- [RoboMME data](https://huggingface.co/datasets/Yinpei/robomme_data_h5),
  [released models](https://huggingface.co/Yinpei/mme_vla_suite).
- [Official policy ecosystem](https://github.com/XPolicyLab/XPolicyLab),
  [StarVLA RoboDojo bridge](https://github.com/JinhuiYE/XPolicyLab/tree/fix/starvla-hf-robodojo-eval).

Published scores remain author-reported until a complete, content-bound local
evaluation exists. Source inspection depth and implementation status must be
shown beside each baseline; code availability does not mean reproduction.
