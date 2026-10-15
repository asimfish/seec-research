# Actual validation progress — 2026-10-16

The native learner has executed real updates, and prospective student-state
oracle labeling now works. Neither establishes an absorption benefit. The
candidate currently aliases full-history BC/DAgger under the same inputs and
labels; it is not an additional method arm or a novelty result.

## Native optimization

Two matched views of the original train-0 correction, seed 7, batch size 1,
8 updates each, on two available A100 80GB GPUs. The pinned 3.44B model retains
its original image-encoder freeze, 3.027B trainable parameters, AdamW,
10,000-step warmup and EMA 0.999. No smaller model or memory-only freeze was
substituted. Both views start with trainable fingerprint
`6a8e7e1451978727fae1baebc9d53ce45db589a3c1d0c69756646754ffed5de6`.

The full-history arm passed optimization and checkpoint reload: 8 finite,
nonzero-gradient updates; trainable weights changed to
`bf3f2aee1206ae295526802988b8c5261057755aa8d8c02cfe5d4d75375bcd47`.
The expected and reloaded EMA fingerprints both equal
`1eca5ca90f0b45982211b766245beebab18f309ca94d82dc3779e8ca173572fd`.
The recent-history arm also passed all 8 updates and exact EMA reload. Its
trainable fingerprint changed to `59ad6a6200c5114b0bb1e755de5be50ac80df5adc9c13a917905cc44936721c7`;
expected and reloaded EMA both equal `88eb17424322fe275adfe3e8646c2fe2fc33fae0d5e7b29d661cf26929b91b07`. See `native_training_20261016/qualification-run1/`.

Full-history first compile/update: 114.65 seconds. Median of the seven warm
updates: 0.1645 seconds. JAX recorded peak allocation of 37,878,626,560 bytes
per device (largest of the two devices); this is an allocator statistic, not
a total physical VRAM measurement. These timings are interface profiling,
not end-to-end method cost or a learning effect.

Both completed arms used the original committed qualification runner, SHA-256
`6548e93fa882979cd4d6eb98cf6bffbd11a9646792cc59a81483400842214c0c`.
Its complete source is retained beside the result. The first strict audit
finished naturally; no optimizer run was aborted or restarted. A later
per-shard byte comparison is implemented and CPU-tested but did not replace
the checks in this completed job.

The long audit was observed repeatedly in JAX's global shard-to-host array
assembly. On the same shared host, a 32MB strided float32 copy took 19–28 s
with default allocation behavior, and 0.0165–0.0188 s in three probes using
`NUMPY_MADVISE_HUGEPAGE=0`. See `host_memory_probe.json`. Future jobs record and
use this process-level allocation setting for every arm. No global OS
setting changed. A full-model timing comparison of that setting is NOT_RUN.
The newer audit compares all expected GPU shards with restored CPU bytes,
including frozen parameters; its one CPU fingerprint follows that independent
comparison and is not described as two independently computed hashes.

The shared NAS had little free space. Verified input weights use a bounded,
sealed private RAM cache; checkpoints use the job's writable scratch. The
local verified source copy remains intact. 689,702,192 bytes of this job's
JAX libraries were moved to its own NAS directory with hashes unchanged,
leaving enough checkpoint headroom. No other workload or shared package was
modified. Full checkpoint transfer back to local storage is in progress.

## Prospective source screen and a strong baseline

Plan `robomme-absorption-screen-5c9451b5a5b9791c`, SHA-256
`f9f8101b8dd997a3386a03f5f0d0a1cd8e641daf738f8f6aa089dac26860397d`,
was committed before execution. PatternLock and PickHighlight each use train
instances 1–8: 1–4 supply source data, 5–8 are held back only from this new
absorption dataset as development diagnostics. Their pretraining exposure is
not established; they are not described as new rules unseen by the released
model. Previously inspected instance 0 is excluded from this screen.

At the closed-case snapshot, 7/16 instances completed: PatternLock 1 success
58 controls; PickHighlight 1 success 215; PatternLock 2 failure 110;
PickHighlight 2 success 433; PatternLock 3 success 152; PickHighlight 3 failure
621; PatternLock 4 success 22. All are retained. No partial-cohort success
rate or training-seed inference is reported. The first service-not-ready
attempt stopped before environment creation/control and is retained; run2
uses the same registered cases. A short pause stopped only the parent
scheduler while its current episode continued; it was resumed without any
policy restart or interrupted episode.

Source PatternLock 1 received four eligible, prospectively fixed queries at
controls 0/16/32/48. The proposed control-64 query is ineligible because the
student already terminated at 58; no substitute was chosen. Public-prefix
checks passed on all four forks: 64/80/96/112 frames, 352 checks total. The
oracle executed 20/20/20/14 controls: 74 total, plus 96 student replay
controls and four resets. Internal reset demonstration-generation cost is
unmeasured, not zero. The last fork reached terminal success; the other three
ended at their label budget and have null terminal outcomes.

Three complete 20-action chunks are now materialized for each history view;
the partial 14-action query remains in costs but is never padded. Target-set
SHA-256 is `76441f3cf9a6f82414eaec2765c21645ed07dde7fe4ca0e990fa287d89c7364e`.
All chunk anchors are student-visited. Inputs come from the autonomous source
history; oracle futures provide action labels only. The stick's unused -1
gripper slot follows the original dataset's 7-to-8 action layout. This is a
DAgger/OPD data path, not a trained DAgger result. It remains separate from
the retrospective recovery suffix, where six of seven anchors were expert-visited.

RoboMME provides demonstration/video conditioning during reset. “Autonomous”
here means no additional expert intervention after that benchmark-provided
conditioning, not absence of demonstrations. Extra oracle queries, applied
controls, resets and uninstrumented setup costs are kept separate.

## Complete reference evaluation and RoboDojo

The independent fixed-model validation continues unchanged. Its audited
snapshot is 109/800, 43,599 controls, 2,777 checked action programs and 2,995
hashed files. BinFill completed 22/50 (44%); StopCube completed 20/50 (40%).
The remaining tasks are incomplete and no full-suite score is reported.
Both earlier infrastructure aborts and all task failures remain. Test is
untouched. This snapshot can lag the live journal.

RoboDojo's matched 14D QwenOFT weight and all run/config/statistics hashes
passed the upstream acquisition validator. The pinned Qwen3 implementation
already forces SDPA; FlashAttention installation is not required by that
code path. A fresh native reset captured the production proprioception and
instruction with the three actual RGB cameras. Native policy initialization
reached GPU transfer but failed under concurrent shared VRAM use. This is
retained as an infrastructure failure, with zero policy controls, not a task
failure. The independent forecast is queued after all 16 source cases finish,
when only the dedicated source policy can be stopped to release its GPU.
The 800-case validation service is unaffected. Native RoboDojo rollout is
still NOT_RUN.

## What remains to test

Next: complete the prospective source collection, qualify the same student-
state label path across both tasks, materialize native batches, then freeze
a bounded matched absorption pilot using the measured training profile.
Primary outcomes are autonomous task-balanced terminal success and success
at common extra-expert budgets, with actual query/control cost, recovery,
retention and history interventions. Use paired training seeds; all methods
share data, initial weights, architecture and optimizer updates. Identical
full-history BC/DAgger/candidate views train once as aliases.

The official benchmark scores remain separate from train-side development.
No new-rule ICL, multigeneration acceleration, expert-dependence reduction,
method efficacy or top-conference readiness has been established.


Implementation validation: 35 focused tests passed, one optional bf16 test
skipped in the system Python (ml_dtypes absent). A separate real two-device
CPU JAX probe exercised float32/bfloat16 shard comparison and rejected a
changed second shard. Ruff and bytecode compilation are checked on owned
runner/helper files. These checks are code-contract evidence, not efficacy.
Ten additional query plans were committed in `2fe1e43` before their batch
started; `run_robomme_query_batch.py` explicitly assigns GPUs 4/5/7, permits no
automatic retries and records a timeout separately from task outcome.

## Subsequent same-day query completion

All ten additional registered queries completed on PatternLock train-2 and
PickHighlight train-1. Each supplied one complete 20-control chunk; all 980
additional prefix-frame checks matched exactly. Together with train-1 above:
14 queries, 1,332 exact frame checks, 274 oracle controls, 416 replayed student
controls and 14 resets; 13 complete chunks per history view. The partial
14-control terminal query remains excluded from chunk targets and included
in cost. No task or infrastructure failure was retried. Both 5-chunk views
passed the source/plan/trace/target binding validator; inputs remain strictly
causal. Native model feature materialization of the first three chunks is
running on the learning server. No trained-policy benefit is reported.
