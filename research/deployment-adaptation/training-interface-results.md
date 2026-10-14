# Fresh-source and eight-update training interface

2026-10-15. **Engineering PASS; method efficacy NOT_RUN.**

Two newly sampled LIBERO-10 source starts (fixed tasks 0 and 4) yielded two
successful full-program teacher rollouts. These produced **66 eight-action
anchor annotations** and 520 executed policy steps. The original QwenOFT
training forward completed **8 AdamW updates** on batches of two source anchors.
Only its original action head was trainable. Its weights changed, the entire
frozen VLM backbone fingerprint stayed identical, and the saved/reloaded head
matched the final head bytes. No development or query rollout was evaluated.

## Identities and records

- Code: `f320ce0da0ac94353bac8b902bf0751619e7aeaa`.
- Plan: `program-interface-bb0c5867d76a36a1`.
- Plan digest: `1a1eb8e700ceea23a31473d34cd00da9812b5f4ea0581e8644c0290a742b813f`.
- Reference checkpoint SHA256: `80d7a2ab0b033ce49d84070eb1124a2f9718e5b1f0ef8e2db6e4c44478b05df9`.
- Generated bank: `fc2730617885f31e4668efdcbc440b7ccb7b079e5f82ec07eed9f3d514f2012d`.
- View: `71e4832991d159caddb8c83835074465229792ebf2608fe8254029762a5f5ff0`.
- Normalizer SHA256: `64bb15c9525653c8efc476496d2a0e43d678401bc4ccc65b61fa6e314c836687`.
- Final head checkpoint SHA256: `d3b285921b28d53eee452463583bf9148eec72689e64dc48106809d8d2c58c96`.

`plan.json` is the pre-execution registration; `artifacts/` retains the original
state bank metadata, replay result, collection journal/manifest/receipt,
training metadata, every update and final result. Images, generated XML/state
arrays and the head checkpoint remain on the authorized server. The archived
upstream source still lacks an exact release Git identity. Current files do
not silently relabel this run as an official LIBERO suite score or full VLA SFT.

## Source and split qualification

Six generated starts have distinct numeric state hashes and no overlap with
the 500 reference state hashes. Source, development and query each have one
state per fixed task. Development and query states were only generated,
not labelled, trained on or policy-evaluated. Two source starts were restored
from bound XML+state in fresh environments twice each; all initial and
16-prefix-step state/camera/success signatures match exactly. Prefix
qualification uses 104 simulation steps and four logical episode resets.
This does not qualify arbitrary contact checkpoint restoration.

| Source case | Terminal success | Policy steps | Expert program queries |
|---|---|---:|---:|
| Task 0, new source start | Yes | 298 | 38 |
| Task 4, new source start | Yes | 222 | 28 |

All 520 executed commands were expert-controlled. The 66 forecasts retain
their source anchor and future-output offsets. Final partial forecasts are
masked to the executed episode length: 3,640 valid scalar action targets.
This is teacher-only data; no actual student-failure recovery occurred here.

## Optimizer and storage qualification

Seed 14015; original head, bfloat16 source precision; AdamW lr=1e-4,
betas=(0.9,0.95), eps=1e-8, weight_decay=1e-8; gradient clip 1.0; eight updates,
batch size two; final checkpoint only. 65,623,047 head parameters were trainable.
The training process reports peak allocated CUDA memory 9,890,684,416 bytes
and 66.42 seconds for its own load/hash/train/save phase; this is not total
pipeline wall time or a budget advantage.

All eight losses and gradients are finite. Losses were approximately
0.000862, 0.032409, 0.036288, 0.027280, 0.028382, 0.011041, 0.012949, 0.025529.
They came from different sampled batches and are not monotonically decreasing.
The first is near zero because the labels came from the initial released
policy. This cannot establish a learning gain; the run is an update/reload
qualification, not training to convergence. No best step is selected.

## Complete recorded cost and caveats

Generation: six logical reset episodes, no explicit control steps.
Replay qualification: four logical reset episodes, 104 steps.
Source collection: two logical reset episodes, 540 simulation steps including
20 stabilization steps; 66 expert model calls and 520 expert-controlled steps.
Training: eight model forward/backward calls and optimizer updates, no
simulation steps or extra teacher calls. Total explicit simulation steps: 644.
The reset counts denote logical case initialization, matching the reference
ledger convention; environment construction and XML model reconstruction are
additional initialization operations, not independently measured reset costs.
Their time is not asserted to be zero. GPU 7 was idle at launch and free again
at pipeline completion. Other workloads and installed environments were unchanged.

A premature archive-unpack command failed before transfer completed; it
generated no states, labels or gradients. The same completed code archive
was then unpacked and registered once. This transport attempt is recorded
in `execution.json`. There was no failed scientific run or data-driven seed change.

70 focused tests pass across the new and existing experiment/bridge boundaries.
The known unrelated legacy Planar test remains unresolved; no all-green core
verdict is claimed.

## Next scientific frontier

See `../ABSORPTION_STUDY_DRAFT.md`. A meaningful learner/expert gap and recovery
ability at actual student failures must precede five-seed absorption results.
Self-labelling the identical released policy is mostly an identity target;
it is not evidence of expert correction adding competence. Compare ordinary
BC/SFT, actual-student-state OPD and recovery under the same model/executor and
budgets. History use needs an actual context-capable learner and fresh-query
empty/shuffled-history controls. Current training bridge is single-anchor and
cannot be presented as in-context learning, generational RSI or method efficacy.
