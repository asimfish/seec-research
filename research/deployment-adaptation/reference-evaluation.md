# Released-policy reference evaluation — 2026-10-14

Purpose: establish a learned robot reference on LIBERO-10 using the released
Qwen3-VL-OFT-LIBERO-4in1 checkpoint already present on authorized node 30109.
This precedes a concrete post-training comparison; it is not a proposed-method
result, an independent training reproduction, unknown-rule ICL, or evidence
for recursive improvement.

Register before evaluation: all ten tasks, official initial-state indices
0–49 per task, environment seed 0, inference-server seed 0, 520 policy steps
plus ten upstream stabilization steps. No checkpoint selection. An exported
StarVLA/LIBERO source without `.git` is identified by all copied content hashes,
not assigned an upstream revision. The existing environments are read-only;
their package snapshots and the checkpoint/base-model hashes are recorded.

Three processes visit disjoint task sets [0,1,2,3], [4,5,6], [7,8,9] on GPUs
4,5,7, after an idle-resource check. Each invokes the original evaluator with
the original image rotation, PIL resize, eight-action cache, action
unnormalization, gripper conversion and LIBERO success predicate. Wrappers
record inputs, commands, outcomes and cost. The standard ten stabilization
actions are explicitly counted and are not student decisions. These processes
are evaluation shards, not three statistical training seeds.

Every case produces a metadata record and low-dimensional controller-command
trace; all failures additionally retain both pre-resize camera streams. Videos
use the upstream path. No failure is selected for retention by task or outcome
severity. An eight-GiB free-space reserve stops a shard; such a run is incomplete.
Source-binding, runtime or simulator errors are recorded rather than relabeled
as task failures. No full-suite success rate is emitted with missing or aborted
cases. Report per-task and task-balanced success, inference requests,
simulation steps, resets, wall time, and zero updates/expert queries. Checking
`done` against the same LIBERO predicate is not an independent verifier.

The checkpoint trained on these task families; nominal performance cannot be
called transfer or adaptation. Failure trajectories are development material
for specifying later recovery cohorts. A fresh, source-disjoint query set and
paired training-seed post-training study require their own plan. The legacy
Planar stop rules and negative Cooking/credit/Alchemy results are unchanged.

Execution tools: `scripts/prepare_deployment_reference_snapshot.py` creates a
write-once source snapshot and bound plan; `robocore.adaptation.reference_audit`
instruments its evaluator. `summarize` rejects duplicate/unregistered cases,
counts simulator aborts separately and requires exact registered coverage.
