# Program training-interface qualification, 2026-10-15

Status: registered engineering run, not an absorption efficacy study.

## Objective and fixed execution

Exercise the actual source-bound QwenOFT training forward on new LIBERO RGB
program annotations, confirm finite loss/gradients, update only the action
head, save/reload that head exactly, and keep the frozen backbone unchanged.
No teacher-free score, checkpoint selection or adaptation claim is produced.

Tasks 0 and 4 of LIBERO-10 are fixed before generation/collection; they are the
first two cases of the earlier interface ladder, not selected for failure or
gain. Generate one source, one development and one query state per task by
the original environment's placement sampler with seeds 1401500..1401505
(task order 0 then 4; within task source/development/test). Bind the resulting
state arrays and model XML. All six state hashes must be distinct, and none
may equal any of the prior 500 official initial-state hashes. Different RNG
seeds alone do not prove different states. The generated reset distribution
is an extension, not the official 500-case reference score.

Only the two source states receive teacher rollouts. Development/test states
are generated and stored but not stepped, labelled or used for training.
Each source rollout executes the complete eight-action teacher program,
with the same cameras, controller, 520-step horizon, ten stabilization steps
and success predicate as the reference. A failure is retained and is not
promoted to a successful demonstration. Collection budget: at most 1,060
simulation steps, 132 expert program calls, two collection resets. State
generation and exact restore qualification cost are recorded separately.

Before collection, each source start is restored twice in fresh environments
from its XML+numeric state and followed by the same ten stabilization actions
and 16 deterministic dummy commands. Camera sequences, numeric simulator
states and success labels must match byte-for-byte. This is qualification of
episode-start replay, not arbitrary contact-checkpoint restoration. Failure
stops collection; no tolerance is introduced to make the check pass.

## Learner and update registration

Use the content-bound released Qwen3-VL-OFT-LIBERO-4in1 checkpoint and its
normalizer. The vision/language backbone stays frozen and in eval mode.
Train its original action head via QwenOFT.forward, without substituting a
compact model. Only source expert program anchors enter the batch, current
dual images plus stable instruction, normalized seven-dimensional actions,
tail-mask expanded across dimensions. Failed student commands are never
fabricated as expert labels. No history arm is tested here.

Fixed seed 14015; AdamW lr=1e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=1e-8;
gradient clip 1.0; batch size 2 anchors sampled with replacement; **8 optimizer
updates**; no accumulation; all-eight action offsets trained unless the
registered episode-tail mask excludes them. No hyperparameter search or best
checkpoint. Save only the final head and its metadata; do not create another
10GB full checkpoint. Zero updates or finite but unchanged parameters do not
pass. Log the initial loss and all update losses, exact sampled anchor IDs,
view/normalizer/source/code identities, initial/final head hashes and final
head reload hash. Log CUDA peak memory and all model forward counts.

The effect of eight updates on this tiny corpus is not an estimate of learning
performance. A loss reduction is not a robot-success result. Formal absorption
requires a separate matched-budget manifest, multiple paired training seeds,
student-state collection, expert strength checks, development selection and
fresh query evaluation. The baseline has only one checkpoint; treating it as
500 independent training replicates is invalid.

## Failure handling

Source/content or split mismatch, missing/aborted episodes, restore mismatch,
non-finite loss/gradient, backbone update, no head update, or reload mismatch
fails this interface run. Infrastructure repair can rerun in a new output
directory with a recorded reason, preserving the failed attempt. Task,
controller and scoring parameters are unchanged. No official evaluation case
or image is used as a learning target. Do not use the query starts for smoke.

A concrete JSON manifest with full code revision, source plan/checkpoint hashes,
state-generator parameters, budgets and hyperparameters is committed/exported
before state generation and gradients. Generated artifact references are
recorded as execution receipts; the registration is never edited retrospectively.
