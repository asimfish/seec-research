# Native robot training jobs — 2026-10-16

The user authorized PRO 6000 resources and training. No PRO 6000 SSH alias or
scheduler endpoint is currently configured. These are prepared jobs, not
submitted or running jobs. The existing 800-case reference evaluation keeps
running independently; its model, memory buffer and service are not shared
with training or source collection.

## First job: native RoboMME optimization qualification

Request one RTX PRO 6000 **96 GB** GPU (verify the actual model/capacity),
128 GB host RAM and 80 GB scratch storage. Alternative: two available A100
80 GB GPUs with native FSDP=2. This is a qualification request, not a claim that
either configuration has measured sufficient peak VRAM. No paid rental has
been ordered. A server/scheduler entry is still needed.

The pinned native model has 3,441,899,024 parameters; 3,027,095,328 are
trainable under its official image-encoder freeze rule. Parameter, gradient,
Adam moment and EMA storage totals 62,201,121,344 bytes under the native dtype
assumptions. Activations, compiler workspace, restore/update temporaries and
allocator overhead are excluded. Peak VRAM and step speed remain unmeasured.
This explains why a shared 32 GB card cannot host this unchanged training
configuration. We do not silently substitute a tiny model or freeze all but
the memory module to produce a successful-looking training run.

Inputs: the verified released FrameSamp+Modul checkpoint, pinned upstream
`ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`, and the actual PatternLock train-0
expert correction. Each of the matched recent-history and full-history views
has seven complete 20-action chunks. Targets are identical; the current RGB,
state, prompt, backbone, native quantile normalization, optimizer, update count
and seed remain matched. Only causal memory frame selection differs. Full
history is the candidate alias, not a third novel arm. These data are too small
for an efficacy experiment and are not dense student-state DAgger labels.

Two sequential qualifications, 8 updates each, batch size 1, seed 7, native
AdamW/warmup/EMA. Reuse the same explicit GPU allocation. Preserve every loss,
gradient norm, selected row, cold and warm step time, saved checkpoint and
failure. A finite forward loss already passed on both views; gradients,
checkpoint save/reload and post-training autonomous evaluation are NOT_RUN.
The native 10,000-step warmup is retained; eight updates are not a learning
budget intended to establish improvement.

From a copied robocore checkout and a compatible isolated JAX runtime:

```bash
PYTHONPATH=. python scripts/run_robomme_native_training_qualification.py \
  --checkout /path/to/pinned/robomme_policy_learning \
  --checkpoint /path/to/verified/released-model \
  --batches /path/to/native-correction-batches \
  --view full_history_intervention_bc --gpus 0 --steps 8 --seed 7 \
  --output /path/to/new/qualification-full
```

Run the same command for `recent_history_intervention_bc` with a distinct
output directory. For two allocated A100s use `--gpus 0,1`; these are explicit
physical indices, not permission to use occupied devices. The runner requires
90,000 MiB free for one device, or 45,000 MiB per device for two, and 30 GiB free
output space before initialization. Preflight can run with `--preflight-only`.
The current local check refuses training with 24,428 MiB free. It executed
zero gradients; it is not an out-of-memory training result.

The native optimization branch and distributed save are implemented but have
not executed on qualifying hardware. The first job must verify finite
nonzero gradients, actual changed trainable weights, native checkpoint reload,
and peak VRAM/throughput before a larger training allocation is requested.
An infrastructure failure is retained and diagnosed; no rule or task
parameter is adjusted to convert the failure into a method result.

## Second job: prospective train-side failures and oracle labels

Use an independent policy instance and train episode IDs fixed before running;
never interleave memory/reset RPCs with the ongoing validation service. Expand
past the single retrospective PatternLock prefix. Qualify teacher recovery and
prefix replay for each task, then record expert queries and executed controls.
Dense DAgger/OPD labels must originate at student-visited states; expert-run
recovery suffixes remain intervention BC and cannot be renamed DAgger.

Select tasks across RoboMME memory categories from source-side feasibility,
retain failures, and keep official val/test episodes out of data generation.
Freeze exact source episodes, query rule and collection cap before collection.
The existing train-0 prefix was chosen after seeing its failure and remains a
development diagnostic, not a prospective recovery-rate estimate.

## Third job: matched absorption pilot, after data and native training qualify

Common frozen targets/backbone/initial checkpoint and equal optimizer updates:
recent-history intervention BC versus full-history intervention BC, extra
expert BC/SFT, and dense full-history DAgger/OPD. Identical data/model views
train once as aliases. Compare the frozen released model without updating it.
Use paired training seeds; determine a feasible main budget from the first
native training profile and a separately registered pilot, not from loss on
these seven chunks.

Primary: official task-balanced autonomous success from reset with no expert.
Also report success at 0/1/2/5 expert-chunk budgets, actual expert control count,
recovery from fixed prefixes, old-task retention, and all source/replay/compute
costs. Include history reset/shuffle interventions. A memory benchmark success
does not establish new-rule ICL or multi-generation acceleration. No claim of
method novelty or efficacy is established by this interface work.

RoboDojo has a separate simulation/policy job: pinned Isaac Sim 5.1,
IsaacLab/CuRobo/XPolicyLab and a matched 14D ARX checkpoint. Its 7D LIBERO
checkpoint is not a valid replacement. Keep RTX rendering on supported RTX
hardware; A100s serve the learning or RoboMME path, not unsupported Isaac RTX
rendering. Native scene and policy qualification precede its full task suite.
