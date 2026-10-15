# RoboMME native training interface — 2026-10-16

Actual robot correction data now pass the pinned released FrameSamp+Modul
model's input transforms, history feature encoder, observation parser and
original training flow-matching loss. There are **zero gradient updates** and
no post-training autonomous evaluation. This qualifies inputs and forward
loss, not absorption efficacy, a memory advantage or a new method.

The successful PatternLock train-0 expert recovery provides 26 executed native
actions. Its seven complete 20-action chunks are shared by two views:
recent 32 causal frames and the official even sampling over the full causal
prefix. Both use 512 tokens. Current front/wrist RGB, state, instruction,
normalized action tensors and target IDs are checked equal. Reset-conditioning
frames are inputs only; no labels are fabricated on them or padded at terminal.
Only the first of seven chunk anchors is student-visited; this is intervention
BC, not dense DAgger/OPD. The current candidate aliases full-history BC.

The public input history's SHA-256 is
`be58794fa84874a43150972976e12ba6d1e88eeb5517c2b8ec5df8f9676d8945`;
the common raw target-set SHA-256 is
`546b590387b66202fde028d493784619135800bc077f7f899239e0f6f4ee0d70`.
Upstream policy source is `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`.
Checkpoint archive SHA-256 is
`2bfde48a0e9c616c87afcac5359b69f281689765e1af3fecbbec5c918e6faa62`.
Batch and execution metadata are in `native_training_20261016/`.

`materialize_robomme_native_corrections.py` uses the unchanged upstream vision
encoder, frame-feature packing, quantile statistics, joint delta transform,
8-to-32 dimension padding, prompt tokenizer and `compute_loss(train=True)`.
The model boundary has public cameras/state/goal and precomputed public memory;
the expert's hidden state and subgoal are excluded. The original script's
observation type requires JAX arrays for memory features; an initial NumPy-only
attempt was rejected and corrected at that parser boundary. No training ran
during the failed attempt.

Fourteen numeric NPZ batch files (7 per view) are reloadable without pickle.
bf16 vision features are stored as exactly representable float32 values;
NumPy would otherwise reload their custom dtype as opaque bytes. This has a
regression check in the actual model environment. Every file is hashed and
the future training runner verifies those hashes before initialization.

The final native forward losses are 0.01041163225 (recent) and 0.01206420176
(full history), using one fixed seed and the first matched chunk. They are
neither evaluation metrics nor a comparison of trained policies. An earlier
process produced 0.01050857455 and 0.01212229393; cross-process bitwise numeric
reproducibility is not claimed. Both views within each process share the
computed feature cache. Both runs are retained, and no loss was selected as a
model checkpoint or a headline result.

The native model contains 3,441,899,024 parameters, of which 3,027,095,328 are
trainable under its image-encoder freeze rule. The analytical parameter,
gradient, Adam-state and EMA storage total is 62,201,121,344 bytes. Activations,
compile buffers, temporary copies and allocator overhead are excluded. This
is not a measured VRAM peak. The local card had 24,428 MiB available, so the
new native 8-update runner refused training before allocating the model.
No OOM or successful gradient run is being claimed.

`run_robomme_native_training_qualification.py` uses the unchanged upstream
`init_train_state`, `train_step`, optimizer and checkpoint writer. The GPU
optimization branch is **NOT_RUN**, including FSDP and save/reload. A PRO 6000
96GB or two A100 80GB allocation is requested for its first execution; the
server/scheduler entry is pending. See `TRAINING_TASKS_20261016.md` for exact
commands, resources, acceptance criteria and the subsequent source-data and
multi-seed experimental sequence.

Verification: 88 related checks passed in the qualified CPU test runtime;
all five new boundary checks also passed in the model runtime with unrelated
pytest plugin autoload disabled. The added checks reject eval-split labels,
modified histories, future memory and changed targets; they check explicit
public input fields, exact batch roundtrips and numeric bf16 persistence.
Ruff and Python compilation are separate checks. These are software contracts,
not robot learning results.
