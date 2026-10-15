Latest follow-up (2026-10-16): [learner-prefix expert recovery and validation continuation](RECOVERY_RESULTS.md). The earlier launch below did not finish; retain its infrastructure aborts.

# RoboMME closed-loop qualification and validation launch

2026-10-15. Fixed released FrameSamp+Modul, seed 7, zero gradients.
This is reference execution evidence, not new-method efficacy.

| Task | Split / episode | Official outcome | Control steps |
|---|---|---|---:|
| PickXtimes | train / 0 | success | 512 |
| VideoUnmask | train / 0 | success | 178 |
| PickHighlight | train / 0 | success | 312 |
| PatternLock | train / 0 | fail | 68 |

All four registered cases completed. No missing or infrastructure aborts in the
completed run. Total 1,070 control steps, 69 predicted chunks, 1,188 public
frames. The read-only audit hashed 81 artifacts and checked every executed
action against its source chunk and every stored public frame. It did not
independently re-simulate these episodes. Initial demonstration simulation
inside reset remains uninstrumented and is not counted as zero.

The first connection attempt stopped before reset because the service was not
ready. The first actual rollout attempt then stopped before the first action:
a missing `add_buffer` wire flag made the upstream server treat the memory
packet as inference. The source assertion and failed receipt are retained.
A regression now checks the flag. A fresh plan and output directory were used;
no task failure was replaced by a successful retry.

Completed source plan: robomme-reference-2bd7ed18a9e44765, SHA-256
75df454a53ef68323feb2c476e81e4840c7c24310ac60788f280403c11aed4c8.
Execution source snapshot: 77b0344b439aa712987427f1dcac40986688f57a;
its six implementation-file hashes match main commit
938b8cc729b91b9db4473c13547359f08768b0ac. The snapshot was made on local
storage while a NAS Git commit waited on filesystem I/O; the source bundle
and parent research identity are retained.

The model is the immutable official checkpoint already fully SHA verified.
Policy runs in an isolated local RTX 5090 environment and simulation on 30109
GPU 1, through a private loopback SSH forward. JAX 0.5.3, Flax 0.10.2,
Orbax 0.11.13, TensorStore 0.1.74, NumPy 1.26.0; the simulator is the previously
qualified Torch 2.6 compatibility environment. This is a pinned-source
compatibility evaluation, not an exact release-runtime reproduction.
The benchmark is fae9f7300347f1146c2ae4844faa4eef7e38098e and the
policy source is ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b. That policy
repository pins an older benchmark submodule, 856bc3a189d4172f3f47dbee4424d585f8d78db3;
this run therefore does not reproduce the paper-source benchmark revision.

Cross-process forecasts before persistent compilation caching differed by up
to 0.003417 action units on the four fixed inputs, with no gripper-sign changes.
The next process using the persistent compilation cache reproduced the four
cached forecast file hashes. These are bounded observations, not a universal
bitwise-determinism claim. Evaluation seed 7 is not a training replicate.

The separate all-task validation plan is
robomme-reference-e0ad4d742f53cd15, SHA-256
15cb16087a715842b765bedbd408492b30f4185e31f643b98a2a365a6392bfbb:
16 tasks × 50 val episodes = 800. The source qualification audit is required
before its launcher starts. The fixed 1300-control-step limit, 20-step
prediction / 16-step execution, observation boundary and model are unchanged.
The supervisor owns and closes its policy service and SSH tunnel. Test is
untouched; no correction absorption, new training, or method benefit is run.
Ongoing coverage is in validation-progress.json; no partial full-suite success
rate is reported. This website is a timestamped snapshot, not a live scheduler.

RoboDojo's entire asset cache is hash verified. A 464,495,655-byte subset for the
previously chosen stack_bowls scene was checked with USD core. The camera-stand
asset references a developer's absolute texture path and remote MDL modules;
Isaac material resolution and physics remain pending. No substitute material,
14D control result or RoboDojo success score is claimed.
