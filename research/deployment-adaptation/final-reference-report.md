# LIBERO-10 reference and expert interface qualification

2026-10-14. **Complete, zero-training reference. New-method efficacy NOT_RUN.**

The fixed released Qwen3-VL-OFT-LIBERO-4in1 checkpoint succeeds on **476/500
episodes, task-balanced success 95.2%**. All ten tasks use the same fifty
official initial-state indices, fixed seed and 520-control-step horizon.
The three task shards all exited with code zero; missing 0, aborted 0.
They are parallel execution shards, not independent training seeds. No
seed-level confidence interval or claim of new-method superiority follows.

| Task | Successes / 50 | Success rate |
|---|---:|---:|
| 0 | 48 | 96% |
| 1 | 50 | 100% |
| 2 | 50 | 100% |
| 3 | 45 | 90% |
| 4 | 45 | 90% |
| 5 | 50 | 100% |
| 6 | 42 | 84% |
| 7 | 50 | 100% |
| 8 | 48 | 96% |
| 9 | 48 | 96% |

## Identity, evidence and cost

- Plan: `libero-reference-f1770364fd96ddaa`.
- Plan digest: `9adbfd4d27a6b53412b7d68ebda209eb0cd64aaf44d0e21f9f4ddba331ac626b`.
- Execution code: `0e827270cbe9f20ebc5bdd91adcbf1d7bf4f24d0`.
- Checkpoint SHA256: `80d7a2ab0b033ce49d84070eb1124a2f9718e5b1f0ef8e2db6e4c44478b05df9`.
- Artifact audit code: `646f22c8f8062578a828aa0e31a0c3b1195c9d71`.
- Audit JSON SHA256: `0af623eb179bf5123a44a7630be07d29d5127c9c2b1ac2116226028e4b5fb1d8`.

`final_audit.json` verifies all 500 numeric trace checksums/shapes/outcomes
and all 24 failed-camera archive checksums and exact inference-input hashes.
`final_shards/` preserves the three original journals, summaries, bindings
and exit codes. Local copies match their remote audit hashes and independently
reproduce the aggregate from those journals. This is artifact verification;
it does not rerun the dynamics. The predicate recheck is the same LIBERO
predicate, not an independent V. Success-camera arrays were not retained,
although source videos and their recorded sequence hashes remain available.

| Operation | Simulation steps | Policy steps | Resets | Gradient updates |
|---|---:|---:|---:|---:|
| Full reference | 134,304 | 129,304 | 500 | 0 |
| Native adapter parity, 3 cases | 776 | 746 | 3 | 0 |
| First-action teacher projection, 3 cases | 1,186 | 1,156 | 3 | 0 |
| Full-program parity, 3 cases | 776 | 746 | 3 | 0 |
| Total, including qualifications | 137,042 | 131,952 | 509 | 0 |

Reference inference calls: 16,381. The first-action control issued 1,156 calls.
Stabilization is ten steps per reset and is never a training target.
Inference calls are not student expert interventions; this reference cannot
establish reduced expert dependence. Shard wall times were 8,332.11, 3,898.76
and 7,205.33 seconds; concurrent walls are not summed as elapsed wall time.

Both private exported source trees lack Git metadata. This is a
content-bound archived integration, with exact checkpoint-release source
revision unavailable. The fixed checkpoint was not trained or selected here.
`progress.json` and `REPORT.md` retain the earlier 80-case timestamped snapshot;
the final artifacts supersede its running status, not its recorded history.

## Control-interface finding

The three cases were registered before the control qualification: tasks
0/4/7, initial state 0. Native adapter parity and full-program parity each
match the original cameras, commands, rewards, initial states, success and
instructions exactly, 3/3. Full-program code:
`77de939d360a78873f288f07992b8c65fc7c0298`.

| Case | Original program | Fresh first action per control step |
|---|---|---|
| Task 0, state 0 | Success, 257 policy steps | Failure at 520 steps |
| Task 4, state 0 | Success, 230 policy steps | Success, 333 policy steps |
| Task 7, state 0 | Success, 259 policy steps | Success, 303 policy steps |

First camera inputs and first commands match; commands first diverge at index
1. Replanning cadence changed while the simulator control interval,
checkpoint, initial state and preprocessing stayed fixed. The projected
teacher is **TEACHER_NOT_QUALIFIED** for the single-action collector. This is
an interface diagnostic on three reused cases, not a general finding about
DAgger, robot learning or a novel mechanism. Full-program parity at nominal
starts does not prove expert strength at arbitrary off-nominal student states.
The frozen negative plan/result and prefix analysis are retained.

## Implemented continuation and remaining experiment

`programs.py` and `program_collector.py` now provide immutable, source-bound
program annotation, NPZ/JSON roundtrip, expert-only/DAgger/recovery/history
views and collection at actual student anchors. Mixing and takeover are per
program; every query and executed command is costed. Output offset j belongs
to the anchor input, not the later student observation. Final partial targets
are masked consistently. Filtering targets leaves the observations and actual
failure history intact. Eleven program tests and seven artifact-audit tests
pass; 63 focused tests across the affected experiment/bridge boundary pass.

Not yet implemented/run: a qualified program learner/trainer, collection on
new disjoint source starts, matched-budget absorption, assisted deployment
dependence, fresh-query history adaptation and generations 0–3. Controller
programs still need the upstream training normalization and image pipeline;
they must not be fed directly to a model expecting normalized actions.

All future data-use arms share their program executor, action horizon,
architecture, initialization and update budget. The 500 reference cases and
their failure frames cannot become training data for a policy subsequently
confirmed on those same cases. Fresh source/development/query state identities
and exact budgets are the next manifest. The baseline is near ceiling, so
overall success and preregistered recovery outcomes must both remain visible.
No task subgroup is selected from these evaluation outcomes for a claimed
confirmatory gain.

The old stopped carriers, negative Cooking/credit/Alchemy results and unrelated
legacy Planar CI failure remain intact. This work neither revives those routes
nor asserts an all-green repository, robot RSI efficacy or conference readiness.
