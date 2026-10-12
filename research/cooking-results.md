# Cooking diagnostic · actual result

Valid result: run2 only. Plan identity: `2df82215046e3d0da7b7ee4bd4f501baa94ae9d1c0bc8419a3f8026637064484`.

24 held-out hidden mappings; 125 goal instances; 5 search tie-order seeds; 6 symbolic policies; each arm 3,125 exploitation episodes. All arms complete 3,125/3,125. This is a success-rate ceiling, not evidence of a better learned method.

Reset-history: 12.88960 execution steps. Full-history and sufficient-fact memory: 10.30400. Paired saving 2.58560 steps (20.060% execution-only); the 24-mapping cluster-bootstrap interval is [2.251851851851852, 2.8225352112676054]. Including the common 20-action exploration cost amortized over 5 executions, total cost is 84.44800 vs 71.52000, a 15.309% saving. Repeated executions within a task and repeated search orders are not independent test conditions.

## Registered predictive checks

- H1: sufficient information explains the result. Full-history vs structured facts: exactly equal on 3750 corresponding records. Exact predictive residual 0. No extra mechanism earns its complexity.
- H2: terminal-success filtering is sufficient from the first execution. Falsified for action cost: first-trial cost 12.8896 vs 10.3040. It catches up on the second execution because the first successful episode supplies the facts. It is not falsified for success rate.
- H3: retry alone explains the cost drop. Reset-history stays 12.8896 for every attempt. It still reaches 100% success, so no success improvement can be attributed here.
- H4: query selection matters. Same exploration budget random policy averages 10.69568 steps vs 10.3040 directed. This tests the hand-coded query schedule, not a learned exploration algorithm.

Stop reason: discrimination exhausted for the proposed new-method advantage on this diagnostic. Structured memory and the full-history reference are behaviorally identical; do not tune this evaluated panel or shorten its horizon. Next experiment must introduce independently registered task-relevant vs irrelevant evidence and train the learning program against DREAM/AD-style references.

## Integrity and history

Coverage: 22500/22500 records, missing 0, aborted 0. Independent reset-and-action replay: 279120 steps; all observations/rewards exact. The raw maximum difference is 0. This is not snapshot restoration E4 or robotics validation.

run1 is INVALID: the adapter swapped upstream up/down action numbers. It is preserved with its plan and failure annotation. Only the adapter was corrected, and the entire registered panel rerun. The exhaustive oracle direction check covers all 64 assignments and all valid goal pairs. There was no task, horizon, split or budget tuning.

Gradient updates: 0. Teacher model calls: 0. Robot/physics/VLA executions: 0. Source/development mappings are reserved; no trained DREAM/RL²/AD baseline is claimed.
