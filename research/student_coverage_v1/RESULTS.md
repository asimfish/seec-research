# Student histories × factual learning: verified result

Four arms × three paired seeds, all initialized from the same action-only parent. Two collect/update rounds, 600 additional updates each. Teacher-free final evaluation; no checkpoint selection.

| Arm | Mean success | Seeds 7 / 19 / 41 | Memory reset | New collection actions |
|---|---:|---|---:|---:|
| Offline + actions | 30.03% | 27.20% / 33.60% / 29.28% | 17.87% | 85,293 |
| Offline + facts | 60.53% | 59.36% / 65.60% / 56.64% | 21.33% | 85,293 |
| Student + actions | 81.33% | 66.72% / 94.56% / 82.72% | 67.73% | 101,383 |
| Student + facts | 78.51% | 81.92% / 82.56% / 71.04% | 63.47% | 101,328 |

| Paired contrast | Mean difference (percentage points) | Seeds (percentage points) |
|---|---:|---|
| coverage_action | +51.31 | +39.52 / +60.96 / +53.44 |
| coverage_facts | +17.97 | +22.56 / +16.96 / +14.40 |
| facts_offline | +30.51 | +32.16 / +32.00 / +27.36 |
| facts_student | -2.83 | +15.20 / -12.00 / -11.68 |

Interaction (facts gain under student collection minus offline): -33.33 points.
Registered coverage gain check: True; facts survive coverage: False.

Both checks require at least five percentage points in the seed mean and positive difference in all three seeds. A failed prediction is retained, not tuned away.

All arms apply the common observable guard: once bowl contents already contradict the goal, ignore the action-positive loss while retaining the actual history and factual targets. Parent metrics are untouched. This is not an isolated estimate of the guard benefit.

The base source contains 1,062 histories; round pools have 1,239 and 1,416 per seed. Optimizer steps, sequence batch size and batch indices match. Actual collection actions and valid labels differ and are reported. No equal-token or budget-advantage claim is made.

Coverage: 18,000 evaluation records, 4,248 new histories. Independently replayed 373,297 collected actions and 292,359 evaluation actions; all raw differences zero. Student collector actions and all final outputs recomputed from frozen checkpoint bytes.

This reuses the prior panel and training seeds; three paired seeds are descriptive. It is a DAgger-style comparator, not original GMPS, DREAM, VariBAD or OPD reproduction. Real robotics, execution uncertainty and the future-adaptation selector remain untested.
