# Alchemy E0 — a small memory benefit, insufficient registered gap

The fixed panel completed: 8 unique chemistries × 4 arms × 10 native trials,
6,400 actions / 32 resets, **zero gradient updates**. Each arm has identical
chemistry and trial-start item observations for a seed. Six boundary tests and
independent replay passed. The descriptive E0 gap gate **did not pass**.

| Policy | Mean trial reward | Last-five mean reward | First-trial reward |
|---|---:|---:|---:|
| Visible random | 16.3875 | 15.7500 | 17.8750 |
| Bayesian greedy, reset every trial | 22.9875 | 22.6750 | 27.1250 |
| Bayesian greedy, retain belief | 24.9000 | 23.8750 | 27.1250 |
| Privileged greedy reference | 25.4375 | 23.8750 | 30.7500 |

This is raw reward, not success percentage. Max theoretical trial reward is 45;
it is not necessarily achievable for a given graph and generated potions. The
privileged reference is local greedy control, **not an optimal ceiling**.

The registered last-five paired gap is **+1.20 reward units**, below 2.0, with
5/8 positive seeds, below 6/8. The whole-ten-trial gap +1.9125 is reported but
does not replace the registered last-five outcome.

| Chemistry seed | Carry minus reset, last-five mean |
|---|---:|
| 4101 | +0.20 |
| 4102 | −2.40 |
| 4103 | 0.00 |
| 4104 | +5.60 |
| 4105 | +0.40 |
| 4106 | 0.00 |
| 4107 | +3.00 |
| 4108 | +2.80 |

## What this instrument revealed

Belief retention works at the information level: the median number of possible
worlds at trial starts falls from 5,232 to 36 after one trial, then to 14 / 14 /
8.5 / 6 / 5 / 5 / 5 / 5. The reset policy starts each trial with thousands of
worlds. These counts are posterior diagnostics, not a learning-score replacement.

Retained-belief and privileged-greedy last-five means match **in every one of
the eight chemistries**. We therefore did not establish sufficient remaining
headroom for a new learned-credit experiment under these local decisions.
This does not show that Alchemy lacks meaningful task learning: full-horizon
planning, longer-term value of information and learned meta-policies were not
tested. An uncertain belief can already support the same myopic action, while
uncertain structure may still matter to a better planner.

Raw first-to-later curves also mix learning with changing item difficulty.
For example, retained belief earns 27.125 in trial one but averages 23.875 in
the last five, despite beating reset on matched instances. A declining curve
alone is not evidence of forgetting. Use paired same-instance comparisons and
absolute budgets, not each policy's last-minus-first contrast.

These are post-run descriptive localizations from fixed outputs, not newly
registered gates or a new theoretical discovery. No tuning, replacement seeds,
changed domain or new source reward was used to convert the E0 result into PASS.

## Integrity and budget

All 32 episodes are present, missing/aborted 0. Chemistry remains fixed across
each episode, with 8 unique hashes; every arm shares the ten trial-start item
keys. The 39-dimensional float32 observation contains no hidden chemistry.
Visible posterior updates match the official tracker in the interface fixture.
No boundary potion outcome needed an internal tracker in the scientific run
(skip count 0).

Independent replay: all 6,400 observations/action transitions and policy actions,
all rewards and 160 trial-start posterior counts reproduced exactly. Extra
verification cost: 6,400 actions / 32 resets; zero gradients. Scientific wall
time was 9.98 seconds, excluding initial precomputed-map load and software tests.
Bootstrap/fixture costs are separate software work, not claimed as free method
certification. Raw logs and original-source hashes remain available locally.

## Next frontier

Do not launch another large credit training panel using this E0 as a PASS.
First register a bounded planning calibration: compare the myopic reference with
an exact known-chemistry planner and, within an explicit search-time budget, the
official full belief-space observer on the unchanged native task. Preserve this
E0's fail and costs; report any timeout, not a selectively solved subset.

If planning leaves a meaningful adaptation gap, then register a learned pilot
with absolute return at matched action budgets, attainable-reference comparisons
and retained competence on separate rule subfamilies. A Bayesian filter or local
planner is an existing method ingredient, not our novelty. Multi-generation
experience production, teacher-free learned Alchemy efficacy, robotics and
top-conference readiness remain unestablished.
