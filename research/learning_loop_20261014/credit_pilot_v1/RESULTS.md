# Credit-ranking development pilot — completed, gate not passed

2026-10-14. Plan `cooking-credit-5d8bf5ccf37f8781`; three new seeds 101/103/107,
eight fixed candidate batches per seed, 51 trained models, 11,136 updates.
This is a Cooking training-side diagnostic, not a full evolving producer,
SOAR/Meta-Teaching/DREAM reproduction or robot result. The finite rule family and
split have been used in earlier work. Do not merge these rates with the old
20-step-preface Cooking rates.

## Registered result

The short adaptation-gain score ranked longer-training adaptation **proxy**
utility better on average. However, every experience batch it selected violated
the registered longer-training competence guard. The fixed outcome is therefore
`NO_NEW_CREDIT_SUPPORT_RETAIN_SIMPLER_EXPLANATIONS`; no threshold, objective or
checkpoint was changed to manufacture a pass.

| Short credit | Mean Spearman, 3 seeds | Selected long audit proxy gain |
|---|---:|---:|
| Source success | 0.008 | +5.79 pp |
| Source progress | 0.169 | +9.26 pp |
| Cold-start performance gain | −0.288 | +5.79 pp |
| Total-performance gain | 0.136 | +10.19 pp |
| Final-attempt performance gain | 0.319 | +10.19 pp |
| Adaptation-gain credit | **0.537** | **+14.35 pp** |
| Uniform eligible candidate | not a score | +7.25 pp |

The adaptation correlation is positive for all seeds: 0.624 / 0.361 / 0.626.
These are descriptive correlations over eight candidates per seed, not
confirmatory significance or six independently trained methods. All scores
reuse the same candidate models. Uniform means expected value of selecting an
eligible candidate at random, not training on pooled data.

Five ranking/selection checks passed; the competence check failed. Plotting only
the above table would misrepresent this study.

## Why a larger proxy is not useful learning improvement here

The registered quantity is `Q_adapt = mean(R1,R3,R5) − R0`. An exact post-hoc
decomposition, which does not change the registered decision, gives:

| Adaptation-credit-selected batches | Before | After | Change |
|---|---:|---:|---:|
| First-attempt success | 86.11% | 72.92% | **−13.19 pp** |
| Later success, mean R1/R3/R5 | 88.89% | 90.05% | +1.16 pp |
| Adaptation proxy | 2.78 pp | 17.13 pp | +14.35 pp |
| Total performance, mean R0/R1/R3/R5 | 88.19% | 85.76% | −2.43 pp |

Identity: `delta proxy = delta later success − delta first success`.
Most of the apparent proxy increase is attributable to lower first-attempt
performance. The seed-107 selected model performs worse on **both** first and
later attempts, while its adaptation proxy increases by 12.50 pp.

First-attempt success allows the policy to interact within that episode. It has
no prior exploration preface, but is not zero-interaction accuracy. These initial
learners already perform well in this small domain; the result also cautions that
there is limited cross-attempt learning headroom.

## The guard did not transport across rule panels

The same selections were safe on short-training feedback rules but already
unsafe on short-training audit rules, before the longer 300-update training:

| Seed / selection | Short feedback first change | Short audit first change | Long audit first change |
|---|---:|---:|---:|
| 101 / batch-02 | +6.25 pp | −16.67 pp | −10.42 pp |
| 103 / batch-01 | 0.00 pp | −8.33 pp | −12.50 pp |
| 107 / batch-01 | 0.00 pp | −16.67 pp | −16.67 pp |

Consequently, the failure cannot be attributed solely to a short-versus-long
training horizon. Feedback-to-audit rule transport already fails at the short
horizon. This table is post-hoc localization from stored cases, not another pass
gate or independent experiment. We have not identified whether goal coverage,
rule strata, teacher-label density or another factor causes the transport error.

## Integrity and cost

Independent verification passed: 1,638 source histories, 3,354 evaluation
histories / 20,124 episodes, raw-observation maximum difference 0; replayed all
123,807 source actions and 241,313 evaluation actions; recomputed 41,302 student
collector actions and all 241,313 evaluation policy actions; bound 51 checkpoints.
One fixed first 64-update branch reproduced exact weights, optimizer and sampling.
This is not 51 repeated trainings or physics snapshot E4.

| Cost | Scientific execution | Additional verification |
|---|---:|---:|
| Initial / short / long optimizer updates | 2,400 / 1,536 / 7,200 | 64 |
| Environment actions | 365,120 | 365,120 |
| Resets | 30,660 | 30,660 |
| Observable-teacher label function calls | 247,614 | 247,614 |

Execution wall time was 426.71 seconds on two CPU threads. Software fixture tests
are outside these scientific execution totals. No compute or budget advantage is
claimed. All rejected candidates, label abstentions and negative seed outcomes
remain in the logs and summaries.

## Next frontier

Do not start a large closed loop using this baseline-subtracted proxy. Do not
re-score this panel with a new objective and call it a registered success.

The next domain check should use Symbolic Alchemy: chemistry stays fixed across
trials while objects change, supporting structure inference beyond repetition of
the same goal. First establish the public environment and the no-learning /
ideal-observer adaptation gap; then register absolute performance at matched
interaction cost or cumulative regret, plus retention tested across independent
rule subfamilies. Compare ordinary meta-curriculum and post-training total-return
credit again. This is a proposed next experiment, not an executed new mechanism.

The present study supports a concrete measurement/selection failure boundary.
It does not establish a novel general law, a successful recursive learner or a
method exceeding prior meta-learning work.
