# Deployment corrections: first robot absorption study

2026-10-15. Draft, **NOT REGISTERED / NOT RUN**. The eight-update interface run
does not instantiate this study. Choices below are candidates that must be
resolved in a complete manifest before method-result collection or training.

## Scientific question and the two effects

After receiving deployment failures and expert corrections, does a policy's
next version (i) complete and recover autonomously more often and (ii) make
better use of its own subsequent interaction history? These are different
effects. Ordinary data aggregation can explain the first without the second;
having an external teacher solve a failed attempt proves neither.

For deployment condition z, let executed support history be
H_b={(o,a,r,termination)} over b attempts and let the policy produce an action
program A_t=pi_theta(o_t,H_b). The slow update theta'=U(theta,D) uses source
data only. A query rollout starts at a fresh state, with theta' frozen and no
expert. Measure task-balanced query success S(theta',b), including b=0. The
history contrast is S(theta',b;valid H)-S(theta',b;empty/shuffled H), with the
same support budget and input capacity. The absorption contrast compares
theta' across data recipes using paired training seeds. Improved retries on
the same start are not sufficient for transferable history adaptation.

Expert outputs are complete eight-action programs conditioned on the actual
student anchor. Output offset j is not a state-local label at time t+j.
Execution cadence and the program-level mixing rule are shared by all arms.
The projection diagnostic is an executor check, not the scientific mechanism.

## Baselines and what can be isolated

| Comparator | Training target/data operation | Interpretation |
|---|---|---|
| Fixed released policy | No update | Starting competence and executor control |
| Additional expert BC/SFT | Source expert rollouts with the same query/update allowance | More ordinary demonstrations |
| Actual-student-state DAgger/OPD | Labels at genuinely visited student program anchors | Student state coverage |
| Recovery suffix BC | Student prefix preserved, expert takeover targets | Corrective state coverage, not full PLD residual RL |
| History-conditioned correction | Same target programs with executed failure/support history | History use; needs an actual context-capable model |
| Context interventions | Empty, valid, shuffled, unrelated histories on the same trained model | Whether success uses informative own feedback |

DAgger/OPD semantics reuse upstream code concepts; a local adapter is not a
full published-baseline reproduction. PLD, RECAP and RoboTTT need separately
identified compatible source implementations and accounting. The current
QwenOFT training bridge has current-anchor RGB+language only and explicitly
rejects history rows. It cannot be reported as a history-conditioned arm.

Before comparing history organization, all relevant arms must share the same
model, context capacity and target loss. A larger memory model beating a
smaller BC model would conflate architecture and data. Identical views are
aliases, not separately trained comparisons. No new method's novelty follows
from adding familiar OPD, memory and recovery components.

## Benchmark and source distribution

First domain: LIBERO manipulation with official tasks and success predicates,
full original horizons/cameras/controller/action executor. All ten LIBERO-10
tasks stay in the first study; task subsets cannot be chosen from the 95.2%
reference outcomes to create a favorable difference. The original fixed
checkpoint's exact release-source commit is unavailable, so that fidelity
limit remains visible. Extend to Spatial/Object/Goal and a second domain
(SimplerEnv) once one-round absorption is qualified.

Use a generated, content-bound source/development/query state bank, checking
XML, arrays and split overlap. Report this generated-start protocol separately
from the official 500-case score. No prior reference case or its failure image
may become training data for subsequent confirmation on that case. A source
sampler is frozen before outcomes. Arbitrary contact checkpoint recovery is
not assumed: replay a recorded prefix from a qualified initial state and count
its cost. Keep full-from-step-zero rollouts and conditional recovery separate.

Proposed starting study scale: all ten tasks; eight source, four development
and eight fresh query starts per task; five paired training seeds. These counts
are a feasibility proposal, not permission to silently select or replace hard
states. Initial-state content, collection/action budgets, expert identity,
checkpoint rule and exact seed values need the final manifest. A short warmup
must demonstrate a usable student before failure collection; a randomly reset
head is not labelled an already deployed competent robot.

## Critical teacher/student check

Read-only source inventory on 2026-10-15 found an existing converted LIBERO-10
demonstration corpus: metadata reports 379 episodes and 101,469 frames at 20 Hz;
the episode manifest sums agree. This is a candidate for a conventional
low-data BC student, not yet a qualified training dataset. Its task ordinals
are a permutation of the benchmark ordinals (dataset 0 is benchmark 4,
dataset 5 is benchmark 0); match exact instructions before task selection or
per-task metrics. Do not feed dataset ordinals directly into environment resets.
The no-noops conversion's timing/targets, selected video/parquet checksums,
provenance and original simulator-state overlap still need auditing. Its
converted metadata does not contain full source simulator start arrays. The
inventory is training-use NOT_RUN and no pickle index was loaded.

Only one released teacher checkpoint is currently qualified. Initializing the
student to that exact same deterministic policy and querying it as its own
teacher at the same anchors creates mostly identity targets. That is not a
valid test of whether expert correction adds missing competence. Before an
absorption efficacy run, register an independently meaningful student/expert
gap: a standard BC learner from a fixed low-data pool, a distinct weaker
released checkpoint, or a documented limited-capacity student. Test the
teacher's recovery competence at actual source student failures. A teacher's
nominal 95.2% cannot substitute for this recovery check. The architecture/
initialization choice is shared across the data-use arms, not selected after
looking at their query results.

The current eight-update run retains the pretrained head precisely because it
qualifies its optimizer/storage interface, not because self-labelling the
released policy would prove correction efficacy. If suitable expert recovery
does not exist, preserve that negative result and reconsider this teacher or
benchmark pairing before spending on five-seed method training.

## Primary metrics and cost

- Primary: task-balanced teacher-free terminal success under the frozen horizon,
  per task and training seed, paired seed-level contrasts. Do not count episodes
  as independent training replicates. Report all seeds and aborted coverage.
- Recovery: success after a preregistered failure prefix, alongside complete
  policy-controlled success from step zero. Do not choose only recoverable failures.
- Adaptation: absolute success at support budgets 0/1/2/3/5, fresh-query success,
  cumulative successes and empty/shuffled-history contrasts. Do not make
  later-minus-first slope the main metric; decreasing initial competence can
  manufacture improvement room, as the prior credit diagnostic demonstrated.
- Assisted dependence: success versus a common expert-control budget, queries
  and fraction of policy steps executed by expert. Zero teacher queries during
  teacher-free evaluation is tautological, not reduced dependence. The public
  takeover rule and expert must be shared across arms and fixed before outcomes.
- Retention: old-task success after absorption, with paired evaluation cases.
- Efficiency: total collection/prefix replay/stabilization steps, resets,
  attempted/rejected annotations, model calls, optimizer updates, compute and
  storage. Align optimizer updates rather than epochs and report admitted data.

Success times include censoring for failures. Both raw successful durations
and failure rates remain visible; a policy that fails quickly is not faster.
The released baseline is near ceiling. Development-only floor/ceiling checks
can determine feasibility, but test difficulty and thresholds cannot be tuned
until a desired gain appears. Method and full-suite results must retain the
overall outcome beside any registered recovery cohort.

## Sequence and interpretation

1. Exact initial-state replay, program labels/transforms, true upstream forward,
   finite head updates and strict reload. This is the current engineering run.
2. Meaningful learner/expert pairing and source-failure recovery audit.
3. Frozen one-round, matched-budget pilot with all seeds and ordinary BC/OPD
   controls; new method effects are absent until those results exist.
4. Fresh-query own-history contrasts, using a qualified context learner.
5. Only if prior effects exist: generations 0/1/2/3 versus fixed data and ordinary
   OPD, with new query cohorts, old-skill retention and total cost.

If ordinary coverage explains the gain, say so. If own-history contrasts are
zero, say absorption works without evidence of improved learning from feedback.
If expert assistance solves failures but the next autonomous policy does not,
that locates an absorption failure rather than proving the closed loop works.
All earlier stopped carriers and negative experiments remain part of the record.
