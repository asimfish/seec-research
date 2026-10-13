# Experience-credit ranking v1 — development pilot

Status: this specification becomes registered only when `plan.json` is written,
its code/parent/environment identities bind, and the run starts with that plan.
Plan identity excludes the registration timestamp; every ledger row also binds
the complete plan file SHA-256. Changing a semantic parameter changes the ID.
Do not modify registered code or overwrite output. No training has happened at
the time this document is authored.

Question: does 64-update adaptation credit rank the 300-update adaptation utility
of a fixed bank of candidate student histories better than ordinary task success,
source learning progress, cold-start training gain, total performance gain and
meta-adaptation final performance gain?

## Common substrate and prior boundary

Unmodified DREAM Cooking dynamics at d52204c94067641df6e6649b19abb359b87ff028.
Use the previous 32/8/24 rule split as source/feedback/audit, all on the training
side of this development study. None is a new sealed or confirmatory test set.
New independent learner seeds are 101/103/107. The learner, causal encoder and
observable teacher functions are reused exactly from frozen prior code; no fact
auxiliary loss. Inputs contain visible observations, goals, actual previous
actions/rewards and episode markers, not environment identity or teacher labels.

Fresh base data covers probe prefixes of 0/8/20 actions, each with observable
teacher behavior epsilon 0/.1, then six goal attempts. This supplies real cold
start examples; removing an old teacher prefix without recollection is forbidden.
1062 histories are shared across the three 800-update initial models. All initial
teacher actions are from visible causal history. Irreparably corrupted bowls have
action loss abstention but complete observation histories; they are never called
successful recovery. All labels are independently reconstructed from events.

## Candidate bank and branches

Eight preregistered recipes: one/two/three distinct ingredient rule strata with
repeat/mixed goals where valid, plus uniform repeat/mixed/all goals. 24 source
task histories per recipe. Exact task IDs are in the plan, sampled with replacement
using 17014+recipe index; duplicate histories are weights, not independent data.
Same candidate tasks for all seeds. All collectors are the corresponding initial
student, acting from step zero, no initial probe, teacher action prefix or teacher
takeover. Annotation is offline, causal and identical across candidates.

Each candidate independently branches from the same initial weights **and Adam
state** for 64 and 300 updates; long training does not continue short training.
Every update uses 12 shared base histories and 12 candidate histories, with common
sampling indices within each seed/stage. Learning rate .003, hidden width 96,
gradient clip 1, batch 24, two CPU threads, deterministic Torch algorithms.
Total 51 models, 11136 optimizer updates. Exact valid target/token/action counts
are reported, rather than pretending equal sequence counts imply equal cost.

## Evaluation and credit

Each case starts with empty recurrent memory and makes six own-policy goal
attempts under the same hidden rule. Memory persists across these resets and is
cleared before another case. No model weight updates or teacher calls during
evaluation. Each actual final response is ingested, including a last-action
pickup, before the next reset. Success uses the original terminal bowl condition.
No exploration preface precedes attempt zero.

Budget b=0/1/3/5 is the number of previous task attempts. Macro average over rule
identity, then over the one/two valid goals per rule. Feedback has 16 goal cases;
audit has 46. Q_total=mean(R0,R1,R3,R5); Q_adapt=mean(R1,R3,R5)-R0. Report all six
attempts, R0, final success and full costs; positive adaptation does not excuse
lower initial or total performance.

Short scores: source success before training; source Q_total progress; feedback
R0 gain; feedback Q_total gain; feedback R5 gain; feedback Q_adapt gain. A common
eligible set requires short feedback R0 and Q_total to each fall no more than
.05. Lexical batch ID breaks ties for every selector. Uniform is expected audit
utility of choosing uniformly among eligible candidates, not pooled-data training.
No eligible candidate is a failed feasibility result, not a reason to relax guards.
Audit never enters score or eligibility. All candidates and fixed long checkpoints
are reported, even where a selector would have rejected them.

Primary outcome is per-seed Spearman ranking against long audit Q_adapt gain,
plus audit utility of short-score-selected candidates. Ties use average ranks;
constant arrays return UNDEFINED, never zero or an artificial correlation. No
correlation is computed across seeds to manufacture extra independent samples.

The discriminating prediction passes only if mean adaptation rank correlation
is >=.3 and exceeds the strongest other score's mean by >=.1, is positive in all
three seeds, and adaptation-selected long audit gain exceeds uniform AND the
strongest other selector by >=.02 on average. Also report long R0/Q_total damage.
Failure or ties retain ordinary coverage/general meta-curriculum explanations.
This is a descriptive gate, not a hypothesis test or proof of originality.

## Costs, integrity and stop

All source/evaluation actions, resets, valid target queries, base training, short
branches, long branches, candidates rejected by guards and wall time count.
Source upper bound 1062*140 + 576*120 actions; evaluation upper bound is the plan's
history count *120. Time cap 5400 seconds. No early stopping on scientific scores.
All inputs/evaluations are hash-chained and all checkpoint weights are bound.
An independent verifier replays observations, actions, labels, memory and metrics,
checks split/plan membership, and retrains a fixed first short branch from its
initial weights/optimizer to check exact training repeatability. This is action
replay, not physics state-bundle E4 or full repeated training of all 51 models.

One fixed run. Infrastructure errors are recorded as invalid with cost; they do
not authorize score-based tuning. No full closed loop, new-rule robot efficacy,
SOAR/Meta-Teaching reproduction, budget advantage or top-conference readiness is
claimed by this experiment.
