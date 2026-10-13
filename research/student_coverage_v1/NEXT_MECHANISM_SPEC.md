# Next separating test: which failure is evidence about a rule?

Status: design specification, **NOT_RUN / NOT_PREREGISTERED**. This is not a
new result or an approved main experiment. The completed Cooking experiment is
unchanged. A future extension must have its own build, plan, costs and stopping
rules before training. No hidden rule is supplied at deployment.

## Concrete question

A robot attempts to open a door and observes no movement. A persistent latch
rule may require a preceding operation, or the attempted operation may not have
been executed because of a temporary actuator interruption. The next action
may be the same retry in both cases; what should be retained across later
attempts is different. A terminal failure label cannot identify either cause.

Use a minimal public-task extension before physics: a persistent prerequisite
rule and a temporary execution omission, with student-visible operation
receipts. Do not silently add omissions to the original DREAM benchmark or call
new prerequisite combinations appearance generalization. A valid test must also
contain executed informative failures; refusing every negative observation is
not the desired behavior.

## Proposed learned object and nearest alternatives

Let `H` be the actual observed prefix, `tau` the persistent rule, `z` a temporary
execution condition, and `r` an observable receipt. The learner receives only
the same history and receipt as all comparators. It proposes a distributional
task update, not a hidden ground-truth label or a teacher action at inference.

Ordinary Bayes marginalizing `z` is the required strong reference. If omission
is exogenous and its visible evidence is independent of `tau`, its task update
is the identity. If the rule itself prevents execution, omission can be
informative. If execution is unknown, marginalize it. These are ordinary
inference properties, not our invention.

A candidate training target contrasts two physically consistent branches from
the same prior information: executed discriminating intervention and exogenous
unexecuted intervention. It must preserve true observations and reset semantics;
an impossible token substitution is not an execution counterfactual. Use a
visible-likelihood posterior target, never the sampled hidden rule as input.
The contrastive update loss must be compared with ordinary posterior auxiliary
supervision; otherwise it merely renames Humplik/BMIL-style belief training.

For experience selection, freeze the current policy and compare future return
after its supported memory update with a registered no-update memory branch.
Physical state, future goals, randomness, action budget and teacher access must
match. Exclude the current attempt from this future-return target and separately
record its completion value. Charge every reference branch. This is a proposed
training signal, not an established causal theorem or guaranteed improvement.

## Required controls and measurements

1. Full-history action imitation, the current DAgger-style correction method,
   ordinary task-posterior auxiliary learning, task/local-state inference and
   structured observable Bayes. Give all of them identical receipt information.
2. Withhold contrastive update targets while keeping model size, actual student
   histories and update count fixed. Frozen common histories separate target
   effects from the system-level occupancy effect seen in the completed pilot.
3. For selection, compare uniform replay, uncertainty/information gain, current
   repair advantage (RE-OPD proximity), and future-attempt return. Equalize total
   physical and annotation budgets; count discarded verification branches.
4. Report unjustified task-posterior collapse after exogenous omissions and
   failure to update after executed informative interventions, alongside belief
   calibration, attempts 1/2/3/5, final success and all costs. Neither calibrated
   beliefs without behavior nor behavior without a separating control suffices.
5. Hold out rule combinations and prerequisite graphs, with no task-ID cue;
   reset persistent memory between tasks and retain it between attempts. Use a
   new untouched panel and new training seeds after the exploratory pilots.

## Advancement and rejection

First check the observable receipt and identifiability oracle without training.
If Bayes cannot distinguish the registered branches from the visible data,
reject that test rather than adding hidden labels to rescue it. If the learner
only benefits from the receipt or ordinary DAgger coverage, retain that baseline
explanation. If ordinary posterior supervision matches the contrastive target,
there is no demonstrated contribution from that loss. If future-credit
selection does not improve later behavior at matched total cost, reject it as
the core mechanism. A useful engineering system alone is not proof of method
novelty or recursive acceleration.

The nearest priors and their overlap are in `NOVELTY.md`. Implementation,
registered budgets, trained comparisons, new-rule transfer and robot validation
are all still NOT_RUN for this candidate.
