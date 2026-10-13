# Separating factual learning from student-history coverage

This is a new internally registered diagnostic after the verified nine-model
pilot. Original parent plans, source and checkpoints remain unchanged.
No environment/task/horizon/split tuning follows this experiment. All four
arms start from the same action-only parent checkpoint for a paired seed;
they do not start from different objective-trained policies.

The competing explanations are: (a) fixed offline histories cause compounding
execution errors; (b) historical-fact supervision adds benefit even after
student-state coverage; (c) a generic extra-update budget explains the gain;
(d) the finite model/skill mapping remains inadequate. A 2x2 experiment separates
collection (offline epsilon-teacher vs actual student) from learning objective
(action-only vs action plus grounded facts). This is a DAgger-style comparator,
not an original OPD algorithm or a reproduction of GMPS.

Each arm gets two collect/update rounds, 177 training-only mapping-goal histories
per round and 300 optimizer updates per round (600 total). Every history contains
20 exploration actions at most and five 20-step execution attempts. Offline
controls use the observable teacher with fixed epsilon=0.1 and paired RNG.
Student arms use beta=0: every action, including exploration step zero, comes
from that arm's current checkpoint. Expert queries label actual visited inputs
without changing executed actions. Intermediate collectors are preserved.

All arms aggregate the same 1,062 parent histories plus 177 histories per round;
sequence-count, minibatch indices, architecture, initial weights, learning rate,
auxiliary weight and updates match. Collected lengths and action-label counts
can differ: they are explicitly reported, not called token-budget matched.
Sampling uses full causal sequences (including terminal feedback), never a
window reconstructed from only admitted rows. Auxiliary labels use only actual
prior pickup inventory responses. Hidden assignments exist in the evaluator's
environment factory, not the input or teacher function.

The parent audit found all failures eventually corrupt the irreversible bowl.
In *all* arms and old/new training histories, once the observed bowl already
contradicts the goal, action loss abstains (-100) until episode reset. These
failed histories remain as inputs and factual-supervision targets. The guard
does not erase preceding actions or observations, change termination, reset the
environment early, or imply a repair exists. This common label-validity rule is
an explicit post-parent diagnostic choice, not a change to the parent's metrics.

Adam is restarted from the common parent weights and maintained across the two
rounds; no optimizer state is imported. Batch 24 sequences, learning rate 0.003,
fact weight 0.2, norm clip 1, GRU 96 and two CPU threads match the parent. Seeds
7/19/41 are reused as paired pilot seeds, not independent confirmation. All
twelve final checkpoints are finalized before the 24-map,125-goal evaluation.
No dev selection, teacher prefix, auxiliary labels, oracle or symbolic memory
enters autonomous inference. Memory-reset evaluation uses the same checkpoints.

Primary differences are student_action minus offline_action and student_facts
minus student_action; thresholds are 5 percentage points in the seed mean and
positive direction in all three seeds. The interaction (fact gain with student
collection minus fact gain offline) is descriptive. A positive interaction is
not necessary for the baseline DAgger explanation. An advantage disappearing
under student collection eliminates the claim that the auxiliary objective is
the essential missing mechanism. All results and costs are reported.

The finite evaluated panel has been seen in prior diagnostics; this is not an
untouched test or method-superiority test. No p-values treat episodes as training
replicates. GMPS already extends DAgger to meta-learning; belief-supervised and
joint belief-imitation approaches are prior art. A novel contribution would
require a further demonstrated delta, not these parts' combination alone.

Stop after this fixed 12-run budget. Fail closed on input/identity/leakage errors;
preserve partial runs and register a code-only correction if needed. No second
set of hyperparameters is searched on this panel. Report physical actions,
expert labels, resets, prior reused source and training separately from the
additional 7,200 optimizer updates. Reset/action replay is not snapshot E4.
