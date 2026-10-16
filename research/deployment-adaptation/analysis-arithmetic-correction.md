# Analysis v1.1: exact success-count differences

Recorded before any trained-model validation result exists. The training
queue is initialized under the frozen training plan; data, model, seed,
optimizer, update budget, evaluation cases, thresholds and inference family
are unchanged. Its immutable 8156edb source remains the training provenance.

A synthetic regression found that `(0.4-0.3)-(0.3-0.2)` evaluates to
5.55e-17. Calling that a nonzero interaction could create six artificial
positive seeds and p=0.03125 when all six mathematical interactions are zero.
This is an arithmetic defect, not a new statistical test or threshold.

The final analyzer now computes each gain by subtracting integer terminal
success counts, and computes the interaction by subtracting those integer
gains, before dividing by 100 registered cases. Consequently a mathematical
tie is exactly zero. Bootstrap, seed unit, tie handling, two-sided sign test,
MCID and hierarchical gatekeeping remain as frozen. A regression requires
30→40 and 20→30 successes to produce equal 0.1 gains, zero interaction,
six ties and p=1. The coordinator binds the corrected analyzer fingerprint;
the original analyzer in the immutable training snapshot is retained and is
not used for the final inference. This change uses no learner outcomes.
