# Symbolic Alchemy E0 — zero-training domain qualification

Written before the E0 scientific run. Registration is the exact `plan.json` plus
bound authored sources, protocol and runtime/source manifest. This is local
content registration, not an independent preregistration service.

Use the official `alchemy/perceptual_mapping_randomized_with_random_bottleneck`
at revision `68a26254b5c0f15e84fa0c15d66bf0c626ede8e0`. All source and data files
remain unchanged. Protobuf bindings are generated from unchanged definitions;
NumPy 1.23.5 preserves the original generator-to-stack API. No rotation in this
registered level; rotation generalization is not examined.

Eight environment/chemistry seeds: 4101–4108. Four arms use the same initial
chemistry and the same generated items in each of 10 trials. Every trial uses
the native 20-action limit, all original three stones and twelve potions, native
rewards and discrete actions. No end-trial shortcut or bulk settlement. Each
policy pays 200 actions / one reset per episode. Total: 32 episodes, 320 trials,
6,400 actions, 32 resets, zero gradient updates. Wall budget 1,800 seconds.
Infrastructure failures are retained as incomplete, with cost; no scientific
score changes, selective seed removal or replacement.

Arms:

1. `visible_random`: visible legal stone/potion sampling, no cross-trial memory.
2. `bayes_reset`: the official Bayesian belief update reconstructed solely from
   production observations, reset before each trial, with authored one-step
   expected cash-value decisions.
3. `bayes_carry`: the same belief update and decision rule, retained across trials
   and reset before a new chemistry/episode.
4. `privileged_greedy`: knows the actual graph and latent item effects; one-step
   greedy reference. It is **not** an optimal oracle or an upper bound.

The three nonprivileged arms receive only `symbolic_obs` (39 float32 dimensions),
their own actions, and the within-trial action index. No environment/tracker,
seed, chemistry identifier or ground truth enters their decision interface.
The Bayesian prior is the official full level prior: 8 stone maps, 48 potion
maps, 109 graphs. No true map narrows the prior. The same visible settlement rule
puts maximum-value stones into the cauldron and reserves actions for remaining
positive stones. Ties use sorted slot order.

The native environment emits a new trial observation after its 20th action.
If the old potion outcome is no longer visible, the policy skips that update
and records it; it never reconstructs a lost outcome from an internal tracker
or the next trial's stone. Visible updates are fixture-tested against the
official tracker for the same actual events.

Qualification outcome: paired carry-minus-reset mean reward per trial over the
last five trials, all eight seeds reported. The descriptive domain gate requires
mean gap >=2 reward units, positive in >=6/8 seeds, both Bayes arms above the
all-zero floor, and carry below the theoretical maximum 45 reward/trial. Gate
failure means the E0 learning gap is not established; it does not permit tuning
this panel until it passes. First-trial scores, all ten trial curves, per-seed
total reward, posterior counts, fixed action/reset costs and boundary skips are
also reported. Trials are dependent within chemistry, not independent samples.

Carry and reset can choose different actions; their contrast is a system effect
of retained belief, not an isolated causal estimate for a single observation.
The official full-horizon IdealObserverBot is **NOT_RUN**. The authored myopic
score has no explicit information bonus, and this is neither original-method
performance reproduction nor a learned meta-policy.

Independent replay verifies every raw observation, real action, reward, policy
decision, trial posterior count, plan binding and ledger link. This is action
replay, not physical snapshot E4. Verification costs are additional. A pass only
allows a new, separately registered learner pilot; it does not establish a
credit mechanism, recursive improvement, robotics efficacy or novelty.
