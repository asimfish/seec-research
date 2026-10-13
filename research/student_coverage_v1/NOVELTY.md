# Innovation triage: which delta survives?

Updated 2026-10-13. These are bounded claim searches and primary-source reading,
not proof that no paper exists. Cross-model reviewer MCP is unavailable in this
session; no independent novelty score or cross-review is invented. Paper identity
verification and reading-depth records are separate from method reproduction.

## Claims that cannot be contributions

| Proposed claim | Direct prior | Judgment |
|---|---|---|
| Learn a task belief along with an action policy | [Humplik, 2019](https://arxiv.org/abs/1905.06424), [BMIL](https://proceedings.mlr.press/v115/gangwani20a.html) | Auxiliary-belief supervision and joint belief-imitation objectives already exist. Our fixed Cooking auxiliary head is a comparator. |
| Aggregate expert corrections on student states to learn faster task adaptation | [GMPS, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/d324a0cc02881779dcda44a675fdcaaa-Paper.pdf), §4.1 | GMPS explicitly extends DAgger into meta-learning. Our two-round recurrent experiment is not a novel GMPS or OPD algorithm. |
| Separate persistent task uncertainty from temporary state uncertainty | [Akuzawa et al., L4DC 2021](https://arxiv.org/abs/2105.06660) | Hierarchical global-task/local-state belief variables are direct prior art; renaming the local variable “execution” is insufficient. |
| Use structured beliefs to resolve conflicting robot action labels and choose probing | [BEACON, September 2026 preprint](https://arxiv.org/html/2609.22730v1), §§2–4 | Explicit Bayesian filtering, belief uncertainty and latched-door probing already cover this broad story. Their configured estimator differs from learning an update rule, but that alone does not establish a new contribution. |
| Keep failed consequences without imitating failed actions | [FACT, August 2026 preprint](https://arxiv.org/html/2608.10232v1), §3 | Masking failed action losses while retaining consequence supervision is already explicit. Our irreversible-bowl guard is an integrity choice, not a contribution. |
| Distill verified local teacher corrections on student failures | [RE-0 / RE-OPD, September 2026 preprint](https://arxiv.org/html/2609.32416v1), §§3.1–3.3 | Same-state local counterfactual credit and teacher-free absorption already exist; a future-attempt selector needs a distinct supported delta. |
| Absorb multi-episode learning histories | [Algorithm Distillation, ICLR 2023](https://arxiv.org/abs/2210.14215) | Distilling a learning procedure into context adaptation is direct prior art. |
| Select/relabel/replay experience under a teacher-query budget | [ASkDAgger, TMLR 2025](https://askdagger.github.io/) | Active gating, plan feedback and prioritized replay already exist; uncertainty-based selection alone is not the delta. |

BMIL is UAI 2019 with proceedings published in PMLR 2020. FACT's project page
lists CoRL 2026; here it is labeled as a primary arXiv preprint unless the venue
record is independently checked. Do not promote an arXiv label to acceptance.

## One narrower candidate, with a falsifiable difference

**Teach the policy which observed failures should change a persistent task
hypothesis, by contrasting executed and unexecuted interventions on student
histories; absorb only updates that improve later teacher-free adaptation.**

The intended difference is the *target and admission rule*: corrective action
labels address what to do now; a posterior-update target addresses what the
agent is justified in learning from this particular execution. The experience
selector would score improvement on subsequent attempts, not just immediate
recovery, prediction fidelity, generic uncertainty or final success. Neither
piece alone is new; the proposed concrete contrastive training protocol and its
cross-rule evidence have not yet been established here.

Let $\tau$ be a task rule persistent across attempts, $z_t$ a transient execution
state, $r_t$ a student-visible receipt and $H_t$ the actual history. A required
reference is ordinary marginalized Bayesian filtering:

$$b_{t+1}(\tau)\propto b_t(\tau)\sum_z p(r_t,o_{t+1}\mid H_t,a_t,z,\tau)\,p(z\mid H_t,a_t,\tau).$$

It is not our invention. Under the *registered* assumption that an unexecuted
action yields task-independent receipt/observation likelihood, that evidence
must not collapse the task posterior. Without that assumption (e.g., a task rule
itself prevents execution), the receipt may be informative. Unknown execution
is marginalized rather than treated as a negative fact. “No effect” alone does
not determine which case occurred.

A candidate learned update $U_\theta$ would be trained on paired, physically
consistent histories at the same prior information state: an executed
discriminating intervention versus an exogenous execution omission. Training
targets must be computed from the visible likelihood/receipt, not a hidden task
ID. Perturbing a token to an impossible observation is not a valid intervention.
The policy never receives future success, hidden rule or teacher at deployment.

An experience's candidate score is the difference in future teacher-free return
or interaction cost after the supported update versus the registered no-update
reference, with current physical state and future rollout budget matched. The
score itself requires simulations and is fully charged. It is a training target,
not a causal-identification theorem. Current masks and fact labels do not
implement this selector.

## What would make the candidate carry

1. Standard hidden-rule tasks plus an explicitly labeled execution-uncertainty
   extension. Original tasks and their parameter settings remain separate.
2. Same visible receipts, teacher access, initialization, update steps and query
   cost for full-history learning, task-belief supervision, DAgger/GMPS-style
   collection, hierarchical state/task inference and structured Bayes reference.
3. Control current completion value against subsequent adaptation value; equal
   physical histories and collection cost. Include uniform replay, information
   gain and uncertainty-prioritized experience controls (ASkDAgger proximity).
4. Distinguish unknown rules/precedence graphs from unseen appearance or reset
   states. Evaluate new combinations with no task-ID cue and reset memory at each
   new task. Report trials, belief calibration and final task success together.
5. The candidate must reduce unjustified rule updates without simply refusing all
   evidence, and improve future behavior at a cost a strong inference baseline
   cannot already explain. If Bayes or ordinary DAgger explains the gain, keep
   the baseline explanation rather than inventing a new mechanism.

Recommendation: **PROCEED WITH CAUTION on the narrow candidate; no established
new algorithm yet.** The thinness is specific: its proposed filtering behavior
is ordinary Bayesian inference, its experience aggregation resembles GMPS, and
its consequence/action separation resembles FACT. A defensible paper needs an
observable additional finding or cost-effective learning mechanism that survives
those controls. The current pilot cannot supply that novelty by itself.

## Search scope and reproducibility

Queries used combinations of “task belief auxiliary imitation”, “DAgger
meta-learning student histories”, “persistent task vs transient execution
failure”, “counterfactual task belief”, “execution belief update”, and
“adaptation experience selection”, with 2025/2026 variants. Primary PMLR,
NeurIPS, arXiv and author pages were read; search aggregators were discovery
only. Existing DREAM/AD/VariBAD/RE-0 reading records remain applicable. No
restricted database or unavailable cross-model reviewer is claimed as searched.
