# Learned task-history pilot: actual results

The fixed pilot used three optimizer seeds and nine training runs. It is recurrent behavioral cloning with a matched auxiliary-label ablation, not trained DREAM/AD/RL2 or OPD. No hidden assignment, symbolic memory, or teacher is supplied to a learned policy at evaluation.

| Arm | Mean task success | Three seed success rates | Cost steps (failure = 20) | Reset-memory success |
|---|---:|---|---:|---:|
| Action-only history BC | 24.21% | 27.84%, 24.80%, 20.00% | 17.2853 | 18.67% |
| Current-state auxiliary | 34.29% | 34.56%, 36.48%, 31.84% | 16.5515 | 18.13% |
| Observed-fact auxiliary | 56.80% | 57.92%, 60.80%, 51.68% | 14.4139 | 24.00% |
| Symbolic full-history reference | 100.00% | 100.00%, 100.00%, 100.00% | 10.3040 | — |

The primary metric averages the five task attempts. Success-by-attempt curves and each seed are retained in the JSON. The same 20-action exploration cap and 20-step execution horizon apply to every arm. All exploration actions come from each learned policy; there is no teacher prefix. Source is the same registered teacher/perturbed history pool, so this is offline supervision rather than student-state distillation.

Observed-fact auxiliary minus action-only: +32.59 percentage points; minus current-state auxiliary: +22.51 points. Paired seed differences are [0.30080000000000007, 0.36, 0.3168] and [0.23360000000000003, 0.24319999999999997, 0.19840000000000002].

Registered H1 (>=5 points over both, positive at every seed, memory drop >=5 points): **SUPPORTED IN THIS PILOT**. All-arm <20% floor: False. Stop: registered-budget-completed; no horizon, loss or checkpoint tuning on this panel.

The first run was interrupted for a terminal-feedback input bug and is INVALID. Its plan, code snapshot and partial source/training remain preserved. The valid run adds real terminal feedback-only tokens before episode resets, without executing extra simulator actions or changing task settings. All final checkpoint bytes were saved before development and test evaluation. Each recorded action, observation, reward, outcome, cost and aggregate has been independently recomputed.

Three seeds do not support confirmatory inference. The mapping panel reuses the prior symbolic diagnostic split, so it is not an independent confirmatory test set. Symbolic reference numbers are skill/memory upper references, not another trained model. A grounded auxiliary benefit would still require stronger trained DREAM/AD/Bayes and student-state collection controls; a failure does not prove that task-rule learning is unnecessary.

Plan: `cooking-learned-v1-29bf00c0b57b1c9f`; source tokens: 86,276; actual source steps: 79,904; optimizer updates: 7,200; model decisions independently checked: 295,280.

## Matched-history follow-up (registered after the primary result)

The nine fixed checkpoints receive the same real 20-step prefix; only prefix inventory feedback is redacted at the policy input. Execution observations remain original. This is a supplied-prefix diagnostic, not primary teacher-free evaluation. No checkpoint or optimizer update changes.

| Model | Shared real prefix | Inventory feedback redacted | Real minus redacted |
|---|---:|---:|---:|
| Action-only | 24.21% | 26.99% | −2.77 pp |
| Current-state auxiliary | 34.29% | 32.91% | +1.39 pp |
| Observed-fact auxiliary | 56.80% | 29.81% | +26.99 pp |

All 1,125 autonomous exploration prefixes equal the common prefix. The primary differences therefore do not come from collecting different exploration information on these cases. Redaction collapses the 125 prefix input sequences to one. Independent replay checks 11,250 records and 185,786 policy decisions, with raw observation maximum difference zero. All nine checkpoint identities are unchanged.

The fact-supervised model uses historical feedback; removing it substantially hurts. Redaction slightly improves the action-only model, so information retention is not universally beneficial to these trained policies. The intervention changes the input distribution and was registered after primary results. It does not establish new causal-credit theory or auxiliary-learning novelty. Humplik et al. (2019, arXiv:1905.06424), VariBAD (ICLR 2020) and UNREAL (ICLR 2017) are direct prior art.
