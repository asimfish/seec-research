# Current frontier — 14 October 2026

Alchemy E0 is complete and independently replayed. Plan:
`alchemy-e0-321d904e1a98af37`. **Qualification gate FAIL; replay PASS.**

8 unique chemistries, 32 episodes, 320 native trials, 6,400 environment actions,
32 resets, no gradients. Retained-minus-reset last-five mean reward: +1.20,
positive in 5/8 seeds. Registered requirements were >=2.0 and >=6/8.
Keep that fail; the whole-ten-trial +1.9125 cannot replace the primary outcome.

The visible belief filter is functional: its posterior shrinks from thousands
to a median five possible worlds by later trials. Yet retained-belief and the
privileged one-step greedy reference match in every seed's last-five mean.
No meaningful learning headroom was established under this decision rule.
The privileged reference is not an optimal ceiling; full belief-space planning
has not run. Do not conclude that Alchemy itself lacks task learning.

Next is a separately registered, bounded planning calibration on the unchanged
native task: exact known-chemistry planning and full official ideal-observer
search with explicit compute/timeout accounting. Only after a useful gap exists
should another credit-learning panel be registered. No neural Alchemy training,
multi-generation learning loop, original-method reproduction or robot efficacy.

Read `RESULTS.md`, `run1/analysis.json`, `run1/verification.json`. Frozen sources,
plan, protocol and runtime remain unchanged; all prior Cooking results and the
credit retention failure remain separate. Existing robot scenes and historical
Lift/Planar stops remain preserved.
