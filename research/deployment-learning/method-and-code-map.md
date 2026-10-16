# Deployment learning: implementation and review boundary

Design review snapshot, 2026-10-17. This is not a new registered experiment.

Current implementation at research revision b3c1473c2e8d098d8dc3e237974d2c8a76507385:
π0.5 / RoboMME FrameSamp+Modul, fixed student-state corpus, oracle action chunks,
one offline absorption round. At most 32 uniformly spanning frames versus the
most recent 32 frames, same 512-token budget. The native initial demo is part
of this history; the experiment does not isolate learning from self-feedback.
The historical full_history identifier does not mean exhaustive history.

Reuse map (research snapshot; not yet merged into main):
- robocore/adaptation/data.py: SourceTrace, SupervisedRow, target masking that
  retains observed history. Current persisted trace schema: robot-interaction-trace-v2.
- robomme_stream.py / robomme_rollout.py: native initial video, public observations,
  8D action conventions and official task outcomes.
- robomme_corrections.py / robomme_absorption.py: causal history view and isolated
  model sessions. Frame sampling/modulation itself is RoboMME's contribution.
- scripts/train_robomme_absorption.py: original native JAX training, not a second
  training framework; serve/evaluate scripts bind the native runtime.
- analyze_robomme_absorption.py / verify_robomme_absorption_trial.py: complete
  all-seed analysis and independent engineering verification.

Debug entry at that research snapshot:
python -m pytest -q tests/test_deployment_robomme_absorption.py tests/test_deployment_robomme_native_training.py tests/test_deployment_robomme_student_query.py

These are CPU contracts, not complete native GPU validation. The main repository
CI covers CPU tests, imports and registry checks. Runtime source/code and exact
plans come from the experiment supervisor, not a command to restart an old job.
Repository: https://github.com/asimfish/robocore (access may require permission).

Choose A: strong history-conditioned DAgger, plus fixed-demo and teacher-removal
diagnostics. B is an unimplemented training-only evidence-acquisition/recovery
teacher; its novelty is not established relative to A2D/CritiQ and related work.
Future comparisons must fix demo access separately from self-interaction history,
match native models/targets/updates, and account for teacher/planning cost.

G1/G2 are hypotheses with withdrawal conditions, not certified common failures.
Full source reading depths and five pending follow-ups are in the linked JSON.
No raw experiment paths, logs, checkpoints or private teacher states are included.
Method text goes through PR review. No model training was started by this update.
