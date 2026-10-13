# SEEC · Research in progress

**Latest research update (2026-10-13):** https://asimfish.github.io/seec-research/learning.html

Nine recurrent BC models (three optimizer seeds) are now trained and independently replayed. Observed-fact auxiliary success is 56.80%, versus 24.21% action-only and 34.29% current-state auxiliary. A separate shared-prefix feedback-redaction diagnostic reports all models, including the action-only improvement. Task-belief auxiliary learning has direct prior art; this is not new-method, OPD, RSI or robot efficacy. Six earlier hand-coded controls and all historical stops remain separate. Code, plans, results, invalid-run costs and figures are provided; raw logs and checkpoints remain local.

**New separating experiment:** twelve paired continuation runs cross student-history coverage with factual auxiliary supervision. Student-history action supervision reaches 81.33%, compared with 78.51% for student-history factual auxiliary, 30.03% offline action supervision and 60.53% offline factual supervision. The registered prediction that factual gains survive student-state coverage fails. Ordinary DAgger-style coverage is retained as the simpler explanation; it is not a new algorithm. The page preserves the earlier pilot, new costs, negative findings, and five additional direct precedents (26 checked identifiers).

**Student-Executable Experience Consolidation** studies which teacher
demonstrations can become executable supervision at the student's information
boundary. This repository is the public presentation package, separate from
the private scientific working repository.

**Website:** https://asimfish.github.io/seec-research/?lang=en&view=overview

**中文:** https://asimfish.github.io/seec-research/?lang=zh&view=overview

The site has a method overview, categorized experiment records, a research
roadmap, all 18 current manuscript figures and a downloadable ICML-style draft.
Use `lang=en|zh` and `view=overview|experiments|roadmap|resources` for direct links.
No framework, external CDN, analytics or server is required. GitHub Pages
publishes the root of the `docs/site` branch; `.nojekyll` keeps the static files.

## Current scientific status

Main efficacy **DEFERRED**. Earlier neural feasibility routes **STOPPED**.
The original five-seed pilot has zero selected held-out successes. Strong
certificate-free finite controls match SEEC retrieval on the reused panel.
The newest 600-record diagnostic supplies a successful hand-designed
known-model feedback witness, not a learned-policy or certificate-benefit result.
Historical failures and one verifier/evaluator numerical disagreement remain.

Artifact/protocol checks do not certify scientific efficacy, submission
readiness or conference acceptance. Public-task reference learners have a
different task/memory boundary and are reported separately.

## Public package and updating

- `index.html`, `styles.css`, `app.js`: static bilingual presentation.
- `data/progress.json`: curated source-linked research snapshot.
- `data/feedback-summary.json`, `data/frequency-summary.json`: bounded results.
- `data/delivery-verification.json`: artifact checks with explicit unavailable evidence.
- `assets/seec-paper.pdf`, `assets/figures/`: authored manuscript and figures.
- `publication-manifest.json`: SHA-256/size for every exported file, excluding itself.

Updates are exported with an explicit allowlist from the scientific repository,
visually checked in a browser and published to this separate repository.
No private repository history, raw training pool, checkpoint, credential or
unselected log is exported. Full raw records remain private. The manifest binds
the public snapshot to the recorded research revision; it is not a claim that
all underlying raw data are publicly available.

Presentation references: [research report](https://anonymous-report-421.github.io/public-website/?lang=en&view=),
[RoboDojo evaluation report](https://robodojo-benchmark.com/report/gpt-6-astra-eval),
[Awesome Astra Embodied AI](https://github.com/zjwzcx/Awesome-Astra-Embodied-AI).
Layout inspiration only. Their results, code and media are not our research.

**Recovered robot scenes and staged-resume audit:** https://asimfish.github.io/seec-research/scenes.html

The original reconstructed A/B/C laboratory and four complete PathTracing demonstrations are preserved. Their 20 camera videos are verified locally; the public page presents four station videos. Localized kinematic carry recovery passed, but the full-scene gate remains FAIL, nonzero velocities are discarded by the archived adapter, and a cross-process multistage physical chain is unproven. Rendering replay is distinct from dynamic continuation. No new physics or robot-learning result is claimed.
