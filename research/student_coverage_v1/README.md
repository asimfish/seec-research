# Student-history coverage versus factual supervision

The new 2x2 experiment continues from the **same action-only parent weights**
for each paired seed. It is a DAgger-style baseline, not a new OPD/GMPS method.
Read `PROTOCOL.md`, immutable `plan.json`, `RESULTS.md`, `parent_audit.json` and
`NOVELTY.md`. Parent histories/checkpoints and task transitions are unchanged.
The new common action-label guard is explicitly registered before this run.

Four arms × three paired seeds; two collection/update rounds; 177 training-only
histories per round; 300 updates per round. Every history retains the full actual
input sequence and terminal feedback. Teacher labels use only actual observed
pickups, with no hidden rule or task ID. Inference controls from the initial step,
with no teacher prefix, factual labels or symbolic memory.

Python 3.10.12, NumPy 2.2.6, PyTorch 2.13.0+cu130 on CPU, Gym 0.23.1 and
gym_notices 0.1.0. The pinned official DREAM revision is
`d52204c94067641df6e6649b19abb359b87ff028`. Upstream code is not redistributed.
The registration is internal timestamp/hash binding, not an external service.

From this authored folder (or the exact public copy), after reproducing the
parent learned pilot described in the sibling `learning-pilot/README.md`:

```bash
python run_coverage.py --parent /path/to/verified-parent-run \
  --upstream /path/to/pinned/dream --plan plan.json --output /path/to/new-run
python verify_coverage.py --parent /path/to/verified-parent-run \
  --upstream /path/to/pinned/dream --folder /path/to/new-run
python analyze_coverage.py --folder /path/to/new-run
```

The public folder is `research/student_coverage_v1`. The exact parent policy
module is additionally copied to `research/learner_pilot_v1/` to preserve this
runner's immutable relative import and plan hashes. `research/learning-pilot/`
retains the primary pilot code, plan and reports. No source pools or checkpoints
are uploaded; regenerate them from the fixed parent plan. For unit tests, place
the pinned checkout at `/tmp/robot-task-dream-20261013`, then run from the public
repository root:

```bash
python -m pytest -q research/student_coverage_v1/test_coverage.py
```

Costs: 373,297 new collection actions, 25,488 collection resets, 7,200 additional
optimizer updates, 292,359 evaluation actions and 18,000 evaluation resets.
The prior 79,904 source actions and parent training are reused, not free new
collection. Matching updates/sequences is not matching physical actions or
tokens. The summary field `expert_action_queries` denotes candidate action-label
sites (including abstentions); the verifier separately counts 315,911 valid
correction labels and 57,386 ignored corrupted-state sites. These are local
observable teacher computations, not measured external API/human queries.
Offline collector teacher decisions are additional and not labeled as costless
human supervision. No budget advantage is claimed.

All actual collection and evaluation actions are independently reset/replayed;
student collector checkpoints and final checkpoint outputs are reconstructed.
This is not optimizer reproducibility, snapshot E4, robot efficacy or ICML
readiness. The three seeds and finite evaluation panel reuse the parent diagnostic;
no new hyperparameters or horizon are tuned after reading results.
