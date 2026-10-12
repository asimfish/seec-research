# Interactive task learning · 2026-10-13

Start with [RESEARCH_UPDATE.md](RESEARCH_UPDATE.md). Read [PRE_REGISTRATION.md](PRE_REGISTRATION.md) before interpreting the result. This is a controlled diagnostic on DREAM Cooking, not a reproduction of its trained method.

Upstream: https://github.com/ezliu/dream at `d52204c94067641df6e6649b19abb359b87ff028`. No upstream code is vendored or changed. Upstream supplies no explicit repository license; consult its authors before redistributing it. Our adapter uses Gym 0.23.1 and restores NumPy's removed integer dtype alias; transitions, observations and rewards remain unchanged.

```bash
git clone https://github.com/ezliu/dream.git /tmp/dream-cooking
git -C /tmp/dream-cooking checkout d52204c94067641df6e6649b19abb359b87ff028
python3 -m venv --system-site-packages /tmp/cooking-runtime
/tmp/cooking-runtime/bin/pip install --no-deps gym==0.23.1 gym_notices==0.1.0
/tmp/cooking-runtime/bin/python run_cooking_diagnostic.py --upstream /tmp/dream-cooking --plan plan.json --output /tmp/new-cooking-result
/tmp/cooking-runtime/bin/python verify_cooking_diagnostic.py --upstream /tmp/dream-cooking --folder /tmp/new-cooking-result
```

`plan.json` binds the runner bytes and registered split/seeds before execution. `run1` is INVALID due to an up/down adapter error, and its original plan is preserved. `run2` is the valid result. Its observations/rewards are independently replayed, outcomes and hash chain recomputed. The source revision identifies the base checkout; runner and plan hashes identify the added experiment exactly.

The 5 seeds change a symbolic search tie order. Statistical clusters are the 24 unknown mappings; no learned model was trained. All policies solve 100% of goals. Full-history and sufficient-fact memory are deliberately identical policies on this environment. The result supports an information-retention diagnostic, not algorithmic superiority or robot learning.
