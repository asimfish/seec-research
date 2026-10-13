"""Register the fixed E0 identities before running any scientific seeds."""

import datetime
import hashlib
import json
from pathlib import Path

from run_e0 import canonical, sha

FOLDER = Path(__file__).parent
NAMES = [
    "policies.py",
    "run_e0.py",
    "analyze_e0.py",
    "verify_e0.py",
    "test_e0.py",
    "register_e0.py",
    "PROTOCOL.md",
    "RUNTIME.json",
]


def register():
    target = FOLDER / "plan.json"
    if target.exists():
        raise FileExistsError("Existing registration is immutable")
    plan = {
        "schema": "alchemy-e0-plan-v1",
        "scope": "Zero-training domain qualification, not method efficacy",
        "upstream_revision": "68a26254b5c0f15e84fa0c15d66bf0c626ede8e0",
        "level": "alchemy/perceptual_mapping_randomized_with_random_bottleneck",
        "environment_seeds": list(range(4101, 4109)),
        "arms": ["visible_random", "bayes_reset", "bayes_carry", "privileged_greedy"],
        "native_trials": 10,
        "native_steps_per_trial": 20,
        "native_stones": 3,
        "native_potions": 12,
        "expected_actions": 6400,
        "expected_resets": 32,
        "gradient_updates": 0,
        "max_wall_seconds": 1800,
        "code_hashes": {name: sha(FOLDER / name) for name in NAMES},
        "qualification": {
            "last_five_mean_reward_gap": 2.0,
            "positive_seeds": 6,
            "first_trial_and_all_seeds_reported": True,
        },
        "full_ideal_observer": "NOT_RUN",
    }
    identity = hashlib.sha256(canonical(plan)).hexdigest()
    plan["plan_id"] = "alchemy-e0-" + identity[:16]
    plan["registered_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "plan_id": plan["plan_id"],
                "semantic_sha256": identity,
                "plan_file_sha256": sha(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    register()
