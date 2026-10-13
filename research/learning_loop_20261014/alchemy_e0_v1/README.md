# Symbolic Alchemy E0 qualification

This is a zero-training diagnostic, not a trained meta-policy or robot result.
See the immutable `PROTOCOL.md`, `plan.json` and `RUNTIME.json` before using a
number. Plan: `alchemy-e0-321d904e1a98af37`; exact plan SHA-256:
`43f936f78c66b284adb5b8589f147eba72aae57ac2ba19dcd67161215a6d74a3`.

Official source: https://github.com/google-deepmind/dm_alchemy at
`68a26254b5c0f15e84fa0c15d66bf0c626ede8e0`.
All 140 tracked source/data files are byte-identical in the build copy. The
generated protobuf bindings are additional files; no source transition, graph,
reward, object generator, observation or action rule was modified.

## Reproduce the isolated runtime

Use Python 3.10.12 and a fresh dependency target. These commands install into a
task-specific directory, not the user's global Python environment:

```sh
python3 -m pip --isolated install --target ALCHEMY_DEPS \
  numpy==1.23.5 scipy==1.14.1 absl-py==2.1.0 dm-env==1.6 \
  dm-env-rpc==1.1.7 dm-tree==0.1.8 docker==7.1.0 \
  grpcio==1.71.0 grpcio-tools==1.71.0 protobuf==5.29.5 \
  googleapis-common-protos==1.70.0 portpicker==1.6.0 frozendict==2.4.6
```

Make an archive copy of the pinned official checkout as `ALCHEMY_BUILD`. Generate
bindings with the delivered helper. It checks every original file, reads the
exact ten paths from unchanged `setup.py`, and checks generated identities:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=ALCHEMY_DEPS \
  python3 research/learning_loop_20261014/alchemy_e0_v1/build_bindings.py --build ALCHEMY_BUILD
```

Do not run setup or compilation inside the original read-only checkout.
Generated file identities and every original file are recorded in `RUNTIME.json`.

The runtime does not start Docker, Unity or a GPU; their Python imports are
required by the original package initializer. Bootstrap failures and resolutions
are recorded in the runtime manifest. In particular, NumPy 1.26 rejects a
generator-to-stack call in the old official code; pinning 1.23.5 preserves it
without a source patch. RPC 1.1.7 requires the protobuf 5.29 generated-code API.

From the authored repository root, replace the directory placeholders:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=ALCHEMY_BUILD:ALCHEMY_DEPS:research/learning_loop_20261014/alchemy_e0_v1 \
  python3 research/learning_loop_20261014/alchemy_e0_v1/test_e0.py
PYTHONNOUSERSITE=1 PYTHONPATH=ALCHEMY_BUILD:ALCHEMY_DEPS:research/learning_loop_20261014/alchemy_e0_v1 \
  python3 research/learning_loop_20261014/alchemy_e0_v1/run_e0.py --output FRESH_OUTPUT
PYTHONNOUSERSITE=1 PYTHONPATH=ALCHEMY_BUILD:ALCHEMY_DEPS:research/learning_loop_20261014/alchemy_e0_v1 \
  python3 research/learning_loop_20261014/alchemy_e0_v1/analyze_e0.py --run FRESH_OUTPUT
PYTHONNOUSERSITE=1 PYTHONPATH=ALCHEMY_BUILD:ALCHEMY_DEPS:research/learning_loop_20261014/alchemy_e0_v1 \
  python3 research/learning_loop_20261014/alchemy_e0_v1/verify_e0.py --run FRESH_OUTPUT
```

Never overwrite the original run or re-register its changed sources. Exact raw
replay uses float32 production observations, native rewards and the full pinned
environment. Public outputs contain summaries, analysis, verification, authored
adapters and the registered content; raw event logs stay local. Runtime validation
enforces Python/NumPy versions and every original source / generated binding
hash; the other active dependency versions are recorded for reproduction.

## Interpretation boundary

`bayes_reset` and `bayes_carry` reconstruct the official belief filter from
visible observations, with our **one-step expected cash-value** decisions.
The full-horizon official IdealObserverBot has not run. `privileged_greedy`
knows the actual latent effects but is not an optimal oracle or a ceiling.
Eight environment seeds, all ten trial returns and paired differences are kept.
Qualification FAIL is distinct from software/replay PASS. Earlier Cooking
credit failure remains unchanged.
