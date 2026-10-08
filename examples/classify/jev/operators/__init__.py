"""Operator-contract adapters for the classify_rl rollout workers.

Each submodule exposes a backend class matching the shapes in
`examples/classify/jev/operator.schema.json`. The AReno package itself
depends on none of them; a rollout worker imports exactly the backend
it wants.

Currently shipped:
  * `sts2_sim`  — wraps the in-house pure-Python simulator under
    `examples/classify/jev/sts2_sim/`. Covers Ironclad (A0-A10) in one
    Act 1 combat per episode, with every Ironclad card and Act 1
    monster of STS2 v0.107.1. This replaced the earlier r33hab/sts2 NativeAOT shim; see
    git history for the removed `r33hab_sts2.py` if cross-referencing
    the migration.
"""
