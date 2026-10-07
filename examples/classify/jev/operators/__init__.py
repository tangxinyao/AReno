"""Operator-contract adapters for the classify_rl rollout workers.

Each submodule wraps a third-party STS environment behind the common contract
documented in `examples/classify/jev/operator.schema.json`. The AReno package
does not depend on any of them; a rollout worker imports exactly the backend
it wants, and gets a clear ImportError (with the build URL) when the native
pieces are missing.
"""
