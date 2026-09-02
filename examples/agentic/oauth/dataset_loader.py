"""Dataset loader for the OAuth agentic-RL records.

Validates each generated record and returns it unchanged (prompt included as
``prompt``).  Loaded by areno via the ``--dataset-loader-fn`` hook with the
signature ``load_training_dataset(dataset_path, *, default_loader, ...)``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import oauth_env  # noqa: E402  (needed only for the expected combo-key set)


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Validate the generator's flat JSONL records; prompts pass through."""

    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        spec = json.loads(str(record["task_spec"]))
        if int(record.get("context_budget", 0)) != oauth_env.DEFAULT_CONTEXT_BUDGET:
            raise ValueError("OAuth records must use the canonical context budget")
        if int(record.get("max_turns", 0)) < 1:
            raise ValueError("OAuth records must define a positive safety turn limit")

        combo_keys = {d.key for d in oauth_env.oauth_matrix.google_email_dimensions()}
        combo = spec.get("combo") or {}
        if set(combo) != combo_keys:
            raise ValueError("record combo axes do not match the google-email matrix")
        expected = spec.get("expected") or {}
        if expected.get("outcome") not in ("reach", "blocked"):
            raise ValueError("record task_spec carries no expected outcome ruler")
        record["prompt"] = str(record["prompt"])
        records.append(record)
    return records
