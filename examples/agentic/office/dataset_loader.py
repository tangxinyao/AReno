"""Dataset loader for generated Office agentic-RL records."""

from __future__ import annotations

import json


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Validate the generator's flat JSONL records and keep prompts unchanged."""

    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        spec = json.loads(str(record["task_spec"]))
        if int(record.get("context_budget", 0)) != 10_000:
            raise ValueError("Office demo records must use a 10k context budget")
        if int(record.get("max_turns", 0)) < 1:
            raise ValueError("Office demo records must define a positive safety turn limit")
        if spec.get("output_file") != record.get("output_file"):
            raise ValueError("Office output_file does not match task_spec")
        record["prompt"] = str(record["prompt"])
        records.append(record)
    return records
