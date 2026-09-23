"""Dataset loader for the Ling-3.0-flash-Fin knowledge-injection SFT rows."""

from __future__ import annotations

from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load train.jsonl (or the data/ directory holding it) as prompt/response rows.

    eval.jsonl sits next to train.jsonl with a different schema, so a directory
    path is narrowed to train.jsonl instead of loading every file in it.
    """

    path = Path(dataset_path)
    if path.is_dir():
        path = path / "train.jsonl"
    return [
        {"prompt": str(row["prompt"]).strip(), "response": str(row["response"]).strip()}
        for row in default_loader(str(path))
    ]
