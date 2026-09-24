"""Dataset loader for Draco Malfoy dialogue SFT rows from extract_dialogue.py."""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Turn {speaker, prompt_line, line} rows into prompt/response pairs.

    The prompt is what the other character says to Draco; the SFT trainer masks
    it, so loss only falls on Draco's reply.
    """

    return [
        {"prompt": f"{row['speaker']}: {str(row['prompt_line']).strip()}", "response": str(row["line"]).strip()}
        for row in default_loader(dataset_path)
    ]
