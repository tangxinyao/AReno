"""Dataset loader for the STS2 ReAct example: one record per combat episode.

Each JSONL row names an encounter, a simulator seed and an ascension. The
prompt is the first observation of that fight, so the same row always starts
from the same board.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import Episode, Sts2ReActEnv  # noqa: E402
from sim import DATA_ROOT  # noqa: E402

DEFAULT_ACT = "overgrowth"
DEFAULT_ROOM_TYPES = ("monster", "elite")


def load_training_dataset(dataset_path: str | None, *, default_loader=None, **_: object) -> list[dict[str, Any]]:
    """Load JSONL episodes and attach each one's first observation as its prompt."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def generate_records(count: int = 256, seed: int = 2026) -> list[dict[str, Any]]:
    """Cycle through Act 1 normal and elite fights with fresh simulator seeds."""

    encounters = _encounter_ids()
    rng = random.Random(seed)
    return [
        {
            "id": f"sts2-{idx:05d}",
            "encounter": encounters[idx % len(encounters)],
            "seed": rng.randrange(2**31),
            "ascension": 0,
        }
        for idx in range(count)
    ]


def _encounter_ids() -> list[str]:
    with (DATA_ROOT / "encounters.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    ids = [
        str(row["encounter_id"])
        for row in rows
        if row.get("act") == DEFAULT_ACT and row.get("room_type") in DEFAULT_ROOM_TYPES
    ]
    if not ids:
        raise ValueError(f"no {DEFAULT_ACT} encounters found under {DATA_ROOT}")
    return ids


def _load_records(dataset_path: str | None) -> list[dict[str, Any]]:
    path = Path(dataset_path).expanduser() if dataset_path else None
    if path is None or not path.exists():
        return generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict[str, Any], index: int) -> dict[str, Any]:
    episode = Episode(
        encounter=str(raw["encounter"]),
        seed=int(raw["seed"]),
        ascension=int(raw.get("ascension", 0)),
    )
    env = Sts2ReActEnv(episode)
    try:
        prompt = env.reset()
    finally:
        env.close()
    return {
        "id": raw.get("id", f"sts2-{index:05d}"),
        "prompt": prompt,
        "encounter": episode.encounter,
        "seed": episode.seed,
        "ascension": episode.ascension,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSONL path to write.")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with Path(args.output).expanduser().open("w", encoding="utf-8") as handle:
        for record in generate_records(args.count, args.seed):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
