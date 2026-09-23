"""Split a trajectory SFT dataset into per-phase sub-datasets ("subtasks").

Phase tags come from ``ARENO_OPC_PHASES=1`` in ``dataset_loader.py``: every
assistant turn is labelled by its action genre (explore / read / implement /
run / report). This script groups rows by genre and writes one JsonL file per
genre, so a long task like ``five-step-pipeline`` becomes several smaller SFT
tasks you can train and inspect independently (and, e.g., weight or filter per
stage).

Usage::

    python examples/sft/opc/split_phases.py \
        --dataset examples/sft/opc/data \
        --out examples/sft/opc/split
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

_GENRES = ("explore", "read", "implement", "run", "report")


def _load_loader():
    spec = importlib.util.spec_from_file_location("opc_loader", HERE / "dataset_loader.py")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("cannot load dataset_loader.py")
    spec.loader.exec_module(module)
    return module


def split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group loader records by ``phase_genre``, in a stable genre order."""

    groups: dict[str, list[dict[str, Any]]] = {genre: [] for genre in _GENRES}
    for record in records:
        genre = record["phase_genre"]
        groups.setdefault(genre, []).append(record)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(HERE / "data"))
    parser.add_argument("--out", default=str(HERE / "split"))
    args = parser.parse_args(argv)

    os.environ["ARENO_OPC_PHASES"] = "1"
    loader = _load_loader()
    records = loader.load_training_dataset(args.dataset, default_loader=lambda _: [])

    groups = split(records)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: list[tuple[str, int, int]] = []
    for genre, rows in groups.items():
        if not rows:
            continue
        payload = [
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "tokens": row["tokens"],
                "prompt_mask": row["prompt_mask"],
                "loss_mask": row["loss_mask"],
                "source_trial": row["source_trial"],
                "phase_index": row["phase_index"],
            }
            for row in rows
        ]
        path = out_dir / f"{genre}.jsonl"
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n", encoding="utf-8")
        trials = len({row["source_trial"] for row in rows})
        counts.append((genre, len(payload), trials))

    print(f"wrote {len(counts)} phase datasets to {out_dir}")
    for genre, n, trials in counts:
        print(f"  {genre:10s} rows={n:>3}  from {trials} trial(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
