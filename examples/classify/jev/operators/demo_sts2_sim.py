"""End-to-end smoke demo of the in-house `sts2_sim` operator shim.

Replaces the earlier `demo_r33hab.py` and no longer depends on any
NativeAOT emulator. Builds the backend, resets a fresh run, drives the
decision loop with a greedy "first legal candidate" policy, and emits
one compact JSON line per StatePacket. Victory, defeat, or hitting
`--steps` all cause clean termination.

Usage:

    python3.11 examples/classify/jev/operators/demo_sts2_sim.py --seed DEMO01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))  # AReno root

from examples.classify.jev.operators.sts2_sim import Sts2SimBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", default=None, help="deterministic seed label; None lets the backend pick")
    parser.add_argument("--steps", type=int, default=40, help="max /step calls before giving up")
    parser.add_argument("--ascension", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    backend = Sts2SimBackend()
    caps = backend.capabilities()
    print(json.dumps({"stage": "capabilities", **caps}, ensure_ascii=False))

    reset_payload = {
        "character": "ironclad",
        "ascension": args.ascension,
        "episode_scope": "full_run",
        "seed": args.seed,
    }
    packet = backend.reset(reset_payload)
    _emit("reset", packet)
    episode_id = packet["episode_id"]

    for turn in range(args.steps):
        if packet["done"]:
            break
        if not packet["candidates"]:
            print(json.dumps({"stage": "no_candidates", "step": packet["step"]}))
            break
        chosen = packet["candidates"][0]["id"]
        packet = backend.step({"episode_id": episode_id, "action_id": chosen, "old_logp": None})
        _emit(f"step_{turn+1}", packet, chosen=chosen)

    backend.close(episode_id)
    print(json.dumps({"stage": "closed", "episode_id": episode_id}))


def _emit(stage: str, packet: dict, *, chosen: str | None = None) -> None:
    """One compact JSON line per state packet (candidates truncated for readability)."""

    candidates = packet["candidates"]
    row = {
        "stage": stage,
        "episode_id": packet["episode_id"],
        "step": packet["step"],
        "done": packet["done"],
        "reward": packet["reward"],
        "decision_point": packet["decision_point"],
        "chosen": chosen,
        "state_text": packet["state_text"].replace("\n", " | "),
        "candidate_count": len(candidates),
        "candidate_ids": [c["id"] for c in candidates[:8]],
        "info_subset": {
            key: packet["info"].get(key)
            for key in ("hp", "max_hp", "turn", "energy", "enemies_alive", "outcome", "backend")
        },
    }
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
