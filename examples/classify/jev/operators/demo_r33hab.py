"""End-to-end smoke demo of the r33hab/sts2 operator shim.

Builds a backend, resets a fresh run, takes three legal actions chosen by a
greedy "first legal action" policy, and prints the resulting StatePackets.
This is the minimum thing that proves /reset -> /step -> /step -> /step
goes through the operator contract against a real NativeAOT emulator.

    STS2_LIB_PATH=/tmp/r33hab_sts2/out \\
    PYTHONPATH=/tmp/r33hab_sts2/src \\
    python3.11 examples/classify/jev/operators/demo_r33hab.py --seed DEMO01

Prints are one JSON line per packet so the output is grep-friendly; omit
--seed for a time-based random one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))  # AReno root

from examples.classify.jev.operators.r33hab_sts2 import R33habSts2Backend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", default=None, help="deterministic seed string; None => backend picks")
    parser.add_argument("--steps", type=int, default=3, help="number of /step calls to make")
    parser.add_argument("--ascension", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    backend = R33habSts2Backend()
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
        step_payload = {"episode_id": episode_id, "action_id": chosen, "old_logp": None}
        packet = backend.step(step_payload)
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
            for key in ("hp", "max_hp", "floor", "act", "outcome", "episode_scope", "backend")
        },
    }
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
