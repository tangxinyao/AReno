"""Drive a live Slay the Spire 2 game through STS2MCP with a Jev policy.

Loop: GET game state -> candidates (operators/sts2mcp_live.py) -> score via
serve_decisions.py -> POST the chosen action back to the game. Questions
are built with `decision_prompt.build_question`, the same builder the sim
rollout uses, so a checkpoint trained on sts2_sim sees identical prompt
structure in the live game.

  python3.11 examples/classify/jev/play_sts2mcp_live.py \
      --server-url http://127.0.0.1:8123 --temperature 0 --log-jsonl live.jsonl

  # Inspect candidates without acting:
  python3.11 examples/classify/jev/play_sts2mcp_live.py --dry-run

Stops at game_over (it does not return to the main menu on its own).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset_loader / decision_prompt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # AReno root

from decision_prompt import DecisionClient, build_question, render  # noqa: E402
from examples.classify.jev.operators.sts2mcp_live import (  # noqa: E402
    DEFAULT_BASE_URL,
    Sts2McpClient,
    Sts2McpLiveBackend,
)


def choose(logits: list[float], temperature: float, rng: random.Random) -> tuple[int, float]:
    """Argmax when temperature <= 0, otherwise sample softmax(logits / T)."""

    t = temperature if temperature > 0 else 1.0
    scaled = [v / t for v in logits]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    probs = [v / total for v in exps]
    if temperature <= 0:
        idx = max(range(len(probs)), key=probs.__getitem__)
    else:
        idx = rng.choices(range(len(probs)), weights=probs)[0]
    return idx, math.log(max(probs[idx], 1e-12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcp-url", default=DEFAULT_BASE_URL, help="STS2MCP mod base URL")
    parser.add_argument("--server-url", default="http://127.0.0.1:8123", help="serve_decisions.py base URL")
    parser.add_argument("--server-model", default="sts2_live")
    parser.add_argument("--server-timeout", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy argmax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--poll-interval", type=float, default=0.5,
                        help="seconds between polls while the game has no legal candidates")
    parser.add_argument("--max-idle-polls", type=int, default=600,
                        help="give up after this many consecutive empty polls")
    parser.add_argument("--dry-run", action="store_true", help="print the current candidates and exit")
    parser.add_argument("--log-jsonl", default=None, help="append one row per decision (prompt/candidates/chosen/logp)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    backend = Sts2McpLiveBackend(Sts2McpClient(args.mcp_url))
    packet = backend.reset()

    if args.dry_run:
        print(packet["state_text"])
        print(f"decision_point={packet['decision_point']}")
        for c in packet["candidates"]:
            print(f"  {c['id']:40s} {c['text']}")
        return

    client = DecisionClient(args.server_url, args.server_model, args.server_timeout)
    rng = random.Random(args.seed)
    log = open(args.log_jsonl, "a", encoding="utf-8") if args.log_jsonl else None
    idle = 0
    try:
        for _ in range(args.max_steps):
            if packet["done"]:
                logging.info("stage=game_over state=%s", packet["state_text"].splitlines()[1])
                break
            candidates = packet["candidates"]
            if not candidates:
                idle += 1
                if idle > args.max_idle_polls:
                    raise RuntimeError(f"no legal candidates after {idle} polls (state={packet['info']})")
                time.sleep(args.poll_interval)
                packet = backend.observe()
                continue
            idle = 0

            if len(candidates) == 1:
                idx, logp = 0, 0.0
                question = None
            else:
                question = build_question(packet["decision_point"], candidates)
                idx, logp = choose(client.logits(packet["state_text"], question), args.temperature, rng)
            chosen = candidates[idx]["id"]
            logging.info("stage=act point=%s k=%d chosen=%s", packet["decision_point"], len(candidates), chosen)
            if log is not None and question is not None:
                prompt, rendered = render(packet["state_text"], question)
                row = {"prompt": prompt, "candidates": rendered, "chosen": idx, "old_logp": logp,
                       "decision_point": packet["decision_point"], "action_id": chosen}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            packet = backend.step({"action_id": chosen})
            if not packet["info"]["settled"]:
                logging.warning("stage=unsettled after=%s; acting on the latest state anyway", chosen)
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
