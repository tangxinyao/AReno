"""Rollout worker: drives the in-house sts2_sim via `Sts2SimBackend` and
dumps decisions.jsonl in the shape `train_rl.py` consumes.

Each row the trainer expects:

    {"prompt": str, "candidates": [str, ...],
     "chosen": int, "old_logp": float, "advantage": float}

This script runs `--episodes` episodes with a uniform-random behavior
policy (or `--policy greedy` to just pick the first candidate), scores
each decision with its episode's terminal reward discounted back to that
step, z-scores advantages across the whole batch, and writes one JSONL
line per decision. Decisions with fewer than two legal candidates (Neow
`skip`, game-over `terminal`) are filtered because the grouped-softmax
PPO loss needs at least two options per group.

    python examples/classify/jev/rollout_sts2_sim.py \\
        --episodes 256 --out /tmp/sts2_sim_decisions.jsonl

Pair with `train_rl.py` for the training step; `run_classify_rl.sh`
runs both back-to-back.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # AReno root

from examples.classify.jev.operators.sts2_sim import Sts2SimBackend  # noqa: E402


STATE_MARK = "<|sts2:state|>"
QUESTION_MARK = "<|sts2:question|>"
ANSWER_MARK = "<|sts2:answer|>"


@dataclass(slots=True)
class _DecisionRecord:
    """One multi-candidate decision point captured during rollout."""

    episode_index: int
    step_index: int
    prompt: str
    candidates: list[str]
    chosen: int
    old_logp: float
    terminal_reward: float = 0.0
    steps_until_terminal: int = 0


def render_prompt(state_text: str, decision_point: str) -> str:
    return (
        f"{STATE_MARK}\n"
        f"{state_text}\n"
        f"{QUESTION_MARK}\n"
        f"decision_point={decision_point}\n"
    )


def render_candidate(index: int, cand: dict) -> str:
    return f"[{index}] id={cand['id']} :: {cand['text']}\n{ANSWER_MARK}"


def _pick(policy: str, k: int, rng: random.Random) -> int:
    if policy == "greedy":
        return 0
    return rng.randrange(k)


def run_episode(
    backend: Sts2SimBackend,
    episode_index: int,
    seed_label: str,
    *,
    policy: str,
    max_steps: int,
    rng: random.Random,
) -> tuple[list[_DecisionRecord], float]:
    """Returns the sampled decision records and the final terminal reward."""

    packet = backend.reset({
        "character": "ironclad",
        "ascension": 0,
        "episode_scope": "full_run",
        "seed": seed_label,
    })
    episode_id = packet["episode_id"]
    records: list[_DecisionRecord] = []

    for step in range(max_steps):
        if packet["done"]:
            break
        candidates = packet["candidates"]
        if not candidates:
            break
        chosen_idx = _pick(policy, len(candidates), rng)
        if len(candidates) >= 2:
            # Only record decisions the trainer can consume; single-candidate
            # points like Neow's skip and the game-over terminal are dropped.
            records.append(_DecisionRecord(
                episode_index=episode_index,
                step_index=packet["step"],
                prompt=render_prompt(packet["state_text"], packet["decision_point"]),
                candidates=[render_candidate(i, c) for i, c in enumerate(candidates)],
                chosen=chosen_idx,
                old_logp=-math.log(len(candidates)),
            ))
        chosen_id = candidates[chosen_idx]["id"]
        packet = backend.step({
            "episode_id": episode_id,
            "action_id": chosen_id,
            "old_logp": -math.log(max(1, len(candidates))),
        })

    terminal_reward = float(packet["reward"]) if packet["done"] else 0.0
    # Back-fill each record's distance to the terminal step so advantage
    # computation can discount properly.
    for i, record in enumerate(records):
        record.terminal_reward = terminal_reward
        record.steps_until_terminal = len(records) - 1 - i
    backend.close(episode_id)
    return records, terminal_reward


def _z_score_advantages(records: list[_DecisionRecord], gamma: float) -> list[float]:
    returns = [
        r.terminal_reward * (gamma ** r.steps_until_terminal) for r in records
    ]
    if not returns:
        return []
    mean = sum(returns) / len(returns)
    centered = [v - mean for v in returns]
    var = sum(v * v for v in centered) / len(centered)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return [v / std for v in centered]


def _write_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--policy", choices=["random", "greedy"], default="random")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--sampling-seed", type=int, default=17,
                        help="seed for the behavior-policy RNG (not the sim's run seeds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    backend = Sts2SimBackend()
    rng = random.Random(args.sampling_seed)

    all_records: list[_DecisionRecord] = []
    terminal_rewards: list[float] = []
    wins = 0
    for ep in range(args.episodes):
        seed_label = f"ROLLOUT-{args.seed_start + ep:05d}"
        records, terminal = run_episode(
            backend, episode_index=ep, seed_label=seed_label,
            policy=args.policy, max_steps=args.max_steps, rng=rng,
        )
        all_records.extend(records)
        terminal_rewards.append(terminal)
        if terminal > 0:
            wins += 1
        if (ep + 1) % args.log_every == 0 or ep + 1 == args.episodes:
            logging.info(
                "stage=rollout ep=%d/%d decisions=%d wins=%d win_rate=%.3f",
                ep + 1, args.episodes, len(all_records), wins, wins / (ep + 1),
            )

    advantages = _z_score_advantages(all_records, args.gamma)
    rows = []
    for record, advantage in zip(all_records, advantages, strict=True):
        rows.append({
            "prompt": record.prompt,
            "candidates": record.candidates,
            "chosen": record.chosen,
            "old_logp": record.old_logp,
            "advantage": advantage,
        })
    _write_jsonl(rows, args.out)

    stats = {
        "episodes": args.episodes,
        "decisions": len(rows),
        "wins": wins,
        "win_rate": wins / max(1, args.episodes),
        "mean_terminal_reward": sum(terminal_rewards) / max(1, len(terminal_rewards)),
    }
    logging.info("stage=rollout_done out=%s stats=%s", args.out, stats)


if __name__ == "__main__":
    main()
