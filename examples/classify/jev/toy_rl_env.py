"""A tiny 3-step 'pick-the-sum' MDP + a decisions.jsonl generator.

Episode: draw a target T from {5, 7, 9}; over 3 steps the agent picks one of
{add 1, add 2, add 3}, accumulating a sum. All step rewards are zero; the
terminal reward is `-|final_sum - T|` (0 for exact, down to -4). The optimal
policy is deterministic in (T, S, R).

Run this file to roll out `--episodes` episodes under a uniform-random
behavior policy, score each decision with its discounted return minus the
batch-mean baseline, and write one JSONL row per decision to the output path.

    python examples/classify/jev/toy_rl_env.py --episodes 2048 --out /tmp/toy_decisions.jsonl

The resulting file is the exact shape `train_rl.py` consumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

TARGETS = (5, 7, 9)
STEPS_PER_EPISODE = 3
CANDIDATES = ("add 1", "add 2", "add 3")
ACTION_VALUES = (1, 2, 3)
STATE_MARK = "<|toy:state|>"
QUESTION_MARK = "<|toy:question|>"
ANSWER_MARK = "<|toy:answer|>"


@dataclass(slots=True)
class ToyStep:
    """One decision taken by the behavior policy."""

    target: int
    remaining: int
    sum_so_far: int
    chosen: int
    old_logp: float


def render_prompt(target: int, remaining: int, sum_so_far: int) -> str:
    """The score head reads the last token; keep the template fixed across prompts."""

    return (
        f"{STATE_MARK}\n"
        f"target={target}\n"
        f"remaining={remaining}\n"
        f"sum_so_far={sum_so_far}\n"
        f"{QUESTION_MARK}\n"
        f"choice: pick a number to add\n"
    )


def render_candidate(index: int) -> str:
    return f"[{index}] {CANDIDATES[index]}\n{ANSWER_MARK}"


def rollout_episode(rng: random.Random) -> tuple[list[ToyStep], float]:
    """One uniform-random episode; returns the step log and the terminal reward."""

    target = rng.choice(TARGETS)
    sum_so_far = 0
    uniform_logp = -math.log(len(CANDIDATES))
    steps: list[ToyStep] = []
    for turn in range(STEPS_PER_EPISODE):
        chosen = rng.randrange(len(CANDIDATES))
        steps.append(
            ToyStep(
                target=target,
                remaining=STEPS_PER_EPISODE - turn,
                sum_so_far=sum_so_far,
                chosen=chosen,
                old_logp=uniform_logp,
            )
        )
        sum_so_far += ACTION_VALUES[chosen]
    terminal_reward = -abs(sum_so_far - target)
    return steps, float(terminal_reward)


def to_decision_row(step: ToyStep, advantage: float) -> dict:
    return {
        "prompt": render_prompt(step.target, step.remaining, step.sum_so_far),
        "candidates": [render_candidate(index) for index in range(len(CANDIDATES))],
        "chosen": step.chosen,
        "old_logp": step.old_logp,
        "advantage": advantage,
        "target": step.target,  # kept for inspection; ignored by the trainer
        "remaining": step.remaining,
    }


def generate_decisions(num_episodes: int, *, seed: int, gamma: float) -> tuple[list[dict], dict]:
    """Return a list of decision rows and a stats dict for logging."""

    rng = random.Random(seed)
    all_steps: list[ToyStep] = []
    returns: list[float] = []
    terminal_rewards: list[float] = []
    for _ in range(num_episodes):
        steps, terminal = rollout_episode(rng)
        terminal_rewards.append(terminal)
        # Sparse terminal reward: G_t = gamma^(steps_left_after_t - 1) * terminal.
        for turn, step in enumerate(steps):
            discount = gamma ** (STEPS_PER_EPISODE - 1 - turn)
            returns.append(discount * terminal)
            all_steps.append(step)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    centered = [value - mean_return for value in returns]
    std = math.sqrt(sum(value * value for value in centered) / len(centered)) if centered else 1.0
    advantages = [value / std if std > 1e-8 else value for value in centered]
    rows = [to_decision_row(step, advantage) for step, advantage in zip(all_steps, advantages, strict=True)]
    stats = {
        "episodes": num_episodes,
        "decisions": len(rows),
        "mean_terminal_reward": sum(terminal_rewards) / len(terminal_rewards),
        "exact_hit_rate": sum(1.0 for value in terminal_rewards if value == 0.0) / len(terminal_rewards),
        "mean_return": mean_return,
        "return_std": std,
    }
    return rows, stats


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True, help="where to write decisions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows, stats = generate_decisions(args.episodes, seed=args.seed, gamma=args.gamma)
    write_jsonl(rows, args.out)
    logging.info("stage=toy_rl_env_done out=%s stats=%s", args.out, stats)


if __name__ == "__main__":
    main()
