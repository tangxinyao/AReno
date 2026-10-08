"""Rollout worker: drives sts2_sim and dumps decisions.jsonl.

Renders prompts and candidates through `dataset_loader.question_prefix` /
`render_candidate` so the strings a `--policy server` rollout saves match
byte-for-byte what `serve_decisions.py` encodes internally — otherwise
training-time tokenization would diverge from the policy that produced
`old_logp` and PPO's ratio would be miscalibrated.

Three policies:

  --policy random   uniform over legal candidates. `old_logp = -log(K)`.
                    Use for the cold start (iter 0).
  --policy greedy   always picks index 0. `old_logp = 0` (deterministic).
                    Debug mode; not valid for PPO's off-policy correction.
  --policy server   POSTs each decision to a running `serve_decisions.py`,
                    parses `logits`, applies `--temperature`, samples a
                    candidate, records `old_logp = log_softmax(logits/T)[chosen]`.
                    This is the loop iter k>=1 drives: load ckpt_{k-1}
                    into the server, rollout against it, train on the
                    collected decisions.

Each JSONL row the trainer consumes:

    {"prompt": str, "candidates": [str, ...],
     "chosen": int, "old_logp": float, "advantage": float}

Decisions with fewer than two legal candidates (Neow `skip`, game-over
`terminal`) are dropped since grouped-softmax PPO needs K>=2 per group.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # dataset_loader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # AReno root

from dataset_loader import ANSWER_MARK, question_prefix, render_candidate  # noqa: E402
from examples.classify.jev.operators.sts2_sim import Sts2SimBackend  # noqa: E402


INSTRUCTION_BY_POINT = {
    "neow_bonus":   "Pick a Neow bonus to accept.",
    "map_select":   "Choose which map room to enter next.",
    "combat_play":  "Pick an action in combat (play a card or end the turn).",
    "card_reward":  "Pick a card reward (or skip).",
    "rest_site":    "Decide what to do at the rest site.",
    "shop":         "Decide what to buy at the shop (or skip).",
    "event_choice": "Pick an event option.",
    "boss_relic":   "Pick a boss relic.",
    "game_over":    "Game over.",
}


@dataclass(slots=True)
class _DecisionRecord:
    """One multi-candidate decision point captured during rollout."""

    prompt: str
    candidates: list[str]
    chosen: int
    old_logp: float
    terminal_reward: float = 0.0
    steps_until_terminal: int = 0


def _build_question(decision_point: str, candidates: list[dict]) -> dict:
    return {
        "type": "choice",
        "instructions": INSTRUCTION_BY_POINT.get(decision_point, f"Pick an action at {decision_point}."),
        "criteria": {c["id"]: c["text"] for c in candidates},
    }


def _render(state_text: str, question: dict) -> tuple[str, list[str]]:
    prompt = question_prefix(state_text, question)
    rendered = [
        render_candidate(question, i) + "\n" + ANSWER_MARK
        for i in range(len(question["criteria"]))
    ]
    return prompt, rendered


# ---------------------------------------------------------------------------
# Policies

def _policy_random(k: int, rng: random.Random) -> tuple[int, float]:
    idx = rng.randrange(k)
    return idx, -math.log(k)


def _policy_greedy(k: int) -> tuple[int, float]:
    del k
    return 0, 0.0


class _ServerPolicy:
    """HTTP client against serve_decisions.py's /api/alpha/decisions."""

    def __init__(self, server_url: str, temperature: float, model_name: str, timeout: float):
        self._url = server_url.rstrip("/") + "/api/alpha/decisions"
        self._temperature = max(temperature, 1e-6)
        self._model = model_name
        self._timeout = timeout

    def pick(self, state_text: str, candidates: list[dict], rng: random.Random) -> tuple[int, float]:
        question = _build_question("", candidates)  # decision_point ignored for server call
        # The server does its own prompt rendering via question_prefix, so we
        # only need to pass raw state + the question block. The resulting
        # logits are keyed by the candidate ids we provided in criteria.
        body = {
            "model": self._model,
            "state": state_text if state_text else "<empty>",
            "questions": {"q": question},
        }
        try:
            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                self._url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"server decision call failed: {exc}") from exc

        answer = reply["answers"]["q"]
        logits = answer["logits"]  # dict: candidate_id -> raw score
        ids_in_order = [c["id"] for c in candidates]
        scaled = [logits[c] / self._temperature for c in ids_in_order]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        total = sum(exps)
        probs = [v / total for v in exps]
        # Sample from the empirical distribution.
        r = rng.random()
        acc = 0.0
        chosen = len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                chosen = i
                break
        return chosen, math.log(max(probs[chosen], 1e-12))


# ---------------------------------------------------------------------------
# Episode loop

def run_episode(
    backend: Sts2SimBackend,
    seed_label: str,
    *,
    policy: str,
    server_policy: _ServerPolicy | None,
    max_steps: int,
    rng: random.Random,
) -> tuple[list[_DecisionRecord], float]:
    """Returns the captured decision records and the final terminal reward."""

    packet = backend.reset({
        "character": "ironclad",
        "ascension": 0,
        "episode_scope": "full_run",
        "seed": seed_label,
    })
    episode_id = packet["episode_id"]
    records: list[_DecisionRecord] = []

    for _ in range(max_steps):
        if packet["done"]:
            break
        candidates = packet["candidates"]
        if not candidates:
            break
        k = len(candidates)
        question = _build_question(packet["decision_point"], candidates)
        prompt_text, rendered = _render(packet["state_text"], question)

        if policy == "random":
            chosen_idx, old_logp = _policy_random(k, rng)
        elif policy == "greedy":
            chosen_idx, old_logp = _policy_greedy(k)
        elif policy == "server":
            assert server_policy is not None
            chosen_idx, old_logp = server_policy.pick(packet["state_text"], candidates, rng)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        if k >= 2:
            records.append(_DecisionRecord(
                prompt=prompt_text,
                candidates=rendered,
                chosen=chosen_idx,
                old_logp=old_logp,
            ))

        chosen_id = candidates[chosen_idx]["id"]
        packet = backend.step({
            "episode_id": episode_id,
            "action_id": chosen_id,
            "old_logp": old_logp,
        })

    terminal_reward = float(packet["reward"]) if packet["done"] else 0.0
    for i, record in enumerate(records):
        record.terminal_reward = terminal_reward
        record.steps_until_terminal = len(records) - 1 - i
    backend.close(episode_id)
    return records, terminal_reward


def _z_score_advantages(records: list[_DecisionRecord], gamma: float) -> list[float]:
    returns = [r.terminal_reward * (gamma ** r.steps_until_terminal) for r in records]
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
    parser.add_argument("--policy", choices=["random", "greedy", "server"], default="random")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--sampling-seed", type=int, default=17)

    parser.add_argument("--server-url", default="http://127.0.0.1:8125",
                        help="serve_decisions.py base URL for --policy server")
    parser.add_argument("--server-model", default="sts2_sim",
                        help="`model` field submitted to serve_decisions (echoed, not validated)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature for --policy server (also recorded in old_logp)")
    parser.add_argument("--server-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    backend = Sts2SimBackend()
    rng = random.Random(args.sampling_seed)
    server_policy = None
    if args.policy == "server":
        server_policy = _ServerPolicy(
            server_url=args.server_url,
            temperature=args.temperature,
            model_name=args.server_model,
            timeout=args.server_timeout,
        )
        logging.info("stage=policy server_url=%s temperature=%.3f", args.server_url, args.temperature)

    all_records: list[_DecisionRecord] = []
    terminal_rewards: list[float] = []
    wins = 0
    for ep in range(args.episodes):
        seed_label = f"ROLLOUT-{args.seed_start + ep:05d}"
        records, terminal = run_episode(
            backend, seed_label,
            policy=args.policy, server_policy=server_policy,
            max_steps=args.max_steps, rng=rng,
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
        "policy": args.policy,
    }
    logging.info("stage=rollout_done out=%s stats=%s", args.out, stats)


if __name__ == "__main__":
    main()
