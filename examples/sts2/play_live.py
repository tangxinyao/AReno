"""Play one live STS2 game through STS2MCP with a ReAct policy served over HTTP.

Run this on the machine that runs the game (for example a Mac). The policy is
an OpenAI-compatible endpoint, typically `areno serve` on the DGX Spark reached
through an SSH tunnel:

  ssh -L 8000:127.0.0.1:8000 <dgx-host>
  python3 examples/sts2/play_live.py --model-url http://127.0.0.1:8000/v1 \
      --log-jsonl live.jsonl

  # Print the observation and legal actions for the current screen, then exit:
  python3 examples/sts2/play_live.py --dry-run

Only the standard library is needed. Combat screens are rendered in the same
text layout the simulator uses in training (`env.Sts2ReActEnv.observation`), so
the policy sees the format it was trained on. Other screens use the adapter's
state digest.

Stops at game over. It does not return to the main menu on its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "live"))

from env import MAX_INVALID_ACTIONS, MAX_TURNS, SYSTEM_PROMPT, parse_action  # noqa: E402
from sts2mcp_live import DEFAULT_BASE_URL, Sts2McpClient, Sts2McpLiveBackend  # noqa: E402

logger = logging.getLogger("sts2_play_live")


def render_observation(packet: dict[str, Any]) -> str:
    """Text the policy sees: the sim layout for combat, the digest otherwise."""

    state = packet["state_struct"]
    lines: list[str] = []
    if state.get("state_type") in ("monster", "elite", "boss") and "battle" in state:
        player = state.get("player") or {}
        battle = state["battle"]
        lines.append(
            f"Turn {battle.get('round', '?')} | HP {player.get('hp', '?')}/{player.get('max_hp', '?')} | "
            f"Block {player.get('block', 0)} | Energy {player.get('energy', '?')}/{player.get('max_energy', '?')}"
        )
        lines.append("Enemies:")
        for idx, enemy in enumerate(battle.get("enemies") or []):
            if enemy.get("hp", 0) <= 0:
                continue
            intents = "/".join(str(i.get("label") or i.get("type") or "?") for i in enemy.get("intents") or [])
            lines.append(
                f"- [{idx}] {enemy.get('name', '?')} HP {enemy.get('hp', '?')}/{enemy.get('max_hp', '?')} "
                f"Block {enemy.get('block', 0)} Intent: {intents or 'unknown'}"
            )
    else:
        lines.append(packet["state_text"])
    lines.append("Legal actions:")
    for candidate in packet["candidates"]:
        lines.append(f"- {candidate['id']} :: {candidate['text']}")
    return "\n".join(lines)


def chat_completion(model_url: str, api_key: str, messages: list[dict[str, Any]], *, timeout: float) -> str:
    """One non-streaming request to an OpenAI-compatible /chat/completions endpoint."""

    body = json.dumps({"model": "policy", "messages": messages, "stream": False}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        model_url.rstrip("/") + "/chat/completions", data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"].get("content") or ""


def wait_for_candidates(backend: Sts2McpLiveBackend, packet: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """Re-read the game while the screen is animating and no action is offered."""

    deadline = time.monotonic() + timeout
    while not packet["candidates"] and not packet["done"] and time.monotonic() < deadline:
        time.sleep(0.5)
        packet = backend.observe()
    return packet


def play(args: argparse.Namespace) -> None:
    backend = Sts2McpLiveBackend(
        Sts2McpClient(args.server_url, timeout=args.http_timeout),
        settle_timeout=args.settle_timeout,
    )
    packet = backend.reset()
    if args.dry_run:
        print(render_observation(packet))
        return

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    invalid = 0
    log_handle = Path(args.log_jsonl).open("a", encoding="utf-8") if args.log_jsonl else None
    try:
        for turn in range(1, MAX_TURNS + 1):
            packet = wait_for_candidates(backend, packet, timeout=args.settle_timeout)
            if packet["done"]:
                break
            observation = render_observation(packet)
            messages.append({"role": "user", "content": observation})
            text = chat_completion(args.model_url, args.api_key, messages, timeout=args.model_timeout)
            messages.append({"role": "assistant", "content": text})

            action = parse_action(text)
            legal = [candidate["id"] for candidate in packet["candidates"]]
            if action is None or action not in legal:
                invalid += 1
                problem = "The reply had no Action line." if action is None else f"{action!r} is not legal now."
                logger.warning("turn %d invalid (%d/%d): %s", turn, invalid, MAX_INVALID_ACTIONS, problem)
                _log(log_handle, turn, observation, text, action, packet, accepted=False)
                if invalid >= MAX_INVALID_ACTIONS:
                    logger.error("stopping after %d invalid replies", invalid)
                    break
                messages.append({"role": "user", "content": f"Observation: {problem}\n{observation}"})
                continue

            logger.info("turn %d action %s", turn, action)
            _log(log_handle, turn, observation, text, action, packet, accepted=True)
            packet = backend.step({"action_id": action})
            if not backend.last_settled:
                logger.warning("game state did not settle within %.1fs", args.settle_timeout)
            messages.append({"role": "user", "content": f"Observation:\n{render_observation(packet)}"})
    finally:
        if log_handle is not None:
            log_handle.close()
    logger.info("finished: decision_point=%s done=%s", packet["decision_point"], packet["done"])


def _log(handle, turn: int, observation: str, reply: str, action: str | None, packet: dict[str, Any], *, accepted: bool) -> None:
    if handle is None:
        return
    record = {
        "turn": turn,
        "decision_point": packet["decision_point"],
        "observation": observation,
        "reply": reply,
        "action": action,
        "accepted": accepted,
        "settled": packet["info"]["settled"],
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-url", default=DEFAULT_BASE_URL, help="STS2MCP base URL on this machine.")
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="", help="Bearer token for the model endpoint, if it needs one.")
    parser.add_argument("--settle-timeout", type=float, default=10.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--log-jsonl", default=None, help="Append one JSON line per model reply.")
    parser.add_argument("--dry-run", action="store_true", help="Print the current observation and exit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    play(args)


if __name__ == "__main__":
    main()
