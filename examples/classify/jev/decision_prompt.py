"""Shared decision-question building + serve_decisions client.

Used by both the sim rollout (`rollout_sts2_sim.py`) and the live-game
driver (`play_sts2mcp_live.py`). Keeping one `build_question` guarantees
the question a server scores at rollout time is byte-identical to the one
rendered into the training prompt, so PPO's `old_logp` stays consistent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from dataset_loader import ANSWER_MARK, question_prefix, render_candidate


INSTRUCTION_BY_POINT = {
    "neow_bonus":    "Pick a Neow bonus to accept.",
    "map_select":    "Choose which map room to enter next.",
    "combat_play":   "Pick an action in combat (play a card or end the turn).",
    "hand_select":   "Pick a card from your hand for the current card effect.",
    "rewards":       "Pick a combat reward to claim (or proceed).",
    "card_reward":   "Pick a card reward (or skip).",
    "rest_site":     "Decide what to do at the rest site.",
    "shop":          "Decide what to buy at the shop (or skip).",
    "treasure":      "Pick a treasure relic (or proceed).",
    "event_choice":  "Pick an event option.",
    "card_select":   "Pick a card from your deck for the current effect.",
    "bundle_select": "Pick a card bundle.",
    "boss_relic":    "Pick a boss relic.",
    "crystal_sphere": "Pick a Crystal Sphere move.",
    "game_over":     "Game over.",
}


def build_question(decision_point: str, candidates: list[dict]) -> dict:
    return {
        "type": "choice",
        "instructions": INSTRUCTION_BY_POINT.get(decision_point, f"Pick an action at {decision_point}."),
        "criteria": {c["id"]: c["text"] for c in candidates},
    }


def render(state_text: str, question: dict) -> tuple[str, list[str]]:
    """Training-time prompt + per-candidate strings (matches serve_decisions)."""

    prompt = question_prefix(state_text, question)
    rendered = [
        render_candidate(question, i) + "\n" + ANSWER_MARK
        for i in range(len(question["criteria"]))
    ]
    return prompt, rendered


class DecisionClient:
    """Minimal HTTP client for serve_decisions.py's /api/alpha/decisions."""

    def __init__(self, server_url: str, model_name: str, timeout: float):
        self._url = server_url.rstrip("/") + "/api/alpha/decisions"
        self._model = model_name
        self._timeout = timeout

    def logits(self, state_text: str, question: dict) -> list[float]:
        """Raw scores aligned with `question["criteria"]` insertion order."""

        body = {
            "model": self._model,
            "state": state_text if state_text else "<empty>",
            "questions": {"q": question},
        }
        try:
            req = urllib.request.Request(
                self._url, data=json.dumps(body).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"server decision call failed: {exc}") from exc
        by_id = reply["answers"]["q"]["logits"]
        return [float(by_id[cid]) for cid in question["criteria"]]
