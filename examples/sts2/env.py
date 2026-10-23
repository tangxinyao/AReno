"""ReAct-facing wrapper around the in-house STS2 single-combat simulator.

Each episode is one Ironclad fight against a fixed encounter. The Neow screen is
skipped on reset so the agent starts on the first combat decision. Observations
are plain text; the legal actions are the simulator's own candidate ids, so the
agent can only pick ids the game currently offers.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim.run import RunLoop  # noqa: E402

_ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# Model calls allowed per episode, and illegal or missing actions before the
# episode is abandoned. Shared by the rollout loop and the reward replay.
MAX_TURNS = 48
MAX_INVALID_ACTIONS = 3

SYSTEM_PROMPT = (
    "You are playing one Slay the Spire 2 combat as the Ironclad. Defeat every enemy "
    "while keeping your HP high. Reply in the ReAct format on every turn:\n"
    "Thought: one or two sentences about the best next play.\n"
    "Action: exactly one id copied from the Legal actions list, such as end_turn.\n"
    "Play one action per reply. After each action you receive an Observation with the "
    "updated state and the next list of legal actions."
)


def parse_action(text: str) -> str | None:
    """Return the id on the last ``Action:`` line of a model response."""

    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip("`'\" ")


@dataclass(frozen=True)
class Episode:
    """Immutable identity of one combat: enough to rebuild it deterministically."""

    encounter: str
    seed: int
    ascension: int = 0


class Sts2ReActEnv:
    """One single-combat episode driven one legal action at a time."""

    def __init__(self, episode: Episode, *, max_steps: int = 2000) -> None:
        self.episode = episode
        self._loop = RunLoop(
            encounter=episode.encounter,
            seed=episode.seed,
            ascension=episode.ascension,
            max_steps=max_steps,
        )
        self._packet: dict[str, Any] = {}
        self._enemy_hp_start = 0

    def reset(self) -> str:
        """Start the fight and return the first observation."""

        packet = self._loop.reset()
        # The Neow screen offers a single "skip" option; take it to reach combat.
        packet = self._loop.step(packet["candidates"][0]["id"])
        self._packet = packet
        self._enemy_hp_start = sum(m.max_hp for m in self._loop.state.combat.monsters)
        return self.observation()

    def legal_ids(self) -> list[str]:
        return [candidate["id"] for candidate in self._packet.get("candidates", [])]

    def step(self, action_id: str) -> str:
        """Apply one legal action. Raises ``ValueError`` if it is not legal now."""

        if self.done:
            raise ValueError("episode is already over")
        if action_id not in self.legal_ids():
            raise ValueError(f"illegal action {action_id!r}")
        self._packet = self._loop.step(action_id)
        return self.observation()

    @property
    def done(self) -> bool:
        return bool(self._packet.get("done"))

    @property
    def outcome(self) -> str:
        return str(self._packet.get("outcome", "undecided"))

    @property
    def won(self) -> bool:
        return self.outcome == "victory"

    def enemy_damage_fraction(self) -> float:
        """Share of the enemies' starting HP that is gone, in [0, 1]."""

        if self._enemy_hp_start <= 0:
            return 0.0
        remaining = sum(max(m.hp, 0) for m in self._loop.state.combat.monsters)
        return min(1.0, max(0.0, 1.0 - remaining / self._enemy_hp_start))

    def observation(self) -> str:
        """Render the current combat state and legal actions as text."""

        if self.done:
            return f"The fight is over. Outcome: {self.outcome}."
        state = self._loop.state
        player = state.player
        combat = state.combat
        lines = [
            f"Turn {combat.turn} | HP {player.hp}/{player.max_hp} | Block {player.block} | "
            f"Energy {player.energy}/{player.max_energy}",
            "Enemies:",
        ]
        for idx, monster in enumerate(combat.monsters):
            if monster.hp <= 0:
                continue
            intent = monster.queued_move or "unknown"
            lines.append(
                f"- [{idx}] {monster.name} HP {monster.hp}/{monster.max_hp} "
                f"Block {monster.block} Intent: {intent}"
            )
        lines.append("Legal actions:")
        for candidate in self._packet.get("candidates", []):
            lines.append(f"- {candidate['id']} :: {candidate['text']}")
        return "\n".join(lines)

    def close(self) -> None:
        self._loop.close()
