"""CPU tests for the STS2 ReAct example: parsing, env, loader and reward replay.

The rollout loop in `examples/sts2/run_agent.py` needs an OpenAI-compatible
endpoint and is not covered here; these tests replay scripted policies instead.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "sts2"))

import dataset_loader  # noqa: E402
import env  # noqa: E402
import reward  # noqa: E402

EPISODE = env.Episode(encounter="slimes_weak", seed=7)


def _scripted_actions(episode_env: env.Sts2ReActEnv, limit: int = 300) -> list[str]:
    """Play the first Strike or Bash while possible, otherwise end the turn."""

    chosen: list[str] = []
    for _ in range(limit):
        if episode_env.done:
            break
        legal = episode_env.legal_ids()
        attack = next((a for a in legal if a.startswith(("play_card:strike", "play_card:bash"))), None)
        action = attack or "end_turn"
        episode_env.step(action)
        chosen.append(action)
    return chosen


def _record(actions: list[str], *, episode: env.Episode = EPISODE, extra: list[dict] | None = None):
    messages = [{"role": "assistant", "content": f"Thought: go\nAction: {action}"} for action in actions]
    messages.extend(extra or [])
    return types.SimpleNamespace(
        source_record={"encounter": episode.encounter, "seed": episode.seed, "ascension": episode.ascension},
        messages=messages,
    )


class ParseActionTest(unittest.TestCase):
    def test_takes_last_action_line(self) -> None:
        text = "Thought: first\nAction: end_turn\nThought: again\nAction: `play_card:bash:0`"
        self.assertEqual(env.parse_action(text), "play_card:bash:0")

    def test_missing_action_line_returns_none(self) -> None:
        self.assertIsNone(env.parse_action("Thought: no action here"))
        self.assertIsNone(env.parse_action(""))


class Sts2ReActEnvTest(unittest.TestCase):
    def test_reset_skips_neow_and_lists_combat_actions(self) -> None:
        episode_env = env.Sts2ReActEnv(EPISODE)
        try:
            observation = episode_env.reset()
        finally:
            episode_env.close()
        self.assertTrue(observation.startswith("Turn 1 | HP 80/80"))
        self.assertIn("Legal actions:", observation)
        self.assertIn("end_turn", observation)

    def test_illegal_action_is_rejected_without_advancing(self) -> None:
        episode_env = env.Sts2ReActEnv(EPISODE)
        try:
            episode_env.reset()
            before = episode_env.legal_ids()
            with self.assertRaises(ValueError):
                episode_env.step("play_card:not_a_card:0")
            self.assertEqual(episode_env.legal_ids(), before)
        finally:
            episode_env.close()

    def test_same_episode_replays_identically(self) -> None:
        first = env.Sts2ReActEnv(EPISODE)
        second = env.Sts2ReActEnv(EPISODE)
        try:
            first.reset()
            second.reset()
            actions = _scripted_actions(first)
            for action in actions:
                second.step(action)
            self.assertEqual(first.outcome, second.outcome)
            self.assertEqual(first.observation(), second.observation())
        finally:
            first.close()
            second.close()


class DatasetLoaderTest(unittest.TestCase):
    def test_generated_records_carry_episode_fields_and_prompt(self) -> None:
        records = dataset_loader.load_training_dataset(None)
        self.assertEqual(len(records), 256)
        first = records[0]
        self.assertEqual(first["prompt"].splitlines()[0][:6], "Turn 1")
        self.assertTrue(first["encounter"])
        self.assertIsInstance(first["seed"], int)

    def test_generate_records_is_reproducible(self) -> None:
        self.assertEqual(
            dataset_loader.generate_records(8, seed=1),
            dataset_loader.generate_records(8, seed=1),
        )


class RewardReplayTest(unittest.TestCase):
    def test_replay_matches_live_score(self) -> None:
        live = env.Sts2ReActEnv(EPISODE)
        try:
            live.reset()
            actions = _scripted_actions(live)
            expected = reward.score(live)
        finally:
            live.close()
        self.assertAlmostEqual(reward.reward_fn(_record(actions)), expected)

    def test_too_many_invalid_actions_gets_penalty(self) -> None:
        bad = {"role": "assistant", "content": "Action: play_card:not_a_card:0"}
        record = _record([], extra=[bad] * env.MAX_INVALID_ACTIONS)
        self.assertEqual(reward.reward_fn(record), reward.INVALID_REWARD)

    def test_user_messages_are_ignored(self) -> None:
        record = _record([], extra=[{"role": "user", "content": "Action: end_turn"}])
        episode_env = env.Sts2ReActEnv(EPISODE)
        try:
            episode_env.reset()
            expected = reward.score(episode_env)
        finally:
            episode_env.close()
        self.assertAlmostEqual(reward.reward_fn(record), expected)


if __name__ == "__main__":
    unittest.main()
