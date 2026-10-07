"""CPU test for the operator contract schema at examples/classify/jev/operator.schema.json.

Validates that the schema loads as JSON, declares the expected $defs, and that
a hand-written DecisionLogRow is accepted verbatim by `encode_decision` of
the classify_rl trainer. We do not run a full JSON Schema validator (would
add a dependency); the goal here is to catch typos and keep the DecisionLogRow
shape in lock-step with the trainer.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev" / "operator.schema.json"

EXPECTED_DEFS = {
    "DecisionPoint",
    "Character",
    "EpisodeScope",
    "Candidate",
    "Info",
    "StatePacket",
    "ResetRequest",
    "StepRequest",
    "SnapshotRequest",
    "SnapshotResponse",
    "RestoreRequest",
    "Capabilities",
    "DecisionLogRow",
}

EXPECTED_DECISION_POINTS = {
    "combat_play",
    "map_select",
    "card_reward",
    "rest_site",
    "event_choice",
    "shop",
    "boss_relic",
    "neow_bonus",
    "game_over",
}

EXPECTED_CHARACTERS = {"ironclad", "silent", "defect", "watcher", "necrobinder", "regent"}


class OperatorSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.defs = cls.schema["$defs"]

    def test_defs_cover_the_contract(self):
        self.assertEqual(set(self.defs), EXPECTED_DEFS)

    def test_decision_point_and_character_enums(self):
        self.assertEqual(set(self.defs["DecisionPoint"]["enum"]), EXPECTED_DECISION_POINTS)
        self.assertEqual(set(self.defs["Character"]["enum"]), EXPECTED_CHARACTERS)

    def test_state_packet_requires_fields_rollout_worker_needs(self):
        required = set(self.defs["StatePacket"]["required"])
        self.assertEqual(
            required,
            {"episode_id", "step", "done", "reward", "decision_point", "state_text", "candidates", "info"},
        )

    def test_decision_log_row_matches_trainer_input(self):
        required = set(self.defs["DecisionLogRow"]["required"])
        self.assertEqual(required, {"prompt", "candidates", "chosen", "old_logp", "advantage"})
        properties = self.defs["DecisionLogRow"]["properties"]
        self.assertIn("sft_target", properties)
        self.assertIn("meta", properties)

    def test_canned_row_is_accepted_by_encode_decision(self):
        """A row that matches the DecisionLogRow $def must round-trip through the trainer."""

        from areno.experimental.classify.rl_trainer import encode_decision

        row = {
            "prompt": "state_text at decision time",
            "candidates": ["[strike_0] Strike enemy 0 for 6", "[strike_1] Strike enemy 1 for 6", "[end_turn] End turn"],
            "chosen": 1,
            "old_logp": -1.0986122886681098,
            "advantage": 0.42,
            "sft_target": None,
            "meta": {"episode_id": "ep-1", "decision_point": "combat_play", "step": 7, "seed": "SEED"},
        }

        class _Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(char) % 128 for char in text]

        decision = encode_decision(row, _Tokenizer())
        self.assertEqual(len(decision.leaves), 3)
        self.assertEqual(decision.chosen, 1)
        self.assertAlmostEqual(decision.old_logp, row["old_logp"])
        self.assertAlmostEqual(decision.advantage, row["advantage"])
        self.assertIsNone(decision.sft_target)

    def test_capabilities_declares_episode_scope_and_characters(self):
        required = set(self.defs["Capabilities"]["required"])
        self.assertEqual(required, {"backend", "characters", "ascensions", "episode_scopes", "game_versions"})

    def test_step_request_carries_optional_old_logp(self):
        properties = self.defs["StepRequest"]["properties"]
        self.assertEqual(set(self.defs["StepRequest"]["required"]), {"episode_id", "action_id"})
        self.assertIn("old_logp", properties)


if __name__ == "__main__":
    unittest.main()
