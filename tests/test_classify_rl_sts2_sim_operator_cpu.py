"""Operator-contract tests for the in-house `sts2_sim` shim.

Replaces the previous `test_classify_rl_r33hab_operator_cpu.py` and no
longer needs a fake `sts2_gym` module: the sim is pure Python, so we
just construct `Sts2SimBackend` directly and drive it through the
operator surface.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _import_shim():
    mod_name = "classify_jev_sts2_sim_shim_for_tests"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    shim_path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "classify" / "jev" / "operators" / "sts2_sim.py"
    )
    spec = importlib.util.spec_from_file_location(mod_name, shim_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


class CapabilitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = _import_shim()

    def test_capabilities_fields(self) -> None:
        caps = self.shim.Sts2SimBackend().capabilities()
        self.assertEqual(caps["backend"], "sts2-sim")
        self.assertEqual(caps["characters"], ["ironclad"])
        self.assertEqual(caps["ascensions"], [0])
        self.assertIn("full_run", caps["episode_scopes"])
        self.assertTrue(caps["paired_seed_bit_exact"])
        self.assertFalse(caps["supports_snapshot"])


class ResetValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = _import_shim()
        self.backend = self.shim.Sts2SimBackend()

    def test_rejects_non_ironclad(self) -> None:
        with self.assertRaisesRegex(ValueError, "ironclad"):
            self.backend.reset({"character": "silent", "ascension": 0, "episode_scope": "full_run"})

    def test_rejects_wrong_episode_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_run"):
            self.backend.reset({"character": "ironclad", "ascension": 0, "episode_scope": "battle"})

    def test_rejects_unsupported_ascension(self) -> None:
        with self.assertRaisesRegex(ValueError, "ascension"):
            self.backend.reset({"character": "ironclad", "ascension": 5, "episode_scope": "full_run"})

    def test_missing_fields_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            self.backend.reset({"character": "ironclad"})


class ResetShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = _import_shim()
        self.backend = self.shim.Sts2SimBackend()
        self.packet = self.backend.reset({
            "character": "ironclad",
            "ascension": 0,
            "episode_scope": "full_run",
            "seed": "DEMO01",
        })

    def test_state_packet_core_keys(self) -> None:
        for key in ("episode_id", "step", "done", "reward", "decision_point",
                    "state_text", "state_struct", "candidates", "info"):
            self.assertIn(key, self.packet, f"missing key {key}")

    def test_initial_decision_point_is_neow(self) -> None:
        self.assertEqual(self.packet["decision_point"], "neow_bonus")
        self.assertEqual([c["id"] for c in self.packet["candidates"]], ["skip"])
        self.assertFalse(self.packet["done"])
        self.assertEqual(self.packet["step"], 0)
        self.assertEqual(self.packet["reward"], 0.0)

    def test_info_fields(self) -> None:
        info = self.packet["info"]
        self.assertEqual(info["backend"], "sts2-sim")
        self.assertEqual(info["character"], "ironclad")
        self.assertEqual(info["ascension"], 0)
        self.assertEqual(info["episode_scope"], "full_run")
        self.assertEqual(info["seed"], "DEMO01")
        self.assertEqual(info["hp"], 80)
        self.assertEqual(info["max_hp"], 80)
        self.assertIsNone(info["outcome"])

    def test_state_text_mentions_neow(self) -> None:
        self.assertIn("neow", self.packet["state_text"])


class SeedDeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = _import_shim()

    def test_same_seed_label_produces_identical_opening_hand(self) -> None:
        b1 = self.shim.Sts2SimBackend()
        b2 = self.shim.Sts2SimBackend()
        r1 = b1.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run", "seed": "SEEDA"})
        r2 = b2.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run", "seed": "SEEDA"})
        s1 = b1.step({"episode_id": r1["episode_id"], "action_id": "skip"})
        s2 = b2.step({"episode_id": r2["episode_id"], "action_id": "skip"})
        self.assertEqual(
            [c["id"] for c in s1["candidates"]],
            [c["id"] for c in s2["candidates"]],
        )

    def test_different_seeds_diverge(self) -> None:
        b1 = self.shim.Sts2SimBackend()
        b2 = self.shim.Sts2SimBackend()
        r1 = b1.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run", "seed": "SEEDA"})
        r2 = b2.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run", "seed": "SEEDZ"})
        s1 = b1.step({"episode_id": r1["episode_id"], "action_id": "skip"})
        s2 = b2.step({"episode_id": r2["episode_id"], "action_id": "skip"})
        # Enemy HP rolls and shuffle both pull from the master seed; at least one
        # of (candidates, info.hp) should differ for distinct seeds.
        same_candidates = [c["id"] for c in s1["candidates"]] == [c["id"] for c in s2["candidates"]]
        # Compare worm HP by peeking at info; both should be in [40,44].
        # Note: the HPs could still coincidentally match, so just check the whole packet.
        self.assertFalse(same_candidates and s1["state_text"] == s2["state_text"])


class EndToEndFlowTest(unittest.TestCase):
    """Reset -> skip -> combat_play -> end_turn until win or step budget."""

    def setUp(self) -> None:
        self.shim = _import_shim()
        self.backend = self.shim.Sts2SimBackend()
        self.reset_packet = self.backend.reset({
            "character": "ironclad",
            "ascension": 0,
            "episode_scope": "full_run",
            "seed": "E2E01",
        })
        self.episode_id = self.reset_packet["episode_id"]

    def test_skip_transitions_to_combat_play(self) -> None:
        packet = self.backend.step({"episode_id": self.episode_id, "action_id": "skip"})
        self.assertEqual(packet["decision_point"], "combat_play")
        self.assertFalse(packet["done"])
        ids = {c["id"] for c in packet["candidates"]}
        self.assertIn("end_turn", ids)

    def test_greedy_policy_reaches_terminal(self) -> None:
        packet = self.backend.step({"episode_id": self.episode_id, "action_id": "skip"})
        for _ in range(50):
            if packet["done"]:
                break
            chosen = packet["candidates"][0]["id"]
            packet = self.backend.step({"episode_id": self.episode_id, "action_id": chosen})
        self.assertTrue(packet["done"])
        self.assertEqual(packet["decision_point"], "game_over")
        self.assertIn(packet["info"]["outcome"], ("win", "loss"))

    def test_terminal_reward_sign_matches_outcome(self) -> None:
        packet = self.backend.step({"episode_id": self.episode_id, "action_id": "skip"})
        for _ in range(50):
            if packet["done"]:
                break
            chosen = packet["candidates"][0]["id"]
            packet = self.backend.step({"episode_id": self.episode_id, "action_id": chosen})
        self.assertTrue(packet["done"])
        if packet["info"]["outcome"] == "win":
            self.assertAlmostEqual(packet["reward"], 1.0)
        else:
            self.assertAlmostEqual(packet["reward"], -1.0)


class EpisodeIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = _import_shim()
        self.backend = self.shim.Sts2SimBackend()

    def test_unknown_episode_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown episode_id"):
            self.backend.step({"episode_id": "nope", "action_id": "skip"})

    def test_close_removes_episode(self) -> None:
        r = self.backend.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run"})
        self.backend.close(r["episode_id"])
        with self.assertRaisesRegex(ValueError, "unknown episode_id"):
            self.backend.step({"episode_id": r["episode_id"], "action_id": "skip"})

    def test_two_episodes_independent(self) -> None:
        r1 = self.backend.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run"})
        r2 = self.backend.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run"})
        self.assertNotEqual(r1["episode_id"], r2["episode_id"])
        self.backend.step({"episode_id": r1["episode_id"], "action_id": "skip"})
        # Episode 2 is still at neow.
        p2 = self.backend.step({"episode_id": r2["episode_id"], "action_id": "skip"})
        self.assertEqual(p2["step"], 1)


class ActionShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shim = _import_shim()
        self.backend = self.shim.Sts2SimBackend()
        r = self.backend.reset({"character": "ironclad", "ascension": 0, "episode_scope": "full_run"})
        self.episode_id = r["episode_id"]

    def test_rejects_non_string_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            self.backend.step({"episode_id": self.episode_id, "action_id": 42})

    def test_rejects_illegal_action_at_neow(self) -> None:
        with self.assertRaisesRegex(ValueError, "RunLoop rejected"):
            self.backend.step({"episode_id": self.episode_id, "action_id": "play:bash:0"})


if __name__ == "__main__":
    unittest.main()
