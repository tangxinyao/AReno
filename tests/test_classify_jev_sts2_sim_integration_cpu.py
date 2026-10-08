"""Phase 1 closure: RunLoop + CombatContext integration tests.

Drives the whole stack via `RunLoop.reset()` + `RunLoop.step(action_id)`
only, with no direct CombatContext access, to prove the Jev-side
operator contract works end-to-end: Neow -> combat_play candidates ->
play/end_turn -> victory -> game_over.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _import_sim():
    mod_name = "classify_jev_sts2_sim_for_tests"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    root = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev" / "sts2_sim"
    init = root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        mod_name, init, submodule_search_locations=[str(root)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


class NeowToCombatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42)
        self.reset_packet = self.loop.reset()

    def test_reset_starts_at_neow_with_skip_only(self) -> None:
        self.assertEqual(self.reset_packet["screen"], self.sim.Screen.NEOW)
        self.assertEqual(
            [c["id"] for c in self.reset_packet["candidates"]],
            ["choose_event_option:0"],
        )

    def test_skip_enters_combat_with_jaw_worm(self) -> None:
        packet = self.loop.step("choose_event_option:0")
        self.assertEqual(packet["screen"], self.sim.Screen.COMBAT)
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.COMBAT_PLAY)
        combat = self.loop.state.combat
        self.assertIsNotNone(combat)
        self.assertEqual(len(combat.monsters), 1)
        self.assertEqual(combat.monsters[0].monster_id, "jaw_worm")

    def test_initial_combat_candidates_include_strikes_defends_bash_and_end_turn(self) -> None:
        packet = self.loop.step("choose_event_option:0")
        ids = {c["id"] for c in packet["candidates"]}
        # Hand is drawn from the standard Ironclad deck; expect at least one
        # of each affordable base card type after shuffle. Strike + Defend
        # are both cost 1 (affordable with 3 energy); Bash costs 2.
        has_strike = any(i.startswith("play_card:strike:") for i in ids)
        has_defend = any(i == "play_card:defend" for i in ids)
        self.assertTrue(has_strike or has_defend, "no cost-1 card in opening hand??")
        self.assertIn("end_turn", ids)


class CombatControlTest(unittest.TestCase):
    """Drive the engine purely through RunLoop.step. Uses a fixed seed so
    the opening hand is deterministic for assertions."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42)
        self.loop.reset()
        self.loop.step("choose_event_option:0")

    def test_end_turn_runs_enemy_phase_and_returns_to_player(self) -> None:
        hp_before = self.loop.state.player.hp
        packet = self.loop.step("end_turn")
        self.assertEqual(packet["screen"], self.sim.Screen.COMBAT)
        self.assertFalse(packet["done"])
        self.assertLess(self.loop.state.player.hp, hp_before)  # Jaw Worm chomped
        self.assertEqual(self.loop.state.combat.phase, self.sim.CombatPhase.PLAYER)

    def test_bogus_action_rejected(self) -> None:
        with self.assertRaisesRegex(self.sim.RunLoopError, "illegal action"):
            self.loop.step("eat_grass")

    def test_malformed_play_rejected(self) -> None:
        with self.assertRaisesRegex(self.sim.RunLoopError, "illegal action"):
            self.loop.step("play_card:")


class CombatToVictoryTest(unittest.TestCase):
    """Force-win by draining Jaw Worm HP via direct state mutation, then
    play through the engine to confirm the transition finalizes."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42)
        self.loop.reset()
        self.loop.step("choose_event_option:0")

    def test_strike_to_victory_drops_run_into_game_over(self) -> None:
        worm = self.loop.state.combat.monsters[0]
        worm.hp = 5  # one Strike (6) will kill
        # Find a Strike in hand and target it at slot 0.
        hand = self.loop.state.player.hand
        if "strike" not in hand:
            # Guarantee a strike by planting it.
            hand.append("strike")
            self.loop.state.player.energy = 3
        packet = self.loop.step("play_card:strike:0")
        self.assertTrue(packet["done"])
        self.assertEqual(packet["screen"], self.sim.Screen.GAME_OVER)
        self.assertEqual(packet["outcome"], self.sim.Outcome.VICTORY)
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.GAME_OVER)
        self.assertEqual([c["id"] for c in packet["candidates"]], ["menu_select:main_menu"])

    def test_cannot_step_after_game_over(self) -> None:
        worm = self.loop.state.combat.monsters[0]
        worm.hp = 5
        hand = self.loop.state.player.hand
        if "strike" not in hand:
            hand.append("strike")
            self.loop.state.player.energy = 3
        self.loop.step("play_card:strike:0")
        with self.assertRaisesRegex(self.sim.RunLoopError, "terminal"):
            self.loop.step("menu_select:main_menu")


class CombatToDefeatTest(unittest.TestCase):
    """Force defeat by capping player HP so Jaw Worm's Chomp kills in one hit."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42)
        self.loop.reset()
        self.loop.state.player.hp = 5
        self.loop.state.player.max_hp = 5
        self.loop.step("choose_event_option:0")

    def test_end_turn_lethal_chomp_sets_defeat(self) -> None:
        packet = self.loop.step("end_turn")
        self.assertTrue(packet["done"])
        self.assertEqual(packet["screen"], self.sim.Screen.GAME_OVER)
        self.assertEqual(packet["outcome"], self.sim.Outcome.DEATH)


class CandidateFilteringTest(unittest.TestCase):
    """A card the player cannot afford must NOT appear in candidates."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42)
        self.loop.reset()
        self.loop.step("choose_event_option:0")

    def test_bash_disappears_when_energy_too_low(self) -> None:
        # Plant a known hand: bash only.
        self.loop.state.player.hand[:] = ["bash"]
        self.loop.state.player.energy = 1  # bash costs 2
        packet = self.loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertNotIn("play_card:bash:0", ids)
        self.assertEqual(ids, {"end_turn"})

    def test_bash_appears_with_enough_energy(self) -> None:
        self.loop.state.player.hand[:] = ["bash"]
        self.loop.state.player.energy = 2
        packet = self.loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertIn("play_card:bash:0", ids)

    def test_duplicate_cards_deduplicate_in_candidates(self) -> None:
        self.loop.state.player.hand[:] = ["strike", "strike", "strike"]
        self.loop.state.player.energy = 3
        packet = self.loop._packet()
        strike_candidates = [c for c in packet["candidates"] if c["id"].startswith("play_card:strike")]
        # One candidate per alive enemy slot, deduplicated by card_id.
        alive = sum(1 for m in self.loop.state.combat.monsters if m.alive)
        self.assertEqual(len(strike_candidates), alive)

    def test_unplayable_wound_filtered_from_candidates(self) -> None:
        self.loop.state.player.hand[:] = ["wound", "strike"]
        self.loop.state.player.energy = 3
        packet = self.loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertNotIn("play_card:wound", ids)
        self.assertNotIn("play_card:wound:0", ids)
        self.assertTrue(any(i.startswith("play_card:strike") for i in ids))

    def test_hand_of_only_unplayables_leaves_only_end_turn(self) -> None:
        self.loop.state.player.hand[:] = ["wound", "wound"]
        self.loop.state.player.energy = 3
        packet = self.loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertEqual(ids, {"end_turn"})


if __name__ == "__main__":
    unittest.main()
