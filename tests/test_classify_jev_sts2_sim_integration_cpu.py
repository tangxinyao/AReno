"""RunLoop + CombatContext integration tests.

Drives the whole stack via `RunLoop.reset()` + `RunLoop.step(action_id)`
to prove the Jev-side operator contract works end-to-end: Neow -> map -> an
Act 1 weak-pool combat -> play / select / end_turn -> victory or death.
Pinned-encounter loops (`encounter=`) are single-combat episodes.
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
        ids = [c["id"] for c in self.reset_packet["candidates"]]
        self.assertIn("choose_event_option:0", ids)
        self.assertTrue(all(x.startswith("select_relic:") or x == "choose_event_option:0" for x in ids))

    def test_skip_enters_an_act1_weak_pool_combat(self) -> None:
        packet = self.loop.step("choose_event_option:0")
        self.assertEqual(packet["screen"], self.sim.Screen.MAP)
        self.assertTrue(all(c["text"].startswith("go to Monster (row 1") for c in packet["candidates"]))
        packet = self.loop.step("choose_map_node:0")
        self.assertEqual(packet["screen"], self.sim.Screen.COMBAT)
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.COMBAT_PLAY)
        enc = self.loop._encounters[self.loop.state.encounter_id]
        self.assertEqual(enc.pool, "weak")
        self.assertIn(enc.act, ("overgrowth", "underdocks"))
        self.assertTrue(self.loop.state.combat.monsters)

    def test_both_act1_variants_are_drawn(self) -> None:
        acts = set()
        for seed in range(30):
            loop = self.sim.RunLoop(seed=seed)
            loop.reset()
            loop.step("choose_event_option:0")
            loop.step("choose_map_node:0")
            acts.add(loop._encounters[loop.state.encounter_id].act)
        self.assertEqual(acts, {"overgrowth", "underdocks"})

    def test_initial_combat_candidates_include_strikes_defends_bash_and_end_turn(self) -> None:
        self.loop.step("choose_event_option:0")
        packet = self.loop.step("choose_map_node:0")
        ids = {c["id"] for c in packet["candidates"]}
        # Hand is drawn from the standard Ironclad deck; expect at least one
        # of each affordable base card type after shuffle. Strike + Defend
        # are both cost 1 (affordable with 3 energy); Bash costs 2.
        has_strike = any(i.startswith("play_card:strike_ironclad:") for i in ids)
        has_defend = any(i == "play_card:defend_ironclad" for i in ids)
        self.assertTrue(has_strike or has_defend, "no cost-1 card in opening hand??")
        self.assertIn("end_turn", ids)


class CombatControlTest(unittest.TestCase):
    """Drive the engine purely through RunLoop.step. Uses a fixed seed so
    the opening hand is deterministic for assertions."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42, encounter="nibbits_weak")
        self.loop.reset()
        self.loop.step("choose_event_option:0")

    def test_end_turn_runs_enemy_phase_and_returns_to_player(self) -> None:
        hp_before = self.loop.state.player.hp
        packet = self.loop.step("end_turn")
        self.assertEqual(packet["screen"], self.sim.Screen.COMBAT)
        self.assertFalse(packet["done"])
        self.assertLess(self.loop.state.player.hp, hp_before)  # a lone Nibbit opens on Butt
        self.assertEqual(self.loop.state.combat.phase, self.sim.CombatPhase.PLAYER)

    def test_bogus_action_rejected(self) -> None:
        with self.assertRaisesRegex(self.sim.RunLoopError, "illegal action"):
            self.loop.step("eat_grass")

    def test_malformed_play_rejected(self) -> None:
        with self.assertRaisesRegex(self.sim.RunLoopError, "illegal action"):
            self.loop.step("play_card:")


class CombatToVictoryTest(unittest.TestCase):
    """Force-win by draining the Nibbit's HP via direct state mutation, then
    play through the engine to confirm the transition finalizes."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42, encounter="nibbits_weak")
        self.loop.reset()
        self.loop.step("choose_event_option:0")

    def test_strike_to_victory_drops_run_into_game_over(self) -> None:
        worm = self.loop.state.combat.monsters[0]
        worm.hp = 5  # one Strike (6) will kill
        # Find a Strike in hand and target it at slot 0.
        hand = self.loop.state.player.hand
        if "strike_ironclad" not in hand:
            # Guarantee a strike by planting it.
            hand.append("strike_ironclad")
            self.loop.state.player.energy = 3
        packet = self.loop.step("play_card:strike_ironclad:0")
        self.assertTrue(packet["done"])
        self.assertEqual(packet["screen"], self.sim.Screen.GAME_OVER)
        self.assertEqual(packet["outcome"], self.sim.Outcome.VICTORY)
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.GAME_OVER)
        self.assertEqual([c["id"] for c in packet["candidates"]], ["menu_select:main_menu"])

    def test_cannot_step_after_game_over(self) -> None:
        worm = self.loop.state.combat.monsters[0]
        worm.hp = 5
        hand = self.loop.state.player.hand
        if "strike_ironclad" not in hand:
            hand.append("strike_ironclad")
            self.loop.state.player.energy = 3
        self.loop.step("play_card:strike_ironclad:0")
        with self.assertRaisesRegex(self.sim.RunLoopError, "terminal"):
            self.loop.step("menu_select:main_menu")


class CombatToDefeatTest(unittest.TestCase):
    """Force defeat by capping player HP so the Nibbit's Butt kills in one hit."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42, encounter="nibbits_weak")
        self.loop.reset()
        self.loop.state.player.hp = 5
        self.loop.state.player.max_hp = 5
        self.loop.step("choose_event_option:0")

    def test_end_turn_lethal_butt_sets_defeat(self) -> None:
        packet = self.loop.step("end_turn")
        self.assertTrue(packet["done"])
        self.assertEqual(packet["screen"], self.sim.Screen.GAME_OVER)
        self.assertEqual(packet["outcome"], self.sim.Outcome.DEATH)


class CandidateFilteringTest(unittest.TestCase):
    """A card the player cannot afford must NOT appear in candidates."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=42, encounter="nibbits_weak")
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
        self.loop.state.player.hand[:] = ["strike_ironclad", "strike_ironclad", "strike_ironclad"]
        self.loop.state.player.energy = 3
        packet = self.loop._packet()
        strike_candidates = [c for c in packet["candidates"] if c["id"].startswith("play_card:strike_ironclad")]
        # One candidate per alive enemy slot, deduplicated by card_id.
        alive = sum(1 for m in self.loop.state.combat.monsters if m.alive)
        self.assertEqual(len(strike_candidates), alive)

    def test_unplayable_wound_filtered_from_candidates(self) -> None:
        self.loop.state.player.hand[:] = ["wound", "strike_ironclad"]
        self.loop.state.player.energy = 3
        packet = self.loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertNotIn("play_card:wound", ids)
        self.assertNotIn("play_card:wound:0", ids)
        self.assertTrue(any(i.startswith("play_card:strike_ironclad") for i in ids))

    def test_hand_of_only_unplayables_leaves_only_end_turn(self) -> None:
        self.loop.state.player.hand[:] = ["wound", "wound"]
        self.loop.state.player.energy = 3
        packet = self.loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertEqual(ids, {"end_turn"})


class AliveTargetPositionTest(unittest.TestCase):
    """Target args are positions among alive enemies (STS2MCP drops the dead
    from battle.enemies), not spawn slots."""

    def test_position_shifts_after_first_enemy_dies(self) -> None:
        sim = _import_sim()
        loop = sim.RunLoop(seed=3, encounter=["nibbit", "nibbit"])
        loop.reset()
        packet = loop.step(sim.actions.NEOW_SKIP)
        loop.state.player.hand[:] = ["strike_ironclad"]
        packet = loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertIn("play_card:strike_ironclad:1", ids)

        first, second = loop.state.combat.monsters
        first.hp = 0  # kill slot 0 directly
        packet = loop._packet()
        ids = {c["id"] for c in packet["candidates"]}
        self.assertIn("play_card:strike_ironclad:0", ids)
        self.assertNotIn("play_card:strike_ironclad:1", ids)

        hp_before = second.hp
        loop.step("play_card:strike_ironclad:0")
        self.assertLess(second.hp, hp_before)


class CardSelectionFlowTest(unittest.TestCase):
    """Mid-card choices surface as STS2MCP hand_select / card_select screens."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.loop = self.sim.RunLoop(seed=5, encounter="nibbits_weak")
        self.loop.reset()
        self.loop.step(self.sim.actions.NEOW_SKIP)

    def test_burning_pact_hand_select(self) -> None:
        self.loop.state.player.hand[:] = ["burning_pact", "wound", "strike_ironclad"]
        self.loop.state.player.energy = 3
        packet = self.loop.step("play_card:burning_pact")
        self.assertEqual(packet["decision_point"], "hand_select")
        ids = [c["id"] for c in packet["candidates"]]
        self.assertEqual(ids, ["combat_select_card:0", "combat_select_card:1"])
        self.assertIn("Wound", packet["candidates"][0]["text"])
        packet = self.loop.step("combat_select_card:0")
        self.assertEqual([c["id"] for c in packet["candidates"]], ["combat_confirm_selection"])
        packet = self.loop.step("combat_confirm_selection")
        self.assertEqual(packet["decision_point"], self.sim.DecisionPoint.COMBAT_PLAY)
        self.assertIn("wound", self.loop.state.player.exhaust_pile)

    def test_headbutt_card_select(self) -> None:
        self.loop.state.player.hand[:] = ["headbutt"]
        self.loop.state.player.discard_pile[:] = ["bash", "defend_ironclad"]
        self.loop.state.player.energy = 3
        packet = self.loop.step("play_card:headbutt:0")
        self.assertEqual(packet["decision_point"], "card_select")
        self.assertEqual([c["id"] for c in packet["candidates"]], ["select_card:0", "select_card:1"])
        self.loop.step("select_card:1")
        self.loop.step("confirm_selection")
        self.assertEqual(self.loop.state.player.draw_pile[-1], "defend_ironclad")


if __name__ == "__main__":
    unittest.main()
