"""CPU tests for the STS2MCP live-game adapter.

State fixtures follow the JSON shapes documented in Gennadiyev/STS2MCP
`docs/raw-full.md`; no game or network is involved.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _import_live():
    mod_name = "classify_jev_sts2mcp_live_for_tests"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = Path(__file__).resolve().parents[1] / "examples" / "classify" / "jev" / "operators" / "sts2mcp_live.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _card(index, cid, name, *, target="None", can_play=True, upgraded=False, cost="1"):
    return {
        "index": index, "id": cid, "name": name, "type": "Attack", "cost": cost,
        "description": f"{name} text", "target_type": target, "can_play": can_play,
        "is_upgraded": upgraded,
    }


def _combat_state():
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 2, "ascension": 0},
        "player": {
            "hp": 70, "max_hp": 80, "gold": 99, "block": 0, "energy": 3, "max_energy": 3,
            "hand": [
                _card(0, "STRIKE_R", "Strike", target="AnyEnemy"),
                _card(1, "STRIKE_R", "Strike", target="AnyEnemy"),
                _card(2, "DEFEND_R", "Defend", target="Self"),
                _card(3, "BASH", "Bash", target="AnyEnemy", upgraded=True, cost="2"),
                _card(4, "WOUND", "Wound", can_play=False),
            ],
            "potions": [
                {"slot": 0, "name": "Fire Potion", "description": "Deal 20.", "can_use_in_combat": True,
                 "target_type": "AnyEnemy"},
                {"slot": 1, "name": "Swift Potion", "description": "Draw 3.", "can_use_in_combat": True,
                 "target_type": "None"},
            ],
            "max_potion_slots": 3,
            "draw_pile_count": 5, "discard_pile_count": 0, "exhaust_pile_count": 0,
        },
        "battle": {
            "round": 1, "turn": "player", "is_play_phase": True,
            "enemies": [
                {"entity_id": "LOUSE_0", "name": "Louse", "hp": 10, "max_hp": 12, "block": 0,
                 "status": [], "intents": [{"type": "Attack", "label": "6"}]},
                {"entity_id": "LOUSE_1", "name": "Louse", "hp": 11, "max_hp": 12, "block": 0,
                 "status": [{"name": "Strength", "amount": 1}], "intents": [{"type": "Buff"}]},
            ],
        },
    }


class CombatCandidatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.live = _import_live()
        self.cands = self.live.enumerate_candidates(_combat_state())
        self.by_id = {c.id: c for c in self.cands}

    def test_ids(self) -> None:
        self.assertEqual(
            [c.id for c in self.cands],
            [
                "play_card:strike_r:0", "play_card:strike_r:1", "play_card:defend_r",
                "play_card:bash+1:0", "play_card:bash+1:1",
                "use_potion:0:0", "use_potion:0:1", "use_potion:1",
                "end_turn",
            ],
        )

    def test_duplicate_cards_play_first_copy_with_entity_target(self) -> None:
        self.assertEqual(
            self.by_id["play_card:strike_r:1"].request,
            {"action": "play_card", "card_index": 0, "target": "LOUSE_1"},
        )
        self.assertEqual(self.by_id["play_card:defend_r"].request, {"action": "play_card", "card_index": 2})
        self.assertEqual(self.by_id["play_card:bash+1:0"].request["card_index"], 3)

    def test_potion_and_end_turn_requests(self) -> None:
        self.assertEqual(self.by_id["use_potion:0:0"].request, {"action": "use_potion", "slot": 0, "target": "LOUSE_0"})
        self.assertEqual(self.by_id["use_potion:1"].request, {"action": "use_potion", "slot": 1})
        self.assertEqual(self.by_id["end_turn"].request, {"action": "end_turn"})

    def test_not_play_phase_has_no_candidates(self) -> None:
        state = _combat_state()
        state["battle"]["is_play_phase"] = False
        self.assertEqual(self.live.enumerate_candidates(state), [])

    def test_all_ids_parse_as_mcp_actions(self) -> None:
        for c in self.cands:
            name, _ = self.live.A.parse_action(c.id)
            self.assertEqual(name, c.request["action"])

    def test_state_text(self) -> None:
        text = self.live.render_state_text(_combat_state())
        self.assertIn("hand=strike_r x2, defend_r, bash+1, wound", text)
        self.assertIn("Louse#1[11/12 block=0 intent=Buff Strength=1]", text)
        self.assertIn("potions=Fire Potion, Swift Potion", text)


class ScreenCandidatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.live = _import_live()

    def _ids_and_requests(self, state):
        return [(c.id, c.request) for c in self.live.enumerate_candidates(state)]

    def test_hand_select(self) -> None:
        state = {"state_type": "hand_select", "hand_select": {
            "cards": [_card(0, "STRIKE_R", "Strike"), _card(1, "DEFEND_R", "Defend")], "can_confirm": True}}
        self.assertEqual(self._ids_and_requests(state), [
            ("combat_select_card:0", {"action": "combat_select_card", "card_index": 0}),
            ("combat_select_card:1", {"action": "combat_select_card", "card_index": 1}),
            ("combat_confirm_selection", {"action": "combat_confirm_selection"}),
        ])

    def test_rewards_and_card_reward(self) -> None:
        rewards = {"state_type": "rewards", "player": {"potions": [], "max_potion_slots": 3},
                   "rewards": {"items": [{"index": 0, "type": "gold", "description": "25 gold"},
                                         {"index": 1, "type": "card", "description": "card"}],
                               "can_proceed": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(rewards)],
                         ["claim_reward:0", "claim_reward:1", "proceed"])
        card_reward = {"state_type": "card_reward", "card_reward": {
            "cards": [_card(0, "UPPERCUT", "Uppercut")], "can_skip": True}}
        self.assertEqual(self._ids_and_requests(card_reward), [
            ("select_card_reward:0", {"action": "select_card_reward", "card_index": 0}),
            ("skip_card_reward", {"action": "skip_card_reward"}),
        ])

    def test_full_potion_belt_offers_discard_on_rewards(self) -> None:
        potions = [{"slot": i, "name": f"P{i}"} for i in range(3)]
        state = {"state_type": "rewards", "player": {"potions": potions, "max_potion_slots": 3},
                 "rewards": {"items": [], "can_proceed": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(state)],
                         ["discard_potion:0", "discard_potion:1", "discard_potion:2", "proceed"])

    def test_map_event_rest(self) -> None:
        m = {"state_type": "map", "map": {"next_options": [
            {"index": 0, "type": "RestSite", "row": 3, "col": 2, "leads_to": [{"type": "Elite"}]}]}}
        self.assertEqual(self._ids_and_requests(m), [("choose_map_node:0", {"action": "choose_map_node", "index": 0})])
        ev = {"state_type": "event", "event": {"event_id": "NEOW", "in_dialogue": False, "options": [
            {"index": 0, "title": "A", "is_locked": False}, {"index": 1, "title": "B", "is_locked": True}]}}
        self.assertEqual([i for i, _ in self._ids_and_requests(ev)], ["choose_event_option:0"])
        self.assertEqual(self.live.decision_point(ev), "neow_bonus")
        ev["event"]["in_dialogue"] = True
        self.assertEqual([i for i, _ in self._ids_and_requests(ev)], ["advance_dialogue"])
        rest = {"state_type": "rest_site", "rest_site": {"options": [
            {"index": 0, "name": "Rest", "is_enabled": True}, {"index": 1, "name": "Smith", "is_enabled": False}],
            "can_proceed": False}}
        self.assertEqual([i for i, _ in self._ids_and_requests(rest)], ["choose_rest_option:0"])

    def test_shop_filters_unaffordable(self) -> None:
        shop = {"state_type": "shop", "shop": {"items": [
            {"index": 0, "category": "card", "price": 50, "is_stocked": True, "can_afford": True, "card_name": "X"},
            {"index": 1, "category": "relic", "price": 300, "is_stocked": True, "can_afford": False},
            {"index": 2, "category": "card_removal", "price": 75, "is_stocked": False, "can_afford": True},
        ], "can_proceed": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(shop)], ["shop_purchase:0", "proceed"])
        fake = {"state_type": "fake_merchant", "fake_merchant": {"shop": shop["shop"]}}
        self.assertEqual([i for i, _ in self._ids_and_requests(fake)], ["shop_purchase:0", "proceed"])

    def test_treasure_select_overlays_and_game_over(self) -> None:
        tr = {"state_type": "treasure", "treasure": {"relics": [{"index": 0, "name": "Lantern"}], "can_proceed": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(tr)], ["claim_treasure_relic:0", "proceed"])
        self.assertEqual(self._ids_and_requests({"state_type": "treasure", "treasure": {"message": "Opening chest..."}}), [])
        cs = {"state_type": "card_select", "card_select": {
            "cards": [_card(0, "STRIKE_R", "Strike")], "can_confirm": False, "can_cancel": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(cs)], ["select_card:0", "cancel_selection"])
        bs = {"state_type": "bundle_select", "bundle_select": {
            "bundles": [{"index": 0, "cards": []}, {"index": 1, "cards": []}], "can_confirm": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(bs)],
                         ["select_bundle:0", "select_bundle:1", "confirm_bundle_selection"])
        rs = {"state_type": "relic_select", "relic_select": {"relics": [{"index": 0, "name": "Black Star"}], "can_skip": True}}
        self.assertEqual([i for i, _ in self._ids_and_requests(rs)], ["select_relic:0", "skip_relic_selection"])
        self.assertEqual(self.live.decision_point(rs), "boss_relic")
        go = {"state_type": "game_over", "game_over": {"options": ["main_menu"]}}
        self.assertEqual(self._ids_and_requests(go), [("menu_select:main_menu", {"action": "menu_select", "option": "main_menu"})])

    def test_crystal_sphere(self) -> None:
        state = {"state_type": "crystal_sphere", "crystal_sphere": {
            "clickable_cells": [{"x": 4, "y": 7}], "tool": "big",
            "can_use_big_tool": True, "can_use_small_tool": True, "can_proceed": False}}
        self.assertEqual(self._ids_and_requests(state), [
            ("crystal_sphere_click_cell:4:7", {"action": "crystal_sphere_click_cell", "x": 4, "y": 7}),
            ("crystal_sphere_set_tool:small", {"action": "crystal_sphere_set_tool", "tool": "small"}),
        ])

    def test_menu_and_unknown_have_no_candidates(self) -> None:
        for st in ("menu", "overlay", "unknown"):
            self.assertEqual(self.live.enumerate_candidates({"state_type": st}), [])


class _FakeClient:
    def __init__(self, states):
        self.states = list(states)
        self.posts = []

    def get_state(self):
        return self.states[0]

    def post(self, body):
        self.posts.append(body)
        self.states.pop(0)
        return {"status": "ok", "message": "done"}


class LiveBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.live = _import_live()

    def test_step_posts_mapped_request_and_reobserves(self) -> None:
        go = {"state_type": "game_over", "game_over": {"options": ["main_menu"]}}
        fake = _FakeClient([_combat_state(), go])
        backend = self.live.Sts2McpLiveBackend(fake)
        packet = backend.reset()
        self.assertEqual(packet["decision_point"], "combat_play")
        self.assertFalse(packet["done"])
        packet = backend.step({"action_id": "play_card:strike_r:1"})
        self.assertEqual(fake.posts, [{"action": "play_card", "card_index": 0, "target": "LOUSE_1"}])
        self.assertTrue(packet["done"])
        self.assertEqual(packet["step"], 1)

    def test_rejects_unknown_action(self) -> None:
        backend = self.live.Sts2McpLiveBackend(_FakeClient([_combat_state()]))
        backend.reset()
        with self.assertRaises(ValueError):
            backend.step({"action_id": "play_card:carnage"})

    def test_surfaces_mcp_error(self) -> None:
        class _ErrClient(_FakeClient):
            def post(self, body):
                return {"status": "error", "error": "Card requires a target."}

        backend = self.live.Sts2McpLiveBackend(_ErrClient([_combat_state()]))
        backend.reset()
        with self.assertRaisesRegex(RuntimeError, "requires a target"):
            backend.step({"action_id": "end_turn"})


if __name__ == "__main__":
    unittest.main()
