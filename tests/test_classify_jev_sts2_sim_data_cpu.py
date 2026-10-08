"""sts2_sim data-contract tests: loader validation and content accuracy.

Content targets STS2 v0.107.1. The spot checks below pin numbers, costs,
rarities and keywords to the game's own values (cross-checked against the
game's data via Spire Codex and r33hab/sts2 when the data was built), so a
bad hand-edit fails here before the engine sees it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
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


class _Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = _import_sim()
        cls.powers, cls.cards, cls.monsters = cls.sim.load_all()
        cls.encounters = cls.sim.load_encounters(monster_ids=set(cls.monsters))


class CardCatalogTest(_Data):
    def ironclad(self):
        return {cid: c for cid, c in self.cards.items() if c.color == "ironclad"}

    def test_all_87_ironclad_cards_with_upgrades(self) -> None:
        base = [c for c in self.ironclad().values() if not c.upgraded]
        self.assertEqual(len(base), 87)
        for c in base:
            self.assertIn(c.upgrade_of, self.cards, c.card_id)
            self.assertEqual(self.cards[c.upgrade_of].game_id, c.game_id)

    def test_ids_are_lowercased_game_ids(self) -> None:
        for cid, c in self.cards.items():
            self.assertEqual(cid.removesuffix("+1"), c.game_id.lower())

    def test_every_playable_ironclad_card_has_an_effect(self) -> None:
        from importlib import import_module

        effects = import_module(self.sim.__name__ + ".card_effects").CARD_EFFECTS
        missing = sorted({c.game_id for c in self.ironclad().values()
                          if not c.multiplayer_only and c.game_id not in effects})
        self.assertEqual(missing, [])

    def test_descriptions_present_and_numbers_match_vars(self) -> None:
        # Vars that are not printed as a number in the text.
        unprinted = {"calculation_base", "calculation_extra", "calculated_damage", "colossus",
                     "repeat", "power", "attacks", "energy"}
        for cid, c in self.ironclad().items():
            self.assertTrue(c.description, cid)
            self.assertNotIn("[", c.description, cid)
            nums = {int(n) for n in re.findall(r"\d+", c.description)}
            for k, v in c.vars.items():
                if k in unprinted or v == 0:
                    continue
                if k == "extra_damage" and c.vars.get("calculation_base") == 0:
                    continue  # Body Slam: a multiplier on Block, not a printed number
                self.assertIn(v, nums, f"{cid}: var {k}={v} not in {c.description!r}")

    def test_spot_values(self) -> None:
        # (card_id, cost, type, rarity, target, vars subset)
        spots = [
            ("strike_ironclad", 1, "attack", "basic", "single_enemy", {"damage": 6}),
            ("strike_ironclad+1", 1, "attack", "basic", "single_enemy", {"damage": 9}),
            ("defend_ironclad+1", 1, "skill", "basic", "self", {"block": 8}),
            ("bash", 2, "attack", "basic", "single_enemy", {"damage": 8, "vulnerable": 2}),
            ("bash+1", 2, "attack", "basic", "single_enemy", {"damage": 10, "vulnerable": 3}),
            ("bloodletting", 0, "skill", "common", "self", {"hp_loss": 3, "energy": 2}),
            ("bloodletting+1", 0, "skill", "common", "self", {"energy": 3}),
            ("hemokinesis", 1, "attack", "uncommon", "single_enemy", {"hp_loss": 2, "damage": 15}),
            ("cinder", 2, "attack", "common", "single_enemy", {"damage": 18}),
            ("break", 1, "attack", "ancient", "single_enemy", {"damage": 20, "vulnerable": 5}),
            ("corruption", 3, "power", "ancient", "self", {}),
            ("corruption+1", 2, "power", "ancient", "self", {}),
            ("body_slam+1", 0, "attack", "common", "single_enemy", {}),
            ("barricade+1", 2, "power", "rare", "self", {}),
            ("whirlwind", 0, "attack", "uncommon", "all_enemies", {"damage": 5}),
            ("sword_boomerang", 1, "attack", "common", "random_enemy", {"damage": 3, "repeat": 3}),
            ("shrug_it_off+1", 1, "skill", "common", "self", {"block": 11, "cards": 1}),
            ("perfected_strike", 2, "attack", "common", "single_enemy",
             {"calculation_base": 6, "extra_damage": 2}),
            ("unrelenting", 2, "attack", "uncommon", "single_enemy", {"damage": 14}),
            ("demon_form+1", 3, "power", "rare", "self", {"strength": 3}),
            ("crimson_mantle", 1, "power", "rare", "self", {"crimson_mantle": 8}),
        ]
        for cid, cost, ctype, rarity, target, vars_ in spots:
            card = self.cards[cid]
            self.assertEqual((card.cost, card.card_type, card.rarity, card.target),
                             (cost, ctype, rarity, target), cid)
            for k, v in vars_.items():
                self.assertEqual(card.vars[k], v, f"{cid}.{k}")
        self.assertTrue(self.cards["whirlwind"].x_cost)
        self.assertTrue(self.cards["cascade"].x_cost)

    def test_keywords_and_restrictions(self) -> None:
        c = self.cards
        self.assertTrue(c["impervious"].exhausts)
        self.assertTrue(c["offering"].exhausts)
        self.assertTrue(c["dominate"].exhausts)
        self.assertIn("innate", c["aggression+1"].keywords)
        self.assertNotIn("innate", c["aggression"].keywords)
        self.assertFalse(c["feed"].generated_in_combat)
        self.assertTrue(c["tank"].multiplayer_only)
        self.assertTrue(c["demonic_shield"].multiplayer_only)
        self.assertFalse(c["demonic_shield+1"].exhausts, "Demonic Shield+ loses Exhaust")
        self.assertIn("strike", c["pommel_strike"].tags)

    def test_status_and_curse_cards(self) -> None:
        c = self.cards
        self.assertTrue(c["wound"].unplayable)
        self.assertTrue(c["dazed"].unplayable and c["dazed"].ethereal)
        self.assertEqual((c["slimed"].cost, c["slimed"].exhausts), (1, True))
        self.assertTrue(c["burn"].unplayable)
        self.assertEqual(c["infection"].description,
                         "At the end of your turn, if this is in your Hand, take 3 damage.")
        self.assertEqual(c["beckon"].cost, 1)
        for cid, card in c.items():
            if card.card_type in ("status", "curse"):
                self.assertIsNone(card.upgrade_of, cid)


class Sts2CardIdAlignmentTest(_Data):
    """Sim card ids must be real STS2 ids (lower-cased) so live STS2MCP
    candidates (`play_card:<id>`) line up with sim candidates."""

    def test_every_sim_card_is_an_sts2_id(self) -> None:
        sts2 = json.loads((self.sim.DATA_ROOT / "sts2_card_ids.json").read_text())
        bases = {cid.removesuffix("+1") for cid in self.cards}
        self.assertEqual(sorted(bases - set(sts2["all"])), [])

    def test_starting_deck_uses_sts2_ironclad_ids(self) -> None:
        from importlib import import_module

        run = import_module(self.sim.__name__ + ".run")
        self.assertEqual(sorted(set(run.IRONCLAD_STARTING_DECK)), ["bash", "defend_ironclad", "strike_ironclad"])


class PowerCatalogTest(_Data):
    def test_every_referenced_power_exists(self) -> None:
        for m in self.monsters.values():
            for pid, *_ in m.innate_powers:
                self.assertIn(pid, self.powers)
            for move in m.moves.values():
                for step in move.effects:
                    if step.verb == "apply_power":
                        self.assertIn(step.args["power"], self.powers)

    def test_kinds(self) -> None:
        self.assertEqual(self.powers["vulnerable"].kind, "debuff")
        self.assertEqual(self.powers["artifact"].kind, "buff")
        self.assertIn("{amount}", self.powers["demon_form"].description)


class MonsterCatalogTest(_Data):
    def test_roster(self) -> None:
        # 51 Act 1 monsters (Overgrowth + Underdocks), 26 Hive, 25 Glory.
        self.assertEqual(len(self.monsters), 102)
        for mid in ("nibbit", "vantom", "ceremonial_beast", "kin_priest", "lagavulin_matriarch",
                    "soul_fysh", "waterfall_giant", "terror_eel", "knowledge_demon", "the_insatiable",
                    "crusher", "rocket", "queen", "test_subject", "aeonglass", "decimillipede_segment_front"):
            self.assertIn(mid, self.monsters)

    def test_act23_spot_values(self) -> None:
        m = self.monsters
        self.assertEqual((m["exoskeleton"].hp, m["exoskeleton"].hp_asc), ((24, 28), (25, 29)))
        self.assertEqual(m["exoskeleton"].innate_powers, (("hard_to_kill", 9, 9, 0),))
        self.assertEqual(m["test_subject"].hp, (100, 100))
        self.assertEqual(m["queen"].hp_asc, (419, 419))
        self.assertEqual(m["frog_knight"].starting_block, (15, 19, 8))
        bees = m["entomancer"].moves["BEES"].effects[0].args
        self.assertEqual((bees["damage"], bees["hits"]), (3, [7, 8, 9]))
        self.assertEqual(m["owl_magistrate"].moves["VERDICT"].effects[0].args["damage"], [33, 36, 9])

    def test_spot_values(self) -> None:
        m = self.monsters
        self.assertEqual((m["nibbit"].hp, m["nibbit"].hp_asc), ((42, 46), (44, 48)))
        self.assertEqual(m["vantom"].hp, (173, 173))
        self.assertEqual(m["vantom"].innate_powers, (("slippery", 8, 9, 8),))
        self.assertEqual(m["nibbit"].moves["BUTT"].effects[0].args["damage"], [12, 13, 9])
        lash = m["phrog_parasite"].moves["LASH"].effects[0]
        self.assertEqual((lash.args["damage"], lash.args["hits"]), ([4, 5, 9], 4))
        self.assertEqual(m["lagavulin_matriarch"].starting_block, 12)
        self.assertEqual(m["terror_eel"].innate_powers, (("shriek", 70, 75, 8),))
        self.assertEqual(m["nibbit"].moves["SLICE"].intents, ("attack", "defend"))

    def test_every_move_reachable(self) -> None:
        stun_targets = {"ceremonial_beast": {"BEAST_CRY"}, "terror_eel": {"TERROR"},
                        "waterfall_giant": {"ABOUT_TO_BLOW"}, "axebot": {"BOOT_UP"},
                        "test_subject": {"RESPAWN"},
                        **{f"decimillipede_segment_{p}": {"DEAD"} for p in ("front", "middle", "back")}}
        for mid, m in self.monsters.items():
            seen: set[str] = set()
            todo = [m.ai_initial, *stun_targets.get(mid, ())]
            while todo:
                nid = todo.pop()
                if nid in seen:
                    continue
                seen.add(nid)
                node = m.ai_nodes[nid]
                if node.next:
                    todo.append(node.next)
                todo.extend(b.target for b in node.branches)
            reached = {m.ai_nodes[n].move for n in seen if m.ai_nodes[n].node_type == "move"}
            self.assertEqual(reached, set(m.moves), mid)


class EncounterCatalogTest(_Data):
    def test_pools_per_act(self) -> None:
        counts: dict[tuple[str, str], int] = {}
        for e in self.encounters.values():
            counts[(e.act, e.pool)] = counts.get((e.act, e.pool), 0) + 1
        self.assertEqual(counts, {
            ("overgrowth", "weak"): 4, ("overgrowth", "normal"): 12,
            ("overgrowth", "elite"): 3, ("overgrowth", "boss"): 3,
            ("underdocks", "weak"): 4, ("underdocks", "normal"): 10,
            ("underdocks", "elite"): 3, ("underdocks", "boss"): 3,
            ("hive", "weak"): 4, ("hive", "normal"): 10, ("hive", "elite"): 3, ("hive", "boss"): 3,
            ("glory", "weak"): 3, ("glory", "normal"): 9, ("glory", "elite"): 3, ("glory", "boss"): 3,
        })

    def test_generated_rosters_use_known_monsters(self) -> None:
        import random
        from importlib import import_module

        build = import_module(self.sim.__name__ + ".encounters").build_roster
        for e in self.encounters.values():
            for seed in range(5):
                roster = build(e, random.Random(seed))
                self.assertTrue(roster, e.encounter_id)
                for mid, _flags in roster:
                    self.assertIn(mid, self.monsters)


class LoaderValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, payload) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload))
        return path

    @staticmethod
    def _card(**kw):
        base = {"card_id": "x", "game_id": "X", "name": "X", "color": "ironclad", "card_type": "attack",
                "rarity": "common", "target": "single_enemy", "cost": 1, "description": "Deal 1 damage."}
        base.update(kw)
        return base

    @staticmethod
    def _monster(**kw):
        base = {"monster_id": "m", "game_id": "M", "name": "M", "kind": "normal", "hp": [5, 6], "hp_asc": [6, 7],
                "moves": {"HIT": {"name": "Hit", "intents": ["attack"],
                                  "effects": [{"verb": "attack", "args": {"damage": 3}}]}},
                "ai": {"initial": "HIT", "nodes": {"HIT": {"type": "move", "move": "HIT", "next": "HIT"}}}}
        base.update(kw)
        return base

    def test_accepts_minimal_card_and_monster(self) -> None:
        self.sim.load_cards([self._write("c.json", [self._card()])])
        self.sim.load_monsters([self._write("m.json", [self._monster()])], power_ids=set(), card_ids=set())

    def test_rejects_unknown_card_type_and_keyword(self) -> None:
        with self.assertRaisesRegex(self.sim.SimDataError, "card_type"):
            self.sim.load_cards([self._write("c.json", [self._card(card_type="spell")])])
        with self.assertRaisesRegex(self.sim.SimDataError, "keywords"):
            self.sim.load_cards([self._write("c.json", [self._card(keywords=["fleeting"])])])

    def test_rejects_duplicate_and_broken_upgrade_link(self) -> None:
        with self.assertRaisesRegex(self.sim.SimDataError, "duplicate"):
            self.sim.load_cards([self._write("c.json", [self._card(), self._card()])])
        with self.assertRaisesRegex(self.sim.SimDataError, "upgrade_of"):
            self.sim.load_cards([self._write("c.json", [self._card(upgrade_of="x+1")])])

    def test_rejects_status_with_upgrade(self) -> None:
        cards = [self._card(card_type="status", upgrade_of="x+1"),
                 self._card(card_id="x+1", upgraded_from="x")]
        with self.assertRaisesRegex(self.sim.SimDataError, "must not declare upgrade_of"):
            self.sim.load_cards([self._write("c.json", cards)])

    def test_rejects_unknown_monster_verb_and_bad_state(self) -> None:
        bad_verb = self._monster(moves={"HIT": {"name": "Hit", "intents": ["attack"],
                                                "effects": [{"verb": "lasers", "args": {}}]}})
        with self.assertRaisesRegex(self.sim.SimDataError, "verb unknown"):
            self.sim.load_monsters([self._write("m.json", [bad_verb])], power_ids=set(), card_ids=set())
        bad_next = self._monster(ai={"initial": "HIT", "nodes": {"HIT": {"type": "move", "move": "HIT",
                                                                        "next": "NOPE"}}})
        with self.assertRaisesRegex(self.sim.SimDataError, "unknown state"):
            self.sim.load_monsters([self._write("m.json", [bad_next])], power_ids=set(), card_ids=set())

    def test_rejects_unknown_power_card_and_ascension_shape(self) -> None:
        pw = self._monster(moves={"HIT": {"name": "Hit", "intents": ["buff"], "effects": [
            {"verb": "apply_power", "args": {"power": "zeal", "amount": 1, "target": "self"}}]}})
        with self.assertRaisesRegex(self.sim.SimDataError, "power unknown"):
            self.sim.load_monsters([self._write("m.json", [pw])], power_ids={"strength"}, card_ids=set())
        cd = self._monster(moves={"HIT": {"name": "Hit", "intents": ["status"], "effects": [
            {"verb": "add_card", "args": {"card": "goo", "pile": "discard", "count": 1}}]}})
        with self.assertRaisesRegex(self.sim.SimDataError, "card unknown"):
            self.sim.load_monsters([self._write("m.json", [cd])], power_ids=set(), card_ids={"wound"})
        asc = self._monster(moves={"HIT": {"name": "Hit", "intents": ["attack"], "effects": [
            {"verb": "attack", "args": {"damage": [3, 4, 7]}}]}})
        with self.assertRaisesRegex(self.sim.SimDataError, "ascension"):
            self.sim.load_monsters([self._write("m.json", [asc])], power_ids=set(), card_ids=set())

    def test_rejects_encounter_with_unknown_monster(self) -> None:
        enc = [{"encounter_id": "e", "name": "E", "act": "overgrowth", "pool": "weak", "room_type": "monster",
                "monsters": ["ghost"]}]
        with self.assertRaisesRegex(self.sim.SimDataError, "unknown monster"):
            self.sim.load_encounters(self._write("e.json", enc), monster_ids={"nibbit"})


if __name__ == "__main__":
    unittest.main()
