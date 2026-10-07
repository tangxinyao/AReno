"""Phase 1 data-contract tests: schemas, loader validation, data sanity.

These tests do NOT exercise the engine (there is none yet). They only verify
that the authored JSON matches the schemas module's allowed vocabulary and
that every cross-reference (power ids, upgrade links, enemy move ids) is
resolved.

Author-level regressions ("Cultist's first move must be Incantation",
"Strike base damage is 6") live here because they catch bad hand-edits
before the engine ever sees them.
"""

from __future__ import annotations

import importlib.util
import json
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


class PowersLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        self.powers = self.sim.load_powers()

    def test_six_phase1_powers_present(self) -> None:
        self.assertEqual(
            set(self.powers.keys()),
            {"strength", "dexterity", "vulnerable", "weak", "frail", "ritual"},
        )

    def test_vulnerable_is_turn_scoped_debuff(self) -> None:
        v = self.powers["vulnerable"]
        self.assertEqual(v.kind, "debuff")
        self.assertEqual(v.duration, "turns")

    def test_ritual_ticks_at_end_of_turn(self) -> None:
        r = self.powers["ritual"]
        self.assertEqual(r.kind, "buff")
        self.assertEqual(r.duration, "end_of_turn_tick")


class CardsLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        powers = self.sim.load_powers()
        self.power_ids = set(powers.keys())
        self.cards = self.sim.load_cards(power_ids=self.power_ids)

    def test_card_entry_count(self) -> None:
        # 15 base attack/skill/power cards + 15 upgrades + 1 status (Wound)
        # after Phase 2a mechanics scaffold.
        self.assertEqual(len(self.cards), 31)

    def test_each_regular_base_card_has_upgrade_link(self) -> None:
        bases = [
            c for c in self.cards.values()
            if c.upgraded_from is None and c.card_type not in ("status", "curse")
        ]
        self.assertEqual(len(bases), 15)
        for base in bases:
            self.assertIsNotNone(base.upgrade_of, f"{base.card_id} missing upgrade_of")
            upgraded = self.cards[base.upgrade_of]
            self.assertEqual(upgraded.upgraded_from, base.card_id)

    def test_status_cards_have_no_upgrade_links(self) -> None:
        statuses = [c for c in self.cards.values() if c.card_type == "status"]
        self.assertGreaterEqual(len(statuses), 1)
        for s in statuses:
            self.assertIsNone(s.upgrade_of, f"{s.card_id} should not declare upgrade_of")
            self.assertIsNone(s.upgraded_from, f"{s.card_id} should not declare upgraded_from")

    def test_wound_is_unplayable_zero_cost_status(self) -> None:
        wound = self.cards["wound"]
        self.assertEqual(wound.card_type, "status")
        self.assertEqual(wound.cost, 0)
        self.assertTrue(wound.unplayable)
        self.assertEqual(wound.effects, ())

    def test_pummel_exhausts_on_play(self) -> None:
        self.assertTrue(self.cards["pummel"].exhaust_on_play)
        self.assertTrue(self.cards["pummel+1"].exhaust_on_play)

    def test_carnage_is_ethereal(self) -> None:
        self.assertTrue(self.cards["carnage"].ethereal)
        self.assertTrue(self.cards["carnage+1"].ethereal)

    def test_strike_damage_six(self) -> None:
        strike = self.cards["strike"]
        self.assertEqual(strike.cost, 1)
        self.assertEqual(strike.card_type, "attack")
        self.assertEqual(strike.effects[0].verb, "deal_damage")
        self.assertEqual(strike.effects[0].args["amount"], 6)

    def test_strike_plus_upgrades_to_nine(self) -> None:
        self.assertEqual(self.cards["strike+1"].effects[0].args["amount"], 9)

    def test_bash_applies_vulnerable(self) -> None:
        bash = self.cards["bash"]
        self.assertEqual(len(bash.effects), 2)
        self.assertEqual(bash.effects[1].verb, "apply_power")
        self.assertEqual(bash.effects[1].args["power_id"], "vulnerable")
        self.assertEqual(bash.effects[1].args["amount"], 2)

    def test_cleave_is_aoe(self) -> None:
        cleave = self.cards["cleave"]
        self.assertEqual(cleave.target, "all_enemies")
        self.assertEqual(cleave.effects[0].args["target_scope"], "all_enemies")

    def test_twin_strike_is_two_hits(self) -> None:
        self.assertEqual(self.cards["twin_strike"].effects[0].args["hits"], 2)

    def test_anger_uses_copy_to_discard(self) -> None:
        verbs = [e.verb for e in self.cards["anger"].effects]
        self.assertIn("copy_to_discard", verbs)


class EnemiesLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        powers = self.sim.load_powers()
        self.enemies = self.sim.load_enemies(power_ids=set(powers.keys()))

    def test_act1_enemies_present(self) -> None:
        self.assertEqual(
            set(self.enemies.keys()),
            {
                "jaw_worm", "cultist", "red_louse", "green_louse", "acid_slime_m",
                "blue_slaver", "red_slaver", "fungi_beast",
            },
        )

    def test_cultist_first_move_is_incantation(self) -> None:
        cultist = self.enemies["cultist"]
        first = cultist.movepicker[0]
        self.assertEqual(first.move_id, "incantation")
        self.assertEqual(first.rule, "always_first")
        # Incantation must apply ritual, not deal damage.
        inc = cultist.moves["incantation"]
        self.assertEqual(inc.effects[0].verb, "apply_power")
        self.assertEqual(inc.effects[0].args["power_id"], "ritual")

    def test_jaw_worm_first_move_is_chomp(self) -> None:
        picker = self.enemies["jaw_worm"].movepicker
        first = [s for s in picker if s.rule == "always_first"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].move_id, "chomp")

    def test_acid_slime_m_has_three_moves(self) -> None:
        slime = self.enemies["acid_slime_m"]
        self.assertEqual(set(slime.moves.keys()), {"corrosive_spit", "tackle", "lick"})

    def test_corrosive_spit_applies_weak(self) -> None:
        slime = self.enemies["acid_slime_m"]
        spit = slime.moves["corrosive_spit"]
        self.assertEqual(spit.intent, "attack_debuff")
        verbs = [e.verb for e in spit.effects]
        self.assertEqual(verbs, ["deal_damage", "apply_power"])
        self.assertEqual(spit.effects[1].args["power_id"], "weak")


class LoadAllTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_load_all_cross_resolves_power_ids(self) -> None:
        powers, cards, enemies = self.sim.load_all()
        self.assertEqual(len(powers), 6)
        self.assertEqual(len(cards), 31)
        self.assertEqual(len(enemies), 8)
        # No cross-ref error means every apply_power verb points to a real power.


class EngineContractDocsTest(unittest.TestCase):
    """Guards the engine-development contract — each verb/rule must have
    a non-trivial doc, and the doc keys must track EFFECT_VERBS / MOVE_RULES
    exactly. If someone adds a verb without a doc the Phase 1 engine
    author would silently miss semantics; this test fails loudly instead.
    """

    def setUp(self) -> None:
        self.sim = _import_sim()
        from classify_jev_sts2_sim_for_tests import schemas  # type: ignore
        self.schemas = schemas

    def test_verb_docs_cover_every_effect_verb(self) -> None:
        self.assertEqual(set(self.schemas.VERB_DOCS.keys()), set(self.schemas.EFFECT_VERBS.keys()))

    def test_move_rule_docs_cover_every_move_rule(self) -> None:
        self.assertEqual(set(self.schemas.MOVE_RULE_DOCS.keys()), set(self.schemas.MOVE_RULES))

    def test_each_verb_doc_names_hooks_and_pipeline(self) -> None:
        """Soft content-check: every verb doc must mention 'pipeline' and
        hook names so the engine author has something to grep for."""

        for verb, doc in self.schemas.VERB_DOCS.items():
            self.assertGreater(len(doc), 200, f"{verb} doc looks like a stub")
            self.assertIn("pipeline", doc.lower(), f"{verb} doc missing resolution pipeline")

    def test_deal_damage_doc_covers_all_scaling_terms(self) -> None:
        doc = self.schemas.VERB_DOCS["deal_damage"].lower()
        for term in ("strength", "weak", "vulnerable", "block"):
            self.assertIn(term, doc, f"deal_damage doc missing {term!r}")

    def test_apply_power_doc_covers_ritual_tick_delay(self) -> None:
        doc = self.schemas.VERB_DOCS["apply_power"].lower()
        self.assertIn("ritual", doc)
        self.assertIn("applied_on_turn", doc, "ritual tick-delay rule missing")

    def test_module_docstring_covers_dead_target_rule(self) -> None:
        flat = " ".join((self.schemas.__doc__ or "").split())
        self.assertIn("Dead-target", flat)
        self.assertIn("no-ops silently", flat)


class LoaderValidationTest(unittest.TestCase):
    """Each case writes a bad JSON blob to a temp dir and asserts the loader rejects it."""

    def setUp(self) -> None:
        self.sim = _import_sim()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, content) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def test_rejects_unknown_verb(self) -> None:
        path = self._write(
            "cards.json",
            [{
                "card_id": "junk", "name": "Junk", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgrade_of": "junk+1",
                "effects": [{"verb": "pay_coffee", "args": {}}],
            }, {
                "card_id": "junk+1", "name": "Junk+", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgraded_from": "junk",
                "effects": [],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "verb unknown"):
            self.sim.load_cards(path)

    def test_rejects_bad_target_scope(self) -> None:
        path = self._write(
            "cards.json",
            [{
                "card_id": "weird", "name": "Weird", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgrade_of": "weird+1",
                "effects": [{"verb": "deal_damage", "args": {"amount": 3, "target_scope": "moon", "hits": 1}}],
            }, {
                "card_id": "weird+1", "name": "Weird+", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgraded_from": "weird",
                "effects": [],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "not allowed for verb"):
            self.sim.load_cards(path)

    def test_rejects_card_targeting_player(self) -> None:
        """Only enemies can target the player; cards must not."""

        path = self._write(
            "cards.json",
            [{
                "card_id": "selfhit", "name": "SelfHit", "cost": 0,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgrade_of": "selfhit+1",
                "effects": [{"verb": "deal_damage", "args": {"amount": 1, "target_scope": "player", "hits": 1}}],
            }, {
                "card_id": "selfhit+1", "name": "SelfHit+", "cost": 0,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgraded_from": "selfhit",
                "effects": [],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "cards cannot target 'player'"):
            self.sim.load_cards(path)

    def test_rejects_duplicate_card_id(self) -> None:
        path = self._write(
            "cards.json",
            [
                {"card_id": "a", "name": "A", "cost": 1, "card_type": "attack",
                 "rarity": "common", "target": "single_enemy", "upgrade_of": "a+1",
                 "effects": []},
                {"card_id": "a", "name": "A again", "cost": 1, "card_type": "attack",
                 "rarity": "common", "target": "single_enemy", "upgrade_of": "a+1",
                 "effects": []},
            ],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "duplicate id"):
            self.sim.load_cards(path)

    def test_rejects_broken_upgrade_link(self) -> None:
        path = self._write(
            "cards.json",
            [{
                "card_id": "lonely", "name": "Lonely", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgrade_of": "ghost",   # ghost not defined
                "effects": [],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "not found"):
            self.sim.load_cards(path)

    def test_rejects_card_apply_power_to_unknown_power(self) -> None:
        path = self._write(
            "cards.json",
            [{
                "card_id": "mystic", "name": "Mystic", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgrade_of": "mystic+1",
                "effects": [{"verb": "apply_power",
                             "args": {"power_id": "chronoflux", "amount": 1, "target_scope": "single_enemy"}}],
            }, {
                "card_id": "mystic+1", "name": "Mystic+", "cost": 1,
                "card_type": "attack", "rarity": "common", "target": "single_enemy",
                "upgraded_from": "mystic",
                "effects": [],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "unknown power"):
            self.sim.load_cards(path, power_ids={"strength"})

    def test_rejects_enemy_move_targeting_other_enemies(self) -> None:
        """Enemy moves in Phase 1 may not target other enemies."""

        path = self._write(
            "enemies.json",
            [{
                "enemy_id": "bad", "name": "Bad", "hp_min": 10, "hp_max": 10,
                "moves": {
                    "friendly_fire": {
                        "intent": "attack",
                        "effects": [
                            {"verb": "deal_damage", "args": {"amount": 1, "target_scope": "single_enemy", "hits": 1}}
                        ],
                    }
                },
                "movepicker": [{"move_id": "friendly_fire", "rule": "always_first"}],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "cannot target other enemies"):
            self.sim.load_enemies(path, power_ids={"strength"})

    def test_rejects_movepicker_referencing_unknown_move(self) -> None:
        path = self._write(
            "enemies.json",
            [{
                "enemy_id": "x", "name": "X", "hp_min": 5, "hp_max": 5,
                "moves": {
                    "bite": {"intent": "attack",
                             "effects": [{"verb": "deal_damage",
                                          "args": {"amount": 1, "target_scope": "player", "hits": 1}}]},
                },
                "movepicker": [{"move_id": "nope", "rule": "always_first"}],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "unknown move"):
            self.sim.load_enemies(path, power_ids=set())

    def test_rejects_sequential_without_sequence_index(self) -> None:
        path = self._write(
            "enemies.json",
            [{
                "enemy_id": "y", "name": "Y", "hp_min": 5, "hp_max": 5,
                "moves": {
                    "hit": {"intent": "attack",
                            "effects": [{"verb": "deal_damage",
                                         "args": {"amount": 1, "target_scope": "player", "hits": 1}}]},
                },
                "movepicker": [{"move_id": "hit", "rule": "sequential"}],
            }],
        )
        with self.assertRaisesRegex(self.sim.SimDataError, "sequence_index required"):
            self.sim.load_enemies(path, power_ids=set())


if __name__ == "__main__":
    unittest.main()
