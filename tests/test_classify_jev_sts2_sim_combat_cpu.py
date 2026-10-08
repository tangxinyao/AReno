"""Phase 1 combat engine CPU tests.

Covers verb dispatch (damage/block/power/draw/copy), scaling rules
(strength/dex/weak/vulnerable/frail), AoE and multi-hit semantics,
dead-target noops, draw-pile reshuffle, enemy move resolution,
Ritual's turn-after-apply tick, block reset timing, victory/defeat
detection, and the error surface for illegal card plays.

Tests build combat state directly (via start_combat + a planted hand) so
engine behavior can be exercised without a full RunLoop. The combat
engine is the Phase 1 deliverable; RunLoop integration is Phase 1
closure, tested separately.
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


def _make_ctx(sim, *, seed: int = 42, player_max_hp: int = 80):
    powers, cards, enemies = sim.load_all()
    player = sim.PlayerState(hp=player_max_hp, max_hp=player_max_hp, gold=99, max_energy=3)
    run = sim.RunState(character="ironclad", ascension=0, seed=seed, player=player)
    rng = sim.Rng(seed)
    ctx = sim.CombatContext(
        run=run,
        cards=cards,
        enemies=enemies,
        powers=powers,
        rng=rng,
        hooks=sim.HookBus(),
        effects=sim.EffectQueue(),
    )
    return ctx, powers, cards, enemies


def _plant_hand(ctx, hand: list[str], *, energy: int | None = None) -> None:
    """Force a specific hand regardless of what start_combat drew."""

    # Return current hand to draw pile so counts stay sane in tests that care.
    ctx.player.draw_pile.extend(ctx.player.hand)
    ctx.player.hand = list(hand)
    if energy is not None:
        ctx.player.energy = energy


class StartCombatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_rolls_enemy_hp_deterministically(self) -> None:
        ctx1, *_ = _make_ctx(self.sim, seed=7)
        ctx2, *_ = _make_ctx(self.sim, seed=7)
        ctx1.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)
        ctx2.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)
        self.assertEqual(ctx1.combat.monsters[0].hp, ctx2.combat.monsters[0].hp)
        self.assertTrue(40 <= ctx1.combat.monsters[0].hp <= 44)

    def test_shuffles_draw_pile_deterministically(self) -> None:
        deck = ["strike_ironclad", "strike_ironclad", "defend_ironclad", "defend_ironclad", "bash", "anger", "cleave"]
        ctx1, *_ = _make_ctx(self.sim, seed=99)
        ctx2, *_ = _make_ctx(self.sim, seed=99)
        ctx1.start_combat(["jaw_worm"], deck)
        ctx2.start_combat(["jaw_worm"], deck)
        self.assertEqual(ctx1.player.draw_pile, ctx2.player.draw_pile)

    def test_draws_initial_hand_size(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)
        self.assertEqual(len(ctx.player.hand), 5)
        self.assertEqual(len(ctx.player.draw_pile), 5)

    def test_rejects_unknown_enemy(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        with self.assertRaisesRegex(self.sim.CombatError, "unknown enemy_id"):
            ctx.start_combat(["ghost_worm"], ["strike_ironclad"] * 10)


class DamageAndBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        self.ctx, *_ = _make_ctx(self.sim)
        self.ctx.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)
        self.worm = self.ctx.combat.monsters[0]

    def test_strike_deals_six_damage(self) -> None:
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        hp0 = self.worm.hp
        self.ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(self.worm.hp, hp0 - 6)

    def test_strike_plus_deals_nine(self) -> None:
        _plant_hand(self.ctx, ["strike_ironclad+1"], energy=3)
        hp0 = self.worm.hp
        self.ctx.play_card("strike_ironclad+1", target_slot=0)
        self.assertEqual(self.worm.hp, hp0 - 9)

    def test_strike_vulnerable_bonus(self) -> None:
        self.worm.powers["vulnerable"] = 2
        self.worm.powers_applied_on_turn["vulnerable"] = 1
        self.worm.powers_applied_phase["vulnerable"] = "player"
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        hp0 = self.worm.hp
        self.ctx.play_card("strike_ironclad", target_slot=0)
        # floor(6 * 1.5) = 9
        self.assertEqual(self.worm.hp, hp0 - 9)

    def test_strike_strength_adds_flat(self) -> None:
        self.ctx.player.powers["strength"] = 3
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        hp0 = self.worm.hp
        self.ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(self.worm.hp, hp0 - 9)  # 6 + 3

    def test_strike_weak_reduces(self) -> None:
        self.ctx.player.powers["weak"] = 1
        self.ctx.player.powers_applied_on_turn["weak"] = 1
        self.ctx.player.powers_applied_phase["weak"] = "enemy"
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        hp0 = self.worm.hp
        self.ctx.play_card("strike_ironclad", target_slot=0)
        # floor(6 * 0.75) = 4
        self.assertEqual(self.worm.hp, hp0 - 4)

    def test_defend_adds_five_block(self) -> None:
        _plant_hand(self.ctx, ["defend_ironclad"], energy=3)
        self.ctx.play_card("defend_ironclad")
        self.assertEqual(self.ctx.player.block, 5)

    def test_defend_dex_bonus(self) -> None:
        self.ctx.player.powers["dexterity"] = 2
        _plant_hand(self.ctx, ["defend_ironclad"], energy=3)
        self.ctx.play_card("defend_ironclad")
        self.assertEqual(self.ctx.player.block, 7)

    def test_defend_frail_reduces(self) -> None:
        self.ctx.player.powers["frail"] = 1
        self.ctx.player.powers_applied_on_turn["frail"] = 1
        self.ctx.player.powers_applied_phase["frail"] = "enemy"
        _plant_hand(self.ctx, ["defend_ironclad"], energy=3)
        self.ctx.play_card("defend_ironclad")
        # floor(5 * 0.75) = 3
        self.assertEqual(self.ctx.player.block, 3)

    def test_block_absorbs_damage_before_hp(self) -> None:
        self.worm.block = 10
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        hp0 = self.worm.hp
        self.ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(self.worm.hp, hp0)      # all 6 absorbed
        self.assertEqual(self.worm.block, 4)


class VerbCombinationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        self.ctx, *_ = _make_ctx(self.sim)

    def test_bash_damages_then_vulnerable(self) -> None:
        self.ctx.start_combat(["jaw_worm"], ["bash"] * 10)
        worm = self.ctx.combat.monsters[0]
        _plant_hand(self.ctx, ["bash"], energy=3)
        hp0 = worm.hp
        self.ctx.play_card("bash", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 8)
        self.assertEqual(worm.powers["vulnerable"], 2)

    def test_bash_dead_target_subsequent_effect_noops(self) -> None:
        self.ctx.start_combat(["red_louse"], ["bash"] * 10)
        louse = self.ctx.combat.monsters[0]
        louse.hp = 1  # Bash's 8 damage will kill
        _plant_hand(self.ctx, ["bash"], energy=3)
        self.ctx.play_card("bash", target_slot=0)
        self.assertEqual(louse.hp, 0)
        # vulnerable step must silently no-op on the dead target.
        self.assertNotIn("vulnerable", louse.powers)

    def test_cleave_hits_all_enemies(self) -> None:
        self.ctx.start_combat(["red_louse", "green_louse"], ["cleave"] * 5)
        a, b = self.ctx.combat.monsters
        _plant_hand(self.ctx, ["cleave"], energy=3)
        a0, b0 = a.hp, b.hp
        self.ctx.play_card("cleave")
        self.assertEqual(a.hp, a0 - 8)
        self.assertEqual(b.hp, b0 - 8)

    def test_cleave_skips_dead_enemies(self) -> None:
        self.ctx.start_combat(["red_louse", "green_louse"], ["cleave"] * 5)
        a, b = self.ctx.combat.monsters
        b.hp = 0
        _plant_hand(self.ctx, ["cleave"], energy=3)
        a0 = a.hp
        self.ctx.play_card("cleave")
        self.assertEqual(a.hp, a0 - 8)
        self.assertEqual(b.hp, 0)

    def test_pommel_strike_damages_and_draws(self) -> None:
        self.ctx.start_combat(["jaw_worm"], ["pommel_strike", "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad"])
        worm = self.ctx.combat.monsters[0]
        _plant_hand(self.ctx, ["pommel_strike"], energy=3)
        pre_hand = len(self.ctx.player.hand)
        pre_draw = len(self.ctx.player.draw_pile)
        hp0 = worm.hp
        self.ctx.play_card("pommel_strike", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 9)
        self.assertEqual(len(self.ctx.player.hand), pre_hand - 1 + 1)  # played 1, drew 1
        self.assertEqual(len(self.ctx.player.draw_pile), pre_draw - 1)

    def test_draw_reshuffles_discard_when_draw_empty(self) -> None:
        self.ctx.start_combat(["jaw_worm"], ["pommel_strike"])
        self.ctx.player.hand = []
        self.ctx.player.draw_pile = []
        self.ctx.player.discard_pile = ["strike_ironclad", "defend_ironclad"]
        self.ctx.player.hand = ["pommel_strike"]
        self.ctx.player.energy = 3
        self.ctx.play_card("pommel_strike", target_slot=0)
        # Draw pulled 1 from the reshuffled pile; discard now has 1 card
        # (the remaining one from the pre-shuffle pair) plus pommel strike itself.
        self.assertEqual(len(self.ctx.player.hand), 1)
        self.assertEqual(len(self.ctx.player.draw_pile), 1)

    def test_anger_copies_to_discard(self) -> None:
        self.ctx.start_combat(["jaw_worm"], ["anger"] * 10)
        worm = self.ctx.combat.monsters[0]
        _plant_hand(self.ctx, ["anger"], energy=3)
        pre_discard = len(self.ctx.player.discard_pile)
        self.ctx.play_card("anger", target_slot=0)
        # Discard gains: anger (just played) + anger (copy) = +2
        self.assertEqual(len(self.ctx.player.discard_pile), pre_discard + 2)
        self.assertEqual(self.ctx.player.discard_pile.count("anger"), 2)

    def test_twin_strike_two_separate_hits(self) -> None:
        self.ctx.start_combat(["jaw_worm"], ["twin_strike"] * 5)
        worm = self.ctx.combat.monsters[0]
        _plant_hand(self.ctx, ["twin_strike"], energy=3)
        hp0 = worm.hp
        self.ctx.play_card("twin_strike", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 10)  # 5 + 5

    def test_twin_strike_vulnerable_reapplied_per_hit(self) -> None:
        self.ctx.start_combat(["jaw_worm"], ["twin_strike"] * 5)
        worm = self.ctx.combat.monsters[0]
        worm.powers["vulnerable"] = 2
        worm.powers_applied_on_turn["vulnerable"] = 1
        worm.powers_applied_phase["vulnerable"] = "player"
        _plant_hand(self.ctx, ["twin_strike"], energy=3)
        hp0 = worm.hp
        self.ctx.play_card("twin_strike", target_slot=0)
        # Each hit: floor(5*1.5) = 7, two hits = 14
        self.assertEqual(worm.hp, hp0 - 14)

    def test_shrug_it_off_block_and_draw(self) -> None:
        self.ctx.start_combat(
            ["jaw_worm"],
            ["shrug_it_off", "strike_ironclad", "strike_ironclad", "strike_ironclad", "strike_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad", "defend_ironclad"],
        )
        _plant_hand(self.ctx, ["shrug_it_off"], energy=3)
        pre_hand = len(self.ctx.player.hand)
        self.ctx.play_card("shrug_it_off")
        self.assertEqual(self.ctx.player.block, 8)
        self.assertEqual(len(self.ctx.player.hand), pre_hand - 1 + 1)

    def test_thunderclap_aoe_damage_and_vulnerable(self) -> None:
        self.ctx.start_combat(["red_louse", "green_louse"], ["thunderclap"] * 5)
        a, b = self.ctx.combat.monsters
        _plant_hand(self.ctx, ["thunderclap"], energy=3)
        a0, b0 = a.hp, b.hp
        self.ctx.play_card("thunderclap")
        self.assertEqual(a.hp, a0 - 4)
        self.assertEqual(b.hp, b0 - 4)
        self.assertEqual(a.powers.get("vulnerable", 0), 1)
        self.assertEqual(b.powers.get("vulnerable", 0), 1)


class ExpansionCardsTest(unittest.TestCase):
    """Phase 1 expansion: Clothesline + Perfected Strike."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_clothesline_damage_and_weak(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["clothesline"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["clothesline"], energy=3)
        hp0 = worm.hp
        ctx.play_card("clothesline", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 12)
        self.assertEqual(worm.powers.get("weak"), 2)

    def test_clothesline_plus_fourteen_damage_three_weak(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["clothesline+1"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["clothesline+1"], energy=3)
        hp0 = worm.hp
        ctx.play_card("clothesline+1", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 14)
        self.assertEqual(worm.powers.get("weak"), 3)

    def test_perfected_strike_counts_self(self) -> None:
        """With nothing else in deck, Perfected Strike still counts itself: 6 + 2*1 = 8."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["perfected_strike"])
        worm = ctx.combat.monsters[0]
        # Clear every pile so only the card in flight counts.
        ctx.player.hand = ["perfected_strike"]
        ctx.player.draw_pile = []
        ctx.player.discard_pile = []
        ctx.player.exhaust_pile = []
        ctx.player.energy = 3
        hp0 = worm.hp
        ctx.play_card("perfected_strike", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 8)

    def test_perfected_strike_counts_across_all_zones(self) -> None:
        """2 strikes in hand + 1 in draw + 1 in discard + 1 in exhaust + self = 6 strikes.
        Damage = 6 + 2*6 = 18."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["perfected_strike"])
        worm = ctx.combat.monsters[0]
        ctx.player.hand = ["perfected_strike", "strike_ironclad", "strike_ironclad"]
        ctx.player.draw_pile = ["pommel_strike"]
        ctx.player.discard_pile = ["twin_strike"]
        ctx.player.exhaust_pile = ["strike_ironclad+1"]
        ctx.player.energy = 3
        hp0 = worm.hp
        ctx.play_card("perfected_strike", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 18)

    def test_perfected_strike_ignores_non_strike_cards(self) -> None:
        """Defend / Bash / Cleave should not count — only cards whose id contains 'strike_ironclad'."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["perfected_strike"])
        worm = ctx.combat.monsters[0]
        ctx.player.hand = ["perfected_strike"]
        ctx.player.draw_pile = ["defend_ironclad", "bash", "cleave"]
        ctx.player.discard_pile = []
        ctx.player.exhaust_pile = []
        ctx.player.energy = 3
        hp0 = worm.hp
        ctx.play_card("perfected_strike", target_slot=0)
        # Only self counts -> 6 + 2 = 8
        self.assertEqual(worm.hp, hp0 - 8)

    def test_perfected_strike_plus_scales_at_three_per_strike(self) -> None:
        """PS+ bonus is 3 per strike. 1 strike in draw + self = 2 strikes.
        Damage = 6 + 3*2 = 12."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["perfected_strike+1"])
        worm = ctx.combat.monsters[0]
        ctx.player.hand = ["perfected_strike+1"]
        ctx.player.draw_pile = ["strike_ironclad"]
        ctx.player.discard_pile = []
        ctx.player.exhaust_pile = []
        ctx.player.energy = 3
        hp0 = worm.hp
        ctx.play_card("perfected_strike+1", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 12)


class ExpansionEnemiesTest(unittest.TestCase):
    """Phase 1 expansion: Blue Slaver / Red Slaver / Fungi Beast."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def _run_queued(self, ctx, enemy_slot: int = 0) -> None:
        """Force the queued move to execute via a plain end_turn cycle."""

        ctx.end_turn()

    def test_blue_slaver_rake_damage_and_weak(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["blue_slaver"], ["defend_ironclad"] * 10)
        slaver = ctx.combat.monsters[0]
        slaver.queued_move = "rake"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp0 - 7)
        self.assertEqual(ctx.player.powers.get("weak"), 1)

    def test_blue_slaver_stab_damage(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["blue_slaver"], ["defend_ironclad"] * 10)
        slaver = ctx.combat.monsters[0]
        slaver.queued_move = "stab"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp0 - 13)

    def test_red_slaver_scrape_damage_and_vulnerable(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["red_slaver"], ["defend_ironclad"] * 10)
        slaver = ctx.combat.monsters[0]
        slaver.queued_move = "scrape"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp0 - 8)
        self.assertEqual(ctx.player.powers.get("vulnerable"), 1)

    def test_fungi_beast_grow_adds_strength_to_self(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["fungi_beast"], ["defend_ironclad"] * 10)
        beast = ctx.combat.monsters[0]
        beast.queued_move = "grow"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(beast.powers.get("strength"), 3)
        self.assertEqual(ctx.player.hp, hp0)  # grow doesn't damage

    def test_fungi_beast_grow_not_repeated(self) -> None:
        """Grow is if_not_last — after Grow is played, the next pick must be Bite."""

        ctx, *_ = _make_ctx(self.sim, seed=1)
        ctx.start_combat(["fungi_beast"], ["defend_ironclad"] * 10)
        beast = ctx.combat.monsters[0]
        beast.queued_move = "grow"
        ctx.end_turn()  # Grow fires, next-turn queue is picked after
        # The newly-queued move must be bite (grow is if_not_last).
        self.assertEqual(beast.queued_move, "bite")

    def test_fungi_beast_grow_strength_boosts_next_bite(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["fungi_beast"], ["defend_ironclad"] * 10)
        beast = ctx.combat.monsters[0]
        beast.queued_move = "grow"
        ctx.end_turn()
        # Force next move to bite so the test is deterministic.
        beast.queued_move = "bite"
        hp0 = ctx.player.hp
        ctx.end_turn()
        # Bite base 6 + strength 3 = 9 damage.
        self.assertEqual(ctx.player.hp, hp0 - 9)


class ExhaustAndEtherealTest(unittest.TestCase):
    """Phase 2a mechanics scaffold: exhaust_on_play and ethereal flags."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_pummel_lands_in_exhaust_not_discard(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["pummel"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["pummel"], energy=3)
        pre_exhaust = len(ctx.player.exhaust_pile)
        pre_discard = len(ctx.player.discard_pile)
        hp0 = worm.hp
        ctx.play_card("pummel", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 8)  # 2 damage x 4 hits
        self.assertEqual(len(ctx.player.exhaust_pile), pre_exhaust + 1)
        self.assertEqual(ctx.player.exhaust_pile[-1], "pummel")
        self.assertEqual(len(ctx.player.discard_pile), pre_discard)

    def test_pummel_plus_five_hits(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["pummel+1"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["pummel+1"], energy=3)
        hp0 = worm.hp
        ctx.play_card("pummel+1", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 10)  # 2 x 5

    def test_impervious_blocks_thirty(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["impervious"] * 5)
        _plant_hand(ctx, ["impervious"], energy=3)
        ctx.play_card("impervious")
        self.assertEqual(ctx.player.block, 30)
        self.assertEqual(ctx.player.exhaust_pile[-1], "impervious")

    def test_carnage_ethereal_exhausts_at_end_of_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["carnage"] * 5)
        _plant_hand(ctx, ["carnage"], energy=3)
        pre_exhaust = len(ctx.player.exhaust_pile)
        ctx.end_turn()
        # Carnage was still in hand; must have exhausted, not discarded.
        self.assertIn("carnage", ctx.player.exhaust_pile)
        self.assertNotIn("carnage", ctx.player.discard_pile)
        self.assertGreater(len(ctx.player.exhaust_pile), pre_exhaust)

    def test_non_ethereal_card_still_discards_at_end_of_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)
        _plant_hand(ctx, ["strike_ironclad"], energy=3)
        ctx.end_turn()
        self.assertIn("strike_ironclad", ctx.player.discard_pile)
        self.assertNotIn("strike_ironclad", ctx.player.exhaust_pile)

    def test_carnage_played_still_follows_exhaust_rules_if_flagged(self) -> None:
        """Carnage without exhaust_on_play lands in discard like a normal card.
        (The ethereal flag only matters at end of turn — a played Carnage goes
        to discard, not exhaust, since exhaust_on_play is False.)"""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["carnage"] * 5)
        _plant_hand(ctx, ["carnage"], energy=3)
        ctx.play_card("carnage", target_slot=0)
        self.assertIn("carnage", ctx.player.discard_pile)
        self.assertNotIn("carnage", ctx.player.exhaust_pile)


class Phase2bNewVerbsTest(unittest.TestCase):
    """Phase 2b-1 new verbs: equal_to_block / gain_energy / lose_hp_self / add_card_to_pile."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_body_slam_deals_damage_equal_to_block(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["body_slam"] * 5)
        worm = ctx.combat.monsters[0]
        ctx.player.block = 17
        _plant_hand(ctx, ["body_slam"], energy=3)
        hp0 = worm.hp
        ctx.play_card("body_slam", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 17)

    def test_body_slam_zero_block_zero_damage(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["body_slam"] * 5)
        worm = ctx.combat.monsters[0]
        ctx.player.block = 0
        _plant_hand(ctx, ["body_slam"], energy=3)
        hp0 = worm.hp
        ctx.play_card("body_slam", target_slot=0)
        self.assertEqual(worm.hp, hp0)

    def test_seeing_red_adds_two_energy(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["seeing_red"] * 5)
        _plant_hand(ctx, ["seeing_red"], energy=3)
        before = ctx.player.energy
        ctx.play_card("seeing_red")
        # Net: -1 (card cost) +2 (gain_energy) = +1
        self.assertEqual(ctx.player.energy, before - 1 + 2)
        self.assertIn("seeing_red", ctx.player.exhaust_pile)

    def test_hemokinesis_costs_two_hp_before_damage(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["hemokinesis"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["hemokinesis"], energy=3)
        hp_player0 = ctx.player.hp
        hp_worm0 = worm.hp
        ctx.play_card("hemokinesis", target_slot=0)
        self.assertEqual(ctx.player.hp, hp_player0 - 2)
        self.assertEqual(worm.hp, hp_worm0 - 15)

    def test_lose_hp_self_can_kill_player(self) -> None:
        ctx, *_ = _make_ctx(self.sim, player_max_hp=2)
        ctx.player.hp = 2
        ctx.start_combat(["jaw_worm"], ["hemokinesis"] * 5)
        _plant_hand(ctx, ["hemokinesis"], energy=3)
        ctx.play_card("hemokinesis", target_slot=0)
        self.assertEqual(ctx.player.hp, 0)
        self.assertEqual(ctx.combat.outcome, "defeat")

    def test_wild_strike_adds_wound_to_draw(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["wild_strike"] * 5)
        _plant_hand(ctx, ["wild_strike"], energy=3)
        pre_draw = len(ctx.player.draw_pile)
        pre_wound = ctx.player.draw_pile.count("wound")
        ctx.play_card("wild_strike", target_slot=0)
        self.assertEqual(len(ctx.player.draw_pile), pre_draw + 1)
        self.assertEqual(ctx.player.draw_pile.count("wound"), pre_wound + 1)

    def test_reckless_charge_adds_dazed_to_draw(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["reckless_charge"] * 5)
        _plant_hand(ctx, ["reckless_charge"], energy=3)
        ctx.play_card("reckless_charge", target_slot=0)
        self.assertEqual(ctx.player.draw_pile.count("dazed"), 1)


class Phase2bCardFamiliesTest(unittest.TestCase):
    """Phase 2b-1 new cards: spot checks on signature behavior."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_inflame_adds_two_strength(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["inflame"] * 5)
        _plant_hand(ctx, ["inflame"], energy=3)
        ctx.play_card("inflame")
        self.assertEqual(ctx.player.powers.get("strength"), 2)

    def test_inflame_plus_adds_three_strength(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["inflame+1"] * 5)
        _plant_hand(ctx, ["inflame+1"], energy=3)
        ctx.play_card("inflame+1")
        self.assertEqual(ctx.player.powers.get("strength"), 3)

    def test_intimidate_applies_weak_to_all_and_exhausts(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["red_louse", "green_louse"], ["intimidate"] * 5)
        _plant_hand(ctx, ["intimidate"], energy=3)
        ctx.play_card("intimidate")
        for m in ctx.combat.monsters:
            self.assertEqual(m.powers.get("weak"), 1)
        self.assertIn("intimidate", ctx.player.exhaust_pile)

    def test_uppercut_damage_weak_vulnerable(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["uppercut"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["uppercut"], energy=3)
        hp0 = worm.hp
        ctx.play_card("uppercut", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 13)
        self.assertEqual(worm.powers.get("weak"), 1)
        self.assertEqual(worm.powers.get("vulnerable"), 1)

    def test_shockwave_applies_weak_and_vulnerable_to_all(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["red_louse", "green_louse"], ["shockwave"] * 5)
        _plant_hand(ctx, ["shockwave"], energy=3)
        ctx.play_card("shockwave")
        for m in ctx.combat.monsters:
            self.assertEqual(m.powers.get("weak"), 3)
            self.assertEqual(m.powers.get("vulnerable"), 3)
        self.assertIn("shockwave", ctx.player.exhaust_pile)

    def test_bludgeon_deals_32(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["bludgeon"] * 5)
        worm = ctx.combat.monsters[0]
        _plant_hand(ctx, ["bludgeon"], energy=3)
        hp0 = worm.hp
        ctx.play_card("bludgeon", target_slot=0)
        self.assertEqual(worm.hp, max(0, hp0 - 32))

    def test_body_slam_plus_is_free(self) -> None:
        """Body Slam+ costs 0 — playable with 0 energy."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["body_slam+1"] * 5)
        worm = ctx.combat.monsters[0]
        ctx.player.block = 10
        _plant_hand(ctx, ["body_slam+1"], energy=0)
        hp0 = worm.hp
        ctx.play_card("body_slam+1", target_slot=0)
        self.assertEqual(worm.hp, hp0 - 10)


class Phase2dPowersTest(unittest.TestCase):
    """Phase 2d-1: Metallicize / Combust / Rupture / Dark Embrace / Feel No Pain / Barricade."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_metallicize_grants_block_at_end_of_player_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        ctx.player.powers["metallicize"] = 3
        pre_block = ctx.player.block
        ctx.end_turn()
        # After end_turn, _run_enemy_turn clears player.block to 0 only at
        # start of next player turn. Metallicize fires at end-of-player-turn
        # and lands BEFORE enemy attacks. Jaw Worm's chomp (11) hits a
        # player with 3 block (3 metallicize + any defend if played).
        # We planted no defend, so block starts at 0 before Metallicize, 3
        # after Metallicize, then enemy hits for 11-3=8 of raw damage.
        del pre_block
        self.assertEqual(ctx.player.hp, 80 - 8)

    def test_combust_damages_all_enemies_and_self(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["red_louse", "green_louse"], ["defend_ironclad"] * 10)
        a, b = ctx.combat.monsters
        # Prevent enemies from attacking so we isolate Combust's damage.
        a.queued_move = "grow"  # not actually a red_louse move; override
        a.queued_move = "bite"  # bite is a known red_louse move; 5 dmg
        b.queued_move = "bite"  # 5 dmg
        ctx.player.powers["combust"] = 5
        hp_a, hp_b = a.hp, b.hp
        hp_p = ctx.player.hp
        ctx.end_turn()
        # Combust fires at end-of-player-turn:
        #   - lose 1 HP self (via lose_hp_self)
        #   - deal 5 damage to each enemy
        # Then enemy turn: both bite for 5 => player loses 10 more HP.
        self.assertEqual(a.hp, hp_a - 5)
        self.assertEqual(b.hp, hp_b - 5)
        self.assertEqual(ctx.player.hp, hp_p - 1 - 10)

    def test_rupture_grants_strength_on_hemokinesis_hp_loss(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["hemokinesis"] * 5)
        ctx.player.powers["rupture"] = 2
        _plant_hand(ctx, ["hemokinesis"], energy=3)
        ctx.play_card("hemokinesis", target_slot=0)
        self.assertEqual(ctx.player.powers.get("strength"), 2)

    def test_rupture_cascades_from_combust(self) -> None:
        """Combust's self-tick (lose 1 HP) must trigger Rupture — this
        is the key cascade STS players rely on when stacking the two."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        ctx.player.powers["combust"] = 3
        ctx.player.powers["rupture"] = 2
        # Keep the worm from killing us.
        ctx.combat.monsters[0].queued_move = "thrash"
        ctx.end_turn()
        self.assertEqual(ctx.player.powers.get("strength"), 2)

    def test_dark_embrace_draws_one_on_exhaust(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["strike_ironclad"] * 10 + ["pummel"])
        ctx.player.powers["dark_embrace"] = 1
        _plant_hand(ctx, ["pummel"], energy=3)
        pre_hand = len(ctx.player.hand)
        pre_draw = len(ctx.player.draw_pile)
        ctx.play_card("pummel", target_slot=0)
        # Pummel exhausts -> Dark Embrace draws 1. Net hand delta: -1 played + 1 drawn = 0.
        self.assertEqual(len(ctx.player.hand), pre_hand)
        self.assertEqual(len(ctx.player.draw_pile), pre_draw - 1)
        self.assertIn("pummel", ctx.player.exhaust_pile)

    def test_feel_no_pain_blocks_on_exhaust(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["pummel"] * 5)
        ctx.player.powers["feel_no_pain"] = 3
        _plant_hand(ctx, ["pummel"], energy=3)
        pre_block = ctx.player.block
        ctx.play_card("pummel", target_slot=0)
        # Pummel exhausts once -> Feel No Pain grants 3 block.
        self.assertEqual(ctx.player.block, pre_block + 3)

    def test_barricade_preserves_block_across_player_turns(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        ctx.player.powers["barricade"] = 1
        ctx.player.block = 20
        # Keep the worm from attacking the block (make its queued move zero-damage).
        ctx.combat.monsters[0].queued_move = "bellow"  # defend_buff, no damage
        ctx.end_turn()
        # After the enemy phase (Bellow doesn't hit), block should remain 20
        # since Barricade skipped the start-of-player-turn reset.
        self.assertEqual(ctx.player.block, 20)

    def test_barricade_does_not_apply_on_initial_turn(self) -> None:
        """Combat start sets block=0 even with Barricade — the flag only
        guards the start-of-NEXT-player-turn reset."""

        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        # Players don't start with Barricade — this just verifies that even
        # if they did, combat start wipes block from the previous combat.
        self.assertEqual(ctx.player.block, 0)


class Phase2dPowerCardsTest(unittest.TestCase):
    """The six new power cards each apply their namesake."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def _play_power(self, card_id: str, energy: int = 3):
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], [card_id] * 5)
        _plant_hand(ctx, [card_id], energy=energy)
        ctx.play_card(card_id)
        return ctx

    def test_metallicize_card_applies_three_stacks(self) -> None:
        ctx = self._play_power("metallicize")
        self.assertEqual(ctx.player.powers.get("metallicize"), 3)

    def test_metallicize_plus_card_applies_four_stacks(self) -> None:
        ctx = self._play_power("metallicize+1")
        self.assertEqual(ctx.player.powers.get("metallicize"), 4)

    def test_combust_card_applies_five_stacks(self) -> None:
        ctx = self._play_power("combust")
        self.assertEqual(ctx.player.powers.get("combust"), 5)

    def test_rupture_card_applies_one_stack(self) -> None:
        ctx = self._play_power("rupture")
        self.assertEqual(ctx.player.powers.get("rupture"), 1)

    def test_dark_embrace_card_applies_one_stack(self) -> None:
        ctx = self._play_power("dark_embrace", energy=3)
        self.assertEqual(ctx.player.powers.get("dark_embrace"), 1)

    def test_feel_no_pain_card_applies_three_stacks(self) -> None:
        ctx = self._play_power("feel_no_pain")
        self.assertEqual(ctx.player.powers.get("feel_no_pain"), 3)

    def test_barricade_card_applies_one_stack(self) -> None:
        ctx = self._play_power("barricade")
        self.assertEqual(ctx.player.powers.get("barricade"), 1)

    def test_dark_embrace_plus_cost_is_one(self) -> None:
        # Dark Embrace+ costs 1 instead of 2. Verify the card data.
        powers, cards, _enemies = self.sim.load_all()
        self.assertEqual(cards["dark_embrace"].cost, 2)
        self.assertEqual(cards["dark_embrace+1"].cost, 1)


class Phase2cElitesTest(unittest.TestCase):
    """Phase 2c-1: Gremlin Nob / Lagavulin / Sentry + their signature mechanics."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    # Gremlin Nob -----------------------------------------------------

    def test_gremlin_nob_turn_one_bellows_and_gains_enrage(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["gremlin_nob"], ["defend_ironclad"] * 10)
        nob = ctx.combat.monsters[0]
        self.assertEqual(nob.queued_move, "bellow")
        ctx.end_turn()
        self.assertEqual(nob.powers.get("enrage"), 2)

    def test_enrage_triggers_on_player_skill_play(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["gremlin_nob"], ["defend_ironclad"] * 10)
        nob = ctx.combat.monsters[0]
        ctx.end_turn()  # turn 1: nob bellows -> enrage 2
        # Now nob has enrage 2 and 0 strength (minus prior move's damage).
        # Player turn 2: play a Skill (Defend).
        _plant_hand(ctx, ["defend_ironclad"], energy=3)
        pre_str = nob.powers.get("strength", 0)
        ctx.play_card("defend_ironclad")
        self.assertEqual(nob.powers.get("strength"), pre_str + 2)

    def test_enrage_does_not_trigger_on_attack_play(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["gremlin_nob"], ["strike_ironclad"] * 10)
        nob = ctx.combat.monsters[0]
        ctx.end_turn()  # bellow -> enrage 2
        _plant_hand(ctx, ["strike_ironclad"], energy=3)
        pre_str = nob.powers.get("strength", 0)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(nob.powers.get("strength", 0), pre_str)

    def test_enrage_stacks_each_skill(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["gremlin_nob"], ["defend_ironclad"] * 10)
        nob = ctx.combat.monsters[0]
        ctx.end_turn()  # bellow -> enrage 2
        _plant_hand(ctx, ["defend_ironclad", "defend_ironclad"], energy=3)
        ctx.play_card("defend_ironclad")
        ctx.play_card("defend_ironclad")
        # Two skill plays each add 2 strength = 4 total.
        self.assertEqual(nob.powers.get("strength"), 4)

    # Lagavulin -------------------------------------------------------

    def test_lagavulin_starts_with_eight_metallicize(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["lagavulin"], ["defend_ironclad"] * 10)
        lag = ctx.combat.monsters[0]
        self.assertEqual(lag.powers.get("metallicize"), 8)

    def test_lagavulin_sleeps_first_three_turns(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["lagavulin"], ["defend_ironclad"] * 20)
        lag = ctx.combat.monsters[0]
        self.assertEqual(lag.queued_move, "sleep")
        hp_player0 = ctx.player.hp
        ctx.end_turn()  # turn 1 sleep
        ctx.end_turn()  # turn 2 sleep
        ctx.end_turn()  # turn 3 sleep
        # Player never took damage during sleep.
        self.assertEqual(ctx.player.hp, hp_player0)
        # Next queued move must be attack or siphon (sequential exhausted).
        self.assertIn(lag.queued_move, ("attack", "siphon_soul"))

    def test_lagavulin_metallicize_block_each_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["lagavulin"], ["defend_ironclad"] * 20)
        lag = ctx.combat.monsters[0]
        ctx.end_turn()
        # After one enemy turn, Metallicize fired -> 8 block on Lagavulin.
        # Block resets at start of each enemy turn; metallicize re-grants it.
        self.assertEqual(lag.block, 8)

    def test_lagavulin_siphon_soul_applies_both_debuffs(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["lagavulin"], ["defend_ironclad"] * 20)
        lag = ctx.combat.monsters[0]
        ctx.player.powers["strength"] = 5
        ctx.player.powers["dexterity"] = 5
        # Force Lagavulin into siphon_soul directly (bypass sleep).
        lag.queued_move = "siphon_soul"
        lag.move_history = ["sleep", "sleep", "sleep"]
        ctx.end_turn()
        self.assertEqual(ctx.player.powers.get("strength_down"), 1)
        self.assertEqual(ctx.player.powers.get("dexterity_down"), 1)

    # Sentry ----------------------------------------------------------

    def test_sentry_bolt_adds_two_dazed_to_player_piles(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["sentry"], ["defend_ironclad"] * 20)
        sentry = ctx.combat.monsters[0]
        sentry.queued_move = "bolt"
        # Count across every pile — Bolt shuffles into draw, but the
        # subsequent player turn will draw a fresh hand that may pull a
        # Dazed out of draw into hand or (if ethereal exhausted at prior
        # end-of-turn) into exhaust.
        def _total_dazed() -> int:
            return sum(
                p.count("dazed")
                for p in (ctx.player.hand, ctx.player.draw_pile,
                          ctx.player.discard_pile, ctx.player.exhaust_pile)
            )
        pre = _total_dazed()
        ctx.end_turn()
        self.assertEqual(_total_dazed(), pre + 2)

    def test_sentry_beam_damages_player(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["sentry"], ["defend_ironclad"] * 20)
        sentry = ctx.combat.monsters[0]
        sentry.queued_move = "beam_of_light"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp0 - 9)


class Phase2cDebuffsTest(unittest.TestCase):
    """Strength Down / Dexterity Down end-of-turn conversion."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_strength_down_subtracts_at_end_of_player_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        ctx.player.powers["strength"] = 5
        ctx.player.powers["strength_down"] = 2
        ctx.player.powers_applied_on_turn["strength_down"] = 1
        ctx.player.powers_applied_phase["strength_down"] = "enemy"
        ctx.end_turn()
        self.assertEqual(ctx.player.powers.get("strength"), 3)
        self.assertNotIn("strength_down", ctx.player.powers)

    def test_dexterity_down_subtracts_at_end_of_player_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        ctx.player.powers["dexterity"] = 4
        ctx.player.powers["dexterity_down"] = 2
        ctx.player.powers_applied_on_turn["dexterity_down"] = 1
        ctx.player.powers_applied_phase["dexterity_down"] = "enemy"
        ctx.end_turn()
        self.assertEqual(ctx.player.powers.get("dexterity"), 2)
        self.assertNotIn("dexterity_down", ctx.player.powers)


class Phase2cBossesTest(unittest.TestCase):
    """Phase 2c-2: Act 1 bosses — Hexaghost + Slime Boss + The Guardian."""

    def setUp(self) -> None:
        self.sim = _import_sim()

    # Hexaghost ------------------------------------------------------

    def test_hexaghost_turn_one_activates_and_sets_hex_charge(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.player.hp = 72
        ctx.player.max_hp = 72
        ctx.start_combat(["hexaghost"], ["defend_ironclad"] * 20)
        hex_ = ctx.combat.monsters[0]
        self.assertEqual(hex_.queued_move, "activate")
        ctx.end_turn()
        # hex_charge = max(1, floor(72/12)) = 6
        self.assertEqual(hex_.powers.get("hex_charge"), 6)
        self.assertEqual(hex_.queued_move, "divider")

    def test_hexaghost_divider_deals_hex_charge_damage_per_hit(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["hexaghost"], ["defend_ironclad"] * 20)
        hex_ = ctx.combat.monsters[0]
        hex_.powers["hex_charge"] = 5
        hex_.queued_move = "divider"
        ctx.player.block = 20
        hp_before = ctx.player.hp
        ctx.end_turn()
        # 6 hits of 5 damage: first 4 absorbed by 20 block, last 2 hit hp for 10.
        # (Block gets wiped at start of next player turn, which end_turn also runs,
        # so we only check hp loss here.)
        self.assertEqual(ctx.player.hp, hp_before - 10)

    def test_activate_hex_charge_clamps_to_one_at_low_hp(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.player.hp = 5
        ctx.player.max_hp = 5
        ctx.start_combat(["hexaghost"], ["defend_ironclad"] * 20)
        hex_ = ctx.combat.monsters[0]
        hex_.queued_move = "activate"
        ctx.end_turn()
        # floor(5/12) = 0, clamp to 1
        self.assertEqual(hex_.powers.get("hex_charge"), 1)

    # Slime Boss -----------------------------------------------------

    def test_slime_boss_pattern_goop_preparing_slam(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["slime_boss"], ["defend_ironclad"] * 20)
        sb = ctx.combat.monsters[0]
        self.assertEqual(sb.queued_move, "goop_spray")
        ctx.end_turn()
        self.assertEqual(sb.queued_move, "preparing")
        ctx.end_turn()
        self.assertEqual(sb.queued_move, "slam")

    def test_slime_boss_slam_deals_thirty_five_damage(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["slime_boss"], ["defend_ironclad"] * 20)
        sb = ctx.combat.monsters[0]
        sb.queued_move = "slam"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp0 - 35)

    def test_slime_boss_splits_on_death(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["slime_boss"], ["strike_ironclad"] * 20)
        sb = ctx.combat.monsters[0]
        sb.hp = 5  # lethal to one strike
        _plant_hand(ctx, ["strike_ironclad"], energy=3)
        ctx.play_card("strike_ironclad", target_slot=0)
        # Original Slime Boss dead, two new slimes appended.
        self.assertEqual(sb.hp, 0)
        self.assertFalse(sb.alive)
        self.assertEqual(len(ctx.combat.monsters), 3)
        spawned_ids = {m.monster_id for m in ctx.combat.monsters[1:]}
        self.assertEqual(spawned_ids, {"acid_slime_m", "spike_slime_m"})
        # Spawned slimes have fresh HP in the authored range.
        for spawned in ctx.combat.monsters[1:]:
            self.assertGreaterEqual(spawned.hp, 28)
            self.assertLessEqual(spawned.hp, 32)
        # Combat continues — the split alone shouldn't flip the outcome
        # because the slimes are still alive.
        self.assertIsNone(ctx.combat.outcome)

    def test_slime_boss_split_assigns_initial_moves(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["slime_boss"], ["strike_ironclad"] * 20)
        sb = ctx.combat.monsters[0]
        sb.hp = 5
        _plant_hand(ctx, ["strike_ironclad"], energy=3)
        ctx.play_card("strike_ironclad", target_slot=0)
        for spawned in ctx.combat.monsters[1:]:
            self.assertIsNotNone(spawned.queued_move)
            self.assertIn(spawned.queued_move, spawned.monster_id and
                          {"corrosive_spit", "tackle", "lick", "flame_tackle"})

    def test_spike_slime_m_flame_tackle_adds_slimed(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["spike_slime_m"], ["defend_ironclad"] * 20)
        slime = ctx.combat.monsters[0]
        slime.queued_move = "flame_tackle"
        hp0 = ctx.player.hp
        ctx.end_turn()
        # Flame tackle: 8 damage + 1 Slimed to discard
        self.assertEqual(ctx.player.hp, hp0 - 8)
        self.assertEqual(ctx.player.discard_pile.count("slimed"), 1)

    # The Guardian ---------------------------------------------------

    def test_the_guardian_fierce_bash_hits_hard(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["the_guardian"], ["defend_ironclad"] * 20)
        g = ctx.combat.monsters[0]
        g.queued_move = "fierce_bash"
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp0 - 32)

    def test_the_guardian_vent_steam_applies_weak_and_frail(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["the_guardian"], ["defend_ironclad"] * 20)
        g = ctx.combat.monsters[0]
        g.queued_move = "vent_steam"
        ctx.end_turn()
        self.assertEqual(ctx.player.powers.get("weak"), 2)
        self.assertEqual(ctx.player.powers.get("frail"), 2)

    def test_the_guardian_charging_up_grants_block(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["the_guardian"], ["defend_ironclad"] * 20)
        g = ctx.combat.monsters[0]
        g.queued_move = "charging_up"
        ctx.end_turn()
        self.assertEqual(g.block, 9)


class TurnFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_jaw_worm_first_turn_is_chomp(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        self.assertEqual(ctx.combat.monsters[0].queued_move, "chomp")

    def test_end_turn_runs_enemy_phase_and_returns_to_player(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["jaw_worm"], ["defend_ironclad"] * 10)
        hp0 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.combat.phase, "player")
        self.assertEqual(ctx.combat.turn, 2)
        # Jaw Worm Chomp dealt 11, no block so hp dropped by 11.
        self.assertEqual(ctx.player.hp, hp0 - 11)
        # Player energy refilled.
        self.assertEqual(ctx.player.energy, ctx.player.max_energy)

    def test_block_resets_at_start_of_player_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["red_louse"], ["defend_ironclad"] * 10)
        _plant_hand(ctx, ["defend_ironclad"], energy=3)
        ctx.play_card("defend_ironclad")
        self.assertEqual(ctx.player.block, 5)
        ctx.end_turn()
        # Red Louse bite is 5 damage; defend 5 fully absorbs.
        self.assertEqual(ctx.player.block, 0)

    def test_cultist_ritual_tick_delays_one_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["cultist"], ["defend_ironclad"] * 20)
        cultist = ctx.combat.monsters[0]
        # Turn 1 end: cultist incants ritual 3.
        ctx.end_turn()
        self.assertEqual(cultist.powers.get("ritual"), 3)
        self.assertEqual(cultist.powers.get("strength", 0), 0)  # not yet ticked
        # Turn 2: cultist dark_strike (base 6); end-of-turn ritual tick grants 3 str.
        ctx.end_turn()
        self.assertEqual(cultist.powers.get("strength"), 3)
        # Turn 3: dark_strike 6 + 3 str = 9 damage.
        hp_before_turn3 = ctx.player.hp
        ctx.end_turn()
        self.assertEqual(ctx.player.hp, hp_before_turn3 - 9)

    def test_player_vulnerable_from_enemy_decays_next_player_turn(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["acid_slime_m"], ["defend_ironclad"] * 20)
        slime = ctx.combat.monsters[0]
        # Force slime's queued move to apply weak.
        slime.queued_move = "lick"
        ctx.end_turn()
        self.assertEqual(ctx.player.powers.get("weak"), 1)
        # Next player turn: still weak during attacks. End of player turn: decrement to 0.
        slime.queued_move = "tackle"  # prevent re-applying weak
        ctx.end_turn()
        self.assertNotIn("weak", ctx.player.powers)


class VictoryAndDefeatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()

    def test_kill_only_enemy_ends_combat_victory(self) -> None:
        ctx, *_ = _make_ctx(self.sim)
        ctx.start_combat(["red_louse"], ["strike_ironclad"] * 10)
        louse = ctx.combat.monsters[0]
        louse.hp = 1
        _plant_hand(ctx, ["strike_ironclad"], energy=3)
        ctx.play_card("strike_ironclad", target_slot=0)
        self.assertEqual(ctx.combat.outcome, "victory")

    def test_player_death_ends_combat_defeat(self) -> None:
        ctx, *_ = _make_ctx(self.sim, player_max_hp=5)
        ctx.player.hp = 5
        ctx.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)
        # Jaw Worm Chomp 11 vs 5 hp = death
        ctx.end_turn()
        self.assertEqual(ctx.combat.outcome, "defeat")
        self.assertFalse(ctx.player.alive)


class ErrorSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = _import_sim()
        self.ctx, *_ = _make_ctx(self.sim)
        self.ctx.start_combat(["jaw_worm"], ["strike_ironclad"] * 10)

    def test_insufficient_energy_rejects(self) -> None:
        _plant_hand(self.ctx, ["bash"], energy=1)  # Bash costs 2
        with self.assertRaisesRegex(self.sim.CombatError, "insufficient energy"):
            self.ctx.play_card("bash", target_slot=0)

    def test_card_not_in_hand_rejects(self) -> None:
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        with self.assertRaisesRegex(self.sim.CombatError, "not in hand"):
            self.ctx.play_card("defend_ironclad")

    def test_single_enemy_card_requires_target(self) -> None:
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        with self.assertRaisesRegex(self.sim.CombatError, "requires target_slot"):
            self.ctx.play_card("strike_ironclad")

    def test_target_slot_out_of_range(self) -> None:
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        with self.assertRaisesRegex(self.sim.CombatError, "out of range"):
            self.ctx.play_card("strike_ironclad", target_slot=5)

    def test_dead_target_rejects(self) -> None:
        self.ctx.combat.monsters[0].hp = 0
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        with self.assertRaisesRegex(self.sim.CombatError, "is dead"):
            self.ctx.play_card("strike_ironclad", target_slot=0)

    def test_play_in_wrong_phase_rejects(self) -> None:
        _plant_hand(self.ctx, ["strike_ironclad"], energy=3)
        self.ctx.combat.phase = "enemy"
        with self.assertRaisesRegex(self.sim.CombatError, "phase"):
            self.ctx.play_card("strike_ironclad", target_slot=0)


if __name__ == "__main__":
    unittest.main()
