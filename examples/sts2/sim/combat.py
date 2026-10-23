"""Combat engine for Slay the Spire 2 (Ironclad, v0.107.1 rules).

Owns one combat: start -> player turn (draw / energy / plays) -> end turn ->
enemy turn -> round end -> next player turn, until victory or defeat. All
mutations target the RunState passed in.

Turn order, damage math and power timings follow r33hab/sts2's emulator
(`CombatEngine.EndTurn`, `BuffSystem.IncomingDamage`, `CardEffects`,
`RelicEffects`), which is itself checked against live game captures. Where
the emulator and the card text disagree, the card text wins and the spot
carries a comment.

Card behavior lives in card_effects.py, potions in potion_effects.py,
relics in relics.py, monsters in monster_ai.py. This module provides the
primitives they call (powered/unpowered damage, block, powers, draw,
exhaust, HP loss), card selections, and the turn loop.

Piles hold `CardRef`s (state.py): strings equal to the card id that carry
per-copy state (Rampage growth, Frantic Escape's cost, Bound, Replay,
enchantments ...).

Choices the player makes mid-effect (Brand, Headbutt, Seeker Strike, a
potion's pick ...) are generators that `yield` a SelectionRequest. Effects
that trigger outside a card (Stratagem on shuffle, Toolbox and Gambling Chip
at combat start) are queued as follow-ups and run once the current action
has resolved.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .effects import EffectQueue
from .enums import CombatPhase
from .hooks import HookBus
from .rewards import UNSUPPORTED_CARDS
from .rng import Rng
from .schemas import CardSchema, EncounterSchema, MonsterSchema, PotionSchema, PowerSchema
from .state import (
    CardRef,
    CombatState,
    MonsterState,
    PendingSelection,
    PlayerState,
    RunState,
    as_ref,
    index_ref,
    remove_ref,
)


class CombatError(RuntimeError):
    pass


MAX_HAND = 10
BASE_HAND_SIZE = 5
WITHERING_PRESENCE_CARDS = 6
SURROUNDED_FACING_RIGHT = 1
SURROUNDED_FACING_LEFT = 2
AUTOMATION_DRAWS = 10
PANACHE_CARDS = 5

# Debuffs on the player that tick down once per round.
_DURATION_DEBUFFS = ("vulnerable", "weak", "frail")
# Debuffs Artifact negates. Strength / Dexterity loss counts as a debuff.
_DEBUFF_IDS = frozenset({
    "vulnerable", "weak", "frail", "shrink", "constrict", "tangled", "smoggy", "ringing",
    "no_draw", "no_energy_gain", "slow", "strength_loss", "tender", "hex", "dampen",
    "chains_of_binding", "disintegration", "mind_rot", "sloth", "waste_away", "demise",
})
# Debuffs Rend counts (CardEffects.CountRendDebuffs).
REND_DEBUFFS = ("vulnerable", "weak", "frail", "poison", "burn", "shrink", "tangled", "constrict", "smoggy",
                "hex", "dampen", "disintegration")
# End-of-turn-in-hand status/curse effects: (kind, amount).
_TURN_END_IN_HAND: dict[str, tuple[str, int]] = {
    "burn": ("damage", 2),
    "infection": ("damage", 3),
    "toxic": ("damage", 5),
    "decay": ("damage", 2),
    "wither": ("wither", 3),
    "beckon": ("lose_hp", 6),
    "bad_luck": ("lose_hp", 13),
    "regret": ("lose_hp_hand", 0),
    "doubt": ("weak", 1),
    "shame": ("frail", 1),
    "debt": ("gold", 10),
}
# Knowledge Demon's Curse of Knowledge, per cast.
CURSE_OF_KNOWLEDGE_CURSES = ("mind_rot", "sloth", "waste_away")
CURSE_OF_KNOWLEDGE_DAMAGE = (6, 7, 8)
_CURSE_POWER_AMOUNT = {"mind_rot": 1, "sloth": 3, "waste_away": 1}
# Returned to hand at the start of the next turn (Bolas, Thrumming Hatchet).
_RETURNING = frozenset({"BOLAS", "THRUMMING_HATCHET"})


@dataclass
class Play:
    """One resolution of a card (a replay or auto-play is a new Play)."""

    card: CardSchema
    target: MonsterState | None
    x: int = 0  # energy spent on an X-cost card (Chemical X already added)
    auto: bool = False  # auto-played (Havoc, Cascade ...): selections auto-pick
    hp_lost_from_card: int = 0
    bonus: int = 0  # Rampage / Thrash growth banked by this play
    ref: CardRef | None = None  # the copy being played
    energy_spent: int = 0


@dataclass
class PotionUse:
    potion: PotionSchema
    target: MonsterState | None
    auto: bool = False


@dataclass
class SelectionRequest:
    """Yielded by an effect generator to ask the player to pick cards.

    source "hand" / "discard" / "draw": `candidates` index that pile and the
    generator receives the chosen CardRefs. source "generated": the player
    picks among `options` (card ids) and receives the chosen ids."""

    source: str
    candidates: list[int]
    count: int
    purpose: str
    min_count: int | None = None
    options: list[str] | None = None


@dataclass
class _Pending:
    gen: Iterator[Any]
    finish: Callable[[], None]
    auto: bool = False
    label: str = ""
    play: Play | None = None


@dataclass
class _Scratch:
    rupture_owed: int = 0
    dark_embrace_deferred: int = 0
    pending: _Pending | None = None
    autoplaying: int = 0
    exhaust_autoplay: CardRef | None = None
    playing: Play | None = None  # the card whose effect is resolving
    followups: list[Callable[[], Iterator[Any] | None]] = field(default_factory=list)
    returning: list[CardRef] = field(default_factory=list)
    draws_this_combat: int = 0
    panache_armed: bool = False
    panache_count: int = 0
    attack_skill_played: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class RelicHooks:
    """Every point where relics act on a combat; this base holds no relics.

    relics.RelicEngine overrides these for the relics the player holds."""

    # -- queries ---------------------------------------------------------------
    def has(self, relic_id: str) -> bool: return False
    def conserves_energy(self) -> bool: return False
    def keeps_block(self) -> bool: return False
    def skips_first_flush(self) -> bool: return False
    def retains_hand(self) -> bool: return False
    def makes_ethereal(self, card: CardSchema) -> bool: return False
    def upgrades_played(self, card: CardSchema) -> bool: return False
    def prevent_death(self) -> bool: return False
    # -- modifiers ---------------------------------------------------------------
    def card_damage_bonus(self, card: CardSchema) -> int: return 0
    def card_damage_multiplier(self, card: CardSchema) -> float: return 1.0
    def enchanted_damage_bonus(self) -> int: return 0
    def vulnerable_bonus(self) -> float: return 0.0
    def turn_draw_extra(self, turn: int) -> int: return 0
    def modify_max_energy(self, energy: int, turn: int) -> int: return energy
    def modify_hp_loss(self, amount: int) -> int: return amount
    def modify_card_block(self, amount: int) -> int: return amount
    def modify_strength_gain(self, amount: int) -> int: return amount
    def modify_enemy_debuff(self, power_id: str, amount: int) -> int: return amount
    def modify_gold(self, amount: int) -> int: return amount
    def adjust_x(self, x: int) -> int: return x
    # -- events ----------------------------------------------------------------
    def before_opening_hand(self) -> None: pass
    def combat_start(self) -> None: pass
    def after_block_cleared(self, turn: int) -> None: pass
    def turn_start(self, turn: int, *, attacks_last_turn: int, cards_last_turn: int) -> None: pass
    def after_hand_drawn(self, turn: int) -> None: pass
    def before_card_played(self, card: CardSchema, ref: Any, energy_spent: int) -> None: pass
    def after_card_resolved(self, card: CardSchema) -> None: pass
    def after_card_played(self, card: CardSchema, ref: Any, energy_spent: int) -> None: pass
    def after_exhaust(self, ref: Any, *, ethereal: bool) -> None: pass
    def after_shuffle(self) -> None: pass
    def after_hp_lost(self, amount: int, *, attack: bool, player_turn: bool) -> None: pass
    def hp_changed(self) -> None: pass
    def potions_changed(self) -> None: pass
    def after_potion_used(self, potion: PotionSchema) -> None: pass
    def monster_died(self, monster: MonsterState) -> None: pass
    def hand_emptied(self) -> None: pass
    def turn_end(self, turn: int, *, attacks: int) -> None: pass
    def after_flush(self) -> None: pass


NoRelics = RelicHooks


class CombatContext:
    DEFAULT_HAND_SIZE: int = BASE_HAND_SIZE

    def __init__(
        self,
        *,
        run: RunState,
        cards: dict[str, CardSchema],
        monsters: dict[str, MonsterSchema],
        powers: dict[str, PowerSchema],
        rng: Rng,
        hooks: HookBus,
        effects: EffectQueue,
        encounters: dict[str, EncounterSchema] | None = None,
        potions: dict[str, PotionSchema] | None = None,
    ) -> None:
        if run.player is None:
            raise CombatError("RunState.player must be set before CombatContext")
        self._run = run
        self._card_defs = cards
        self._monster_defs = monsters
        self._power_defs = powers
        self._potion_defs = potions or {}
        self._encounters = encounters or {}
        self._rng = rng
        self._hooks = hooks
        self._effects = effects
        self._player: PlayerState = run.player
        self._combat: CombatState | None = run.combat
        self._s = _Scratch()
        self._hand_size = BASE_HAND_SIZE
        self.relics: Any = NoRelics()
        self.room = "monster"

    # ------------------------------------------------------------------
    # Public surface

    @property
    def combat(self) -> CombatState:
        if self._combat is None:
            raise CombatError("combat not started")
        return self._combat

    @property
    def player(self) -> PlayerState:
        return self._player

    @property
    def run(self) -> RunState:
        return self._run

    @property
    def ascension(self) -> int:
        return self._run.ascension

    @property
    def cards(self) -> dict[str, CardSchema]:
        return self._card_defs

    @property
    def potion_defs(self) -> dict[str, PotionSchema]:
        return self._potion_defs

    @property
    def monster_defs(self) -> dict[str, MonsterSchema]:
        return self._monster_defs

    @property
    def power_defs(self) -> dict[str, PowerSchema]:
        return self._power_defs

    @property
    def rng(self) -> Rng:
        return self._rng

    @property
    def hooks(self) -> HookBus:
        return self._hooks

    @property
    def playing(self) -> Play | None:
        return self._s.playing

    def start_combat(
        self,
        monster_ids: list[str],
        starting_deck: list[str],
        *,
        hand_size: int = DEFAULT_HAND_SIZE,
        monster_flags: list[dict[str, Any]] | None = None,
        room: str = "monster",
    ) -> None:
        """Set up a fresh combat against `monster_ids` (slot order).

        `monster_flags[i]` seeds monster i's flags (encounter starters)."""

        from . import monster_ai

        for mid in monster_ids:
            if mid not in self._monster_defs:
                raise CombatError(f"unknown monster_id {mid!r}")
        for cid in starting_deck:
            if cid not in self._card_defs:
                raise CombatError(f"unknown card_id {cid!r}")

        combat = CombatState(turn=0, phase=CombatPhase.START)
        self._run.combat = combat
        self._combat = combat
        self._s = _Scratch()
        self._hand_size = int(hand_size)
        self.room = room

        kind_counts: dict[str, int] = {}
        for i, mid in enumerate(monster_ids):
            monster = self._new_monster(mid)
            if monster_flags is not None:
                monster.flags.update(monster_flags[i])
            monster.kind_index = kind_counts.get(mid, 0)
            kind_counts[mid] = monster.kind_index + 1
            combat.monsters.append(monster)
        self._reslot()
        for m in combat.monsters:
            monster_ai.enter_combat(self, m)

        p = self._player
        p.hand = []
        p.discard_pile = []
        p.exhaust_pile = []
        p.draw_pile = [self._combat_copy(c) for c in starting_deck]
        self._rng.stream("combat_shuffle").shuffle(p.draw_pile)
        # Innate cards (and Royally Approved ones) start on top of the draw pile.
        innate = [c for c in p.draw_pile if self._is_innate(c)]
        if innate:
            rest = [c for c in p.draw_pile if not self._is_innate(c)]
            p.draw_pile = rest + innate
        p.block = 0
        p.powers = {}
        p.energy = 0
        if any(m.powers.get("back_attack_left", 0) > 0 for m in combat.monsters):
            p.powers["surrounded"] = SURROUNDED_FACING_RIGHT

        for m in combat.monsters:
            monster_ai.choose_first_move(self, m)
        self.relics.before_opening_hand()
        self._hooks.dispatch("on_combat_start", {"ctx": self, "monsters": combat.monsters})
        self._begin_player_turn(initial=True)

    def _combat_copy(self, card_id: str) -> CardRef:
        ref = CardRef(card_id)
        if isinstance(card_id, CardRef):
            ref.enchant = card_id.enchant
            ref.enchant_amount = card_id.enchant_amount
        if ref.enchant == "royally_approved":
            ref.retain = True
        return ref

    def _is_innate(self, ref: str) -> bool:
        return "innate" in self._card_defs[ref].keywords or getattr(ref, "enchant", None) == "royally_approved"

    # -- legality --------------------------------------------------------

    def _hand_copies(self, card_id: str) -> list[CardRef]:
        hand = self._player.hand
        return [as_ref(hand, i) for i, c in enumerate(hand) if c == card_id]

    def hand_instance(self, card_id: str) -> CardRef | None:
        """The copy of `card_id` a play would use: the first playable one in hand."""

        if isinstance(card_id, CardRef) and any(c is card_id for c in self._player.hand):
            return card_id
        copies = self._hand_copies(card_id)
        for ref in copies:
            if self._can_play_ref(ref):
                return ref
        return copies[0] if copies else None

    def effective_cost(self, card_id: str) -> int:
        card = self._card_defs[card_id]
        if card.x_cost:
            return 0
        ref = card_id if isinstance(card_id, CardRef) else self.hand_instance(card_id)
        p = self._player.powers
        cost = card.cost
        if ref is not None and ref.cost_override is not None:
            cost = ref.cost_override
        if ref is not None:
            cost += ref.cost_bump
        if card.game_id == "STOMP":
            cost = max(0, cost - self.combat.attacks_played_this_turn)
        if card.card_type == "attack":
            cost += p.get("tangled", 0)
        if card.card_type == "skill" and p.get("corruption", 0) > 0:
            cost = 0
        if card.card_type == "attack" and p.get("free_attack", 0) > 0:
            cost = 0
        if ref is not None and ref.free_turn:
            cost = 0
        return max(0, cost)

    def can_play(self, card_id: str) -> bool:
        ref = self.hand_instance(card_id)
        return ref is not None and self._can_play_ref(ref)

    def _can_play_ref(self, ref: CardRef) -> bool:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER or combat.outcome is not None:
            return False
        if combat.pending_selection is not None:
            return False
        card = self._card_defs[ref]
        if card.unplayable or card.multiplayer_only:
            return False
        if self.effective_cost(ref) > self._player.energy:
            return False
        p = self._player.powers
        if p.get("ringing", 0) > 0 and combat.cards_played_this_turn > 0:
            return False
        if card.card_type == "skill" and p.get("smoggy", 0) > 0 and combat.skills_played_this_turn > 0:
            return False
        if any(c == "normality" for c in self._player.hand) and combat.cards_played_this_turn >= 3:
            return False
        sloth = p.get("sloth", 0)
        if sloth > 0 and combat.cards_played_this_turn >= sloth:
            return False
        if ref.bound and combat.bound_played_this_turn:
            return False
        return True

    # -- player actions --------------------------------------------------

    def play_card(self, card_id: str, *, target_slot: int | None = None) -> None:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER:
            raise CombatError(f"play_card called in phase {combat.phase!r}")
        if combat.outcome is not None:
            raise CombatError("combat already finished")
        if combat.pending_selection is not None:
            raise CombatError("a card selection is pending")
        if card_id not in self._player.hand:
            raise CombatError(f"card {card_id!r} not in hand")
        ref = self.hand_instance(card_id)
        assert ref is not None
        card = self._card_defs[ref]
        if card.unplayable or card.multiplayer_only:
            raise CombatError(f"{card_id!r} is unplayable")
        cost = self.effective_cost(ref)
        if cost > self._player.energy:
            raise CombatError(f"insufficient energy to play {card_id!r}")
        if not self._can_play_ref(ref):
            raise CombatError(f"{card_id!r} cannot be played now")
        target = self._card_target(card, target_slot)
        if target is not None:
            self.face_toward(target)

        spent = self._player.energy if card.x_cost else cost
        x = self.relics.adjust_x(spent) if card.x_cost else 0
        self._player.energy -= spent
        if not ref.free_turn and card.card_type == "attack" and self._player.powers.get("free_attack", 0) > 0:
            self._change_power(self._player, "free_attack", -1)
        ref.free_turn = False
        if ref.bound:
            combat.bound_played_this_turn = True
        remove_ref(self._player.hand, ref)
        combat.last_card_played = str(ref)
        self.relics.before_card_played(card, ref, spent)
        self._resolve_play(Play(card=card, target=target, x=x, ref=ref, energy_spent=spent), from_hand=True)
        self._settle()

    def _card_target(self, card: CardSchema, target_slot: int | None) -> MonsterState | None:
        if card.target != "single_enemy":
            return None
        combat = self.combat
        if target_slot is None:
            raise CombatError(f"{card.card_id!r} requires target_slot")
        if not (0 <= target_slot < len(combat.monsters)):
            raise CombatError(f"target_slot {target_slot} out of range")
        target = combat.monsters[target_slot]
        if not target.alive:
            raise CombatError(f"target slot {target_slot} is dead")
        return target

    def end_turn(self) -> None:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER:
            raise CombatError(f"end_turn called in phase {combat.phase!r}")
        if combat.outcome is not None:
            raise CombatError("combat already finished")
        if combat.pending_selection is not None:
            raise CombatError("a card selection is pending")
        self._end_player_turn()

    # -- potions ---------------------------------------------------------

    def can_use_potion(self, slot: int) -> bool:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER or combat.outcome is not None or combat.pending_selection is not None:
            return False
        p = self._player
        if not (0 <= slot < len(p.potions)) or p.potions[slot] is None:
            return False
        pid = p.potions[slot]
        return pid in self._potion_defs and pid != "fairy_in_a_bottle"

    def use_potion(self, slot: int, *, target_slot: int | None = None) -> None:
        from .potion_effects import POTION_EFFECTS

        if not self.can_use_potion(slot):
            raise CombatError(f"potion slot {slot} cannot be used now")
        p = self._player
        potion = self._potion_defs[p.potions[slot]]
        target = None
        if potion.target == "single_enemy":
            if target_slot is None:
                raise CombatError(f"{potion.potion_id!r} requires target_slot")
            target = self.combat.monsters[target_slot]
            if not target.alive:
                raise CombatError(f"target slot {target_slot} is dead")
            self.face_toward(target)
        p.potions[slot] = None
        use = PotionUse(potion=potion, target=target)

        def finish() -> None:
            self.relics.after_potion_used(potion)
            self._check_end()

        fn = POTION_EFFECTS.get(potion.game_id)
        self._run_effect(fn, use, finish, label=potion.potion_id)
        self._settle()

    def procure_potion(self, potion_id: str) -> bool:
        """Put a potion in the first empty slot (Alchemize, Entropic Brew ...)."""

        if self.relics.has("sozu"):
            return False
        p = self._player
        if None not in p.potions:
            return False
        p.potions[p.potions.index(None)] = potion_id
        self.relics.potions_changed()
        return True

    # -- selections ------------------------------------------------------

    def toggle_selection(self, index: int) -> None:
        sel = self.combat.pending_selection
        if sel is None:
            raise CombatError("no pending selection")
        if index not in sel.candidates:
            raise CombatError(f"index {index} is not selectable")
        if index in sel.selected:
            sel.selected.remove(index)
        elif len(sel.selected) < sel.max_count:
            sel.selected.append(index)
        else:
            raise CombatError("selection is full")

    def can_confirm_selection(self) -> bool:
        sel = self.combat.pending_selection
        return sel is not None and sel.min_count <= len(sel.selected) <= sel.max_count

    def confirm_selection(self, *, skip: bool = False) -> None:
        sel = self.combat.pending_selection
        if sel is None:
            raise CombatError("no pending selection")
        if skip:
            if sel.min_count > 0:
                raise CombatError("this selection cannot be skipped")
            sel.selected = []
        elif not self.can_confirm_selection():
            raise CombatError("selection count not satisfied")
        pending = self._s.pending
        assert pending is not None
        self.combat.pending_selection = None
        self._s.pending = None
        if sel.source == "generated":
            chosen: list[Any] = [sel.options[i] for i in sorted(sel.selected)]
        else:
            pile = self._pile(sel.source)
            chosen = [as_ref(pile, i) for i in sorted(sel.selected)]
        self._s.playing = pending.play
        self._drive(pending, chosen)
        self._settle()

    def _pile(self, source: str) -> list:
        p = self._player
        return {"hand": p.hand, "discard": p.discard_pile, "draw": p.draw_pile, "exhaust": p.exhaust_pile}[source]

    # ------------------------------------------------------------------
    # Effect resolution

    def _resolve_play(self, play: Play, *, from_hand: bool) -> None:
        """Run a card's effect (plus replays), then route the card."""

        from .card_effects import CARD_EFFECTS

        combat = self.combat
        if play.ref is None:
            play.ref = CardRef(play.card.card_id)
        combat.cards_played_this_turn += 1
        combat.cards_played_this_combat += 1
        if play.card.card_type == "attack":
            combat.attacks_played_this_turn += 1
        elif play.card.card_type == "skill":
            combat.skills_played_this_turn += 1

        def finish() -> None:
            self._after_play(play, from_hand=from_hand)

        fn = CARD_EFFECTS.get(play.card.game_id)
        self._run_card_fn(fn, play, finish, extra_plays=self._extra_plays(play))

    def _extra_plays(self, play: Play) -> int:
        """Replay count, Duplication, then One-Two Punch (CombatEngine.PlayCard order)."""

        p = self._player.powers
        extra = play.ref.replay if play.ref is not None else 0
        if p.get("duplication", 0) > 0:
            self._change_power(self._player, "duplication", -1)
            extra += 1
        if play.card.card_type == "attack" and p.get("one_two_punch", 0) > 0:
            self._change_power(self._player, "one_two_punch", -1)
            extra += 1
        return extra

    def _run_card_fn(self, fn, play: Play, finish: Callable[[], None], *, extra_plays: int = 0) -> None:
        def chain() -> None:
            self._s.playing = None
            # Replays re-run the effect before the card is routed.
            if extra_plays > 0 and self.combat.outcome is None:
                self._run_card_fn(fn, play, finish, extra_plays=extra_plays - 1)
            else:
                finish()

        self._s.playing = play
        self._card_side_effects_before(play)
        if fn is None:
            chain()
            return
        if inspect.isgeneratorfunction(fn):
            gen = fn(self, play)
            self._drive(_Pending(gen=gen, finish=chain, auto=play.auto, label=play.card.card_id, play=play), None)
        else:
            fn(self, play)
            chain()

    def _card_side_effects_before(self, play: Play) -> None:
        """Enchantments that act on play: Adroit's Block, Swift's first-play draw."""

        ref = play.ref
        if ref is None or ref.enchant is None:
            return
        if ref.enchant == "adroit":
            self.gain_block(ref.enchant_amount, powered=False)
        elif ref.enchant == "swift" and not ref.enchant_spent:
            ref.enchant_spent = True
            self.draw(ref.enchant_amount)
        elif ref.enchant == "imbued":
            # Pael's Claw: an Imbued skill grants +N Block on top of its effect.
            # The full STS2 Imbued text is unavailable; this stands in for the
            # bonus so an Imbued Defend is observably stronger than a plain one.
            self.gain_block(ref.enchant_amount, powered=False)

    def _run_effect(self, fn, arg: Any, finish: Callable[[], None], *, label: str, auto: bool = False) -> None:
        """Run a non-card effect (potion, follow-up) that may ask for selections."""

        if fn is None:
            finish()
            return
        if inspect.isgeneratorfunction(fn):
            self._drive(_Pending(gen=fn(self, arg), finish=finish, auto=auto, label=label), None)
        else:
            fn(self, arg)
            finish()

    def _drive(self, pending: _Pending, send: Any) -> None:
        gen = pending.gen
        try:
            request = gen.send(send) if send is not None else next(gen)
        except StopIteration:
            pending.finish()
            return
        while True:
            assert isinstance(request, SelectionRequest)
            cands = list(request.candidates)
            want = min(request.count, len(cands))
            min_count = want if request.min_count is None else min(request.min_count, want)
            chosen: list[Any]
            if want <= 0:
                chosen = []
            elif pending.auto or self._s.autoplaying > 0:
                # Auto-played cards resolve their choices without a prompt, taking the
                # game's autoPick (first hand card / top of a pile / first option).
                if request.source == "generated":
                    chosen = [request.options[i] for i in cands[:want]] if request.options else []
                else:
                    pile = self._pile(request.source)
                    pick = cands[:want] if request.source == "hand" else cands[-want:]
                    chosen = [as_ref(pile, i) for i in pick]
            else:
                self.combat.pending_selection = PendingSelection(
                    source=request.source,
                    candidates=cands,
                    min_count=min_count,
                    max_count=want,
                    purpose=request.purpose,
                    source_card=pending.label,
                    options=list(request.options or []),
                )
                self._s.pending = pending
                self._s.playing = None
                return
            try:
                request = gen.send(chosen)
            except StopIteration:
                pending.finish()
                return

    def queue_followup(self, fn: Callable[[], Iterator[Any] | None]) -> None:
        self._s.followups.append(fn)

    def _settle(self) -> None:
        """Run queued follow-ups once nothing is waiting on the player."""

        combat = self.combat
        while combat.pending_selection is None and combat.outcome is None and self._s.followups:
            factory = self._s.followups.pop(0)
            gen = factory()
            if gen is None:
                continue
            self._drive(_Pending(gen=gen, finish=lambda: None, label="followup"), None)
        if combat.pending_selection is None and combat.outcome is None and combat.phase == CombatPhase.PLAYER:
            if not self._player.hand:
                self.relics.hand_emptied()

    def _after_play(self, play: Play, *, from_hand: bool) -> None:
        combat = self.combat
        card = play.card
        p = self._player
        ref = play.ref
        assert ref is not None
        # Bank per-copy growth (Rampage / Thrash, Momentum).
        if play.bonus:
            ref.bonus += play.bonus
        if ref.enchant == "momentum":
            ref.bonus += ref.enchant_amount
        if card.card_type == "attack":
            p.powers.pop("vigor", None)
            if p.powers.get("gigantification", 0) > 0:
                self._change_power(p, "gigantification", -1)
            # Juggling: a copy of the third Attack each turn.
            if combat.attacks_played_this_turn == 3 and p.powers.get("juggling", 0) > 0:
                for _ in range(p.powers["juggling"]):
                    self._add_to_hand(CardRef(card.card_id))
            rage = p.powers.get("rage", 0)
            if rage > 0:
                self.gain_block(rage, powered=False)
            calamity = p.powers.get("calamity", 0)
            for _ in range(calamity):
                pool = self.generation_pool("attack")
                if pool:
                    self._add_to_hand(CardRef(self._rng.stream("card_generation").choice(pool)))
        self.relics.after_card_resolved(card)
        # Route the played card.
        if from_hand or play.auto:
            if self.relics.upgrades_played(card) and self.is_upgradable(ref):
                ref = CardRef(self._card_defs[ref].upgrade_of, like=ref)
            if card.card_type == "power":
                pass
            elif card.exhausts or (card.card_type == "skill" and p.powers.get("corruption", 0) > 0) \
                    or play.auto and self._s.exhaust_autoplay is play.ref:
                self._s.exhaust_autoplay = None
                self.exhaust_card(ref)
            elif self._nostalgia_takes(card):
                p.draw_pile.append(ref)
            else:
                p.discard_pile.append(ref)
            if card.game_id in _RETURNING:
                self._s.returning.append(ref)
        if card.card_type in ("attack", "skill"):
            self._s.attack_skill_played += 1
        # Rupture owed for HP the card itself cost.
        if self._s.rupture_owed > 0 and self.combat.outcome is None:
            owed, self._s.rupture_owed = self._s.rupture_owed, 0
            self.gain_strength(owed)
        self._after_card_played_powers(card)
        self.relics.after_card_played(card, play.ref, play.energy_spent)
        self._hooks.dispatch("on_card_played", {"ctx": self, "card_id": card.card_id, "card": card})
        self._check_end()

    def _nostalgia_takes(self, card: CardSchema) -> bool:
        """Nostalgia: the first N Attacks/Skills each turn go on top of the draw pile."""

        nostalgia = self._player.powers.get("nostalgia", 0)
        return card.card_type in ("attack", "skill") and nostalgia > self._s.attack_skill_played

    def _after_card_played_powers(self, card: CardSchema) -> None:
        """CombatEngine.ApplyAfterCardPlayedPowers: Curl Up, Tender, enemy reactions, Panache."""

        combat = self.combat
        p = self._player
        for m in combat.monsters:
            if m.flags.pop("curl_armed", False):
                curl = m.powers.pop("curl_up", 0)
                if curl > 0 and m.alive:
                    m.block += curl
        if p.powers.get("tender", 0) > 0:
            self._change_power(p, "strength", -1, allow_negative=True)
            self._change_power(p, "dexterity", -1, allow_negative=True)
            combat.tender_taken += 1
        for m in combat.monsters:
            if not m.alive or m.powers.get("withering_presence", 0) <= 0:
                continue
            self._change_power(m, "withering_presence", -1)
            if m.powers.get("withering_presence", 0) <= 0:
                self._add_to_hand(CardRef("wither"))
                m.powers["withering_presence"] = WITHERING_PRESENCE_CARDS
        living = self.alive_monsters()
        if card.card_type == "skill":
            for m in living:
                enrage = m.powers.get("enrage", 0)
                if enrage > 0:
                    self._change_power(m, "strength", enrage, allow_negative=True)
            spark = max((m.powers.get("vital_spark", 0) for m in living), default=0)
            if spark > 0:
                self._change_power(p, "tainted", spark)
        if card.card_type == "power":
            galvanic = max((m.powers.get("galvanic", 0) for m in living), default=0)
            if galvanic > 0:
                self.damage_player(galvanic)
        panache = p.powers.get("panache", 0)
        if panache > 0:
            if not self._s.panache_armed:
                self._s.panache_armed = True
            else:
                self._s.panache_count += 1
                if self._s.panache_count >= PANACHE_CARDS:
                    self._s.panache_count = 0
                    self.damage_all_unpowered(panache)
        for m in living:
            if m.powers.get("slow", 0) > 0:
                m.flags["slow_count"] = m.flags.get("slow_count", 0) + 1

    # ------------------------------------------------------------------
    # Primitives used by card_effects / potion_effects / relics / monster_ai

    def alive_monsters(self) -> list[MonsterState]:
        return [m for m in self.combat.monsters if m.alive]

    def random_monster(self) -> MonsterState | None:
        alive = self.alive_monsters()
        if not alive:
            return None
        return self._rng.stream("combat_target").choice(alive)

    # -- facing (Kaiser Crab's Surrounded) -----------------------------------

    def face_toward(self, target: MonsterState) -> None:
        """SurroundedPower: targeting the monster at your back turns you round."""

        facing = self._player.powers.get("surrounded", 0)
        if facing == 0:
            return
        behind = (target.powers.get("back_attack_left", 0) > 0 if facing == SURROUNDED_FACING_RIGHT
                  else target.powers.get("back_attack_right", 0) > 0)
        if behind:
            self._player.powers["surrounded"] = (SURROUNDED_FACING_LEFT if facing == SURROUNDED_FACING_RIGHT
                                                 else SURROUNDED_FACING_RIGHT)

    def attacks_from_behind(self, monster: MonsterState) -> bool:
        facing = self._player.powers.get("surrounded", 0)
        if facing == SURROUNDED_FACING_RIGHT:
            return monster.powers.get("back_attack_left", 0) > 0
        if facing == SURROUNDED_FACING_LEFT:
            return monster.powers.get("back_attack_right", 0) > 0
        return False

    # -- damage ------------------------------------------------------------

    def _card_damage_mods(self) -> tuple[int, float]:
        """Flat bonus and multiplier the resolving Attack card adds to each hit."""

        play = self._s.playing
        if play is None or play.card.card_type != "attack":
            return 0, 1.0
        flat = self.relics.card_damage_bonus(play.card)
        ref = play.ref
        if ref is not None and ref.enchant is not None:
            if ref.enchant == "sharp":
                flat += ref.enchant_amount
            flat += self.relics.enchanted_damage_bonus()
        mult = float(self.relics.card_damage_multiplier(play.card))
        if self._player.powers.get("gigantification", 0) > 0:
            mult *= 3
        return flat, mult

    def _powered_amount(self, base: int, attacker: Any, defender: Any) -> int:
        a = attacker.powers
        d = defender.powers
        dmg = float(base) + a.get("strength", 0) + a.get("vigor", 0)
        player_attacking = attacker is self._player
        if player_attacking:
            flat, mult = self._card_damage_mods()
            dmg += flat
        if defender is self._player:
            dmg += d.get("tainted", 0)
        if a.get("weak", 0) > 0:
            dmg *= 0.75
        if a.get("shrink", 0) != 0:
            dmg *= 0.70
        if d.get("flutter", 0) > 0:
            dmg *= 0.5
        if d.get("vulnerable", 0) > 0:
            vmult = 1.5
            if player_attacking:
                vmult += self._player.powers.get("cruelty", 0) / 100.0 + self.relics.vulnerable_bonus()
            dmg *= vmult
        if defender is self._player and isinstance(attacker, MonsterState) and self.attacks_from_behind(attacker):
            dmg *= 1.5
        if player_attacking:
            dmg *= mult
        if d.get("soar", 0) > 0:
            dmg *= 0.5
        out = max(0, int(dmg))
        if isinstance(defender, MonsterState):
            slow = defender.flags.get("slow_count", 0)
            if defender.powers.get("slow", 0) > 0 and slow > 0:
                out = int(out * (1.0 + 0.1 * slow))
        if d.get("intangible", 0) > 0:
            out = min(out, 1)
        return out

    def card_damage_preview(self, card_id: str, base: int, target: MonsterState | None) -> int:
        """Per-hit damage a card would deal (for state text)."""

        if target is None:
            dummy = MonsterState(monster_id="?", name="?", hp=1, max_hp=1, slot=-1)
            return self._powered_amount(base, self._player, dummy)
        return self._powered_amount(base, self._player, target)

    def attack_monster(self, target: MonsterState, base: int, *, hits: int = 1) -> int:
        """Powered attack from the player's card; returns total HP lost."""

        lost = 0
        for _ in range(max(0, hits)):
            if not target.alive or self.combat.outcome is not None:
                break
            lost += self._hit_monster(target, base, powered=True)
        return lost

    def attack_all(self, base: int, *, hits: int = 1) -> None:
        for _ in range(max(0, hits)):
            for m in self.alive_monsters():
                self._hit_monster(m, base, powered=True)
            if not self.alive_monsters():
                break

    def attack_random(self, base: int, *, hits: int = 1) -> None:
        for _ in range(max(0, hits)):
            t = self.random_monster()
            if t is None:
                break
            self._hit_monster(t, base, powered=True)

    def damage_monster_unpowered(self, target: MonsterState, amount: int, *, thorns: bool = False) -> int:
        if not target.alive:
            return 0
        if thorns and target.powers.get("thorns", 0) > 0:
            self.damage_player(target.powers["thorns"])
            if not self._player.alive:
                return 0
        return self._hit_monster(target, amount, powered=False)

    def damage_all_unpowered(self, amount: int) -> None:
        for m in self.alive_monsters():
            self._hit_monster(m, amount, powered=False)

    def lose_hp_monster(self, target: MonsterState, amount: int) -> int:
        """Unblockable HP loss on a monster (Powdered Demise)."""

        if not target.alive or amount <= 0:
            return 0
        from . import monster_ai

        loss = amount
        if target.powers.get("intangible", 0) > 0:
            loss = min(loss, 1)
        cap = target.powers.get("hard_to_kill", 0)
        if cap > 0:
            loss = min(loss, cap)
        loss = self._cap_monster_hp_loss(target, loss)
        target.hp = max(0, target.hp - loss)
        if loss > 0 and target.hp > 0:
            monster_ai.after_hp_lost(self, target, loss)
        if target.hp == 0:
            self._monster_died(target)
        return loss

    def _hit_monster(self, target: MonsterState, base: int, *, powered: bool) -> int:
        from . import monster_ai

        if powered:
            thorns = target.powers.get("thorns", 0)
            if thorns > 0:
                self.damage_player(thorns, powered=False)
                if not self._player.alive:
                    return 0
            dmg = self._powered_amount(base, self._player, target)
        else:
            dmg = max(0, int(base))
            if target.powers.get("intangible", 0) > 0:
                dmg = min(dmg, 1)
        cap = target.powers.get("hard_to_kill", 0)
        if cap > 0:
            dmg = min(dmg, cap)
        absorbed = min(target.block, dmg)
        target.block -= absorbed
        hp_loss = dmg - absorbed
        hp_loss = self._cap_monster_hp_loss(target, hp_loss)
        target.hp = max(0, target.hp - hp_loss)
        if powered and target.powers.get("curl_up", 0) > 0:
            target.flags["curl_armed"] = True
        self._hooks.dispatch("on_damaged", {"ctx": self, "target": target, "amount": hp_loss, "blocked": absorbed,
                                            "powered": powered})
        if hp_loss > 0 and target.hp > 0:
            monster_ai.after_hp_lost(self, target, hp_loss)
        if powered and target.hp > 0:
            monster_ai.after_powered_hit(self, target, hp_loss)
        if powered and hp_loss > 0 and target.hp > 0:
            skittish = target.powers.get("skittish", 0)
            if skittish > 0 and not target.flags.get("skittish_spent"):
                target.flags["skittish_spent"] = True
                target.block += skittish
        if target.hp == 0:
            self._monster_died(target)
        return hp_loss

    def _monster_died(self, target: MonsterState) -> None:
        from . import monster_ai

        monster_ai.on_death(self, target)
        if not target.alive and not monster_ai.is_reviving(target):
            self.relics.monster_died(target)

    def _cap_monster_hp_loss(self, target: MonsterState, hp_loss: int) -> int:
        hardened = target.powers.get("hardened_shell", 0)
        if hardened > 0 and hp_loss > 0:
            hp_loss = min(hp_loss, hardened)
            self._change_power(target, "hardened_shell", -hp_loss, allow_zero=True)
        slippery = target.powers.get("slippery", 0)
        if slippery > 0 and hp_loss >= 1:
            hp_loss = 1
            self._change_power(target, "slippery", -1)
        return hp_loss

    def monster_attack(self, monster: MonsterState, base: int, hits: int = 1) -> int:
        """Powered attack from a monster on the player; returns hits that landed unblocked."""

        landed = 0
        p = self._player
        for _ in range(max(0, hits)):
            if not p.alive or not monster.alive:
                break
            dmg = self._powered_amount(base, monster, p)
            if p.powers.get("colossus", 0) > 0 and monster.powers.get("vulnerable", 0) > 0:
                dmg //= 2
            # Ancient: Diamond Diadem halves incoming damage when the player
            # has played at most 2 cards this turn.
            if (self.relics.has("diamond_diadem")
                    and self.combat.cards_played_this_turn <= 2):
                dmg //= 2
            absorbed = min(p.block, dmg)
            p.block -= absorbed
            unblocked = dmg - absorbed
            if unblocked > 0:
                self.combat.took_unblocked = True
                lost = self._lose_player_hp(unblocked, attack=True)
                if lost > 0:
                    landed += 1
                    suck = monster.powers.get("suck", 0)
                    if suck > 0:
                        self._change_power(monster, "strength", suck)
                    cuts = monster.powers.get("paper_cuts", 0)
                    if cuts > 0:
                        p.max_hp = max(1, p.max_hp - cuts)
                        p.hp = min(p.hp, p.max_hp)
                    if p.powers.get("the_gambit", 0) > 0:
                        p.hp = 0
                        self._end_combat("defeat")
            if monster.powers.get("imbalanced", 0) > 0 and dmg > 0 and absorbed > 0 and unblocked == 0:
                monster.flags["off_balance"] = True
            if p.alive:
                thorns = p.powers.get("thorns", 0)
                if thorns > 0:
                    self.damage_monster_unpowered(monster, thorns)
                fb = p.powers.get("flame_barrier", 0)
                if fb > 0:
                    self.damage_monster_unpowered(monster, fb)
        stabs = monster.powers.get("painful_stabs", 0)
        if stabs > 0 and landed > 0 and p.alive:
            for _ in range(stabs * landed):
                self.add_card_to_pile("wound", "discard")
        return landed

    def damage_player(self, amount: int, *, powered: bool = False) -> int:
        """Blockable, unpowered damage to the player (Burn, Constrict, Thorns)."""

        p = self._player
        dmg = max(0, int(amount))
        if p.powers.get("intangible", 0) > 0:
            dmg = min(dmg, 1)
        absorbed = min(p.block, dmg)
        p.block -= absorbed
        unblocked = dmg - absorbed
        if unblocked > 0:
            self._lose_player_hp(unblocked)
        return unblocked

    def lose_hp(self, amount: int, *, from_card: bool = False) -> int:
        """Unblockable HP loss on the player (card costs, Beckon, Inferno tick)."""

        if amount <= 0 or not self._player.alive:
            return 0
        amt = amount
        if self._player.powers.get("intangible", 0) > 0:
            amt = min(amt, 1)
        return self._lose_player_hp(amt, from_card=from_card)

    def _lose_player_hp(self, amount: int, *, from_card: bool = False, attack: bool = False) -> int:
        p = self._player
        if p.powers.get("buffer", 0) > 0:
            self._change_power(p, "buffer", -1)
            return 0
        amount = self.relics.modify_hp_loss(amount)
        if amount <= 0:
            return 0
        before = p.hp
        p.hp = max(0, p.hp - amount)
        lost = before - p.hp
        if lost <= 0:
            return 0
        combat = self.combat
        combat.hp_loss_events_this_combat += 1
        combat.hp_lost_this_combat += lost
        on_our_turn = combat.phase == CombatPhase.PLAYER
        if on_our_turn:
            combat.hp_lost_this_turn += lost
        if p.hp <= 0 and self._prevent_death():
            pass
        if on_our_turn and p.alive:
            rupture = p.powers.get("rupture", 0)
            if rupture > 0:
                if from_card:
                    self._s.rupture_owed += rupture
                else:
                    self.gain_strength(rupture)
            inferno = p.powers.get("inferno", 0)
            if inferno > 0:
                self.damage_all_unpowered(inferno)
        if p.alive:
            self.relics.after_hp_lost(lost, attack=attack, player_turn=on_our_turn)
        self._hooks.dispatch("on_player_hp_lost", {"ctx": self, "amount": lost})
        if not p.alive:
            self._end_combat("defeat")
        return lost

    def _prevent_death(self) -> bool:
        """Lizard Tail (once per run), then Fairy in a Bottle."""

        p = self._player
        if self.relics.prevent_death():
            return True
        if "fairy_in_a_bottle" in p.potions:
            p.potions[p.potions.index("fairy_in_a_bottle")] = None
            p.hp = max(int(p.max_hp * 0.3), 1)
            self.relics.potions_changed()
            return True
        return False

    def heal_player(self, amount: int) -> None:
        p = self._player
        p.hp = min(p.max_hp, p.hp + max(0, amount))
        self.relics.hp_changed()

    def gain_max_hp(self, amount: int) -> None:
        self._player.max_hp += amount
        self.heal_player(amount)

    def gain_gold(self, amount: int) -> None:
        self._player.gold += self.relics.modify_gold(amount)

    # -- block / energy / stats ------------------------------------------------

    def gain_block(self, amount: int, *, powered: bool = True) -> int:
        p = self._player
        if powered:
            if p.powers.get("no_block", 0) > 0:
                return 0
            blk = float(amount) + p.powers.get("dexterity", 0)
            play = self._s.playing
            fasten = p.powers.get("fasten", 0)
            if fasten > 0 and play is not None and "defend" in play.card.tags:
                blk += fasten
            if p.powers.get("frail", 0) > 0:
                blk *= 0.75
            eff = max(0, int(blk))
            if play is not None:
                eff = self.relics.modify_card_block(eff)
        else:
            eff = max(0, int(amount))
        if eff <= 0:
            return 0
        if powered and p.powers.get("unmovable", 0) > self.combat.block_gains_this_turn:
            eff *= 2
            self.combat.block_gains_this_turn += 1
        p.block += eff
        jug = p.powers.get("juggernaut", 0)
        if jug > 0:
            t = self.random_monster()
            if t is not None:
                self.damage_monster_unpowered(t, jug)
        return eff

    def monster_gain_block(self, m: MonsterState, amount: int) -> int:
        """BuffSystem.IncomingBlock for a monster: Dexterity and Frail apply."""

        blk = float(amount) + m.powers.get("dexterity", 0)
        if m.powers.get("frail", 0) > 0:
            blk *= 0.75
        eff = max(0, int(blk))
        m.block += eff
        return eff

    def gain_energy(self, amount: int) -> None:
        if amount <= 0 or self._player.powers.get("no_energy_gain", 0) > 0:
            return
        self._player.energy += amount

    def gain_strength(self, amount: int) -> None:
        if amount > 0:
            amount = self.relics.modify_strength_gain(amount)
        self._change_power(self._player, "strength", amount, allow_negative=True)

    def gain_temporary_strength(self, amount: int) -> None:
        self.gain_strength(amount)
        self._player.powers["temporary_strength"] = self._player.powers.get("temporary_strength", 0) + amount

    def apply_power(self, target: Any, power_id: str, amount: int, *, source: Any = None) -> bool:
        """Apply `amount` of `power_id` to target. Returns False if Artifact blocked it."""

        if amount == 0:
            return False
        if power_id not in self._power_defs:
            raise CombatError(f"unknown power {power_id!r}")
        if isinstance(target, MonsterState) and not target.alive:
            return False
        is_debuff = power_id in _DEBUFF_IDS or (power_id in ("strength", "dexterity") and amount < 0)
        if is_debuff and isinstance(target, MonsterState) and source is self._player:
            amount = self.relics.modify_enemy_debuff(power_id, amount)
        if is_debuff and target.powers.get("artifact", 0) > 0:
            self._change_power(target, "artifact", -1)
            return False
        if power_id == "dampen" and target is self._player and target.powers.get("dampen", 0) <= 0:
            self._apply_dampen()
        if power_id == "strength" and target is self._player and amount > 0:
            self.gain_strength(amount)
        else:
            self._change_power(target, power_id, amount, allow_negative=power_id in ("strength", "dexterity"))
        if power_id == "vulnerable" and source is self._player and isinstance(target, MonsterState):
            vicious = self._player.powers.get("vicious", 0)
            if vicious > 0:
                self.draw(vicious)
        self._hooks.dispatch("on_power_applied", {"ctx": self, "target": target, "power": power_id,
                                                  "amount": amount, "source": source})
        return True

    def _change_power(self, owner: Any, pid: str, delta: int, *, allow_negative: bool = False,
                      allow_zero: bool = False) -> None:
        cur = owner.powers.get(pid, 0) + delta
        if cur == 0 or (cur < 0 and not allow_negative):
            if allow_zero and cur == 0:
                owner.powers[pid] = 0
                return
            owner.powers.pop(pid, None)
        else:
            owner.powers[pid] = cur

    # -- Dampen ----------------------------------------------------------------

    def _all_piles(self) -> list[list]:
        p = self._player
        return [p.hand, p.draw_pile, p.discard_pile, p.exhaust_pile]

    def _apply_dampen(self) -> None:
        """DampenPower.AfterApplied: every upgraded card is downgraded while it lasts."""

        for pile in self._all_piles():
            for i, cid in enumerate(pile):
                card = self._card_defs[cid]
                if card.upgraded_from is not None:
                    new = CardRef(card.upgraded_from, like=as_ref(pile, i))
                    new.dampened_from = card.card_id
                    pile[i] = new

    def remove_dampen(self) -> None:
        self._player.powers.pop("dampen", None)
        for pile in self._all_piles():
            for i, cid in enumerate(pile):
                ref = as_ref(pile, i)
                if ref.dampened_from is not None:
                    pile[i] = CardRef(ref.dampened_from, like=ref)
                    pile[i].dampened_from = None

    # -- cards / piles ------------------------------------------------------

    def draw(self, amount: int) -> list[str]:
        drawn: list[str] = []
        p = self._player
        for _ in range(max(0, amount)):
            if p.powers.get("no_draw", 0) > 0:
                break
            card_id = self._draw_one()
            if card_id is None:
                break
            drawn.append(card_id)
        return drawn

    def _draw_one(self) -> CardRef | None:
        p = self._player
        if len(p.hand) >= MAX_HAND:
            return None
        if not p.draw_pile:
            if not p.discard_pile:
                return None
            self.shuffle_discard_into_draw()
        ref = as_ref(p.draw_pile, len(p.draw_pile) - 1)
        p.draw_pile.pop()
        combat = self.combat
        chains = p.powers.get("chains_of_binding", 0)
        if chains > 0 and combat.phase == CombatPhase.PLAYER and combat.bound_this_turn < chains:
            combat.bound_this_turn += 1
            ref.bound = True
        p.hand.append(ref)
        self._s.draws_this_combat += 1
        automation = p.powers.get("automation", 0)
        if automation > 0 and self._s.draws_this_combat % AUTOMATION_DRAWS == 0:
            self.gain_energy(automation)
        self._hooks.dispatch("on_card_drawn", {"ctx": self, "card_id": ref})
        card = self._card_defs[ref]
        if card.game_id == "VOID":
            p.energy = max(0, p.energy - card.v("energy"))
        if p.powers.get("hellraiser", 0) > 0 and "Strike" in card.name and card.card_type == "attack":
            self._autoplay_from_hand(ref, random_target=True)
        return ref

    def shuffle_discard_into_draw(self) -> None:
        p = self._player
        p.draw_pile = list(p.discard_pile) + p.draw_pile
        p.discard_pile.clear()
        # Canonical ordering before Fisher-Yates (CardEffects.ShuffleDiscardIntoDraw
        # in r33hab/sts2): sort by (game_id ordinal, upgraded flag). Position is
        # the stable tiebreaker so cards that match on key keep insertion order.
        defs = self._card_defs
        p.draw_pile.sort(key=lambda ref, d=defs: (d[ref].game_id, 1 if str(ref).endswith("+1") else 0))
        self._rng.stream("combat_shuffle").shuffle(p.draw_pile)
        self.relics.after_shuffle()
        for _ in range(p.powers.get("stratagem", 0)):
            self.queue_followup(self._stratagem)
        self._hooks.dispatch("on_shuffle", {"ctx": self})

    def _stratagem(self) -> Iterator[Any] | None:
        pile = self._player.draw_pile
        if not pile:
            return None
        return self._choose_to_hand("draw", list(range(len(pile))), purpose="to_hand")

    def _choose_to_hand(self, source: str, candidates: list[int], *, purpose: str, free: bool = False):
        chosen = yield SelectionRequest(source=source, candidates=candidates, count=1, purpose=purpose)
        pile = self._pile(source)
        for ref in chosen:
            remove_ref(pile, ref)
            if free:
                ref.free_turn = True
            self._add_to_hand(ref)

    def _add_to_hand(self, card_id: str) -> CardRef:
        p = self._player
        ref = card_id if isinstance(card_id, CardRef) else CardRef(card_id)
        if len(p.hand) < MAX_HAND:
            p.hand.append(ref)
        else:
            p.discard_pile.append(ref)
        return ref

    def add_to_hand(self, card_id: str) -> CardRef:
        return self._add_to_hand(card_id)

    def add_card_to_pile(self, card_id: str, pile: str) -> CardRef:
        p = self._player
        if card_id not in self._card_defs:
            raise CombatError(f"unknown card {card_id!r}")
        ref = card_id if isinstance(card_id, CardRef) else CardRef(card_id)
        if pile == "hand":
            self._add_to_hand(ref)
        elif pile == "discard":
            p.discard_pile.append(ref)
        elif pile == "draw_top":
            p.draw_pile.append(ref)
        elif pile == "draw_random":
            pos = self._rng.stream("combat_shuffle").randint(0, len(p.draw_pile))
            p.draw_pile.insert(pos, ref)
        else:
            raise CombatError(f"unknown pile {pile!r}")
        return ref

    def exhaust_card(self, card_id: str, *, ethereal: bool = False) -> None:
        """Put `card_id` (already removed from its pile) into the exhaust pile."""

        p = self._player
        ref = card_id if isinstance(card_id, CardRef) else CardRef(card_id)
        p.exhaust_pile.append(ref)
        self.combat.cards_exhausted_this_turn += 1
        self._hooks.dispatch("on_card_exhausted", {"ctx": self, "card_id": ref})
        card = self._card_defs[ref]
        if card.game_id == "DRUM_OF_BATTLE":
            self.gain_energy(card.v("energy"))
        fnp = p.powers.get("feel_no_pain", 0)
        if fnp > 0:
            self.gain_block(fnp, powered=False)
        de = p.powers.get("dark_embrace", 0)
        if de > 0:
            if ethereal:
                self._s.dark_embrace_deferred += de
            else:
                self.draw(de)
        self.relics.after_exhaust(ref, ethereal=ethereal)

    def exhaust_from_hand(self, card_id: str) -> None:
        hand = self._player.hand
        ref = hand[index_ref(hand, card_id)]
        remove_ref(hand, ref)
        self.exhaust_card(ref)

    def discard_from_hand(self, card_id: str) -> None:
        hand = self._player.hand
        ref = hand[index_ref(hand, card_id)]
        remove_ref(hand, ref)
        self._player.discard_pile.append(ref)

    def upgrade_in_hand(self, index: int) -> None:
        p = self._player
        card = self._card_defs[p.hand[index]]
        if card.upgrade_of is not None:
            p.hand[index] = CardRef(card.upgrade_of, like=as_ref(p.hand, index))

    def upgrade_in_pile(self, pile: list, index: int) -> None:
        card = self._card_defs[pile[index]]
        if card.upgrade_of is not None and card.card_type not in ("status", "curse"):
            pile[index] = CardRef(card.upgrade_of, like=as_ref(pile, index))

    def is_upgradable(self, card_id: str) -> bool:
        card = self._card_defs[card_id]
        return card.upgrade_of is not None and card.card_type not in ("status", "curse")

    def generation_pool(self, card_type: str | None = None, *, color: str = "ironclad") -> list[str]:
        """Cards an in-combat generator may roll (CombatGenerationPool)."""

        key = (card_type, color)
        cache = self._s.extra.setdefault("gen_pools", {})
        if key in cache:
            return cache[key]
        out = []
        for cid, c in self._card_defs.items():
            if c.color != color or c.upgraded or c.multiplayer_only:
                continue
            if c.rarity not in ("common", "uncommon", "rare") or not c.generated_in_combat:
                continue
            if card_type is not None and c.card_type != card_type:
                continue
            if cid in UNSUPPORTED_CARDS:
                continue
            out.append(cid)
        cache[key] = sorted(out)
        return cache[key]

    def distinct_cards(self, pool: list[str], count: int) -> list[str]:
        pool = list(pool)
        self._rng.stream("card_generation").shuffle(pool)
        return pool[:count]

    def autoplay(self, card_id: str, *, exhaust: bool = False, target: MonsterState | None = None) -> None:
        """Play a card for free outside the hand (Havoc, Cascade, Mayhem ...)."""

        ref = card_id if isinstance(card_id, CardRef) else CardRef(card_id)
        card = self._card_defs[ref]
        if card.unplayable:
            # Unplayable cards drawn by Havoc / Cascade just go to their pile.
            if exhaust:
                self.exhaust_card(ref)
            else:
                self._player.discard_pile.append(ref)
            return
        if card.target == "single_enemy" and (target is None or not target.alive):
            target = self.random_monster()
            if target is None:
                self._player.discard_pile.append(ref)
                return
        x = self._player.energy if card.x_cost else 0
        if card.x_cost:
            self._player.energy = 0
            x = self.relics.adjust_x(x)
        if exhaust:
            self._s.exhaust_autoplay = ref
        outer = self._s.playing
        self._s.autoplaying += 1
        try:
            self._resolve_play(Play(card=card, target=target, x=x, auto=True, ref=ref), from_hand=False)
        finally:
            self._s.autoplaying -= 1
            self._s.playing = outer

    def autoplay_top_of_draw(self, count: int) -> None:
        p = self._player
        for _ in range(count):
            if self.combat.outcome is not None:
                return
            if not p.draw_pile:
                if not p.discard_pile:
                    return
                self.shuffle_discard_into_draw()
            ref = as_ref(p.draw_pile, len(p.draw_pile) - 1)
            p.draw_pile.pop()
            self.autoplay(ref)

    def _autoplay_from_hand(self, card_id: str, *, random_target: bool) -> None:
        hand = self._player.hand
        if card_id not in hand:
            return
        ref = hand[index_ref(hand, card_id)]
        remove_ref(hand, ref)
        target = self.random_monster() if random_target else None
        if target is None and self._card_defs[ref].target == "single_enemy":
            self._player.discard_pile.append(ref)
            return
        self.autoplay(ref, target=target)

    def transform_in_hand(self, index: int) -> None:
        """Entropy: a random card becomes another from the same pool."""

        from .rewards import transform_options

        p = self._player
        options = transform_options(self._card_defs, p.hand[index])
        if options:
            p.hand[index] = CardRef(self._rng.stream("card_generation").choice(options))

    # ------------------------------------------------------------------
    # Turn flow

    def _begin_player_turn(self, *, initial: bool) -> None:
        from . import monster_ai

        combat = self.combat
        p = self._player
        combat.turn += 1
        turn = combat.turn
        combat.phase = CombatPhase.PLAYER
        last_attacks = combat.attacks_played_this_turn
        last_cards = combat.cards_played_this_turn
        combat.cards_played_this_turn = 0
        combat.attacks_played_this_turn = 0
        combat.skills_played_this_turn = 0
        combat.cards_exhausted_this_turn = 0
        combat.hp_lost_this_turn = 0
        combat.block_gains_this_turn = 0
        combat.bound_this_turn = 0
        combat.bound_played_this_turn = False
        self._s.panache_count = 0
        self._s.attack_skill_played = 0
        for m in combat.monsters:
            m.flags.pop("slow_count", None)
            m.flags.pop("skittish_spent", None)
            if m.monster_id == "skulking_colony" and m.alive:
                m.powers["hardened_shell"] = 20
        if not initial and p.powers.get("barricade", 0) == 0:
            p.block = min(p.block, 10) if self.relics.keeps_block() else 0
        self.relics.after_block_cleared(turn)
        bnt = p.powers.pop("block_next_turn", 0)
        if bnt > 0:
            self.gain_block(bnt, powered=False)
        max_energy = self.relics.modify_max_energy(p.max_energy, turn) - p.powers.get("waste_away", 0)
        if initial or not self.relics.conserves_energy():
            p.energy = max(0, max_energy)
        else:
            p.energy += max(0, max_energy)
        pyre = p.powers.get("pyre", 0)
        if pyre > 0:
            p.energy += pyre
        radiance = p.powers.get("radiance", 0)
        if radiance > 0:
            self.gain_energy(1)
            self._change_power(p, "radiance", -1)
        # Living Shield's Rampart: every Turret Operator gains Block.
        rampart = max((m.powers.get("rampart", 0) for m in combat.monsters if m.alive), default=0)
        if rampart > 0:
            for m in combat.monsters:
                if m.alive and m.monster_id == "turret_operator":
                    m.block += rampart
        self.relics.turn_start(turn, attacks_last_turn=last_attacks, cards_last_turn=last_cards)
        self._hooks.dispatch("on_player_turn_start_pre_draw", {"ctx": self, "turn": turn, "initial": initial})
        if initial:
            self.relics.combat_start()
        # Start-of-turn powers (CombatEngine.EndTurn order).
        crimson = p.powers.get("crimson_mantle", 0)
        if crimson > 0 and combat.outcome is None:
            self.lose_hp(p.powers.get("crimson_mantle_hp", 1))
            if combat.outcome is None:
                self.gain_block(crimson, powered=False)
        if combat.outcome is None and p.powers.get("demon_form", 0) > 0:
            self.gain_strength(p.powers["demon_form"])
        if combat.outcome is None and p.powers.get("aggression", 0) > 0:
            self._aggression(p.powers["aggression"])
        if combat.outcome is None and p.powers.get("inferno", 0) > 0:
            self.lose_hp(1)
        plating = p.powers.get("plating", 0)
        if plating > 0 and not initial:
            self._change_power(p, "plating", -1)
        prep = p.powers.get("prep_time", 0)
        if prep > 0:
            self._change_power(p, "vigor", prep)
        boulder = p.powers.get("rolling_boulder", 0)
        if boulder > 0 and combat.outcome is None:
            self.damage_all_unpowered(boulder)
            self._change_power(p, "rolling_boulder", 5)
        for _ in range(p.powers.get("entropy", 0)):
            if p.hand:
                self.transform_in_hand(self._rng.stream("card_select").randrange(len(p.hand)))
        if self._check_end():
            return
        for ref in self._s.returning:
            for pile in (p.discard_pile, p.draw_pile, p.exhaust_pile):
                if any(c is ref for c in pile):
                    remove_ref(pile, ref)
                    self._add_to_hand(ref)
                    break
        self._s.returning.clear()
        draws = self._hand_size + self._s.extra.pop("draw_bonus", 0) + self.relics.turn_draw_extra(turn)
        if p.powers.get("clarity", 0) > 0:
            draws += 1
            self._change_power(p, "clarity", -1)
        self.draw(max(0, draws - p.powers.get("mind_rot", 0)))
        if combat.outcome is not None:
            return
        self.relics.after_hand_drawn(turn)
        mayhem = p.powers.get("mayhem", 0)
        if mayhem > 0:
            self.autoplay_top_of_draw(mayhem)
        if combat.outcome is not None:
            return
        if not initial:
            for m in combat.monsters:
                if m.alive or monster_ai.is_reviving(m):
                    monster_ai.choose_next_move(self, m)
        self._hooks.dispatch("on_player_turn_start", {"ctx": self, "turn": turn, "initial": initial})
        if self._check_end():
            return
        self._settle()

    def _aggression(self, count: int) -> None:
        p = self._player
        attacks = [i for i, c in enumerate(p.discard_pile) if self._card_defs[c].card_type == "attack"]
        if not attacks:
            return
        self._rng.stream("card_select").shuffle(attacks)
        for i in sorted(attacks[:count], reverse=True):
            ref = as_ref(p.discard_pile, i)
            p.discard_pile.pop(i)
            if len(p.hand) >= MAX_HAND:
                p.discard_pile.append(ref)
                continue
            if self.is_upgradable(ref):
                ref = CardRef(self._card_defs[ref].upgrade_of or ref, like=ref)
            p.hand.append(ref)

    def _is_ethereal(self, ref: str) -> bool:
        card = self._card_defs[ref]
        if card.ethereal or getattr(ref, "ethereal", False) or self._player.powers.get("hex", 0) > 0:
            return True
        return self.relics.makes_ethereal(card)

    def _end_player_turn(self) -> None:
        from . import monster_ai

        combat = self.combat
        p = self._player
        # End-of-turn auto plays.
        stampede = p.powers.get("stampede", 0)
        for _ in range(stampede):
            attacks = [c for c in p.hand if self._card_defs[c].card_type == "attack"
                       and not self._card_defs[c].unplayable]
            if not attacks or not self.alive_monsters() or combat.outcome is not None:
                break
            pick = self._rng.stream("combat_shuffle").choice(attacks)
            self._autoplay_from_hand(pick, random_target=True)
        for cid in [c for c in p.exhaust_pile if self._card_defs[c].game_id == "HOWL_FROM_BEYOND"]:
            if combat.outcome is not None or not self.alive_monsters():
                break
            remove_ref(p.exhaust_pile, cid)
            self.autoplay(cid)
        if combat.outcome is not None:
            return
        self._hooks.dispatch("on_player_turn_end_pre_flush", {"ctx": self})
        self._s.extra["block_before_end"] = p.block
        plating = p.powers.get("plating", 0)
        if plating > 0:
            self.gain_block(plating, powered=False)
        ritual = p.powers.get("ritual", 0)
        if ritual > 0:
            self.gain_strength(ritual)
        self.relics.turn_end(combat.turn, attacks=combat.attacks_played_this_turn)
        if self._tick_the_bomb() or self._check_end():
            return
        # Turn-scoped powers fall off.
        temp = p.powers.pop("temporary_strength", 0)
        if temp:
            self._change_power(p, "strength", -temp, allow_negative=True)
        if combat.tender_taken:
            taken, combat.tender_taken = combat.tender_taken, 0
            self._change_power(p, "strength", taken, allow_negative=True)
            self._change_power(p, "dexterity", taken, allow_negative=True)
        regen = p.powers.get("regen", 0)
        if regen > 0 and p.alive:
            self.heal_player(regen)
            self._change_power(p, "regen", -1)
        for pid in ("rage", "one_two_punch", "duplication", "tangled", "ringing", "no_draw", "smoggy_used"):
            p.powers.pop(pid, None)
        # Hand flush: in-hand statuses fire, ethereal exhausts, retain stays.
        retain_all = (p.powers.get("retain_hand", 0) > 0
                      or (combat.turn == 1 and self.relics.skips_first_flush())
                      or self.relics.retains_hand())
        hand_size = len(p.hand)
        keep: list[str] = []
        for cid in list(p.hand):
            card = self._card_defs[cid]
            effect = _TURN_END_IN_HAND.get(card.card_id)
            if effect is not None:
                self._turn_end_in_hand(effect, hand_size)
                if combat.outcome is not None:
                    return
            if self._is_ethereal(cid):
                remove_ref(p.hand, cid)
                self.exhaust_card(cid, ethereal=True)
            elif retain_all or "retain" in card.keywords or getattr(cid, "retain", False):
                keep.append(cid)
                remove_ref(p.hand, cid)
            else:
                remove_ref(p.hand, cid)
                p.discard_pile.append(cid)
        p.hand = keep
        if p.powers.get("retain_hand", 0) > 0:
            self._change_power(p, "retain_hand", -1)
        for pile in self._all_piles():
            for c in pile:
                if isinstance(c, CardRef):
                    c.free_turn = False
                    c.bound = False
        constrict = p.powers.get("constrict", 0)
        if constrict > 0:
            self.damage_player(constrict)
            if combat.outcome is not None:
                return
        disintegration = p.powers.get("disintegration", 0)
        if disintegration > 0:
            self.damage_player(disintegration)
            if combat.outcome is not None:
                return
        self.relics.after_flush()
        self._hooks.dispatch("on_player_turn_end", {"ctx": self})
        if self._check_end():
            return
        # Ancient: Pael's Eye may grant an extra player turn — skip the enemy
        # turn and loop straight back into _begin_player_turn.
        if combat.extra_turn_pending:
            combat.extra_turn_pending = False
            self._begin_player_turn(initial=False)
            return
        combat.phase = CombatPhase.ENEMY
        monster_ai.run_enemy_turn(self)
        if combat.outcome is not None:
            return
        p.powers.pop("tainted", None)
        self._round_end()
        if self._check_end():
            return
        if combat.curse_choice is not None:
            self._open_curse_choice()
            return
        self._begin_player_turn(initial=False)

    def _tick_the_bomb(self) -> bool:
        p = self._player
        turns = p.powers.get("the_bomb", 0)
        if turns <= 0:
            return False
        if turns > 1:
            self._change_power(p, "the_bomb", -1)
            return False
        p.powers.pop("the_bomb")
        damage = self._s.extra.pop("the_bomb_damage", 40)
        self.damage_all_unpowered(damage)
        return self.combat.outcome is not None

    def _turn_end_in_hand(self, effect: tuple[str, int], hand_size: int) -> None:
        kind, amount = effect
        p = self._player
        if kind == "damage":
            self.damage_player(amount)
        elif kind == "wither":
            intensity = max((m.flags.get("wither_intensity", 0) for m in self.alive_monsters()), default=0)
            self.damage_player(amount + 3 * intensity)
        elif kind == "lose_hp":
            self.lose_hp(amount)
        elif kind == "lose_hp_hand":
            self.lose_hp(hand_size)
        elif kind == "weak":
            self.apply_power(p, "weak", amount)
        elif kind == "frail":
            self.apply_power(p, "frail", amount)
        elif kind == "gold":
            p.gold = max(0, p.gold - min(amount, p.gold))

    def _round_end(self) -> None:
        """After the enemy turn: duration debuffs tick on both sides."""

        combat = self.combat
        p = self._player
        for m in combat.monsters:
            if not m.alive:
                continue
            for pid in _DURATION_DEBUFFS + ("slumber",):
                if m.powers.get(pid, 0) > 0:
                    self._change_power(m, pid, -1)
            if m.powers.get("shrink", 0) > 0:
                self._change_power(m, "shrink", -1)
            if m.powers.get("intangible", 0) > 0:
                self._change_power(m, "intangible", -1)
            if m.monster_id == "slumbering_beetle" and m.powers.get("slumber", 0) <= 0:
                m.powers.pop("plating", None)
            if m.powers.get("escape_artist", 0) > 1:
                self._change_power(m, "escape_artist", -1)
            if m.powers.get("nemesis", 0) > 0:
                on = not m.flags.get("nemesis_on", False)
                m.flags["nemesis_on"] = on
                if on:
                    m.powers["intangible"] = 1
                else:
                    m.powers.pop("intangible", None)
        snap = combat.player_debuffs_at_round_start
        for pid in _DURATION_DEBUFFS:
            if p.powers.get(pid, 0) > 0 and pid in snap:
                self._change_power(p, pid, -1)
        combat.player_debuffs_at_round_start = {pid for pid in _DURATION_DEBUFFS if p.powers.get(pid, 0) > 0}
        if p.powers.get("intangible", 0) > 0:
            self._change_power(p, "intangible", -1)
        if p.powers.get("colossus", 0) > 0:
            self._change_power(p, "colossus", -1)
        if p.powers.get("no_block", 0) > 0:
            self._change_power(p, "no_block", -1)
        temp_dex = p.powers.pop("temporary_dexterity", 0)
        if temp_dex:
            self._change_power(p, "dexterity", -temp_dex, allow_negative=True)
        for pid in ("flame_barrier", "no_energy_gain"):
            p.powers.pop(pid, None)
        if self._s.dark_embrace_deferred > 0:
            n, self._s.dark_embrace_deferred = self._s.dark_embrace_deferred, 0
            self.draw(n)

    # -- Knowledge Demon's Curse of Knowledge -------------------------------

    def offer_curse_of_knowledge(self, cast: int) -> None:
        cast = max(0, min(cast, len(CURSE_OF_KNOWLEDGE_CURSES) - 1))
        self.combat.curse_choice = (["disintegration", CURSE_OF_KNOWLEDGE_CURSES[cast]],
                                    CURSE_OF_KNOWLEDGE_DAMAGE[cast])

    def _open_curse_choice(self) -> None:
        combat = self.combat
        assert combat.curse_choice is not None
        combat.phase = CombatPhase.PLAYER

        def choose():
            options, damage = combat.curse_choice
            chosen = yield SelectionRequest(source="generated", candidates=list(range(len(options))), count=1,
                                            purpose="curse_of_knowledge", options=list(options))
            combat.curse_choice = None
            pick = chosen[0]
            if pick == "disintegration":
                self._change_power(self._player, "disintegration", damage)
            else:
                self._change_power(self._player, pick, _CURSE_POWER_AMOUNT[pick])

        self._drive(_Pending(gen=choose(), finish=lambda: self._begin_player_turn(initial=False),
                             label="knowledge_demon"), None)

    def _check_end(self) -> bool:
        from . import monster_ai

        combat = self.combat
        if combat.outcome is not None:
            return True
        if not self._player.alive:
            self._end_combat("defeat")
            return True
        if not monster_ai.any_primary_alive(self):
            # Minions abandon combat without their leader.
            for m in combat.monsters:
                if m.alive:
                    m.escaped = True
            self._end_combat("victory")
            return True
        return False

    def _end_combat(self, outcome: str) -> None:
        combat = self.combat
        if combat.outcome is not None:
            return
        combat.outcome = outcome
        combat.phase = CombatPhase.END
        combat.pending_selection = None
        self._s.pending = None
        self._s.followups.clear()
        self._hooks.dispatch("on_combat_end", {"ctx": self, "outcome": outcome})

    # ------------------------------------------------------------------
    # Monsters

    def _hp_band(self, monster_id: str) -> tuple[int, int]:
        mdef = self._monster_defs[monster_id]
        return mdef.hp_asc if self.ascension >= 8 else mdef.hp

    def _new_monster(self, monster_id: str) -> MonsterState:
        # The game rolls a max HP no living enemy already has, when the band allows
        # (CombatState.SetUniqueMonsterHpValue).
        mdef = self._monster_defs[monster_id]
        lo, hi = self._hp_band(monster_id)
        taken = {m.max_hp for m in self.combat.monsters if m.alive}
        free = [v for v in range(lo, hi + 1) if v not in taken]
        stream = self._rng.stream("enemy_hp")
        hp = stream.choice(free) if free else stream.randint(lo, hi)
        return MonsterState(monster_id=monster_id, name=mdef.name, hp=hp, max_hp=hp, slot=0)

    def roll_monster_hp(self, monster_id: str) -> int:
        lo, hi = self._hp_band(monster_id)
        return self._rng.stream("enemy_hp").randint(lo, hi)

    def spawn_monster(self, monster_id: str, *, index: int | None = None, stunned: bool = False,
                      flags: dict[str, Any] | None = None) -> MonsterState:
        from . import monster_ai

        combat = self.combat
        m = self._new_monster(monster_id)
        if flags:
            m.flags.update(flags)
        m.kind_index = sum(1 for o in combat.monsters if o.monster_id == monster_id)
        if index is None:
            combat.monsters.append(m)
        else:
            combat.monsters.insert(index, m)
        self._reslot()
        monster_ai.enter_combat(self, m)
        m.stunned = stunned
        monster_ai.choose_first_move(self, m)
        if combat.phase == CombatPhase.ENEMY:
            # Summoned on the enemy turn: its first intent stands until it has acted.
            m.flags["fresh"] = True
        self._hooks.dispatch("on_enemy_spawned", {"ctx": self, "target": m})
        return m

    def _reslot(self) -> None:
        for i, m in enumerate(self.combat.monsters):
            m.slot = i

    def remove_dead(self) -> None:
        """Drop corpses that hold nothing open (the game removes dead creatures)."""

        from . import monster_ai

        keep = [m for m in self.combat.monsters if m.alive or monster_ai.keeps_corpse(m)]
        if len(keep) != len(self.combat.monsters):
            self.combat.monsters[:] = keep
            self._reslot()


__all__ = ["CombatContext", "CombatError", "MAX_HAND", "NoRelics", "Play", "PotionUse", "REND_DEBUFFS",
           "RelicHooks", "SelectionRequest"]
