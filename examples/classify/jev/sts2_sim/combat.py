"""Combat engine for Slay the Spire 2 (Ironclad, v0.107.1 rules).

Owns one combat: start -> player turn (draw / energy / plays) -> end turn ->
enemy turn -> round end -> next player turn, until victory or defeat. All
mutations target the RunState passed in.

Turn order, damage math and power timings follow r33hab/sts2's emulator
(`CombatEngine.EndTurn`, `BuffSystem.IncomingDamage`, `CardEffects`), which
is itself checked against live game captures. Where the emulator and the
card text disagree, the card text wins and the spot carries a comment.

Card behavior lives in card_effects.py (one function per card, keyed by
game id); monster behavior in monster_ai.py. This module provides the
primitives both call: powered/unpowered damage, block, power application,
draw, exhaust, HP loss, and the turn loop.

Known approximation: piles hold card ids, so per-instance card state
(Rampage / Thrash growth, Infernal Blade's free-this-turn) is tracked per
card id. Two copies of the same card in one combat share it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .effects import EffectQueue
from .enums import CombatPhase
from .hooks import HookBus
from .rng import Rng
from .schemas import CardSchema, EncounterSchema, MonsterSchema, PowerSchema
from .state import CombatState, MonsterState, PendingSelection, PlayerState, RunState


class CombatError(RuntimeError):
    pass


MAX_HAND = 10
BASE_HAND_SIZE = 5

# Debuffs on the player that tick down once per round.
_DURATION_DEBUFFS = ("vulnerable", "weak", "frail")
# Debuffs Artifact negates (monster side). Strength loss counts as a debuff.
_DEBUFF_IDS = frozenset({
    "vulnerable", "weak", "frail", "shrink", "constrict", "tangled", "smoggy", "ringing",
    "no_draw", "no_energy_gain", "slow", "strength_loss",
})
# End-of-turn-in-hand status/curse effects: (kind, amount).
_TURN_END_IN_HAND: dict[str, tuple[str, int]] = {
    "burn": ("damage", 2),
    "infection": ("damage", 3),
    "toxic": ("damage", 5),
    "decay": ("damage", 2),
    "wither": ("damage", 3),
    "beckon": ("lose_hp", 6),
    "bad_luck": ("lose_hp", 13),
    "regret": ("lose_hp_hand", 0),
    "doubt": ("weak", 1),
    "shame": ("frail", 1),
    "debt": ("gold", 10),
}


@dataclass
class Play:
    """One resolution of a card (a replay or auto-play is a new Play)."""

    card: CardSchema
    target: MonsterState | None
    x: int = 0  # energy spent on an X-cost card
    auto: bool = False  # auto-played (Havoc, Cascade ...): selections auto-pick
    hp_lost_from_card: int = 0
    bonus: int = 0  # Rampage / Thrash growth banked by this play


@dataclass
class SelectionRequest:
    """Yielded by a card generator to ask the player to pick cards."""

    source: str  # "hand" | "discard"
    candidates: list[int]
    count: int
    purpose: str
    min_count: int | None = None


@dataclass
class _PendingPlay:
    play: Play
    gen: Iterator[Any] | None
    finish: Callable[[], None]


@dataclass
class _Scratch:
    rupture_owed: int = 0
    dark_embrace_deferred: int = 0
    pending: _PendingPlay | None = None
    autoplaying: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


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
    ) -> None:
        if run.player is None:
            raise CombatError("RunState.player must be set before CombatContext")
        self._run = run
        self._card_defs = cards
        self._monster_defs = monsters
        self._power_defs = powers
        self._encounters = encounters or {}
        self._rng = rng
        self._hooks = hooks
        self._effects = effects
        self._player: PlayerState = run.player
        self._combat: CombatState | None = run.combat
        self._s = _Scratch()

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
    def ascension(self) -> int:
        return self._run.ascension

    @property
    def cards(self) -> dict[str, CardSchema]:
        return self._card_defs

    @property
    def monster_defs(self) -> dict[str, MonsterSchema]:
        return self._monster_defs

    @property
    def rng(self) -> Rng:
        return self._rng

    def start_combat(
        self,
        monster_ids: list[str],
        starting_deck: list[str],
        *,
        hand_size: int = DEFAULT_HAND_SIZE,
        monster_flags: list[dict[str, Any]] | None = None,
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
        p.deck = list(starting_deck)
        p.hand = []
        p.discard_pile = []
        p.exhaust_pile = []
        p.draw_pile = list(starting_deck)
        self._rng.stream("combat_shuffle").shuffle(p.draw_pile)
        # Innate cards start on top of the draw pile (end of list = top).
        innate = [c for c in p.draw_pile if "innate" in self._card_defs[c].keywords]
        if innate:
            rest = [c for c in p.draw_pile if "innate" not in self._card_defs[c].keywords]
            p.draw_pile = rest + innate
        p.block = 0
        p.powers = {}
        p.energy = 0

        for m in combat.monsters:
            monster_ai.choose_first_move(self, m)
        self._hooks.dispatch("on_combat_start", {"monsters": combat.monsters})
        self._begin_player_turn(initial=True)

    # -- legality --------------------------------------------------------

    def effective_cost(self, card_id: str) -> int:
        card = self._card_defs[card_id]
        if card.x_cost:
            return 0
        p = self._player.powers
        cost = card.cost
        if card.game_id == "STOMP":
            cost = max(0, cost - self.combat.attacks_played_this_turn)
        if card.card_type == "attack":
            cost += p.get("tangled", 0)
        if card.card_type == "skill" and p.get("corruption", 0) > 0:
            cost = 0
        if card.card_type == "attack" and p.get("free_attack", 0) > 0:
            cost = 0
        if self.combat.free_this_turn.get(card_id, 0) > 0:
            cost = 0
        return max(0, cost)

    def can_play(self, card_id: str) -> bool:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER or combat.outcome is not None:
            return False
        if combat.pending_selection is not None:
            return False
        if card_id not in self._player.hand:
            return False
        card = self._card_defs[card_id]
        if card.unplayable or card.multiplayer_only:
            return False
        if self.effective_cost(card_id) > self._player.energy:
            return False
        p = self._player.powers
        if p.get("ringing", 0) > 0 and combat.cards_played_this_turn > 0:
            return False
        if card.card_type == "skill" and p.get("smoggy", 0) > 0 and combat.skills_played_this_turn > 0:
            return False
        if any(c == "normality" for c in self._player.hand) and combat.cards_played_this_turn >= 3:
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
        card = self._card_defs[card_id]
        cost = self.effective_cost(card_id)
        if card.unplayable or card.multiplayer_only:
            raise CombatError(f"{card_id!r} is unplayable")
        if cost > self._player.energy:
            raise CombatError(f"insufficient energy to play {card_id!r}")
        if not self.can_play(card_id):
            raise CombatError(f"{card_id!r} cannot be played now")
        target: MonsterState | None = None
        if card.target == "single_enemy":
            if target_slot is None:
                raise CombatError(f"{card_id!r} requires target_slot")
            if not (0 <= target_slot < len(combat.monsters)):
                raise CombatError(f"target_slot {target_slot} out of range")
            target = combat.monsters[target_slot]
            if not target.alive:
                raise CombatError(f"target slot {target_slot} is dead")

        x = self._player.energy if card.x_cost else 0
        self._player.energy -= x if card.x_cost else cost
        if self.combat.free_this_turn.get(card_id, 0) > 0:
            self.combat.free_this_turn[card_id] -= 1
        elif card.card_type == "attack" and self._player.powers.get("free_attack", 0) > 0:
            self._change_power(self._player, "free_attack", -1)
        self._player.hand.remove(card_id)
        combat.last_card_played = card_id
        self._resolve_play(Play(card=card, target=target, x=x), from_hand=True)

    def end_turn(self) -> None:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER:
            raise CombatError(f"end_turn called in phase {combat.phase!r}")
        if combat.outcome is not None:
            raise CombatError("combat already finished")
        if combat.pending_selection is not None:
            raise CombatError("a card selection is pending")
        self._end_player_turn()

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

    def confirm_selection(self) -> None:
        sel = self.combat.pending_selection
        if sel is None:
            raise CombatError("no pending selection")
        if not self.can_confirm_selection():
            raise CombatError("selection count not satisfied")
        pending = self._s.pending
        assert pending is not None
        self.combat.pending_selection = None
        self._s.pending = None
        pile = self._player.hand if sel.source == "hand" else self._player.discard_pile
        chosen = [pile[i] for i in sorted(sel.selected)]
        self._continue_play(pending, chosen, sel)

    # ------------------------------------------------------------------
    # Card resolution

    def _resolve_play(self, play: Play, *, from_hand: bool) -> None:
        """Run a card's effect (plus replays), then route the card."""

        from .card_effects import CARD_EFFECTS

        combat = self.combat
        combat.cards_played_this_turn += 1
        if play.card.card_type == "attack":
            combat.attacks_played_this_turn += 1
        elif play.card.card_type == "skill":
            combat.skills_played_this_turn += 1

        def finish() -> None:
            self._after_play(play, from_hand=from_hand)

        fn = CARD_EFFECTS.get(play.card.game_id)
        self._run_card_fn(fn, play, finish, extra_plays=self._extra_plays(play))

    def _extra_plays(self, play: Play) -> int:
        if play.card.card_type != "attack":
            return 0
        otp = self._player.powers.get("one_two_punch", 0)
        if otp > 0:
            self._change_power(self._player, "one_two_punch", -1)
            return 1
        return 0

    def _run_card_fn(self, fn, play: Play, finish: Callable[[], None], *, extra_plays: int = 0) -> None:
        def chain() -> None:
            # Replays (One-Two Punch) re-run the effect before the card is routed.
            if extra_plays > 0 and self.combat.outcome is None:
                self._run_card_fn(fn, play, finish, extra_plays=extra_plays - 1)
            else:
                finish()

        if fn is None:
            chain()
            return
        if inspect.isgeneratorfunction(fn):
            gen = fn(self, play)
            self._drive(_PendingPlay(play=play, gen=gen, finish=chain), None)
        else:
            fn(self, play)
            chain()

    def _drive(self, pending: _PendingPlay, send: Any) -> None:
        gen = pending.gen
        assert gen is not None
        try:
            request = gen.send(send) if send is not None else next(gen)
        except StopIteration:
            pending.finish()
            return
        while True:
            assert isinstance(request, SelectionRequest)
            cands = list(request.candidates)
            want = min(request.count, len(cands))
            if want <= 0:
                chosen: list[str] = []
            elif pending.play.auto or self._s.autoplaying > 0:
                # Auto-played cards resolve their choices without a prompt, taking
                # the game's autoPick (first hand card / top of discard).
                pile = self._player.hand if request.source == "hand" else self._player.discard_pile
                pick = cands[:want] if request.source == "hand" else cands[-want:]
                chosen = [pile[i] for i in pick]
            else:
                self.combat.pending_selection = PendingSelection(
                    source=request.source,
                    candidates=cands,
                    min_count=want if request.min_count is None else request.min_count,
                    max_count=want,
                    purpose=request.purpose,
                    source_card=pending.play.card.card_id,
                )
                self._s.pending = pending
                return
            try:
                request = gen.send(chosen)
            except StopIteration:
                pending.finish()
                return

    def _continue_play(self, pending: _PendingPlay, chosen: list[str], sel: PendingSelection) -> None:
        del sel
        self._drive(pending, chosen)

    def _after_play(self, play: Play, *, from_hand: bool) -> None:
        combat = self.combat
        card = play.card
        p = self._player
        # Bank per-card growth (Rampage / Thrash).
        if play.bonus:
            combat.bonus_damage[card.card_id] = combat.bonus_damage.get(card.card_id, 0) + play.bonus
        if card.card_type == "attack":
            # Juggling: a copy of the third Attack each turn.
            if combat.attacks_played_this_turn == 3 and p.powers.get("juggling", 0) > 0:
                for _ in range(p.powers["juggling"]):
                    self._add_to_hand(card.card_id)
            rage = p.powers.get("rage", 0)
            if rage > 0:
                self.gain_block(rage, powered=False)
        # Route the played card.
        if from_hand or play.auto:
            if card.card_type == "power":
                pass
            elif card.exhausts or (card.card_type == "skill" and p.powers.get("corruption", 0) > 0) \
                    or play.auto and self._s.extra.get("exhaust_autoplay") == card.card_id:
                self._s.extra.pop("exhaust_autoplay", None)
                self.exhaust_card(card.card_id)
            else:
                p.discard_pile.append(card.card_id)
        # Rupture owed for HP the card itself cost.
        if self._s.rupture_owed > 0 and self.combat.outcome is None:
            owed, self._s.rupture_owed = self._s.rupture_owed, 0
            self.gain_strength(owed)
        # Monster reactions to plays.
        for m in list(combat.monsters):
            if not m.alive:
                continue
            if m.powers.get("slow", 0) > 0:
                m.flags["slow_count"] = m.flags.get("slow_count", 0) + 1
        self._hooks.dispatch("on_card_played", {"card_id": card.card_id})
        self._check_end()

    # ------------------------------------------------------------------
    # Primitives used by card_effects / monster_ai

    def alive_monsters(self) -> list[MonsterState]:
        return [m for m in self.combat.monsters if m.alive]

    def random_monster(self) -> MonsterState | None:
        alive = self.alive_monsters()
        if not alive:
            return None
        return self._rng.stream("combat_target").choice(alive)

    # -- damage ------------------------------------------------------------

    def _powered_amount(self, base: int, attacker: Any, defender: Any) -> int:
        a = attacker.powers
        d = defender.powers
        dmg = float(base) + a.get("strength", 0) + a.get("vigor", 0)
        if a.get("weak", 0) > 0:
            dmg *= 0.75
        if a.get("shrink", 0) != 0:
            dmg *= 0.70
        if d.get("vulnerable", 0) > 0:
            mult = 1.5
            if attacker is self._player:
                mult += self._player.powers.get("cruelty", 0) / 100.0
            dmg *= mult
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

    def damage_monster_unpowered(self, target: MonsterState, amount: int) -> int:
        if not target.alive:
            return 0
        return self._hit_monster(target, amount, powered=False)

    def damage_all_unpowered(self, amount: int) -> None:
        for m in self.alive_monsters():
            self._hit_monster(m, amount, powered=False)

    def _hit_monster(self, target: MonsterState, base: int, *, powered: bool) -> int:
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
        absorbed = min(target.block, dmg)
        target.block -= absorbed
        hp_loss = dmg - absorbed
        hp_loss = self._cap_monster_hp_loss(target, hp_loss)
        target.hp = max(0, target.hp - hp_loss)
        self._hooks.dispatch("on_damaged", {"target": target, "amount": hp_loss, "blocked": absorbed})
        from . import monster_ai

        if hp_loss > 0 and target.hp > 0:
            monster_ai.after_hp_lost(self, target, hp_loss)
        if powered and hp_loss > 0 and target.hp > 0:
            skittish = target.powers.get("skittish", 0)
            if skittish > 0 and not target.flags.get("skittish_spent"):
                target.flags["skittish_spent"] = True
                target.block += skittish
        if target.hp == 0:
            monster_ai.on_death(self, target)
        return hp_loss

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
            absorbed = min(p.block, dmg)
            p.block -= absorbed
            unblocked = dmg - absorbed
            if unblocked > 0:
                self._lose_player_hp(unblocked)
                landed += 1
                suck = monster.powers.get("suck", 0)
                if suck > 0:
                    self._change_power(monster, "strength", suck)
            if p.alive:
                fb = p.powers.get("flame_barrier", 0)
                if fb > 0:
                    self.damage_monster_unpowered(monster, fb)
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

    def _lose_player_hp(self, amount: int, *, from_card: bool = False) -> int:
        p = self._player
        before = p.hp
        p.hp = max(0, p.hp - amount)
        lost = before - p.hp
        if lost <= 0:
            return 0
        combat = self.combat
        combat.hp_loss_events_this_combat += 1
        on_our_turn = combat.phase == CombatPhase.PLAYER
        if on_our_turn:
            combat.hp_lost_this_turn += lost
            rupture = p.powers.get("rupture", 0)
            if rupture > 0:
                if from_card:
                    self._s.rupture_owed += rupture
                else:
                    self.gain_strength(rupture)
            inferno = p.powers.get("inferno", 0)
            if inferno > 0 and p.alive:
                self.damage_all_unpowered(inferno)
        if not p.alive:
            self._end_combat("defeat")
        return lost

    def heal_player(self, amount: int) -> None:
        p = self._player
        p.hp = min(p.max_hp, p.hp + max(0, amount))

    def gain_max_hp(self, amount: int) -> None:
        self._player.max_hp += amount
        self.heal_player(amount)

    # -- block / energy / stats ------------------------------------------------

    def gain_block(self, amount: int, *, powered: bool = True) -> int:
        p = self._player
        if powered:
            blk = float(amount) + p.powers.get("dexterity", 0)
            if p.powers.get("frail", 0) > 0:
                blk *= 0.75
            eff = max(0, int(blk))
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

    def gain_energy(self, amount: int) -> None:
        if amount <= 0 or self._player.powers.get("no_energy_gain", 0) > 0:
            return
        self._player.energy += amount

    def gain_strength(self, amount: int) -> None:
        self._change_power(self._player, "strength", amount, allow_negative=True)

    def apply_power(self, target: Any, power_id: str, amount: int, *, source: Any = None) -> bool:
        """Apply `amount` of `power_id` to target. Returns False if Artifact blocked it."""

        if amount == 0:
            return False
        if power_id not in self._power_defs:
            raise CombatError(f"unknown power {power_id!r}")
        if isinstance(target, MonsterState) and not target.alive:
            return False
        is_debuff = power_id in _DEBUFF_IDS or (power_id in ("strength", "dexterity") and amount < 0)
        if is_debuff and target.powers.get("artifact", 0) > 0:
            self._change_power(target, "artifact", -1)
            return False
        self._change_power(target, power_id, amount, allow_negative=power_id in ("strength", "dexterity"))
        if power_id == "vulnerable" and source is self._player and isinstance(target, MonsterState):
            vicious = self._player.powers.get("vicious", 0)
            if vicious > 0:
                self.draw(vicious)
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

    def _draw_one(self) -> str | None:
        p = self._player
        if len(p.hand) >= MAX_HAND:
            return None
        if not p.draw_pile:
            if not p.discard_pile:
                return None
            p.draw_pile = list(p.discard_pile)
            p.discard_pile.clear()
            self._rng.stream("combat_shuffle").shuffle(p.draw_pile)
        card_id = p.draw_pile.pop()
        p.hand.append(card_id)
        self._hooks.dispatch("on_card_drawn", {"card_id": card_id})
        card = self._card_defs[card_id]
        if card.game_id == "VOID":
            p.energy = max(0, p.energy - card.v("energy"))
        if p.powers.get("hellraiser", 0) > 0 and "Strike" in card.name and card.card_type == "attack":
            self._autoplay_from_hand(card_id, random_target=True)
        return card_id

    def _add_to_hand(self, card_id: str) -> None:
        p = self._player
        if len(p.hand) < MAX_HAND:
            p.hand.append(card_id)
        else:
            p.discard_pile.append(card_id)

    def add_to_hand(self, card_id: str) -> None:
        self._add_to_hand(card_id)

    def add_card_to_pile(self, card_id: str, pile: str) -> None:
        p = self._player
        if card_id not in self._card_defs:
            raise CombatError(f"unknown card {card_id!r}")
        if pile == "hand":
            self._add_to_hand(card_id)
        elif pile == "discard":
            p.discard_pile.append(card_id)
        elif pile == "draw_top":
            p.draw_pile.append(card_id)
        elif pile == "draw_random":
            pos = self._rng.stream("combat_shuffle").randint(0, len(p.draw_pile))
            p.draw_pile.insert(pos, card_id)
        else:
            raise CombatError(f"unknown pile {pile!r}")

    def exhaust_card(self, card_id: str, *, ethereal: bool = False) -> None:
        """Put `card_id` (already removed from its pile) into the exhaust pile."""

        p = self._player
        p.exhaust_pile.append(card_id)
        self.combat.cards_exhausted_this_turn += 1
        self._hooks.dispatch("on_card_exhausted", {"card_id": card_id})
        card = self._card_defs[card_id]
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

    def exhaust_from_hand(self, card_id: str) -> None:
        self._player.hand.remove(card_id)
        self.exhaust_card(card_id)

    def upgrade_in_hand(self, index: int) -> None:
        p = self._player
        card = self._card_defs[p.hand[index]]
        if card.upgrade_of is not None:
            p.hand[index] = card.upgrade_of

    def is_upgradable(self, card_id: str) -> bool:
        card = self._card_defs[card_id]
        return card.upgrade_of is not None and card.card_type not in ("status", "curse")

    def generation_pool(self, card_type: str | None = None) -> list[str]:
        """Ironclad cards an in-combat generator may roll (Infernal Blade, Stoke)."""

        out = []
        for cid, c in self._card_defs.items():
            if c.color != "ironclad" or c.upgraded or c.multiplayer_only:
                continue
            if c.rarity not in ("common", "uncommon", "rare") or not c.generated_in_combat:
                continue
            if card_type is not None and c.card_type != card_type:
                continue
            out.append(cid)
        return sorted(out)

    def autoplay(self, card_id: str, *, exhaust: bool = False, target: MonsterState | None = None) -> None:
        """Play a card for free outside the hand (Havoc, Cascade, Howl, Hellraiser)."""

        card = self._card_defs[card_id]
        if card.unplayable:
            # Unplayable cards drawn by Havoc / Cascade just go to their pile.
            if exhaust:
                self.exhaust_card(card_id)
            else:
                self._player.discard_pile.append(card_id)
            return
        if card.target == "single_enemy" and (target is None or not target.alive):
            target = self.random_monster()
            if target is None:
                self._player.discard_pile.append(card_id)
                return
        x = self._player.energy if card.x_cost else 0
        if card.x_cost:
            self._player.energy = 0
        if exhaust:
            self._s.extra["exhaust_autoplay"] = card_id
        self._s.autoplaying += 1
        try:
            self._resolve_play(Play(card=card, target=target, x=x, auto=True), from_hand=False)
        finally:
            self._s.autoplaying -= 1

    def _autoplay_from_hand(self, card_id: str, *, random_target: bool) -> None:
        if card_id not in self._player.hand:
            return
        self._player.hand.remove(card_id)
        target = self.random_monster() if random_target else None
        if target is None and self._card_defs[card_id].target == "single_enemy":
            self._player.discard_pile.append(card_id)
            return
        self.autoplay(card_id, target=target)

    # ------------------------------------------------------------------
    # Turn flow

    def _begin_player_turn(self, *, initial: bool) -> None:
        from . import monster_ai

        combat = self.combat
        p = self._player
        combat.turn += 1
        combat.phase = CombatPhase.PLAYER
        combat.cards_played_this_turn = 0
        combat.attacks_played_this_turn = 0
        combat.skills_played_this_turn = 0
        combat.cards_exhausted_this_turn = 0
        combat.hp_lost_this_turn = 0
        combat.block_gains_this_turn = 0
        combat.free_this_turn.clear()
        for m in combat.monsters:
            m.flags.pop("slow_count", None)
            m.flags.pop("skittish_spent", None)
            if m.monster_id == "skulking_colony" and m.alive:
                m.powers["hardened_shell"] = 20
        p.energy = p.max_energy
        if not initial:
            if p.powers.get("barricade", 0) == 0:
                p.block = 0
        pyre = p.powers.get("pyre", 0)
        if pyre > 0:
            p.energy += pyre
        # Start-of-turn powers (CombatEngine.EndTurn order).
        crimson = p.powers.get("crimson_mantle", 0)
        if crimson > 0:
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
        if combat.outcome is not None:
            return
        self.draw(self._hand_size)
        if combat.outcome is not None:
            return
        if not initial:
            for m in combat.monsters:
                if m.alive:
                    monster_ai.choose_next_move(self, m)
        self._hooks.dispatch("on_player_turn_start", {"turn": combat.turn})
        self._check_end()

    def _aggression(self, count: int) -> None:
        p = self._player
        attacks = [i for i, c in enumerate(p.discard_pile) if self._card_defs[c].card_type == "attack"]
        if not attacks:
            return
        self._rng.stream("card_select").shuffle(attacks)
        for i in sorted(attacks[:count], reverse=True):
            cid = p.discard_pile.pop(i)
            if len(p.hand) >= MAX_HAND:
                p.discard_pile.append(cid)
                continue
            if self.is_upgradable(cid):
                cid = self._card_defs[cid].upgrade_of or cid
            p.hand.append(cid)

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
            p.exhaust_pile.remove(cid)
            self.autoplay(cid)
        if combat.outcome is not None:
            return
        plating = p.powers.get("plating", 0)
        if plating > 0:
            self.gain_block(plating, powered=False)
        # Turn-scoped powers fall off.
        temp = p.powers.pop("temporary_strength", 0)
        if temp:
            self.gain_strength(-temp)
        for pid in ("rage", "one_two_punch", "tangled", "ringing", "no_draw", "smoggy_used"):
            p.powers.pop(pid, None)
        # Hand flush: in-hand statuses fire, ethereal exhausts, retain stays.
        hand_size = len(p.hand)
        keep: list[str] = []
        for cid in list(p.hand):
            card = self._card_defs[cid]
            effect = _TURN_END_IN_HAND.get(card.card_id)
            if effect is not None:
                self._turn_end_in_hand(effect, hand_size)
                if combat.outcome is not None:
                    return
            if card.ethereal:
                p.hand.remove(cid)
                self.exhaust_card(cid, ethereal=True)
            elif "retain" in card.keywords:
                keep.append(cid)
                p.hand.remove(cid)
            else:
                p.hand.remove(cid)
                p.discard_pile.append(cid)
        p.hand = keep
        constrict = p.powers.get("constrict", 0)
        if constrict > 0:
            self.damage_player(constrict)
            if combat.outcome is not None:
                return
        self._hooks.dispatch("on_player_turn_end", {})
        combat.phase = CombatPhase.ENEMY
        monster_ai.run_enemy_turn(self)
        if combat.outcome is not None:
            return
        self._round_end()
        if self._check_end():
            return
        self._begin_player_turn(initial=False)

    def _turn_end_in_hand(self, effect: tuple[str, int], hand_size: int) -> None:
        kind, amount = effect
        p = self._player
        if kind == "damage":
            self.damage_player(amount)
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
            for pid in _DURATION_DEBUFFS:
                if m.powers.get(pid, 0) > 0:
                    self._change_power(m, pid, -1)
            if m.powers.get("intangible", 0) > 0:
                self._change_power(m, "intangible", -1)
        snap = combat.player_debuffs_at_round_start
        for pid in _DURATION_DEBUFFS:
            if p.powers.get(pid, 0) > 0 and pid in snap:
                self._change_power(p, pid, -1)
        combat.player_debuffs_at_round_start = {pid for pid in _DURATION_DEBUFFS if p.powers.get(pid, 0) > 0}
        if p.powers.get("intangible", 0) > 0:
            self._change_power(p, "intangible", -1)
        if p.powers.get("colossus", 0) > 0:
            self._change_power(p, "colossus", -1)
        for pid in ("flame_barrier", "no_energy_gain"):
            p.powers.pop(pid, None)
        if self._s.dark_embrace_deferred > 0:
            n, self._s.dark_embrace_deferred = self._s.dark_embrace_deferred, 0
            self.draw(n)

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
        self._hooks.dispatch("on_combat_end", {"outcome": outcome})

    # ------------------------------------------------------------------
    # Monsters

    def _new_monster(self, monster_id: str) -> MonsterState:
        # The game rolls a max HP no living enemy already has, when the band allows
        # (CombatState.SetUniqueMonsterHpValue).
        mdef = self._monster_defs[monster_id]
        lo, hi = mdef.hp_asc if self.ascension >= 8 else mdef.hp
        taken = {m.max_hp for m in self.combat.monsters if m.alive}
        free = [v for v in range(lo, hi + 1) if v not in taken]
        stream = self._rng.stream("enemy_hp")
        hp = stream.choice(free) if free else stream.randint(lo, hi)
        return MonsterState(monster_id=monster_id, name=mdef.name, hp=hp, max_hp=hp, slot=0)

    def spawn_monster(self, monster_id: str, *, index: int | None = None, stunned: bool = False) -> MonsterState:
        from . import monster_ai

        combat = self.combat
        m = self._new_monster(monster_id)
        m.kind_index = sum(1 for o in combat.monsters if o.monster_id == monster_id)
        if index is None:
            combat.monsters.append(m)
        else:
            combat.monsters.insert(index, m)
        self._reslot()
        monster_ai.enter_combat(self, m)
        m.stunned = stunned
        monster_ai.choose_first_move(self, m)
        self._hooks.dispatch("on_enemy_spawned", {"target": m})
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


__all__ = ["CombatContext", "CombatError", "MAX_HAND", "Play", "SelectionRequest"]
