"""Phase 1 combat engine.

Owns one combat: start -> player turn (draw/energy/actions) -> end turn ->
enemy turn (move resolution + decay) -> back to player, until victory or
defeat. All mutations target the RunState passed in; CombatContext holds
no state of its own beyond caches for the data tables.

Follows the engine development contract pinned in `schemas.py`. In
particular, see the Resolution order, Dead-target semantics, Hook timing,
and Scaling rules sections of that module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .effects import EffectQueue
from .enums import CombatPhase
from .hooks import HookBus
from .rng import Rng
from .schemas import CardSchema, EffectStep, EnemySchema, PowerSchema, SelectorEntry
from .state import CombatState, MonsterState, PlayerState, RunState


class CombatError(RuntimeError):
    pass


@dataclass
class _EffectContext:
    """Transient context passed down the verb dispatcher.

    Captures the actor identity and the single_enemy target chosen at
    card-play time so later effects of the same card see the same slot
    (see Dead-target semantics in schemas.py)."""

    actor: str  # "player" | "enemy"
    actor_monster: MonsterState | None = None  # set when actor == "enemy"
    source_card: CardSchema | None = None
    source_move_id: str | None = None
    captured_target_slot: int | None = None


class CombatContext:
    DEFAULT_HAND_SIZE: int = 5

    def __init__(
        self,
        *,
        run: RunState,
        cards: dict[str, CardSchema],
        enemies: dict[str, EnemySchema],
        powers: dict[str, PowerSchema],
        rng: Rng,
        hooks: HookBus,
        effects: EffectQueue,
    ) -> None:
        if run.player is None:
            raise CombatError("RunState.player must be set before CombatContext")
        self._run = run
        self._card_defs = cards
        self._enemy_defs = enemies
        self._power_defs = powers
        self._rng = rng
        self._hooks = hooks
        self._effects = effects
        self._player: PlayerState = run.player
        self._combat: CombatState | None = run.combat

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

    def start_combat(
        self,
        enemy_ids: list[str],
        starting_deck: list[str],
        *,
        hand_size: int = DEFAULT_HAND_SIZE,
    ) -> None:
        """Set up a fresh combat on this run."""

        for eid in enemy_ids:
            if eid not in self._enemy_defs:
                raise CombatError(f"unknown enemy_id {eid!r}")
        for cid in starting_deck:
            if cid not in self._card_defs:
                raise CombatError(f"unknown card_id {cid!r}")

        monsters: list[MonsterState] = []
        hp_stream = self._rng.stream("enemy_hp")
        for slot, eid in enumerate(enemy_ids):
            edef = self._enemy_defs[eid]
            hp = hp_stream.randint(edef.hp_min, edef.hp_max)
            monster = MonsterState(
                monster_id=eid,
                name=edef.name,
                hp=hp,
                max_hp=hp,
                slot=slot,
            )
            # Apply any pre-combat powers (Lagavulin's starting Metallicize 8,
            # Guardian's starting Mode Shift buff etc.). applied_on_turn=0 is
            # strictly less than any live turn, so decay / tick logic treats
            # these as if they'd been on the monster forever.
            for pid, stacks in edef.starting_powers:
                monster.powers[pid] = stacks
                monster.powers_applied_on_turn[pid] = 0
                monster.powers_applied_phase[pid] = CombatPhase.START
            monsters.append(monster)

        combat = CombatState(turn=0, phase=CombatPhase.START, monsters=monsters, outcome=None)
        self._run.combat = combat
        self._combat = combat

        self._player.deck = list(starting_deck)
        self._player.hand = []
        self._player.discard_pile = []
        self._player.exhaust_pile = []
        self._player.draw_pile = list(starting_deck)
        self._rng.stream("combat_shuffle").shuffle(self._player.draw_pile)
        self._player.block = 0
        self._player.energy = self._player.max_energy
        self._player.powers = {}
        self._player.powers_applied_on_turn = {}

        self._hand_size = int(hand_size)
        self._register_player_power_hooks()
        self._hooks.dispatch("on_combat_start", {"monsters": monsters})
        self._begin_player_turn(initial=True)

    def play_card(self, card_id: str, *, target_slot: int | None = None) -> None:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER:
            raise CombatError(f"play_card called in phase {combat.phase!r}")
        if combat.outcome is not None:
            raise CombatError("combat already finished")
        if card_id not in self._player.hand:
            raise CombatError(f"card {card_id!r} not in hand")

        card = self._card_defs[card_id]
        if card.cost > self._player.energy:
            raise CombatError(f"insufficient energy to play {card_id!r}")

        if card.target == "single_enemy":
            if target_slot is None:
                raise CombatError(f"{card_id!r} requires target_slot")
            if not (0 <= target_slot < len(combat.monsters)):
                raise CombatError(f"target_slot {target_slot} out of range")
            if not combat.monsters[target_slot].alive:
                raise CombatError(f"target slot {target_slot} is dead")
        else:
            target_slot = None

        self._player.energy -= card.cost
        self._player.hand.remove(card_id)
        combat.last_card_played = card_id

        ctx = _EffectContext(
            actor="player",
            source_card=card,
            captured_target_slot=target_slot,
        )
        for step in card.effects:
            self._apply_effect(step, ctx)
            if combat.outcome is not None:
                break

        if combat.outcome is None:
            if card.exhaust_on_play:
                self._player.exhaust_pile.append(card_id)
                self._hooks.dispatch("on_card_exhausted", {"card_id": card_id, "source": "exhaust_on_play"})
            else:
                self._player.discard_pile.append(card_id)
            self._hooks.dispatch("on_card_played", {"card_id": card_id})
            if not any(m.alive for m in combat.monsters):
                self._end_combat("victory")

    def end_turn(self) -> None:
        combat = self.combat
        if combat.phase != CombatPhase.PLAYER:
            raise CombatError(f"end_turn called in phase {combat.phase!r}")
        if combat.outcome is not None:
            raise CombatError("combat already finished")
        self._end_player_turn()

    # ------------------------------------------------------------------
    # Turn flow

    def _begin_player_turn(self, *, initial: bool) -> None:
        combat = self.combat
        combat.turn += 1
        combat.phase = CombatPhase.PLAYER
        if not initial:
            # Block carries into enemy turn and resets at start of next player
            # turn — see schemas.py gain_block spec. Barricade (Phase 2d) skips
            # the reset so block persists across turns.
            if self._player.powers.get("barricade", 0) == 0:
                self._player.block = 0
            self._player.energy = self._player.max_energy
        # Pick upcoming moves for each alive enemy BEFORE drawing so hooks
        # that read intents on draw see the current telegraph.
        for m in combat.monsters:
            if m.alive:
                self._pick_next_move(m)
        self._draw_cards(self._hand_size)
        self._hooks.dispatch("on_player_turn_start", {"turn": combat.turn})

    def _end_player_turn(self) -> None:
        combat = self.combat
        # Ethereal cards still in hand at end of turn exhaust rather than
        # discard (STS "ethereal." text). Walk hand once, routing cards to
        # exhaust vs discard based on the schema flag.
        kept_discards: list[str] = []
        for card_id in self._player.hand:
            card = self._card_defs.get(card_id)
            if card is not None and card.ethereal:
                self._player.exhaust_pile.append(card_id)
                self._hooks.dispatch("on_card_exhausted", {"card_id": card_id, "source": "ethereal"})
            else:
                kept_discards.append(card_id)
        self._player.discard_pile.extend(kept_discards)
        self._player.hand.clear()
        self._decay_turn_powers(self._player, owner_phase=CombatPhase.PLAYER)
        self._hooks.dispatch("on_player_turn_end", {})
        combat.phase = CombatPhase.ENEMY
        self._run_enemy_turn()

    def _run_enemy_turn(self) -> None:
        combat = self.combat
        for monster in combat.monsters:
            if not monster.alive:
                continue
            monster.block = 0
            self._execute_enemy_move(monster)
            if not self._player.alive:
                self._end_combat("defeat")
                return
        if combat.outcome is not None:
            return
        for monster in combat.monsters:
            if monster.alive:
                self._tick_end_of_turn_powers(monster, owner_phase=CombatPhase.ENEMY)
                self._decay_turn_powers(monster, owner_phase=CombatPhase.ENEMY)
        self._hooks.dispatch("on_enemy_turn_end", {})
        if not any(m.alive for m in combat.monsters):
            self._end_combat("victory")
            return
        if not self._player.alive:
            self._end_combat("defeat")
            return
        self._begin_player_turn(initial=False)

    def _execute_enemy_move(self, monster: MonsterState) -> None:
        move_id = monster.queued_move
        if move_id is None:
            # Shouldn't happen — _pick_next_move always sets one.
            return
        edef = self._enemy_defs[monster.monster_id]
        move = edef.moves[move_id]
        monster.move_history.append(move_id)
        if len(monster.move_history) > 3:
            monster.move_history = monster.move_history[-3:]

        ctx = _EffectContext(
            actor="enemy",
            actor_monster=monster,
            source_move_id=move_id,
        )
        for step in move.effects:
            self._apply_effect(step, ctx)
            if not self._player.alive:
                return

    def _end_combat(self, outcome: str) -> None:
        combat = self.combat
        combat.outcome = outcome
        combat.phase = CombatPhase.END
        self._hooks.dispatch("on_combat_end", {"outcome": outcome})

    # ------------------------------------------------------------------
    # Verb dispatch

    def _apply_effect(self, step: EffectStep, ctx: _EffectContext) -> None:
        verb = step.verb
        args = step.args
        if verb == "deal_damage":
            self._v_deal_damage(args, ctx)
        elif verb == "deal_damage_strike_scaled":
            self._v_deal_damage_strike_scaled(args, ctx)
        elif verb == "deal_damage_equal_to_block":
            self._v_deal_damage_equal_to_block(args, ctx)
        elif verb == "deal_damage_from_power":
            self._v_deal_damage_from_power(args, ctx)
        elif verb == "activate_hex_charge":
            self._v_activate_hex_charge(args, ctx)
        elif verb == "gain_block":
            self._v_gain_block(args, ctx)
        elif verb == "apply_power":
            self._v_apply_power(args, ctx)
        elif verb == "draw_cards":
            self._v_draw_cards(args, ctx)
        elif verb == "copy_to_discard":
            self._v_copy_to_discard(args, ctx)
        elif verb == "gain_energy":
            self._v_gain_energy(args, ctx)
        elif verb == "lose_hp_self":
            self._v_lose_hp_self(args, ctx)
        elif verb == "add_card_to_pile":
            self._v_add_card_to_pile(args, ctx)
        else:
            raise CombatError(f"unhandled verb {verb!r}")

    def _v_deal_damage(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        amount = int(args["amount"])
        scope = args["target_scope"]
        hits = int(args.get("hits", 1))
        for hit_index in range(hits):
            base = amount
            if ctx.actor == "player":
                if ctx.source_card is not None and ctx.source_card.card_type == "attack":
                    base += self._player.powers.get("strength", 0)
                if self._player.powers.get("weak", 0) > 0:
                    base = (base * 3) // 4
            else:
                assert ctx.actor_monster is not None
                base += ctx.actor_monster.powers.get("strength", 0)
                if ctx.actor_monster.powers.get("weak", 0) > 0:
                    base = (base * 3) // 4

            targets = self._resolve_targets(scope, ctx, require_alive=True)
            if not targets:
                return
            for tgt in targets:
                per_hit = base
                if _get_power(tgt, "vulnerable") > 0:
                    per_hit = (per_hit * 3) // 2
                absorbed = min(per_hit, tgt.block)
                tgt.block -= absorbed
                remainder = per_hit - absorbed
                _apply_hp_loss(tgt, remainder)
                self._hooks.dispatch("on_damaged", {
                    "actor": ctx.actor,
                    "source_card_id": ctx.source_card.card_id if ctx.source_card else None,
                    "source_move_id": ctx.source_move_id,
                    "target": tgt,
                    "amount": remainder,
                    "blocked": absorbed,
                    "hit_index": hit_index,
                })
                if isinstance(tgt, MonsterState) and tgt.hp == 0:
                    self._hooks.dispatch("on_enemy_killed", {"target": tgt})
                    # On-death spawns (Slime Boss split). Spawned monsters
                    # join `combat.monsters` at new slots, inheriting fresh
                    # HP rolls + starting_powers + an initial queued move.
                    edef = self._enemy_defs.get(tgt.monster_id)
                    if edef is not None:
                        for spawn_id in edef.on_death_spawn:
                            self._spawn_monster(spawn_id)

    def _v_deal_damage_strike_scaled(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        base = int(args["base"])
        per_strike = int(args["per_strike_bonus"])
        count = 0
        for pile in (
            self._player.hand,
            self._player.draw_pile,
            self._player.discard_pile,
            self._player.exhaust_pile,
        ):
            count += sum(1 for cid in pile if "strike" in cid)
        if ctx.source_card is not None and "strike" in ctx.source_card.card_id:
            count += 1
        amount = base + per_strike * count
        self._v_deal_damage(
            {"amount": amount, "target_scope": args["target_scope"], "hits": int(args.get("hits", 1))},
            ctx,
        )

    def _v_deal_damage_equal_to_block(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        self._v_deal_damage(
            {
                "amount": self._player.block,
                "target_scope": args["target_scope"],
                "hits": int(args.get("hits", 1)),
            },
            ctx,
        )

    def _v_deal_damage_from_power(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        pid = args["power_id"]
        if ctx.actor == "enemy":
            assert ctx.actor_monster is not None
            amount = ctx.actor_monster.powers.get(pid, 0)
        else:
            amount = self._player.powers.get(pid, 0)
        self._v_deal_damage(
            {
                "amount": amount,
                "target_scope": args["target_scope"],
                "hits": int(args.get("hits", 1)),
            },
            ctx,
        )

    def _v_activate_hex_charge(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        del args
        amount = max(1, self._player.hp // 12)
        owner: Any = ctx.actor_monster if ctx.actor == "enemy" else self._player
        owner.powers["hex_charge"] = amount
        owner.powers_applied_on_turn["hex_charge"] = self.combat.turn
        owner.powers_applied_phase["hex_charge"] = self.combat.phase
        self._hooks.dispatch("on_power_applied", {
            "actor": ctx.actor,
            "source_card_id": None,
            "source_move_id": ctx.source_move_id,
            "target": owner,
            "power_id": "hex_charge",
            "amount": amount,
        })

    def _v_gain_energy(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        del ctx
        amount = int(args["amount"])
        self._player.energy += amount
        self._hooks.dispatch("on_energy_gained", {"amount": amount})

    def _v_lose_hp_self(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        del ctx
        amount = int(args["amount"])
        if amount <= 0:
            return
        actual = min(amount, self._player.hp)
        self._player.hp -= actual
        self._hooks.dispatch("on_hp_lost", {"amount": actual, "source": "self"})
        if self._player.hp == 0:
            self._end_combat("defeat")

    def _v_add_card_to_pile(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        del ctx
        card_id = args["card_id"]
        pile = args["pile"]
        amount = int(args["amount"])
        for _ in range(amount):
            if pile == "hand":
                if len(self._player.hand) < 10:
                    self._player.hand.append(card_id)
                    self._hooks.dispatch("on_card_added_to_hand", {"card_id": card_id})
                else:
                    self._player.discard_pile.append(card_id)
                    self._hooks.dispatch("on_card_overdrawn", {"card_id": card_id})
            elif pile == "draw":
                if not self._player.draw_pile:
                    self._player.draw_pile.append(card_id)
                else:
                    pos = self._rng.stream("combat_shuffle").randint(0, len(self._player.draw_pile))
                    self._player.draw_pile.insert(pos, card_id)
                self._hooks.dispatch("on_card_added_to_draw", {"card_id": card_id})
            elif pile == "discard":
                self._player.discard_pile.append(card_id)
                self._hooks.dispatch("on_card_added_to_discard",
                                     {"card_id": card_id, "source": "add_card_to_pile"})
            elif pile == "exhaust":
                self._player.exhaust_pile.append(card_id)
                self._hooks.dispatch("on_card_exhausted",
                                     {"card_id": card_id, "source": "add_card_to_pile"})
            else:
                raise CombatError(f"unknown pile {pile!r}")

    def _v_gain_block(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        base = int(args["amount"])
        if ctx.actor == "player":
            base += self._player.powers.get("dexterity", 0)
            if self._player.powers.get("frail", 0) > 0:
                base = (base * 3) // 4
            self._player.block += max(0, base)
            self._hooks.dispatch("on_block_gained", {
                "actor": "player",
                "source_card_id": ctx.source_card.card_id if ctx.source_card else None,
                "amount": base,
            })
        else:
            assert ctx.actor_monster is not None
            ctx.actor_monster.block += max(0, base)
            self._hooks.dispatch("on_block_gained", {
                "actor": "enemy",
                "source_move_id": ctx.source_move_id,
                "amount": base,
            })

    def _v_apply_power(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        pid = args["power_id"]
        amount = int(args["amount"])
        scope = args["target_scope"]
        if amount == 0:
            return
        if pid not in self._power_defs:
            raise CombatError(f"apply_power references unknown power {pid!r}")
        pdef = self._power_defs[pid]
        targets = self._resolve_targets(scope, ctx, require_alive=True)
        for tgt in targets:
            tgt.powers[pid] = tgt.powers.get(pid, 0) + amount
            if pdef.duration != "permanent" and pid not in tgt.powers_applied_on_turn:
                tgt.powers_applied_on_turn[pid] = self.combat.turn
                # Track phase so decay only skips when applied during the
                # owner's own phase (self-apply); cross-phase applies (e.g.
                # Bash's vul on an enemy) do not get a skip grace turn.
                tgt.powers_applied_phase[pid] = self.combat.phase
            self._hooks.dispatch("on_power_applied", {
                "actor": ctx.actor,
                "source_card_id": ctx.source_card.card_id if ctx.source_card else None,
                "source_move_id": ctx.source_move_id,
                "target": tgt,
                "power_id": pid,
                "amount": amount,
            })

    def _v_draw_cards(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        del ctx
        self._draw_cards(int(args["amount"]))

    def _v_copy_to_discard(self, args: dict[str, Any], ctx: _EffectContext) -> None:
        del args
        if ctx.source_card is None:
            return
        self._player.discard_pile.append(ctx.source_card.card_id)
        self._hooks.dispatch("on_card_added_to_discard", {
            "card_id": ctx.source_card.card_id,
            "source": "copy_to_discard",
        })

    # ------------------------------------------------------------------
    # Target resolution

    def _resolve_targets(
        self,
        scope: str,
        ctx: _EffectContext,
        *,
        require_alive: bool,
    ) -> list[Any]:
        combat = self.combat
        if scope == "self":
            if ctx.actor == "player":
                return [self._player] if (not require_alive or self._player.alive) else []
            assert ctx.actor_monster is not None
            return [ctx.actor_monster] if (not require_alive or ctx.actor_monster.alive) else []
        if scope == "player":
            return [self._player] if (not require_alive or self._player.alive) else []
        if scope == "single_enemy":
            slot = ctx.captured_target_slot
            if slot is None:
                return []
            m = combat.monsters[slot]
            return [m] if (not require_alive or m.alive) else []
        if scope == "all_enemies":
            return [m for m in combat.monsters if m.alive]
        if scope == "random_enemy":
            alive = [m for m in combat.monsters if m.alive]
            if not alive:
                return []
            return [self._rng.stream("combat_target").choice(alive)]
        raise CombatError(f"unknown target_scope {scope!r}")

    # ------------------------------------------------------------------
    # Cards: draw / shuffle

    def _draw_cards(self, amount: int) -> None:
        for _ in range(max(0, amount)):
            if not self._player.draw_pile:
                if not self._player.discard_pile:
                    return
                self._player.draw_pile = list(self._player.discard_pile)
                self._player.discard_pile.clear()
                self._rng.stream("combat_shuffle").shuffle(self._player.draw_pile)
            card_id = self._player.draw_pile.pop()
            if len(self._player.hand) >= 10:
                self._player.discard_pile.append(card_id)
                self._hooks.dispatch("on_card_overdrawn", {"card_id": card_id})
            else:
                self._player.hand.append(card_id)
                self._hooks.dispatch("on_card_drawn", {"card_id": card_id, "source": "draw_cards"})

    # ------------------------------------------------------------------
    # Powers: end-of-turn decay + ritual tick

    def _decay_turn_powers(
        self,
        owner: PlayerState | MonsterState,
        *,
        owner_phase: str,
    ) -> None:
        """Decrement every turn-scoped power on `owner`.

        Skip the first decrement only when the power was applied by the
        owner's own action this same turn (self-apply). Debuffs applied by
        the opposing side do NOT get a grace turn — they decrement at the
        very next end of the owner's turn. See VERB_DOCS["apply_power"]."""

        combat = self.combat
        to_remove: list[str] = []
        for pid, stacks in list(owner.powers.items()):
            pdef = self._power_defs.get(pid)
            if pdef is None or pdef.duration != "turns":
                continue
            applied_on = owner.powers_applied_on_turn.get(pid, combat.turn)
            applied_phase = owner.powers_applied_phase.get(pid)
            if applied_on == combat.turn and applied_phase == owner_phase:
                continue
            owner.powers[pid] = stacks - 1
            if owner.powers[pid] <= 0:
                to_remove.append(pid)
        for pid in to_remove:
            owner.powers.pop(pid, None)
            owner.powers_applied_on_turn.pop(pid, None)
            owner.powers_applied_phase.pop(pid, None)

    def _tick_end_of_turn_powers(
        self,
        owner: PlayerState | MonsterState,
        *,
        owner_phase: str,
    ) -> None:
        """Fire end-of-turn triggers for duration=end_of_turn_tick powers.

        Phase 1 only ritual. Ritual grants `stacks` of strength to the owner
        at the end of every one of their turns, starting the turn AFTER it
        was applied (see Ritual's applied_on_turn note in VERB_DOCS)."""

        combat = self.combat
        for pid, stacks in list(owner.powers.items()):
            pdef = self._power_defs.get(pid)
            if pdef is None:
                continue
            # Enemy-side Metallicize tick (Lagavulin's starting 8).
            # The player side fires via on_player_turn_end hook; monsters
            # don't ride that hook, so handle them inline here. Permanent
            # duration means no applied-this-turn skip is needed.
            if pid == "metallicize" and isinstance(owner, MonsterState):
                owner.block += stacks
                self._hooks.dispatch("on_block_gained", {
                    "actor": "enemy",
                    "source_move_id": None,
                    "amount": stacks,
                })
                continue
            if pdef.duration != "end_of_turn_tick":
                continue
            applied_on = owner.powers_applied_on_turn.get(pid, combat.turn)
            applied_phase = owner.powers_applied_phase.get(pid)
            if applied_on == combat.turn and applied_phase == owner_phase:
                continue
            if pid == "ritual":
                owner.powers["strength"] = owner.powers.get("strength", 0) + stacks
                self._hooks.dispatch("on_power_applied", {
                    "actor": "enemy" if isinstance(owner, MonsterState) else "player",
                    "source_card_id": None,
                    "source_move_id": None,
                    "target": owner,
                    "power_id": "strength",
                    "amount": stacks,
                })

    # ------------------------------------------------------------------
    # Player-side power hooks (Phase 2d)
    #
    # Each handler reads the player's current stacks for its power and
    # acts. Handlers are registered once per combat in start_combat; the
    # stacks check makes a handler for a never-applied power a cheap
    # no-op, so there's no per-apply hook registration/teardown.

    def _register_player_power_hooks(self) -> None:
        self._hooks.register("on_player_turn_end", self._power_tick_player_turn_end, priority=200)
        self._hooks.register("on_hp_lost", self._power_tick_hp_lost, priority=200)
        self._hooks.register("on_card_exhausted", self._power_tick_card_exhausted, priority=200)
        # Enemy-side Enrage (Gremlin Nob): triggers when the player plays a
        # Skill card. The handler iterates monsters so it covers an arbitrary
        # number of Enrage-bearing enemies in future encounters.
        self._hooks.register("on_card_played", self._power_tick_card_played, priority=200)

    def _power_tick_player_turn_end(self, payload: dict[str, Any]) -> None:
        del payload
        met = self._player.powers.get("metallicize", 0)
        if met > 0:
            self._v_gain_block(
                {"amount": met, "target_scope": "self"},
                _EffectContext(actor="player"),
            )
        com = self._player.powers.get("combust", 0)
        if com > 0:
            # Combust self-damage is card-sourced for Rupture's purposes,
            # so route through lose_hp_self (fires on_hp_lost -> cascades).
            self._v_lose_hp_self(
                {"amount": 1, "target_scope": "self"},
                _EffectContext(actor="player"),
            )
            # If Combust's self-tick dropped the player to 0, _v_lose_hp_self
            # has already called _end_combat. Guard before the AoE.
            if self.combat.outcome is None:
                self._v_deal_damage(
                    {"amount": com, "target_scope": "all_enemies", "hits": 1},
                    _EffectContext(actor="player"),
                )
        # Strength Down / Dexterity Down: STS semantics are "At the end of
        # your turn, lose X Strength (or Dex) and this debuff is removed."
        # Fires once per end-of-turn and then clears. Both powers are
        # authored with duration="permanent" so _decay_turn_powers does NOT
        # decrement them before this handler runs — otherwise a 2-stack
        # debuff would only subtract 1 before being consumed.
        for debuff_id, stat_id in (("strength_down", "strength"), ("dexterity_down", "dexterity")):
            delta = self._player.powers.get(debuff_id, 0)
            if delta > 0:
                self._player.powers[stat_id] = self._player.powers.get(stat_id, 0) - delta
                self._player.powers.pop(debuff_id, None)
                self._player.powers_applied_on_turn.pop(debuff_id, None)
                self._player.powers_applied_phase.pop(debuff_id, None)
                self._hooks.dispatch("on_power_applied", {
                    "actor": "player",
                    "source_card_id": None,
                    "source_move_id": None,
                    "target": self._player,
                    "power_id": stat_id,
                    "amount": -delta,
                })

    def _power_tick_hp_lost(self, payload: dict[str, Any]) -> None:
        del payload
        rup = self._player.powers.get("rupture", 0)
        if rup > 0:
            self._v_apply_power(
                {"power_id": "strength", "amount": rup, "target_scope": "self"},
                _EffectContext(actor="player"),
            )

    def _power_tick_card_exhausted(self, payload: dict[str, Any]) -> None:
        del payload
        de = self._player.powers.get("dark_embrace", 0)
        if de > 0:
            self._draw_cards(de)
        fnp = self._player.powers.get("feel_no_pain", 0)
        if fnp > 0:
            self._v_gain_block(
                {"amount": fnp, "target_scope": "self"},
                _EffectContext(actor="player"),
            )

    def _power_tick_card_played(self, payload: dict[str, Any]) -> None:
        """Enemy-side on-skill-played reactions (Gremlin Nob's Enrage).

        Fires for every card the player plays; filters by card_type to
        act only on skills, then iterates alive monsters so a hypothetical
        encounter with multiple Enrage-bearing enemies covers them all.
        """

        card_id = payload.get("card_id")
        if card_id is None:
            return
        card = self._card_defs.get(card_id)
        if card is None or card.card_type != "skill":
            return
        for monster in self.combat.monsters:
            if not monster.alive:
                continue
            enrage = monster.powers.get("enrage", 0)
            if enrage > 0:
                monster.powers["strength"] = monster.powers.get("strength", 0) + enrage
                self._hooks.dispatch("on_power_applied", {
                    "actor": "enemy",
                    "source_card_id": None,
                    "source_move_id": None,
                    "target": monster,
                    "power_id": "strength",
                    "amount": enrage,
                })

    # ------------------------------------------------------------------
    # Monster spawning (Phase 2c-2 Slime Boss split)

    def _spawn_monster(self, enemy_id: str) -> MonsterState:
        """Add a fresh monster to the combat mid-fight.

        Called from the deal_damage pipeline when a dying monster has
        on_death_spawn entries. The new monster:
          * rolls HP from the `enemy_hp` stream (deterministic).
          * inherits starting_powers.
          * gets a `queued_move` picked immediately so the enemy phase
            of the current/next turn has a telegraph to execute.

        Returns the new MonsterState so callers can inspect it.
        """

        edef = self._enemy_defs[enemy_id]
        hp = self._rng.stream("enemy_hp").randint(edef.hp_min, edef.hp_max)
        monster = MonsterState(
            monster_id=enemy_id,
            name=edef.name,
            hp=hp,
            max_hp=hp,
            slot=len(self.combat.monsters),
        )
        for pid, stacks in edef.starting_powers:
            monster.powers[pid] = stacks
            monster.powers_applied_on_turn[pid] = self.combat.turn
            monster.powers_applied_phase[pid] = CombatPhase.START
        self.combat.monsters.append(monster)
        self._pick_next_move(monster)
        self._hooks.dispatch("on_enemy_spawned", {"target": monster})
        return monster

    # ------------------------------------------------------------------
    # Enemy move picker

    def _pick_next_move(self, monster: MonsterState) -> None:
        edef = self._enemy_defs[monster.monster_id]
        combat = self.combat
        last1 = monster.move_history[-1] if monster.move_history else None
        last2 = monster.move_history[-2] if len(monster.move_history) >= 2 else None

        # Collect eligible selectors by rule.
        if combat.turn == 1:
            firsts = [s for s in edef.movepicker if s.rule == "always_first"]
            if firsts:
                monster.queued_move = _weighted_pick(firsts, self._rng.stream("monster_moves"))
                return
        # Sequential: pick the entry whose sequence_index matches the current turn offset.
        # Offset = number of post-first-turn actions this enemy has taken.
        turns_after_first = len(monster.move_history)
        if turns_after_first == 0 and combat.turn > 1:
            turns_after_first = combat.turn - 1
        seq_entries = [s for s in edef.movepicker if s.rule == "sequential"]
        if seq_entries:
            max_seq = max(s.sequence_index or 0 for s in seq_entries)
            weighted_entries = [
                s for s in edef.movepicker
                if s.rule in ("weighted", "if_not_last", "if_not_two")
            ]
            if turns_after_first <= max_seq:
                match = [s for s in seq_entries if s.sequence_index == turns_after_first]
                if match:
                    monster.queued_move = match[0].move_id
                    return
            elif not weighted_entries:
                # Chain exhausted and no weighted fallback — repeat the
                # final sequential (Cultist's Dark Strike loop).
                match = [s for s in seq_entries if s.sequence_index == max_seq]
                if match:
                    monster.queued_move = match[0].move_id
                    return
            # Chain exhausted AND weighted alternatives exist — fall through
            # to the weighted pool below (Lagavulin's wake-up transition).

        pool: list[SelectorEntry] = []
        for s in edef.movepicker:
            if s.rule == "always_first":
                continue
            if s.rule == "sequential":
                continue
            if s.rule == "if_not_last" and last1 == s.move_id:
                continue
            if s.rule == "if_not_two" and (last1 == s.move_id or last2 == s.move_id):
                continue
            pool.append(s)
        if not pool:
            # Fallback: use any entry's move (shouldn't happen if data is sane).
            pool = [s for s in edef.movepicker if s.rule not in ("always_first", "sequential")]
            if not pool:
                return
        monster.queued_move = _weighted_pick(pool, self._rng.stream("monster_moves"))


# ---------------------------------------------------------------------------
# Helpers

def _get_power(owner: Any, pid: str) -> int:
    return owner.powers.get(pid, 0)


def _apply_hp_loss(owner: Any, amount: int) -> None:
    if amount <= 0:
        return
    owner.hp = max(0, owner.hp - amount)


def _weighted_pick(entries: list[SelectorEntry], rng) -> str:
    weights = [max(1, e.weight) for e in entries]
    chosen = rng.choices(entries, weights=weights, k=1)[0]
    return chosen.move_id


__all__ = ["CombatContext", "CombatError"]
