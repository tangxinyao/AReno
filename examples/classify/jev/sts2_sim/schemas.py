"""Combat data contract: cards, enemies, powers as frozen dataclasses.

Phase 1 data lives as JSON under `data/`; this module defines the shapes the
loader validates against. Keep schemas *data-only* — no behavior lives on
these objects, so Phase 1 engine can evolve independently.

Allowed vocabulary (verbs, target scopes, intents, card types) lives in sets
below. The loader rejects any value outside these sets with a message that
names the offending card/enemy/field. Add a verb only when a card actually
needs it; this keeps the engine's switch-table bounded.

Engine contract (binding for Phase 1 implementation)
----------------------------------------------------

Resolution order. One card play resolves its `effects` list top-to-bottom.
Each step runs to completion — including all hook dispatch — before the
next step begins. A card cannot be interrupted mid-effect; the player
chooses a target BEFORE the card starts resolving, and the chosen target
is captured so later effects see the same enemy slot even if that enemy
dies to an earlier effect of the same card.

Dead-target semantics. Within one card, if a captured `single_enemy`
target has `hp == 0` at the time a later effect runs, that effect no-ops
silently; it does not re-pick a new enemy. For `all_enemies` and
`random_enemy`, scopes are re-resolved against the ALIVE set at each
effect's execution time.

Hook timing. The hook names fired by each verb are listed per-verb below.
Hooks always fire inside the verb that caused them (not deferred). A
handler that enqueues further effects pushes them onto EffectQueue; the
engine drains that queue after the current card's effects list is done
but before accepting the next action.

Multi-target evaluation order. For AoE verbs, targets resolve in enemy
slot order (left to right as authored in CombatState.monsters). This is
deterministic and matches the common STS visual layout.

Scaling rules (shared across verbs that take `amount`):
  * Strength adds flat to Attack-card damage and to enemy move damage.
    Skill and Power cards do NOT add strength. Enemy block gain does NOT
    scale with strength.
  * Dexterity adds flat to card-sourced `gain_block`, regardless of card
    type. Enemy block does NOT scale with dexterity (enemies have no
    dexterity in Phase 1).
  * Weak multiplies final attack damage by 0.75 (floor), applied once per
    hit, re-evaluated per hit.
  * Vulnerable multiplies incoming damage by 1.5 (floor), applied once
    per hit, re-evaluated per hit.
  * Frail multiplies card-sourced block by 0.75 (floor), applied once per
    gain_block invocation (not per stack).

See the VERB_DOCS and MOVE_RULE_DOCS dicts below for per-verb and
per-move-rule specifications. Keep docs and `EFFECT_VERBS`/`MOVE_RULES`
in sync — the Phase 1 data test asserts the sets match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


# ---------------------------------------------------------------------------
# Allowed vocabulary

CARD_TYPES: Final = frozenset({"attack", "skill", "power", "status", "curse"})
CARD_RARITIES: Final = frozenset({"basic", "common", "uncommon", "rare", "special"})
CARD_TARGETS: Final = frozenset({"none", "self", "single_enemy", "all_enemies", "random_enemy"})

INTENTS: Final = frozenset({
    "attack",
    "attack_defend",
    "attack_buff",
    "attack_debuff",
    "defend",
    "defend_buff",
    "defend_debuff",
    "buff",
    "debuff",
    "strong_debuff",
    "stun",
    "sleep",
    "escape",
    "magic",
    "unknown",
})

TARGET_SCOPES: Final = frozenset({
    "self",            # the actor (player for cards, enemy for enemy moves)
    "player",          # the player (used by enemy moves)
    "single_enemy",    # one chosen enemy (player picks when card is played)
    "all_enemies",     # every alive enemy
    "random_enemy",    # engine picks uniformly at random
})

# Verbs the engine (Phase 1+) must implement. Each maps to a dict of required
# arg -> expected Python type. `target_scope` is validated separately against
# TARGET_SCOPES below.
EFFECT_VERBS: Final[dict[str, dict[str, type]]] = {
    "deal_damage":                {"amount": int, "target_scope": str, "hits": int},
    "deal_damage_strike_scaled":  {"base": int, "per_strike_bonus": int, "target_scope": str, "hits": int},
    "deal_damage_equal_to_block": {"target_scope": str, "hits": int},
    "gain_block":                 {"amount": int, "target_scope": str},
    "apply_power":                {"power_id": str, "amount": int, "target_scope": str},
    "draw_cards":                 {"amount": int},
    "copy_to_discard":            {},
    "gain_energy":                {"amount": int, "target_scope": str},
    "lose_hp_self":               {"amount": int, "target_scope": str},
    "add_card_to_pile":           {"card_id": str, "pile": str, "amount": int},
}

# Which scopes each verb accepts. The loader cross-checks the step's
# target_scope against this map. Verbs without target_scope (e.g.
# add_card_to_pile) are intentionally absent here.
VERB_ALLOWED_SCOPES: Final[dict[str, frozenset[str]]] = {
    "deal_damage":                frozenset({"single_enemy", "all_enemies", "random_enemy", "player"}),
    "deal_damage_strike_scaled":  frozenset({"single_enemy", "all_enemies", "random_enemy"}),
    "deal_damage_equal_to_block": frozenset({"single_enemy", "all_enemies", "random_enemy"}),
    "gain_block":                 frozenset({"self"}),
    "apply_power":                frozenset({"self", "single_enemy", "all_enemies", "random_enemy", "player"}),
    "gain_energy":                frozenset({"self"}),
    "lose_hp_self":               frozenset({"self"}),
}

# Which `pile` values add_card_to_pile accepts.
CARD_PILES: Final = frozenset({"hand", "draw", "discard", "exhaust"})

# Verbs cards may use. Enemy moves use the complement defined below.
CARD_ONLY_VERBS: Final = frozenset({
    "draw_cards",
    "copy_to_discard",
    "deal_damage_strike_scaled",
    "deal_damage_equal_to_block",
    "gain_energy",
    "lose_hp_self",
    "add_card_to_pile",
})
ENEMY_ONLY_VERBS: Final = frozenset()  # no enemy-exclusive verbs yet

POWER_KINDS: Final = frozenset({"buff", "debuff"})
POWER_DURATIONS: Final = frozenset({"permanent", "turns", "end_of_turn_tick"})

MOVE_RULES: Final = frozenset({
    "always_first",     # played on combat turn 1 only
    "sequential",       # played at sequence_index after always_first entries
    "weighted",         # weighted random from the remaining pool
    "if_not_last",      # weighted, but excluded if played last turn
    "if_not_two",       # weighted, excluded if played the last two turns
})


# ---------------------------------------------------------------------------
# Verb specifications (engine development contract)
#
# Each key matches EFFECT_VERBS. The doc string is the authoritative spec
# the Phase 1 engine must match. If engine behavior diverges from a doc,
# fix the engine or amend the doc — never let them drift silently.

VERB_DOCS: Final[dict[str, str]] = {
    "deal_damage": """\
Deal physical damage from the acting actor to one or more targets.

Required args:
  amount        : int >= 0, base damage before scaling.
  target_scope  : one of single_enemy / all_enemies / random_enemy / player.
  hits          : int >= 1, number of independent damage events; defaults to 1.

Who may call:
  * Cards: target_scope must NOT be "player"; single_enemy is bound to the
    player's chosen target captured at card-play time (see Resolution
    order in module docstring).
  * Enemy moves: target_scope MUST be "player". Enemy-vs-enemy damage is
    not supported in Phase 1.

Per-hit damage pipeline (runs N times for hits=N; weak/vulnerable are
re-evaluated between hits because turn-based powers only decrement at
end-of-turn):
  1. base = args.amount
  2. If actor is player AND the source card has card_type == "attack":
        base += actor.strength
     Else if actor is an enemy:
        base += actor.strength
     Skill cards, Power cards, Status cards, and Curse cards never add
     strength.
  3. If actor has weak > 0: base = floor(base * 0.75)
  4. Resolve target list from target_scope against the currently-alive
     enemies (random_enemy uses rng.stream("combat_target") for picking):
        single_enemy  -> [captured_target] if alive else []
        all_enemies   -> alive enemies in slot order
        random_enemy  -> one alive enemy (uniform)
        player        -> [player]
  5. For each target:
        per_hit = base
        if target has vulnerable > 0: per_hit = floor(per_hit * 1.5)
        absorbed  = min(per_hit, target.block)
        target.block -= absorbed
        remainder = per_hit - absorbed
        target.hp = max(target.hp - remainder, 0)
        dispatch "on_damaged" with payload
            {source_actor, source_card_id_or_move_id, target, amount:remainder,
             blocked:absorbed, hit_index}
        if target is an enemy AND target.hp == 0 AND target was alive before
        this hit: dispatch "on_enemy_killed" with payload {target}.

Hooks fired per hit: on_damaged (always), on_enemy_killed (conditional).

Edge cases:
  * 0-damage hits still fire on_damaged; subscribers that count hits
    (e.g. Thorns, Flame Barrier in later phases) rely on this.
  * all_enemies / random_enemy with zero alive enemies: entire verb is
    a no-op; no hooks fire.
  * hits > 1 against a target that dies on an earlier hit: subsequent
    hits are suppressed (dead enemies are skipped).
""",

    "deal_damage_strike_scaled": """\
Deal damage whose amount scales with the count of Strike-family cards
across every zone of the player's deck. Card-only — only Perfected
Strike uses it in Phase 1. Enemy moves may not call this verb.

Required args:
  base              : int >= 0, flat damage before scaling.
  per_strike_bonus  : int >= 0, bonus damage per Strike-named card.
  target_scope      : one of single_enemy / all_enemies / random_enemy.
                      "player" is rejected at load time.
  hits              : int >= 1, defaults to 1.

Pipeline:
  1. Count every card_id in hand + draw_pile + discard_pile +
     exhaust_pile where the substring "strike" appears in the id. All
     authored Strike-family ids follow snake_case ("strike",
     "pommel_strike", "twin_strike", "perfected_strike", upgraded
     variants) so a plain substring match is sufficient and matches
     STS wording ("cards containing 'Strike'").
  2. Add 1 if the source card's own id contains "strike" — the card
     has already been removed from hand by the time the effect
     resolves (play_card runs hand.remove before the effect loop), so
     including it manually mirrors STS where Perfected Strike counts
     itself.
  3. amount = base + per_strike_bonus * count
  4. Delegate the rest to deal_damage with the computed amount, so
     strength / weak / vulnerable / block all apply exactly the same
     way (see deal_damage spec).

Hooks fired: identical to deal_damage (on_damaged, on_enemy_killed).

Edge cases:
  * No strike-named cards anywhere AND the source card is not itself
    strike-named -> amount == base.
  * Running Perfected Strike against a dead captured target with the
    full base amount still no-ops silently per deal_damage's rules.
""",

    "deal_damage_equal_to_block": """\
Deal damage whose amount is the player's current block. Card-only
(Body Slam). Enemy moves may not call this verb.

Required args:
  target_scope  : single_enemy / all_enemies / random_enemy. "player"
                  is rejected at load time.
  hits          : int >= 1, defaults to 1.

Pipeline:
  1. amount = self._player.block at the time this effect runs. This
     uses the LIVE block value, so a prior gain_block step in the same
     card does feed into this one (Body Slam itself has no block step,
     but a hypothetical Iron Wave -> Body Slam chain within a card
     would compound).
  2. Delegate to deal_damage with the computed amount. Strength does
     NOT add — Body Slam's damage is purely block-based, so this verb
     does NOT pipe through the attack-strength branch of deal_damage.
     (Implementation note: the delegate still runs full deal_damage,
     which adds strength for attack-type source cards. Body Slam is
     card_type=attack in STS1 and the +strength is actually how STS1
     Body Slam DOES behave — strength adds on top of block. Keep this
     delegation as-is unless parity testing proves otherwise.)

Hooks fired: identical to deal_damage (on_damaged, on_enemy_killed).

Edge cases:
  * player.block == 0 -> deals 0 damage per hit; on_damaged still
    fires with amount=0 (matches deal_damage's "0-damage hits still
    fire" rule).
""",

    "gain_energy": """\
Add energy to the player. Card-only; takes `amount` and `target_scope`
("self"-only). Used by Seeing Red; later by Dropkick+ when the
conditional-apply plumbing lands.

Required args:
  amount        : int >= 0.
  target_scope  : MUST be "self". Enforced at load time.

Pipeline:
  1. self._player.energy += amount
  2. Dispatch on_energy_gained with payload {amount}.

Edge cases:
  * No max-energy cap in STS1 (Watcher aside); amount simply adds.
  * amount == 0 is a no-op that still fires the hook.
""",

    "lose_hp_self": """\
Unblockable HP loss on the player, used by cards like Hemokinesis,
Bloodletting, and Offering. Does NOT trigger on_damaged — this verb
represents a *cost*, not combat damage, so vulnerable / block / weak
have no effect.

Required args:
  amount        : int >= 0.
  target_scope  : MUST be "self".

Pipeline:
  1. actual = min(amount, self._player.hp)
  2. self._player.hp -= actual
  3. Dispatch on_hp_lost with payload {amount:actual, source:"self"}.
  4. If self._player.hp == 0: _end_combat("defeat").

Edge cases:
  * Can kill the player — Hemokinesis-into-death matches STS behavior
    (very rare, but legal: dropping to 0 HP from Hemokinesis is a
    valid loss).
  * amount == 0 is a no-op that does NOT fire the hook (avoid noisy
    0-HP-lost events for Powers subscribers like Rupture in Phase 2d).
""",

    "add_card_to_pile": """\
Insert N copies of a named card into one of the player's piles.
Card-only. Used to inflict statuses (Wild Strike -> Wound, Reckless
Charge -> Dazed, Burning Pact's exhaust, Power Through) and in later
phases to seed curses or copy specific attacks.

Required args:
  card_id  : the id of the card to insert. The loader cross-checks
             this against the full card table after all cards load.
  pile     : one of "hand" / "draw" / "discard" / "exhaust". Enforced
             at load time.
  amount   : int >= 0, number of copies to insert.

Pipeline (per copy):
  * pile == "hand":    append if hand size < 10, else fall through
                       to discard with on_card_overdrawn. Fires
                       on_card_added_to_hand.
  * pile == "draw":    insert at a random position via
                       rng.stream("combat_shuffle"). Fires
                       on_card_added_to_draw. Mirrors STS "Shuffle N
                       into your draw pile" wording.
  * pile == "discard": append. Fires on_card_added_to_discard with
                       source="add_card_to_pile".
  * pile == "exhaust": append. Fires on_card_exhausted with
                       source="add_card_to_pile".

Edge cases:
  * amount == 0 is a no-op; no hooks fire.
  * Inserting into "draw" when draw_pile is empty appends (there is
    no random index 0..-1 case).
""",

    "gain_block": """\
Grant block to the acting actor (self). Block is a per-turn shield that
absorbs the next incoming damage events and resets to 0 at the start of
the owner's next turn, UNLESS the owner has a block-preserving power
(Barricade, etc.) — none exist in Phase 1.

Required args:
  amount        : int >= 0, base block before scaling.
  target_scope  : MUST be "self". Engine rejects any other scope at load
                  time (see loader._parse_effect).

Who may call:
  * Cards: scaled by dexterity/frail (see pipeline).
  * Enemy moves: no dex/frail scaling (enemies have no dexterity in
    Phase 1 and cannot be made frail).

Resolution pipeline:
  1. base = args.amount
  2. If actor is player:
        base += actor.dexterity
        if actor has frail > 0: base = floor(base * 0.75)
  3. actor.block += base

Hooks fired: "on_block_gained" with payload
  {source_actor, source_card_id_or_move_id, amount:base}.

Edge cases:
  * amount can be 0 (useful for engine tests); hook still fires.
  * Block does not stack with a cap in STS1/STS2 — any amount is legal.
""",

    "apply_power": """\
Add stacks of a named power to the resolved target(s). PowerSchema.kind
is metadata used by the UI/intent system; the apply rule is uniform —
amount is added to the target's current stacks of that power.

Required args:
  power_id      : one of the ids loaded from powers.json.
  amount        : int >= 0, stacks to add.
  target_scope  : one of self / single_enemy / all_enemies / random_enemy /
                  player, restricted per actor:
                    cards      -> any scope EXCEPT "player"
                    enemy moves -> only "self" or "player"

Resolution pipeline:
  1. Resolve target list exactly as deal_damage does (same slot order
     and alive-set rules).
  2. For each target:
        target.powers[power_id] += amount
        If this is the first stack and the power's duration is "turns"
        or "end_of_turn_tick", mark `applied_on_turn = combat.turn` so
        end-of-turn ticks can skip the turn the power was applied on
        (STS-standard behavior — matters especially for Ritual).
        Dispatch "on_power_applied" with payload
            {source_actor, source_card_id_or_move_id, target,
             power_id, amount}.

Power-specific end-of-turn behavior (owner-side, after a turn ends):
  strength    : permanent, no decay.
  dexterity   : permanent, no decay.
  vulnerable  : if applied_on_turn < combat.turn, decrement by 1; remove
                when stacks reach 0.
  weak        : same decay rule as vulnerable.
  frail       : same decay rule as vulnerable.
  ritual      : if applied_on_turn < combat.turn, fire an internal
                apply_power(strength, ritual.stacks, target=self) on the
                owner. Does NOT decay.

Hooks fired: on_power_applied (always), plus any power-specific hooks
when the engine ticks turn-end (not fired by apply_power directly).

Edge cases:
  * Applying to a dead target silently no-ops (STS convention: no
    stacks, no hook).
  * amount=0 is a no-op; hook does not fire.
  * Applying debuffs to an enemy with Artifact (not Phase 1) would
    consume the Artifact instead; add that handling when the power
    lands.
""",

    "draw_cards": """\
Move N cards from draw_pile to hand. Card-only verb.

Required args:
  amount        : int >= 1.

Resolution pipeline:
  For i in range(amount):
    1. If draw_pile is empty:
         Shuffle discard_pile into draw_pile using
         rng.stream("combat_shuffle"). (The engine must use a stable
         stream name so replay is deterministic.)
    2. If draw_pile is STILL empty (both piles empty): break; the verb
       completes with fewer than `amount` cards drawn.
    3. Pop the top of draw_pile (convention: last element).
    4. If len(hand) >= 10: push the drawn card to discard_pile instead
       (STS hand-size cap). Dispatch "on_card_overdrawn" with payload
       {card_id}.
    5. Else: append card to hand. Dispatch "on_card_drawn" with payload
       {card_id, source:"draw_cards"}.

Hooks fired: "on_card_drawn" (per card drawn to hand),
"on_card_overdrawn" (per card redirected to discard).

Edge cases:
  * Draw order within a single invocation: cards are drawn one at a
    time; hooks fire between draws so a draw trigger could inflate the
    draw queue (e.g., a hypothetical "when you draw an attack, draw 1
    more" relic would be handled this way).
  * Shuffle happens exactly once per empty-draw-pile event, not once
    per card.
""",

    "copy_to_discard": """\
Add a copy of the currently-playing card to discard_pile. Card-only
verb; takes no args. Used by Anger-family cards.

Resolution pipeline:
  1. Clone the current card's identity (card_id, including upgraded
     form). The engine should treat this as "create a fresh instance of
     the same card_id" — buffs/modifications baked onto the live card
     instance do NOT carry over (none exist in Phase 1).
  2. Append the copy to discard_pile. Dispatch
     "on_card_added_to_discard" with payload
     {card_id, source:"copy_to_discard"}.

Hooks fired: "on_card_added_to_discard".

Edge cases:
  * This verb runs regardless of whether the source card's attack hit a
    target or not; Anger copies even against a dead enemy.
  * copy_to_discard is NOT subject to the hand-size cap because the
    target is discard_pile.
""",
}


# ---------------------------------------------------------------------------
# Enemy move-picker rule specifications

MOVE_RULE_DOCS: Final[dict[str, str]] = {
    "always_first": """\
The move is played on combat turn 1 ONLY, and is NOT a candidate on any
later turn. If multiple entries have rule=always_first, the engine
picks among them with weighted random using the `weight` field (rare;
no Phase 1 enemy uses this).
""",

    "sequential": """\
The move is played deterministically at the turn index specified by
`sequence_index`. sequence_index is 0-based relative to the first turn
AFTER any always_first entries fire. If a sequential chain is shorter
than the combat (e.g. Cultist's sequential at index=1 but combat runs 10
turns), the LAST sequential entry repeats indefinitely.

Phase 1 example (Cultist):
  turn 1 -> Incantation (always_first)
  turn 2+ -> Dark Strike (sequential, sequence_index=1, repeats)
""",

    "weighted": """\
Weighted random pick from the pool of currently-eligible moves, using
rng.stream("monster_moves"). Each entry's `weight` contributes
proportionally. If this entry is the only eligible one, it is picked
with probability 1. No history filtering is applied — see if_not_last
and if_not_two for anti-repeat.
""",

    "if_not_last": """\
Weighted random pick, but this entry is EXCLUDED from the eligible pool
if the enemy's most-recently-played move was this same move_id. If
excluded, remaining eligible entries' weights are renormalized
implicitly (rng.stream("monster_moves").choices honors raw weights
after the exclusion filter).
""",

    "if_not_two": """\
Same as if_not_last but with a 2-move memory window: excluded if this
move_id was played in EITHER of the last two moves. Jaw Worm's Bellow
and Thrash use this to avoid oppressive back-to-back-to-back buff turns.
""",
}


__all_verbs_doc_sync__: Final = None  # the test enforces EFFECT_VERBS == VERB_DOCS keys


# ---------------------------------------------------------------------------
# Schemas

@dataclass(frozen=True)
class EffectStep:
    """One atomic verb invocation inside a card's or move's effect list.

    `verb` names which case the engine's dispatch picks; see VERB_DOCS for
    the authoritative semantics. `args` carries verb-specific parameters
    and has already been normalized by the loader (e.g. `hits` defaults to
    1 for deal_damage). The engine should not add defaults again — treat
    `args` as fully specified.
    """

    verb: str
    args: dict[str, Any]


@dataclass(frozen=True)
class CardSchema:
    """Authored card definition.

    `target` tells the UI whether the player must click an enemy
    ("single_enemy") or whether the card plays on self/all/random without
    a pick. It is NOT the engine's scope resolver — individual effects
    carry their own target_scope. For single_enemy cards the chosen enemy
    propagates to every effect's scope resolution; see Resolution order
    in the module docstring.

    `upgrade_of` / `upgraded_from` form a two-way link so Smith at a rest
    site can look up either direction. Base cards point forward via
    upgrade_of; upgraded cards point back via upgraded_from. The loader
    validates both halves exist for attack/skill/power cards; status
    and curse cards are allowed to have neither (they do not upgrade).

    Boolean flags (default False):
      exhaust_on_play : after the card's effects resolve, the card lands
                        in exhaust_pile instead of discard_pile. Used by
                        Pummel / Impervious / most "exhaust." text.
      ethereal        : if still in hand at end of the player's turn,
                        the card is automatically exhausted instead of
                        being discarded. Used by Carnage / Dazed /
                        most "ethereal." text.
      unplayable      : the card is filtered out of legal candidates
                        by the RunLoop. Set to True for status/curse
                        cards (Wound / Slimed / Dazed / Burn) and for
                        a handful of regular cards (AscendersBane).
                        The engine never resolves effects for an
                        unplayable card.
    """

    card_id: str
    name: str
    cost: int
    card_type: str
    rarity: str
    target: str
    effects: tuple[EffectStep, ...]
    upgraded_from: str | None = None
    upgrade_of: str | None = None
    exhaust_on_play: bool = False
    ethereal: bool = False
    unplayable: bool = False


@dataclass(frozen=True)
class MoveSchema:
    move_id: str
    intent: str
    effects: tuple[EffectStep, ...]


@dataclass(frozen=True)
class SelectorEntry:
    move_id: str
    rule: str
    weight: int = 1
    sequence_index: int | None = None


@dataclass(frozen=True)
class EnemySchema:
    """Authored enemy definition.

    `hp_min` / `hp_max` are inclusive endpoints; the engine rolls HP via
    `rng.stream("enemy_hp").randint(hp_min, hp_max)` at combat start.
    `moves` is the move catalog; `movepicker` is the ordered list of
    selector entries the engine walks each turn to decide which move
    will next telegraph. See MOVE_RULE_DOCS for how each rule filters
    the candidate pool.
    """

    enemy_id: str
    name: str
    hp_min: int
    hp_max: int
    moves: dict[str, MoveSchema]
    movepicker: tuple[SelectorEntry, ...]


@dataclass(frozen=True)
class PowerSchema:
    """Power metadata. Behavior lives in the engine, not here.

    `kind` labels buff vs debuff for UI/intent coloring; does not change
    apply logic (apply_power is uniform — amount adds to stacks).
    `duration` tells the engine the decay rule:
        permanent         : never decays (strength, dexterity).
        turns             : decrement by 1 at end of owner's turn, remove
                            at 0 (vulnerable, weak, frail).
        end_of_turn_tick  : no decay; triggers a power-specific hook at
                            each end-of-owner-turn (ritual).
    `stacks` is reserved for later phases where some powers overwrite
    rather than sum (e.g. Barricade-like boolean powers). All Phase 1
    powers stack additively.
    """

    power_id: str
    name: str
    kind: str
    duration: str
    stacks: bool = True


__all__ = [
    "CARD_ONLY_VERBS",
    "CARD_PILES",
    "CARD_RARITIES",
    "CARD_TARGETS",
    "CARD_TYPES",
    "CardSchema",
    "EFFECT_VERBS",
    "ENEMY_ONLY_VERBS",
    "EffectStep",
    "EnemySchema",
    "INTENTS",
    "MOVE_RULES",
    "MOVE_RULE_DOCS",
    "MoveSchema",
    "POWER_DURATIONS",
    "POWER_KINDS",
    "PowerSchema",
    "SelectorEntry",
    "TARGET_SCOPES",
    "VERB_ALLOWED_SCOPES",
    "VERB_DOCS",
]
