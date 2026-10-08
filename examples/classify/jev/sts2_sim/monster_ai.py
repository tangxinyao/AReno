"""Monster behavior: move state machines, move execution, monster powers.

Each monster's moves and AI come from data/monsters/*.json. The AI mirrors
the game's MonsterMoveStateMachine:

  * a `move` node performs one move, then goes to `next`;
  * a `random` node rolls one branch, weighting out branches its repeat rule
    forbids (cannot_repeat: not the last move performed; can_repeat_x: not
    more than `max_times` in a row; use_only_once: once per combat;
    `cooldown` k: not among the last k moves). All blocked -> first branch;
  * a `conditional` node takes the first branch whose condition holds.

Monster-specific mechanics that are not plain moves (Plow, Shriek, Illusion,
Infested, Surprise, Steam Eruption, Ravenous, rat backup ...) live here as
power hooks. Sources: v0.107.1 monster data (names, numbers, intents) and
r33hab/sts2 `EnemyAI.cs` / `CombatEngine.cs` (timings and hidden effects).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .schemas import AiBranch, AiNode, MonsterSchema
from .state import MonsterState

if TYPE_CHECKING:
    from .combat import CombatContext


STUNNED = "__stunned__"
REVIVING = "__reviving__"
MAX_MONSTERS = 6

# Powers that keep a dead monster's slot alive until answered.
_HOLDS_OPEN = ("steam_eruption", "surprise", "infested")


# ---------------------------------------------------------------------------
# Lifecycle

def enter_combat(c: "CombatContext", m: MonsterState) -> None:
    mdef = c.monster_defs[m.monster_id]
    for pid, amount, amount_asc, level in mdef.innate_powers:
        m.powers[pid] = amount_asc if c.ascension >= level else amount
    m.block = mdef.starting_block
    if m.monster_id == "two_tailed_rat":
        m.flags["summon_cooldown"] = 2


def value(c: "CombatContext", raw: Any) -> int:
    """Resolve an ascension-dependent number ([base, asc_value, asc_level])."""

    if isinstance(raw, list):
        base, asc_value, level = raw
        return int(asc_value if c.ascension >= level else base)
    return int(raw)


def choose_first_move(c: "CombatContext", m: MonsterState) -> None:
    mdef = c.monster_defs[m.monster_id]
    start = m.flags.pop("start_state", None) or mdef.ai_initial
    if m.stunned:
        # Spawned stunned (Wrigglers, gremlins): the stun is the first turn and the
        # machine starts from its initial state afterwards.
        m.queued_move = STUNNED
        m.ai_state = None
        m.flags["after_stun"] = start
        return
    _enter(c, m, start)


def choose_next_move(c: "CombatContext", m: MonsterState) -> None:
    """Advance the machine after the monster's turn (called at player turn start)."""

    if not m.alive and not _reviving(m):
        return
    if _reviving(m):
        m.queued_move = REVIVING
        return
    if m.stunned:
        m.queued_move = STUNNED
        return
    nxt = m.flags.pop("after_stun", None)
    if nxt is None:
        node = c.monster_defs[m.monster_id].ai_nodes.get(m.ai_state or "")
        nxt = node.next if node is not None else None
        if node is not None and node.must_perform_once and node.move not in m.performed_once:
            nxt = node.node_id
    if nxt is None:
        nxt = m.ai_state
    if nxt is None:
        return
    _enter(c, m, nxt)


def _enter(c: "CombatContext", m: MonsterState, state_id: str) -> None:
    nodes = c.monster_defs[m.monster_id].ai_nodes
    seen = 0
    while True:
        node = nodes[state_id]
        if node.node_type == "move":
            m.ai_state = node.node_id
            m.queued_move = node.move
            return
        seen += 1
        if seen > 32:
            raise RuntimeError(f"{m.monster_id}: AI branch loop at {state_id}")
        if node.node_type == "random":
            state_id = _pick_random(c, m, node)
        else:
            state_id = _pick_conditional(c, m, node)


def _branch_move(c: "CombatContext", m: MonsterState, branch: AiBranch) -> str | None:
    node = c.monster_defs[m.monster_id].ai_nodes[branch.target]
    return node.move


def _pick_random(c: "CombatContext", m: MonsterState, node: AiNode) -> str:
    if node.node_id.startswith("rat_"):
        return _pick_rat_branch(c, m, node)
    hist = m.move_history
    weights: list[float] = []
    for br in node.branches:
        move = _branch_move(c, m, br)
        w = br.weight
        last = hist[-1] if hist else None
        if br.repeat == "cannot_repeat" and move == last:
            w = 0.0
        elif br.repeat == "can_repeat_x" and br.max_times > 0 and len(hist) >= br.max_times \
                and all(h == move for h in hist[-br.max_times:]):
            w = 0.0
        elif br.repeat == "use_only_once" and move in hist:
            w = 0.0
        if br.cooldown > 0 and move in hist[-br.cooldown:]:
            w = 0.0
        weights.append(w)
    return _weighted(c, node, weights)


def _weighted(c: "CombatContext", node: AiNode, weights: list[float]) -> str:
    roll = c.rng.stream("monster_moves").random() * sum(weights)
    if sum(weights) <= 0:
        return node.branches[0].target
    acc = 0.0
    for br, w in zip(node.branches, weights):
        acc += w
        if w > 0 and roll < acc:
            return br.target
    return [br for br, w in zip(node.branches, weights) if w > 0][-1].target


def _pick_conditional(c: "CombatContext", m: MonsterState, node: AiNode) -> str:
    for br in node.branches:
        if _condition(c, m, br.condition, br.condition_arg):
            return br.target
    return node.branches[-1].target


def _same_kind_alive(c: "CombatContext", m: MonsterState) -> list[MonsterState]:
    return [o for o in c.combat.monsters if o.monster_id == m.monster_id and o.alive]


def _condition(c: "CombatContext", m: MonsterState, cond: str | None, arg: int | None) -> bool:
    if cond is None:
        return True
    if cond == "alone":
        return len(_same_kind_alive(c, m)) <= 1
    if cond == "front":
        kin = _same_kind_alive(c, m)
        return bool(kin) and kin[0] is m
    if cond == "not_front":
        kin = _same_kind_alive(c, m)
        return not kin or kin[0] is not m
    if cond == "kind_index":
        return m.kind_index == arg
    if cond == "starter":
        return m.flags.get("starter") == arg
    if cond == "has_power":
        return m.powers.get(str(arg), 0) > 0
    raise RuntimeError(f"unknown AI condition {cond!r}")


def _pick_rat_branch(c: "CombatContext", m: MonsterState, node: AiNode) -> str:
    """Two-Tailed Rat's RAND: backup weighs 0.75 while it can summon, else 0."""

    others_calling = any(o is not m and o.alive and o.monster_id == m.monster_id
                         and o.queued_move == "CALL_FOR_BACKUP" for o in c.combat.monsters)
    can_summon = (m.flags.get("summon_cooldown", 0) <= 0
                  and c.combat.backup_count < 3
                  and len(c.alive_monsters()) < 5
                  and not others_calling
                  and "CALL_FOR_BACKUP" not in m.move_history)
    ordinary = 1.0 / 12.0 if can_summon else 1.0
    hist = m.move_history
    last = hist[-1] if hist else None
    weights = []
    for br in node.branches:
        move = _branch_move(c, m, br)
        if move == "CALL_FOR_BACKUP":
            weights.append(0.75 if can_summon else 0.0)
            continue
        w = ordinary
        if move == last or (br.cooldown > 0 and move in hist[-br.cooldown:]):
            w = 0.0
        weights.append(w)
    return _weighted(c, node, weights)


# ---------------------------------------------------------------------------
# Enemy turn

def run_enemy_turn(c: "CombatContext") -> None:
    combat = c.combat
    for m in combat.monsters:
        m.flags.pop("skittish_spent", None)
        if m.alive and m.powers.get("burrowed", 0) <= 0:
            m.block = 0
    for m in list(combat.monsters):
        if combat.outcome is not None:
            return
        if _reviving(m):
            _finish_revive(c, m)
            continue
        if not m.alive:
            continue
        _take_turn(c, m)
        if not c.player.alive:
            c._end_combat("defeat")
            return
    for m in combat.monsters:
        if not m.alive:
            continue
        terr = m.powers.get("territorial", 0)
        if terr:
            c._change_power(m, "strength", terr, allow_negative=True)


def _take_turn(c: "CombatContext", m: MonsterState) -> None:
    mdef = c.monster_defs[m.monster_id]
    if m.stunned or m.queued_move == STUNNED:
        m.stunned = False
        _plating_and_sleep(c, m, mdef)
        _restore_temp_strength(m)
        if "after_stun" not in m.flags:
            node = mdef.ai_nodes.get(m.ai_state or "")
            if node is not None and node.next is not None:
                m.flags["after_stun"] = node.next
        return
    move_id = m.queued_move
    if move_id is None:
        return
    move = mdef.moves[move_id]
    vigor = m.powers.get("vigor", 0) if any(s.verb == "attack" for s in move.effects) else 0
    is_buff_only = "buff" in move.intents and "attack" not in move.intents
    for step in move.effects:
        _apply_step(c, m, step.verb, step.args)
        if not c.player.alive or c.combat.outcome is not None:
            break
    m.move_history.append(move_id)
    m.performed_once.add(move_id)
    if m.monster_id == "two_tailed_rat" and move_id != "CALL_FOR_BACKUP":
        if m.flags.get("summon_cooldown", 0) > 0:
            m.flags["summon_cooldown"] -= 1
    _plating_and_sleep(c, m, mdef)
    ritual = m.powers.get("ritual", 0)
    if ritual > 0 and not (is_buff_only and "ritual" in _applied_self_powers(move)):
        c._change_power(m, "strength", ritual, allow_negative=True)
    if vigor > 0:
        c._change_power(m, "vigor", -vigor)
    _restore_temp_strength(m)


def _applied_self_powers(move) -> set[str]:
    return {str(s.args.get("power")) for s in move.effects
            if s.verb == "apply_power" and s.args.get("target") == "self"}


def _plating_and_sleep(c: "CombatContext", m: MonsterState, mdef: MonsterSchema) -> None:
    plating = m.powers.get("plating", 0)
    if plating > 0:
        if c.combat.turn > 1:
            c._change_power(m, "plating", -1)
        m.block += m.powers.get("plating", 0)
    asleep = m.powers.get("asleep", 0)
    if asleep > 0:
        if asleep <= 1:
            m.powers.pop("plating", None)
        c._change_power(m, "asleep", -1)


def _restore_temp_strength(m: MonsterState) -> None:
    loss = m.flags.pop("temporary_strength_loss", 0)
    if loss:
        m.powers["strength"] = m.powers.get("strength", 0) + loss
        if m.powers["strength"] == 0:
            m.powers.pop("strength")


def _apply_step(c: "CombatContext", m: MonsterState, verb: str, args: Any) -> None:
    p = c.player
    if verb == "attack":
        c.monster_attack(m, value(c, args["damage"]), value(c, args.get("hits", 1)))
    elif verb == "block":
        m.block += value(c, args["amount"])
    elif verb == "apply_power":
        amount = value(c, args["amount"])
        target = args["target"]
        pid = args["power"]
        if target == "self":
            c._change_power(m, pid, amount, allow_negative=pid in ("strength", "dexterity"))
        elif target == "player":
            if pid in ("strength", "dexterity") and amount < 0:
                c.apply_power(p, pid, amount)
            else:
                c.apply_power(p, pid, amount)
        else:
            for o in c.alive_monsters():
                if o is not m:
                    c._change_power(o, pid, amount, allow_negative=pid == "strength")
    elif verb == "add_card":
        for _ in range(value(c, args["count"])):
            c.add_card_to_pile(args["card"], args["pile"])
    elif verb == "heal":
        m.hp = min(m.max_hp, m.hp + value(c, args["amount"]))
    elif verb == "summon":
        _summon(c, m, args)
    elif verb == "steal_gold":
        amount = min(value(c, args["amount"]), p.gold)
        if amount > 0 and p.alive:
            p.gold -= amount
            m.stolen_gold += amount
    elif verb == "escape":
        m.escaped = True
    elif verb == "suicide":
        m.hp = 0
        on_death(c, m)
    elif verb == "special":
        SPECIALS[args["name"]](c, m, args)
    else:
        raise RuntimeError(f"unknown monster verb {verb!r}")


def _summon(c: "CombatContext", m: MonsterState, args: Any) -> None:
    mid = args["monster"]
    max_alive = args.get("max_alive")
    for _ in range(int(args.get("count", 1))):
        if len(c.alive_monsters()) >= MAX_MONSTERS:
            return
        if max_alive is not None and sum(1 for o in c.alive_monsters() if o.monster_id == mid) >= max_alive:
            return
        where = args.get("where", "end")
        idx = c.combat.monsters.index(m)
        index = idx if where == "before" else idx + 1 if where == "after" else None
        c.spawn_monster(mid, index=index, stunned=bool(args.get("stunned", False)))


# ---------------------------------------------------------------------------
# Damage / death reactions

def after_hp_lost(c: "CombatContext", m: MonsterState, hp_loss: int) -> None:
    del hp_loss
    shriek = m.powers.get("shriek", 0)
    if shriek > 0 and m.hp <= shriek:
        m.powers.pop("shriek")
        stun(c, m, then="TERROR")
    plow = m.powers.get("plow", 0)
    if plow > 0 and m.hp <= plow:
        m.powers.pop("plow")
        m.powers.pop("strength", None)
        m.flags.pop("temporary_strength_loss", None)
        stun(c, m, then="BEAST_CRY")
    if m.powers.get("asleep", 0) > 0:
        m.powers.pop("asleep")
        m.powers.pop("plating", None)
        stun(c, m, then="SLASH")


def stun(c: "CombatContext", m: MonsterState, *, then: str | None = None) -> None:
    """Stun `m` for its next turn; afterwards its machine resumes at `then`."""

    m.stunned = True
    m.queued_move = STUNNED
    if then is not None:
        m.flags["after_stun"] = then


def on_death(c: "CombatContext", m: MonsterState) -> None:
    combat = c.combat
    if m.monster_id == "shrinker_beetle":
        c.player.powers.pop("shrink", None)
    if m.monster_id == "slithering_strangler":
        c.player.powers.pop("constrict", None)
    if m.powers.get("illusion", 0) > 0:
        for pid in [k for k in m.powers if k in ("vulnerable", "weak", "frail", "shrink")]:
            m.powers.pop(pid)
        m.block = 0
        m.flags["reviving"] = True
        m.queued_move = REVIVING
        return
    if m.powers.pop("surprise", 0) > 0:
        sneaky = c.spawn_monster("sneaky_gremlin", stunned=True)
        del sneaky
        fat = c.spawn_monster("fat_gremlin", stunned=True)
        fat.flags["heist_gold"] = m.stolen_gold
    if m.powers.pop("infested", 0) > 0:
        idx = combat.monsters.index(m) + 1
        for i in range(4):
            if len(c.alive_monsters()) >= MAX_MONSTERS:
                break
            w = c.spawn_monster("wriggler", index=idx + i, stunned=True)
            w.kind_index = i
    heist = m.flags.pop("heist_gold", 0)
    if heist and not m.escaped:
        c.player.gold += heist
    if m.powers.get("steam_eruption", 0) > 0:
        _about_to_blow(c, m)
        return
    for o in combat.monsters:
        if o is m or not o.alive:
            continue
        rav = o.powers.get("ravenous", 0)
        if rav > 0:
            c._change_power(o, "strength", rav, allow_negative=True)
            # A Corpse Slug's eating stun does not advance its cycle: it performs the
            # move it was announcing on the turn after (EnemyAI skips MoveIndex++).
            stun(c, o, then=o.ai_state)


def _about_to_blow(c: "CombatContext", m: MonsterState) -> None:
    eruption = m.powers.get("steam_eruption", 0)
    m.powers = {"steam_eruption": eruption}
    m.max_hp = m.hp = 999_999_999
    m.flags["exploding"] = True
    m.flags.pop("after_stun", None)
    m.stunned = False
    m.ai_state = "ABOUT_TO_BLOW"
    m.queued_move = "ABOUT_TO_BLOW"


def _reviving(m: MonsterState) -> bool:
    return bool(m.flags.get("reviving"))


def _finish_revive(c: "CombatContext", m: MonsterState) -> None:
    m.flags.pop("reviving", None)
    m.hp = m.max_hp
    m.queued_move = None


def keeps_corpse(m: MonsterState) -> bool:
    return _reviving(m) or any(m.powers.get(p, 0) > 0 for p in _HOLDS_OPEN)


def any_primary_alive(c: "CombatContext") -> bool:
    for m in c.combat.monsters:
        if keeps_corpse(m) and not m.alive and not m.escaped:
            return True
        if not m.alive:
            continue
        if m.powers.get("minion", 0) > 0 or m.powers.get("illusion", 0) > 0:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Specials

def _special_explode(c: "CombatContext", m: MonsterState, args: Any) -> None:
    del args
    amount = m.powers.pop("steam_eruption", 0)
    c.monster_attack(m, amount, 1)
    m.hp = 0
    m.flags.pop("exploding", None)


def _special_spike_spit(c: "CombatContext", m: MonsterState, args: Any) -> None:
    del args
    c._change_power(m, "thorns", -2)


def _special_soul_siphon(c: "CombatContext", m: MonsterState, args: Any) -> None:
    amount = value(c, args.get("amount", 2))
    c.apply_power(c.player, "strength", -amount)
    c.apply_power(c.player, "dexterity", -amount)
    c._change_power(m, "strength", amount, allow_negative=True)


def _special_pressure_gun(c: "CombatContext", m: MonsterState, args: Any) -> None:
    base = value(c, args["amount"])
    fired = m.flags.get("pressure_gun_fired", 0)
    c.monster_attack(m, base + 5 * fired, 1)
    m.flags["pressure_gun_fired"] = fired + 1


def _special_rat_backup(c: "CombatContext", m: MonsterState, args: Any) -> None:
    """CallForBackup: a new rat takes the last free of the encounter's 5 slots
    (the starters hold slots 2-4), and the roster stays in slot order."""

    del args
    rats = [o for o in c.combat.monsters if o.monster_id == m.monster_id and o.alive]
    held = {o.flags.get("rat_slot") for o in rats}
    free = [s for s in range(5) if s not in held]
    if not free:
        return
    slot = free[-1]
    index = next((i for i, o in enumerate(c.combat.monsters)
                  if o.monster_id == m.monster_id and o.flags.get("rat_slot", -1) > slot), None)
    rat = c.spawn_monster("two_tailed_rat", index=index)
    rat.flags["rat_slot"] = slot
    c.combat.backup_count += 1


def _special_beckon(c: "CombatContext", m: MonsterState, args: Any) -> None:
    count = value(c, args.get("amount", 2))
    c.add_card_to_pile("beckon", "draw_random")
    for _ in range(count - 1):
        c.add_card_to_pile("beckon", "discard")


SPECIALS: dict[str, Callable[["CombatContext", MonsterState, Any], None]] = {
    "explode": _special_explode,
    "spike_spit": _special_spike_spit,
    "soul_siphon": _special_soul_siphon,
    "pressure_gun": _special_pressure_gun,
    "rat_backup": _special_rat_backup,
    "beckon": _special_beckon,
}


__all__ = [
    "REVIVING",
    "SPECIALS",
    "STUNNED",
    "any_primary_alive",
    "choose_first_move",
    "choose_next_move",
    "enter_combat",
    "keeps_corpse",
    "on_death",
    "run_enemy_turn",
    "stun",
    "value",
]
