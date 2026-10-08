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
Infested, Surprise, Steam Eruption, Reattach, Adaptable, Stock, Crab Rage,
Sandpit, Fabricate ...) live here as power hooks and `SPECIALS`. Sources:
v0.107.1 monster data (names, numbers, intents) and r33hab/sts2
`EnemyAI.cs` / `CombatEngine.cs` (timings and hidden effects).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .enums import CombatPhase
from .schemas import AiBranch, AiNode, MonsterSchema
from .state import CardRef, MonsterState, remove_ref

if TYPE_CHECKING:
    from .combat import CombatContext


STUNNED = "__stunned__"
REVIVING = "__reviving__"
MAX_MONSTERS = 6
FABRICATOR_SLOTS = 5
FABRICATOR_OWN_SLOT = 2
DECIMILLIPEDE_REATTACH_HP = 25

# Powers that keep a dead monster's slot alive until answered.
_HOLDS_OPEN = ("steam_eruption", "surprise", "infested")


# ---------------------------------------------------------------------------
# Lifecycle

def enter_combat(c: "CombatContext", m: MonsterState) -> None:
    mdef = c.monster_defs[m.monster_id]
    for pid, amount, amount_asc, level in mdef.innate_powers:
        m.powers[pid] = amount_asc if c.ascension >= level else amount
    m.block = value(c, mdef.starting_block)
    if m.monster_id == "two_tailed_rat":
        m.flags["summon_cooldown"] = 2
    if m.monster_id == "fabricator":
        m.flags.setdefault("fab_slot", FABRICATOR_OWN_SLOT)
    if m.monster_id.startswith("decimillipede_segment"):
        _decimillipede_hp(c, m)


def _decimillipede_hp(c: "CombatContext", m: MonsterState) -> None:
    """CombatFactory.MakeDecimillipedeHpEvenAndUnique."""

    lo, hi = c._hp_band(m.monster_id)
    first_even = lo + (lo % 2)
    hp = m.max_hp + (m.max_hp % 2)
    if hp > hi:
        hp = first_even
    others = {o.max_hp for o in c.combat.monsters if o is not m and o.monster_id.startswith("decimillipede")}
    for _ in range(8):
        if hp not in others:
            break
        hp += 2
        if hp > hi:
            hp = first_even
    m.max_hp = m.hp = hp


def value(c: "CombatContext", raw: Any) -> int:
    """Resolve an ascension-dependent number ([base, asc_value, asc_level])."""

    if isinstance(raw, (list, tuple)):
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

    if is_reviving(m):
        if m.flags.pop("revive_hold", False):
            return
        if m.flags.get("revive_kind") == "illusion":
            m.queued_move = REVIVING
            return
        node = c.monster_defs[m.monster_id].ai_nodes.get(m.ai_state or "")
        if node is not None and node.next is not None:
            _enter(c, m, node.next)
        return
    if not m.alive:
        return
    if m.flags.pop("fresh", False):
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


def _condition(c: "CombatContext", m: MonsterState, cond: str | None, arg: Any) -> bool:
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
    if cond == "flag":
        return bool(m.flags.get(str(arg)))
    if cond == "side_alive_lt":
        # GetTeammatesOf(this).Count(alive): the whole side, this monster included.
        return len(c.alive_monsters()) < int(arg)
    if cond == "monster_alive":
        return any(o.alive and o.monster_id == arg for o in c.combat.monsters)
    if cond == "below_half_once":
        return m.hp < m.max_hp // 2 and str(arg) not in m.performed_once
    if cond == "move_count_lt":
        move, n = str(arg).split(":")
        return m.move_history.count(move) < int(n)
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
        if is_reviving(m):
            _revive_tick(c, m)
            continue
        if not m.alive:
            continue
        sandpit = m.powers.get("sandpit", 0)
        if sandpit > 0:
            m.powers["sandpit"] = sandpit - 1
            if m.powers["sandpit"] <= 0:
                m.powers.pop("sandpit")
                c.player.hp = 0
                c._end_combat("defeat")
                return
        _take_turn(c, m)
        if not c.player.alive:
            c._end_combat("defeat")
            return
        demise = m.powers.get("demise", 0)
        if demise > 0 and m.alive:
            c.lose_hp_monster(m, demise)
            if c._check_end():
                return
    for m in combat.monsters:
        if not m.alive:
            continue
        for pid in ("high_voltage", "territorial"):
            gain = m.powers.get(pid, 0)
            if gain:
                c._change_power(m, "strength", gain, allow_negative=True)


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
        _apply_step(c, m, move_id, step.verb, step.args)
        if not c.player.alive or c.combat.outcome is not None:
            break
    if not m.alive and not is_reviving(m):
        return
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


def attack_hits(c: "CombatContext", m: MonsterState, move_id: str, args: Any) -> int:
    hits = value(c, args.get("hits", 1))
    if args.get("hits_grow"):
        hits += m.move_history.count(move_id)
    return hits


def attack_base(c: "CombatContext", m: MonsterState, args: Any) -> int:
    base = value(c, args["damage"])
    if args.get("plus_dexterity"):
        base += m.powers.get("dexterity", 0)
    return base


def _apply_step(c: "CombatContext", m: MonsterState, move_id: str, verb: str, args: Any) -> None:
    p = c.player
    if verb == "attack":
        c.monster_attack(m, attack_base(c, m, args), attack_hits(c, m, move_id, args))
    elif verb == "block":
        c.monster_gain_block(m, value(c, args["amount"]))
    elif verb == "apply_power":
        amount = value(c, args["amount"])
        target = args["target"]
        pid = args["power"]
        if target == "self":
            c._change_power(m, pid, amount, allow_negative=pid in ("strength", "dexterity"))
        elif target == "player":
            c.apply_power(p, pid, amount, source=m)
        else:
            for o in c.alive_monsters():
                if o is not m or target == "all":
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


def after_powered_hit(c: "CombatContext", m: MonsterState, hp_loss: int) -> None:
    """Reactions to a powered (Attack card) hit on a living monster."""

    if m.powers.get("burrowed", 0) > 0 and m.block <= 0:
        # Burrowed: losing all its Block stuns it and sends it back to the surface.
        m.powers.pop("burrowed")
        m.block = 0
        stun(c, m, then="BITE")
    if hp_loss > 0 and m.powers.get("flutter", 0) > 0:
        c._change_power(m, "flutter", -1)
        if m.powers.get("flutter", 0) <= 0:
            stun(c, m)
    if hp_loss > 0 and m.powers.get("slumber", 0) > 0:
        c._change_power(m, "slumber", -1)
        if m.monster_id == "slumbering_beetle" and m.powers.get("slumber", 0) <= 0:
            stun(c, m, then="ROLL_OUT")
    hive = m.powers.get("personal_hive", 0)
    if hive > 0:
        for _ in range(hive):
            c.add_card_to_pile("dazed", "draw_random")


def stun(c: "CombatContext", m: MonsterState, *, then: str | None = None) -> None:
    """Stun `m` for its next turn; afterwards its machine resumes at `then`."""

    m.stunned = True
    m.queued_move = STUNNED
    if then is not None:
        m.flags["after_stun"] = then


def on_death(c: "CombatContext", m: MonsterState) -> None:
    combat = c.combat
    p = c.player
    if m.monster_id == "shrinker_beetle":
        p.powers.pop("shrink", None)
    if m.monster_id == "slithering_strangler":
        p.powers.pop("constrict", None)
    if m.monster_id == "torch_head_amalgam":
        for q in combat.monsters:
            if q.alive and q.monster_id == "queen" and q.queued_move == "BURN_BRIGHT_FOR_ME":
                q.ai_state = "ENRAGE"
                q.queued_move = "ENRAGE"
    for stat in ("strength", "dexterity"):
        owed = m.flags.pop(f"possessed_{stat}", 0)
        if owed:
            c._change_power(p, stat, owed, allow_negative=True)
    if m.monster_id == "spectral_knight":
        p.powers.pop("hex", None)
    if m.monster_id == "magi_knight" and p.powers.get("dampen", 0) > 0 \
            and not any(o is not m and o.alive and o.monster_id == "magi_knight" for o in combat.monsters):
        c.remove_dampen()
    if not m.escaped:
        combat.stolen = [(t, card) for t, card in combat.stolen if t is not m]
    if m.powers.get("illusion", 0) > 0:
        for pid in [k for k in m.powers if k in ("vulnerable", "weak", "frail", "shrink")]:
            m.powers.pop(pid)
        m.block = 0
        _start_revive(c, m, "illusion", turns=1, move=None)
        m.queued_move = REVIVING
        return
    if m.powers.get("stock", 0) > 0 and m.monster_id == "axebot":
        _respawn_axebot(c, m)
        return
    if m.powers.get("reattach", 0) > 0 and _other_segment_alive(c, m):
        reattach = m.powers["reattach"]
        m.powers = {"reattach": reattach}
        m.block = 0
        player_turn = combat.phase == CombatPhase.PLAYER
        _start_revive(c, m, "reattach", turns=2 if player_turn else 1, move="DEAD" if player_turn else "REATTACH")
        return
    if m.powers.get("adaptable", 0) > 0 and m.monster_id == "test_subject":
        m.block = 0
        _start_revive(c, m, "adaptable", turns=1, move="RESPAWN")
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
        p.gold += heist
    if m.powers.get("steam_eruption", 0) > 0:
        _about_to_blow(c, m)
        return
    if m.powers.get("crab_rage", 0) > 0:
        _crab_rage(c, m)
    for o in combat.monsters:
        if o is m or not o.alive:
            continue
        rav = o.powers.get("ravenous", 0)
        if rav > 0:
            c._change_power(o, "strength", rav, allow_negative=True)
            # A Corpse Slug's eating stun does not advance its cycle: it performs the
            # move it was announcing on the turn after (EnemyAI skips MoveIndex++).
            stun(c, o, then=o.ai_state)


def _other_segment_alive(c: "CombatContext", m: MonsterState) -> bool:
    return any(o is not m and o.powers.get("reattach", 0) > 0 and (o.alive or is_reviving(o))
               for o in c.combat.monsters)


def _start_revive(c: "CombatContext", m: MonsterState, kind: str, *, turns: int, move: str | None) -> None:
    m.flags["reviving"] = turns
    m.flags["revive_kind"] = kind
    m.stunned = False
    m.flags.pop("after_stun", None)
    if c.combat.phase == CombatPhase.ENEMY:
        # Its next player-turn intent pick must not advance past the revive move.
        m.flags["revive_hold"] = True
    if move is not None:
        m.ai_state = move
        m.queued_move = move


def _revive_tick(c: "CombatContext", m: MonsterState) -> None:
    m.flags["reviving"] -= 1
    if m.queued_move not in (None, REVIVING, STUNNED):
        m.move_history.append(m.queued_move)
        m.performed_once.add(m.queued_move)
    if m.flags["reviving"] > 0:
        return
    m.flags.pop("reviving")
    m.flags.pop("revive_hold", None)
    kind = m.flags.pop("revive_kind")
    if kind == "illusion":
        m.hp = m.max_hp
        m.queued_move = None
    elif kind == "reattach":
        m.hp = min(m.max_hp, m.powers.get("reattach", DECIMILLIPEDE_REATTACH_HP))
    elif kind == "adaptable":
        if m.powers.get("painful_stabs", 0) <= 0:
            hp = value(c, [200, 212, 8])
            m.powers["painful_stabs"] = 1
        else:
            hp = value(c, [300, 313, 8])
            m.powers.pop("adaptable", None)
            m.powers.pop("painful_stabs", None)
            m.powers["nemesis"] = 1
        m.hp = m.max_hp = hp
        m.block = 0


def _respawn_axebot(c: "CombatContext", m: MonsterState) -> None:
    """Stock: a fresh Axebot takes the slot and opens on Boot Up."""

    stock = m.powers.get("stock", 0)
    m.hp = m.max_hp = c.roll_monster_hp(m.monster_id)
    m.block = 0
    m.powers = {"stock": stock - 1} if stock > 1 else {}
    m.move_history.clear()
    m.performed_once.clear()
    m.stunned = False
    m.flags.pop("after_stun", None)
    m.flags.pop("temporary_strength_loss", None)
    _enter(c, m, "BOOT_UP")


def _crab_rage(c: "CombatContext", dead: MonsterState) -> None:
    for o in c.combat.monsters:
        if o is dead or not o.alive or o.powers.get("crab_rage", 0) <= 0:
            continue
        c._change_power(o, "strength", 6, allow_negative=True)
        o.block += 99
        o.powers.pop("crab_rage")
    # Turn to face the survivor if it now stands behind the player.
    living = c.alive_monsters()
    if living and c.attacks_from_behind(living[0]):
        c.face_toward(living[0])


def _about_to_blow(c: "CombatContext", m: MonsterState) -> None:
    eruption = m.powers.get("steam_eruption", 0)
    m.powers = {"steam_eruption": eruption}
    m.max_hp = m.hp = 999_999_999
    m.flags["exploding"] = True
    m.flags.pop("after_stun", None)
    m.stunned = False
    m.ai_state = "ABOUT_TO_BLOW"
    m.queued_move = "ABOUT_TO_BLOW"


def is_reviving(m: MonsterState) -> bool:
    return bool(m.flags.get("reviving"))


def keeps_corpse(m: MonsterState) -> bool:
    return is_reviving(m) or any(m.powers.get(p, 0) > 0 for p in _HOLDS_OPEN)


def any_primary_alive(c: "CombatContext") -> bool:
    for m in c.combat.monsters:
        if keeps_corpse(m) and not m.alive and not m.escaped:
            if m.powers.get("minion", 0) > 0 and m.flags.get("revive_kind") == "illusion":
                continue
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


def _special_dizzy(c: "CombatContext", m: MonsterState, args: Any) -> None:
    del c, args
    m.flags.pop("off_balance", None)


_STEAL_TIERS = (
    lambda r: r == "uncommon",
    lambda r: r in ("common", "rare", "event"),
    lambda r: r in ("basic", "quest"),
    lambda r: True,
)


def _special_steal_card(c: "CombatContext", m: MonsterState, args: Any) -> None:
    """EnemyAI.StealDrawOrDiscardCard: an Uncommon if it can, else Common/Rare, else Basic."""

    del args
    p = c.player
    cands = [(p.draw_pile, i) for i in range(len(p.draw_pile))] + \
            [(p.discard_pile, i) for i in range(len(p.discard_pile))]
    if not cands:
        return
    for tier in _STEAL_TIERS:
        matched = [x for x in cands if tier(c.cards[x[0][x[1]]].rarity)]
        if matched:
            cands = matched
            break
    pile, i = c.rng.stream("card_generation").choice(cands)
    card = pile[i]
    remove_ref(pile, card)
    c.combat.stolen.append((m, str(card)))
    m.powers["swipe"] = 1


def _special_pheromone_spit(c: "CombatContext", m: MonsterState, args: Any) -> None:
    del args
    if m.powers.get("personal_hive", 0) < 3:
        c._change_power(m, "personal_hive", 1)
        c._change_power(m, "strength", 1, allow_negative=True)
    else:
        c._change_power(m, "strength", 2, allow_negative=True)


def _special_lay_eggs(c: "CombatContext", m: MonsterState, args: Any) -> None:
    """Ovicopter: up to three Tough Eggs, left of it, reusing dead eggs' places."""

    del args
    combat = c.combat
    count = min(3, MAX_MONSTERS - len(c.alive_monsters()))
    for _ in range(count):
        corpse = next((o for o in combat.monsters if o.monster_id == "tough_egg" and not o.alive
                       and not keeps_corpse(o)), None)
        if corpse is not None:
            idx = combat.monsters.index(corpse)
            combat.monsters.pop(idx)
        else:
            idx = next((i for i, o in enumerate(combat.monsters) if o.monster_id == "tough_egg"),
                       combat.monsters.index(m))
        c.spawn_monster("tough_egg", index=idx)


def _special_hatch(c: "CombatContext", m: MonsterState, args: Any) -> None:
    lo, hi = value(c, args["lo"]), value(c, args["hi"])
    m.hp = m.max_hp = c.rng.stream("enemy_hp").randint(lo, hi)
    m.powers = {k: v for k, v in m.powers.items() if k == "minion"}


def _special_fabricate(c: "CombatContext", m: MonsterState, args: Any) -> None:
    """Fabricator: a Guardbot/Noisebot (Fabricate only) and a Zapbot/Stabbot."""

    stream = c.rng.stream("monster_moves")

    def pick(first: str, second: str) -> str:
        last = m.flags.get("last_bot")
        pool = [second] if last == first else [first] if last == second else [first, second]
        chosen = stream.choice(pool)
        m.flags["last_bot"] = chosen
        return chosen

    if args.get("defensive") and len(c.alive_monsters()) < MAX_MONSTERS:
        _insert_bot(c, pick("guardbot", "noisebot"))
    if len(c.alive_monsters()) < MAX_MONSTERS:
        _insert_bot(c, pick("zapbot", "stabbot"))


def _insert_bot(c: "CombatContext", bot_id: str) -> None:
    combat = c.combat
    held = {o.flags.get("fab_slot") for o in combat.monsters if o.alive}
    slot = next((s for s in range(FABRICATOR_SLOTS) if s not in held), None)
    if slot is None:
        return
    index = next((i for i, o in enumerate(combat.monsters) if o.flags.get("fab_slot", -1) > slot), None)
    c.spawn_monster(bot_id, index=index, flags={"fab_slot": slot})


def _special_guard(c: "CombatContext", m: MonsterState, args: Any) -> None:
    amount = value(c, args["amount"])
    for o in c.combat.monsters:
        if o.alive and o.monster_id == "fabricator":
            c.monster_gain_block(o, amount)


def _special_boot_up(c: "CombatContext", m: MonsterState, args: Any) -> None:
    c.monster_gain_block(m, value(c, args["block"]))
    gain = max(0, 2 - m.powers.get("stock", 0)) * value(c, args["strength"])
    if gain:
        c._change_power(m, "strength", gain, allow_negative=True)


def _special_possess(c: "CombatContext", m: MonsterState, args: Any) -> None:
    """The Lost / The Forgotten take a stat off the player and keep a tally."""

    stat = args["stat"]
    amount = value(c, args["amount"])
    c._change_power(c.player, stat, -amount, allow_negative=True)
    c._change_power(m, stat, amount, allow_negative=True)
    m.flags[f"possessed_{stat}"] = m.flags.get(f"possessed_{stat}", 0) + amount


def _special_increasing_intensity(c: "CombatContext", m: MonsterState, args: Any) -> None:
    earlier = m.move_history.count("INCREASING_INTENSITY")
    m.flags["wither_intensity"] = m.flags.get("wither_intensity", 0) + 1
    for _ in range(value(c, args["withers"])):
        c.add_card_to_pile("wither", "discard")
    c._change_power(m, "strength", value(c, args["strength"]) + earlier, allow_negative=True)


def _special_liquify_ground(c: "CombatContext", m: MonsterState, args: Any) -> None:
    del args
    c._change_power(m, "sandpit", 4)
    for _ in range(3):
        c.add_card_to_pile("frantic_escape", "draw_random")
    discard = c.player.discard_pile
    stream = c.rng.stream("combat_shuffle")
    for _ in range(3):
        discard.insert(stream.randint(0, len(discard)), CardRef("frantic_escape"))


def _special_curse_of_knowledge(c: "CombatContext", m: MonsterState, args: Any) -> None:
    del args
    c.offer_curse_of_knowledge(m.move_history.count("CURSE_OF_KNOWLEDGE"))


SPECIALS: dict[str, Callable[["CombatContext", MonsterState, Any], None]] = {
    "explode": _special_explode,
    "spike_spit": _special_spike_spit,
    "soul_siphon": _special_soul_siphon,
    "pressure_gun": _special_pressure_gun,
    "rat_backup": _special_rat_backup,
    "beckon": _special_beckon,
    "dizzy": _special_dizzy,
    "steal_card": _special_steal_card,
    "pheromone_spit": _special_pheromone_spit,
    "lay_eggs": _special_lay_eggs,
    "hatch": _special_hatch,
    "fabricate": _special_fabricate,
    "guard": _special_guard,
    "boot_up": _special_boot_up,
    "possess": _special_possess,
    "increasing_intensity": _special_increasing_intensity,
    "liquify_ground": _special_liquify_ground,
    "curse_of_knowledge": _special_curse_of_knowledge,
}


__all__ = [
    "REVIVING",
    "SPECIALS",
    "STUNNED",
    "after_powered_hit",
    "any_primary_alive",
    "attack_base",
    "attack_hits",
    "choose_first_move",
    "choose_next_move",
    "enter_combat",
    "is_reviving",
    "keeps_corpse",
    "on_death",
    "run_enemy_turn",
    "stun",
    "value",
]
