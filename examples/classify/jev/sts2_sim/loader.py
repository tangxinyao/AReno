"""Strict loaders for the sim's game data.

Each `load_*` reads JSON under `data/`, validates every field against the
vocabulary in schemas.py, and returns frozen dataclasses. Violations raise
`SimDataError` with a path such as `cards:bash.vars.damage` or
`monsters:nibbit.moves.SLICE.effects[1]` so authors can find the problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .schemas import (
    AI_CONDITIONS,
    AiBranch,
    AiNode,
    CARD_COLORS,
    CARD_KEYWORDS,
    CARD_PILES,
    CARD_RARITIES,
    CARD_TARGETS,
    CARD_TYPES,
    CardSchema,
    EffectStep,
    EncounterSchema,
    INTENTS,
    MONSTER_KINDS,
    MONSTER_VERBS,
    MonsterSchema,
    MoveSchema,
    POWER_KINDS,
    POWER_STACKS,
    POWER_TARGETS,
    PowerSchema,
    REPEAT_RULES,
)


DATA_ROOT = Path(__file__).resolve().parent / "data"
CARD_FILES = ("ironclad", "status", "curse", "token")
MONSTER_FILES = ("overgrowth", "underdocks")
ENCOUNTER_POOLS = frozenset({"weak", "normal", "elite", "boss"})
ACTS = frozenset({"overgrowth", "underdocks"})
_SPECIALS = frozenset({"explode", "spike_spit", "soul_siphon", "pressure_gun", "rat_backup", "beckon"})
_GENERATORS = frozenset({
    "slimes_weak", "slimes_normal", "flyconid_normal", "slithering_strangler", "ruby_raiders",
    "corpse_slugs_2", "corpse_slugs_3", "two_tailed_rats",
})


class SimDataError(ValueError):
    """Raised when authored data violates the schemas."""


# ---------------------------------------------------------------------------
# Public entry points

def load_powers(path: Path | None = None) -> dict[str, PowerSchema]:
    raw = _read_json(path or DATA_ROOT / "powers.json")
    _require_list(raw, "powers")
    out: dict[str, PowerSchema] = {}
    for i, e in enumerate(raw):
        where = f"powers[{i}]"
        _require_type(e, dict, where)
        p = PowerSchema(
            power_id=_get_str(e, "power_id", where),
            name=_get_str(e, "name", where),
            kind=_get_in(e, "kind", POWER_KINDS, where),
            stack=_get_in(e, "stack", POWER_STACKS, where),
            description=_get_str(e, "description", where),
        )
        _require_unique(out, p.power_id, where="powers")
        out[p.power_id] = p
    return out


def load_cards(paths: list[Path] | None = None) -> dict[str, CardSchema]:
    paths = paths or [DATA_ROOT / "cards" / f"{name}.json" for name in CARD_FILES]
    out: dict[str, CardSchema] = {}
    for path in paths:
        raw = _read_json(path)
        _require_list(raw, f"cards({path.name})")
        for i, entry in enumerate(raw):
            card = _parse_card(entry, where=f"cards({path.name})[{i}]")
            _require_unique(out, card.card_id, where="cards")
            out[card.card_id] = card
    _validate_upgrade_links(out)
    return out


def load_monsters(
    paths: list[Path] | None = None,
    *,
    power_ids: set[str] | None = None,
    card_ids: set[str] | None = None,
) -> dict[str, MonsterSchema]:
    paths = paths or [DATA_ROOT / "monsters" / f"{name}.json" for name in MONSTER_FILES]
    out: dict[str, MonsterSchema] = {}
    for path in paths:
        raw = _read_json(path)
        _require_list(raw, f"monsters({path.name})")
        for i, entry in enumerate(raw):
            m = _parse_monster(entry, where=f"monsters({path.name})[{i}]", power_ids=power_ids, card_ids=card_ids)
            _require_unique(out, m.monster_id, where="monsters")
            out[m.monster_id] = m
    for mid, m in out.items():
        for move in m.moves.values():
            for j, step in enumerate(move.effects):
                if step.verb == "summon" and step.args["monster"] not in out:
                    raise SimDataError(f"monsters:{mid}.moves.{move.move_id}.effects[{j}] summons unknown "
                                       f"monster {step.args['monster']!r}")
    return out


def load_encounters(
    path: Path | None = None,
    *,
    monster_ids: set[str] | None = None,
) -> dict[str, EncounterSchema]:
    raw = _read_json(path or DATA_ROOT / "encounters.json")
    _require_list(raw, "encounters")
    out: dict[str, EncounterSchema] = {}
    for i, e in enumerate(raw):
        where = f"encounters[{i}]"
        _require_type(e, dict, where)
        eid = _get_str(e, "encounter_id", where)
        monsters = tuple(e.get("monsters", ()))
        gen = e.get("generator")
        if bool(monsters) == bool(gen):
            raise SimDataError(f"{where}: exactly one of monsters / generator is required")
        if gen is not None and gen not in _GENERATORS:
            raise SimDataError(f"{where}.generator {gen!r} unknown")
        if monster_ids is not None:
            for mid in monsters:
                if mid not in monster_ids:
                    raise SimDataError(f"{where}.monsters references unknown monster {mid!r}")
        enc = EncounterSchema(
            encounter_id=eid,
            name=_get_str(e, "name", where),
            act=_get_in(e, "act", ACTS, where),
            pool=_get_in(e, "pool", ENCOUNTER_POOLS, where),
            room_type=_get_str(e, "room_type", where),
            monsters=monsters,
            generator=gen,
        )
        _require_unique(out, eid, where="encounters")
        out[eid] = enc
    return out


def load_all() -> tuple[dict[str, PowerSchema], dict[str, CardSchema], dict[str, MonsterSchema]]:
    powers = load_powers()
    cards = load_cards()
    monsters = load_monsters(power_ids=set(powers), card_ids=set(cards))
    return powers, cards, monsters


# ---------------------------------------------------------------------------
# Parsers

def _parse_card(entry: dict, *, where: str) -> CardSchema:
    _require_type(entry, dict, where)
    cid = _get_str(entry, "card_id", where)
    where = f"cards:{cid}"
    cost = entry.get("cost")
    if not isinstance(cost, int) or cost < 0:
        raise SimDataError(f"{where}.cost must be a non-negative int, got {cost!r}")
    keywords = entry.get("keywords", [])
    _require_list(keywords, f"{where}.keywords")
    for kw in keywords:
        if kw not in CARD_KEYWORDS:
            raise SimDataError(f"{where}.keywords {kw!r} not in {sorted(CARD_KEYWORDS)}")
    raw_vars = entry.get("vars", {})
    _require_type(raw_vars, dict, f"{where}.vars")
    for k, v in raw_vars.items():
        if not isinstance(v, int):
            raise SimDataError(f"{where}.vars.{k} must be int, got {v!r}")
    description = entry.get("description")
    if not isinstance(description, str):
        raise SimDataError(f"{where}.description must be a string")
    return CardSchema(
        card_id=cid,
        game_id=_get_str(entry, "game_id", where),
        name=_get_str(entry, "name", where),
        color=_get_in(entry, "color", CARD_COLORS, where),
        card_type=_get_in(entry, "card_type", CARD_TYPES, where),
        rarity=_get_in(entry, "rarity", CARD_RARITIES, where),
        target=_get_in(entry, "target", CARD_TARGETS, where),
        cost=cost,
        description=description,
        vars=MappingProxyType(dict(raw_vars)),
        keywords=frozenset(keywords),
        tags=frozenset(entry.get("tags", [])),
        x_cost=bool(entry.get("x_cost", False)),
        unplayable=bool(entry.get("unplayable", False)),
        multiplayer_only=bool(entry.get("multiplayer_only", False)),
        generated_in_combat=bool(entry.get("generated_in_combat", True)),
        upgrade_of=entry.get("upgrade_of"),
        upgraded_from=entry.get("upgraded_from"),
    )


def _parse_monster(entry: dict, *, where: str, power_ids: set[str] | None,
                   card_ids: set[str] | None) -> MonsterSchema:
    _require_type(entry, dict, where)
    mid = _get_str(entry, "monster_id", where)
    where = f"monsters:{mid}"
    hp = _pair(entry.get("hp"), f"{where}.hp")
    hp_asc = _pair(entry.get("hp_asc"), f"{where}.hp_asc")
    raw_moves = entry.get("moves")
    if not isinstance(raw_moves, dict) or not raw_moves:
        raise SimDataError(f"{where}.moves must be a non-empty object")
    moves: dict[str, MoveSchema] = {}
    for move_id, m in raw_moves.items():
        mw = f"{where}.moves.{move_id}"
        _require_type(m, dict, mw)
        intents = m.get("intents")
        _require_list(intents, f"{mw}.intents")
        for it in intents:
            if it not in INTENTS:
                raise SimDataError(f"{mw}.intents {it!r} not in {sorted(INTENTS)}")
        steps = m.get("effects", [])
        _require_list(steps, f"{mw}.effects")
        moves[move_id] = MoveSchema(
            move_id=move_id,
            name=_get_str(m, "name", mw),
            intents=tuple(intents),
            effects=tuple(_parse_step(s, f"{mw}.effects[{j}]", power_ids, card_ids) for j, s in enumerate(steps)),
        )
    ai = entry.get("ai")
    _require_type(ai, dict, f"{where}.ai")
    raw_nodes = ai.get("nodes")
    _require_type(raw_nodes, dict, f"{where}.ai.nodes")
    nodes: dict[str, AiNode] = {}
    for nid, n in raw_nodes.items():
        nodes[nid] = _parse_node(nid, n, f"{where}.ai.nodes.{nid}", moves)
    initial = _get_str(ai, "initial", f"{where}.ai")
    for nid, n in nodes.items():
        refs = [n.next] if n.next else []
        refs += [b.target for b in n.branches]
        for r in refs + [initial]:
            if r not in nodes:
                raise SimDataError(f"{where}.ai.nodes.{nid} references unknown state {r!r}")
    innate: list[tuple[str, int, int, int]] = []
    for j, p in enumerate(entry.get("innate_powers", [])):
        pw = f"{where}.innate_powers[{j}]"
        pid = _get_str(p, "power", pw)
        if power_ids is not None and pid not in power_ids:
            raise SimDataError(f"{pw}.power unknown {pid!r}")
        base, asc, lvl = _scaled(p.get("amount"), f"{pw}.amount")
        innate.append((pid, base, asc, lvl))
    return MonsterSchema(
        monster_id=mid,
        game_id=_get_str(entry, "game_id", where),
        name=_get_str(entry, "name", where),
        kind=_get_in(entry, "kind", MONSTER_KINDS, where),
        hp=hp,
        hp_asc=hp_asc,
        moves=MappingProxyType(moves),
        ai_nodes=MappingProxyType(nodes),
        ai_initial=initial,
        innate_powers=tuple(innate),
        starting_block=int(entry.get("starting_block", 0)),
        notes=str(entry.get("notes", "")),
    )


def _parse_node(nid: str, n: Any, where: str, moves: dict[str, MoveSchema]) -> AiNode:
    _require_type(n, dict, where)
    ntype = n.get("type")
    if ntype == "move":
        move = _get_str(n, "move", where)
        if move not in moves:
            raise SimDataError(f"{where}.move references unknown move {move!r}")
        return AiNode(node_id=nid, node_type="move", move=move, next=n.get("next"),
                      must_perform_once=bool(n.get("must_perform_once", False)))
    if ntype not in ("random", "conditional"):
        raise SimDataError(f"{where}.type {ntype!r} invalid")
    branches = []
    raw = n.get("branches")
    _require_list(raw, f"{where}.branches")
    if not raw:
        raise SimDataError(f"{where}.branches must not be empty")
    for j, b in enumerate(raw):
        bw = f"{where}.branches[{j}]"
        repeat = b.get("repeat", "can_repeat")
        if repeat not in REPEAT_RULES:
            raise SimDataError(f"{bw}.repeat {repeat!r} invalid")
        cond = b.get("if")
        if cond is not None and cond not in AI_CONDITIONS:
            raise SimDataError(f"{bw}.if {cond!r} not in {sorted(AI_CONDITIONS)}")
        branches.append(AiBranch(
            target=_get_str(b, "to", bw),
            weight=float(b.get("weight", 1.0)),
            repeat=repeat,
            max_times=int(b.get("max_times", 0)),
            cooldown=int(b.get("cooldown", 0)),
            condition=cond,
            condition_arg=b.get("arg"),
        ))
    return AiNode(node_id=nid, node_type=ntype, branches=tuple(branches))


def _parse_step(step: Any, where: str, power_ids: set[str] | None, card_ids: set[str] | None) -> EffectStep:
    _require_type(step, dict, where)
    verb = _get_str(step, "verb", where)
    spec = MONSTER_VERBS.get(verb)
    if spec is None:
        raise SimDataError(f"{where}.verb unknown: {verb!r}")
    args = step.get("args", {})
    _require_type(args, dict, f"{where}.args")
    missing = spec["required"] - set(args)
    if missing:
        raise SimDataError(f"{where}.args missing {sorted(missing)} for verb {verb!r}")
    extra = set(args) - spec["required"] - spec["optional"]
    if extra:
        raise SimDataError(f"{where}.args unexpected {sorted(extra)} for verb {verb!r}")
    for k in ("damage", "hits", "amount", "count"):
        if k in args:
            _scaled(args[k], f"{where}.args.{k}")
    if verb == "apply_power":
        if args["target"] not in POWER_TARGETS:
            raise SimDataError(f"{where}.args.target {args['target']!r} invalid")
        if power_ids is not None and args["power"] not in power_ids:
            raise SimDataError(f"{where}.args.power unknown {args['power']!r}")
    if verb == "add_card":
        if args["pile"] not in CARD_PILES:
            raise SimDataError(f"{where}.args.pile {args['pile']!r} invalid")
        if card_ids is not None and args["card"] not in card_ids:
            raise SimDataError(f"{where}.args.card unknown {args['card']!r}")
    if verb == "special" and args["name"] not in _SPECIALS:
        raise SimDataError(f"{where}.args.name unknown special {args['name']!r}")
    return EffectStep(verb=verb, args=MappingProxyType(dict(args)))


# ---------------------------------------------------------------------------
# Cross-reference checks

def _validate_upgrade_links(cards: dict[str, CardSchema]) -> None:
    for cid, card in cards.items():
        if card.upgrade_of is not None:
            up = cards.get(card.upgrade_of)
            if up is None:
                raise SimDataError(f"cards:{cid}.upgrade_of {card.upgrade_of!r} not found")
            if up.upgraded_from != cid:
                raise SimDataError(f"cards:{cid}.upgrade_of -> {card.upgrade_of!r}, but that card's "
                                   f"upgraded_from = {up.upgraded_from!r}")
        if card.upgraded_from is not None and card.upgraded_from not in cards:
            raise SimDataError(f"cards:{cid}.upgraded_from {card.upgraded_from!r} not found")
        if card.card_type in ("status", "curse") and card.upgrade_of is not None:
            raise SimDataError(f"cards:{cid} is a {card.card_type} and must not declare upgrade_of")


# ---------------------------------------------------------------------------
# Primitives

def _pair(v: Any, where: str) -> tuple[int, int]:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v) and 0 < v[0] <= v[1]):
        raise SimDataError(f"{where} must be [min, max] positive ints, got {v!r}")
    return (v[0], v[1])


def _scaled(v: Any, where: str) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, 0)
    if isinstance(v, list) and len(v) == 3 and all(isinstance(x, int) for x in v) and v[2] in (8, 9):
        return (v[0], v[1], v[2])
    raise SimDataError(f"{where} must be an int or [base, ascension_value, 8|9], got {v!r}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimDataError(f"data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SimDataError(f"{path}: malformed JSON ({exc})") from exc


def _require_type(value: Any, expected: type, where: str) -> None:
    if not isinstance(value, expected):
        raise SimDataError(f"{where} must be {expected.__name__}, got {type(value).__name__}")


def _require_list(value: Any, where: str) -> None:
    if not isinstance(value, list):
        raise SimDataError(f"{where} must be a list, got {type(value).__name__}")


def _require_unique(mapping: dict, key: str, *, where: str) -> None:
    if key in mapping:
        raise SimDataError(f"{where}: duplicate id {key!r}")


def _get_str(entry: dict, field: str, where: str) -> str:
    if field not in entry:
        raise SimDataError(f"{where}.{field} is required")
    value = entry[field]
    if not isinstance(value, str) or not value:
        raise SimDataError(f"{where}.{field} must be a non-empty string, got {value!r}")
    return value


def _get_in(entry: dict, field: str, allowed: frozenset[str], where: str) -> str:
    value = _get_str(entry, field, where)
    if value not in allowed:
        raise SimDataError(f"{where}.{field} {value!r} not in allowed set {sorted(allowed)}")
    return value


__all__ = [
    "CARD_FILES",
    "DATA_ROOT",
    "MONSTER_FILES",
    "SimDataError",
    "load_all",
    "load_cards",
    "load_encounters",
    "load_monsters",
    "load_powers",
]
