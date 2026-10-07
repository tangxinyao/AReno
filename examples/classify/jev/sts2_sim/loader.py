"""Strict loaders for Phase 1 combat data.

Each `load_*` reads the matching JSON file, validates every field against
the schemas module's allowed vocabulary, and returns tuples of frozen
dataclasses. On any violation the loader raises `SimDataError` with a path
like `cards:anger:effects[1]:args.amount` so the author can locate the issue
without diffing the raw file.

Cross-reference checks (card.effects referencing a power_id that doesn't
exist, enemy move referencing an undefined power) are the loader's
responsibility — do them here so the engine can trust its inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import (
    CARD_ONLY_VERBS,
    CARD_RARITIES,
    CARD_TARGETS,
    CARD_TYPES,
    CardSchema,
    EFFECT_VERBS,
    ENEMY_ONLY_VERBS,
    EffectStep,
    EnemySchema,
    INTENTS,
    MOVE_RULES,
    MoveSchema,
    POWER_DURATIONS,
    POWER_KINDS,
    PowerSchema,
    SelectorEntry,
    VERB_ALLOWED_SCOPES,
)


DATA_ROOT = Path(__file__).resolve().parent / "data"


class SimDataError(ValueError):
    """Raised when authored data violates the Phase 1 schemas."""


# ---------------------------------------------------------------------------
# Public entry points

def load_powers(path: Path | None = None) -> dict[str, PowerSchema]:
    path = path or DATA_ROOT / "powers.json"
    raw = _read_json(path)
    _require_list(raw, "powers")
    out: dict[str, PowerSchema] = {}
    for i, entry in enumerate(raw):
        power = _parse_power(entry, where=f"powers[{i}]")
        _require_unique(out, power.power_id, where="powers")
        out[power.power_id] = power
    return out


def load_cards(
    path: Path | None = None,
    *,
    power_ids: set[str] | None = None,
) -> dict[str, CardSchema]:
    path = path or DATA_ROOT / "cards" / "ironclad.json"
    raw = _read_json(path)
    _require_list(raw, "cards")
    out: dict[str, CardSchema] = {}
    for i, entry in enumerate(raw):
        card = _parse_card(entry, where=f"cards[{i}]", power_ids=power_ids)
        _require_unique(out, card.card_id, where="cards")
        out[card.card_id] = card
    _validate_upgrade_links(out)
    return out


def load_enemies(
    path: Path | None = None,
    *,
    power_ids: set[str] | None = None,
) -> dict[str, EnemySchema]:
    path = path or DATA_ROOT / "enemies" / "act1.json"
    raw = _read_json(path)
    _require_list(raw, "enemies")
    out: dict[str, EnemySchema] = {}
    for i, entry in enumerate(raw):
        enemy = _parse_enemy(entry, where=f"enemies[{i}]", power_ids=power_ids)
        _require_unique(out, enemy.enemy_id, where="enemies")
        out[enemy.enemy_id] = enemy
    return out


def load_all() -> tuple[dict[str, PowerSchema], dict[str, CardSchema], dict[str, EnemySchema]]:
    """Convenience: load powers first, then feed power_ids to card/enemy loaders."""

    powers = load_powers()
    power_ids = set(powers.keys())
    cards = load_cards(power_ids=power_ids)
    enemies = load_enemies(power_ids=power_ids)
    return powers, cards, enemies


# ---------------------------------------------------------------------------
# Parsers

def _parse_power(entry: dict, *, where: str) -> PowerSchema:
    _require_type(entry, dict, where)
    power_id = _get_str(entry, "power_id", where)
    kind = _get_str(entry, "kind", where)
    duration = _get_str(entry, "duration", where)
    _require_in(kind, POWER_KINDS, where=f"{where}.kind")
    _require_in(duration, POWER_DURATIONS, where=f"{where}.duration")
    return PowerSchema(
        power_id=power_id,
        name=_get_str(entry, "name", where),
        kind=kind,
        duration=duration,
        stacks=bool(entry.get("stacks", True)),
    )


def _parse_card(entry: dict, *, where: str, power_ids: set[str] | None) -> CardSchema:
    _require_type(entry, dict, where)
    card_id = _get_str(entry, "card_id", where)
    card_type = _get_str(entry, "card_type", where)
    rarity = _get_str(entry, "rarity", where)
    target = _get_str(entry, "target", where)
    _require_in(card_type, CARD_TYPES, where=f"{where}.card_type")
    _require_in(rarity, CARD_RARITIES, where=f"{where}.rarity")
    _require_in(target, CARD_TARGETS, where=f"{where}.target")
    cost = entry.get("cost")
    if not isinstance(cost, int) or cost < 0:
        raise SimDataError(f"{where}.cost must be a non-negative int, got {cost!r}")

    raw_effects = entry.get("effects", [])
    _require_list(raw_effects, f"{where}.effects")
    effects = tuple(
        _parse_effect(step, where=f"{where}.effects[{j}]", source="card", power_ids=power_ids)
        for j, step in enumerate(raw_effects)
    )

    return CardSchema(
        card_id=card_id,
        name=_get_str(entry, "name", where),
        cost=cost,
        card_type=card_type,
        rarity=rarity,
        target=target,
        effects=effects,
        upgraded_from=entry.get("upgraded_from"),
        upgrade_of=entry.get("upgrade_of"),
    )


def _parse_enemy(entry: dict, *, where: str, power_ids: set[str] | None) -> EnemySchema:
    _require_type(entry, dict, where)
    enemy_id = _get_str(entry, "enemy_id", where)
    hp_min = entry.get("hp_min")
    hp_max = entry.get("hp_max")
    if not isinstance(hp_min, int) or not isinstance(hp_max, int) or hp_min <= 0 or hp_max < hp_min:
        raise SimDataError(f"{where}.hp_min/hp_max invalid: {hp_min!r}/{hp_max!r}")

    raw_moves = entry.get("moves", {})
    if not isinstance(raw_moves, dict) or not raw_moves:
        raise SimDataError(f"{where}.moves must be a non-empty object")

    moves: dict[str, MoveSchema] = {}
    for move_id, move_entry in raw_moves.items():
        move_where = f"{where}.moves.{move_id}"
        _require_type(move_entry, dict, move_where)
        intent = _get_str(move_entry, "intent", move_where)
        _require_in(intent, INTENTS, where=f"{move_where}.intent")
        raw_steps = move_entry.get("effects", [])
        _require_list(raw_steps, f"{move_where}.effects")
        steps = tuple(
            _parse_effect(step, where=f"{move_where}.effects[{j}]", source="enemy", power_ids=power_ids)
            for j, step in enumerate(raw_steps)
        )
        moves[move_id] = MoveSchema(move_id=move_id, intent=intent, effects=steps)

    raw_picker = entry.get("movepicker", [])
    _require_list(raw_picker, f"{where}.movepicker")
    if not raw_picker:
        raise SimDataError(f"{where}.movepicker must have at least one entry")

    picker: list[SelectorEntry] = []
    for j, sel in enumerate(raw_picker):
        sel_where = f"{where}.movepicker[{j}]"
        _require_type(sel, dict, sel_where)
        move_ref = _get_str(sel, "move_id", sel_where)
        if move_ref not in moves:
            raise SimDataError(f"{sel_where}.move_id references unknown move {move_ref!r}")
        rule = _get_str(sel, "rule", sel_where)
        _require_in(rule, MOVE_RULES, where=f"{sel_where}.rule")
        weight = int(sel.get("weight", 1))
        if weight < 1:
            raise SimDataError(f"{sel_where}.weight must be >= 1")
        seq_idx = sel.get("sequence_index")
        if rule == "sequential" and not isinstance(seq_idx, int):
            raise SimDataError(f"{sel_where}.sequence_index required when rule=sequential")
        picker.append(SelectorEntry(move_id=move_ref, rule=rule, weight=weight, sequence_index=seq_idx))

    return EnemySchema(
        enemy_id=enemy_id,
        name=_get_str(entry, "name", where),
        hp_min=hp_min,
        hp_max=hp_max,
        moves=moves,
        movepicker=tuple(picker),
    )


def _parse_effect(step: Any, *, where: str, source: str, power_ids: set[str] | None) -> EffectStep:
    _require_type(step, dict, where)
    verb = _get_str(step, "verb", where)
    if verb not in EFFECT_VERBS:
        raise SimDataError(f"{where}.verb unknown: {verb!r}")
    if source == "card" and verb in ENEMY_ONLY_VERBS:
        raise SimDataError(f"{where}.verb {verb!r} is enemy-only, not allowed on cards")
    if source == "enemy" and verb in CARD_ONLY_VERBS:
        raise SimDataError(f"{where}.verb {verb!r} is card-only, not allowed on enemy moves")

    args = step.get("args", {})
    if not isinstance(args, dict):
        raise SimDataError(f"{where}.args must be an object")
    spec = EFFECT_VERBS[verb]
    for arg_name, arg_type in spec.items():
        if arg_name == "hits":
            # optional; default 1 is injected below
            continue
        if arg_name not in args:
            raise SimDataError(f"{where}.args missing required field {arg_name!r} for verb {verb!r}")
        if not isinstance(args[arg_name], arg_type):
            raise SimDataError(
                f"{where}.args.{arg_name} must be {arg_type.__name__}, got {type(args[arg_name]).__name__}"
            )

    # Fill the single optional field so engine sees a stable shape.
    normalized = dict(args)
    if "hits" in spec and "hits" not in normalized:
        normalized["hits"] = 1

    if "target_scope" in spec:
        scope = normalized["target_scope"]
        allowed = VERB_ALLOWED_SCOPES[verb]
        if scope not in allowed:
            raise SimDataError(
                f"{where}.args.target_scope {scope!r} not allowed for verb {verb!r}; "
                f"allowed = {sorted(allowed)}"
            )
        if source == "card" and scope == "player":
            raise SimDataError(f"{where} cards cannot target 'player' (only enemies can)")
        if source == "enemy" and scope in {"single_enemy", "all_enemies", "random_enemy"}:
            raise SimDataError(
                f"{where} enemy moves cannot target other enemies in Phase 1 (scope={scope!r})"
            )

    if verb == "apply_power" and power_ids is not None:
        pid = normalized["power_id"]
        if pid not in power_ids:
            raise SimDataError(f"{where}.args.power_id references unknown power {pid!r}")

    # Numeric sanity: every int-typed arg must be >= 0, except `hits` which
    # must be >= 1 since zero hits is nonsensical for a damage verb.
    for arg_name, arg_type in spec.items():
        if arg_type is not int:
            continue
        value = normalized.get(arg_name)
        if value is None:
            continue
        if arg_name == "hits":
            if value < 1:
                raise SimDataError(f"{where}.args.hits must be >= 1")
        elif value < 0:
            raise SimDataError(f"{where}.args.{arg_name} must be >= 0")

    return EffectStep(verb=verb, args=normalized)


# ---------------------------------------------------------------------------
# Cross-reference checks

def _validate_upgrade_links(cards: dict[str, CardSchema]) -> None:
    for cid, card in cards.items():
        if card.upgrade_of is not None and card.upgrade_of not in cards:
            raise SimDataError(f"cards:{cid}.upgrade_of {card.upgrade_of!r} not found")
        if card.upgraded_from is not None and card.upgraded_from not in cards:
            raise SimDataError(f"cards:{cid}.upgraded_from {card.upgraded_from!r} not found")
    # Base cards (no upgraded_from) must point to a valid upgrade_of; and
    # upgraded cards must point back.
    for cid, card in cards.items():
        if card.upgraded_from is None:
            if card.upgrade_of is None:
                raise SimDataError(f"cards:{cid} base card missing upgrade_of pointer")
            upgraded = cards[card.upgrade_of]
            if upgraded.upgraded_from != cid:
                raise SimDataError(
                    f"cards:{cid}.upgrade_of -> {card.upgrade_of!r}, "
                    f"but that card's upgraded_from = {upgraded.upgraded_from!r}"
                )


# ---------------------------------------------------------------------------
# Primitives

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


def _require_in(value: Any, allowed: frozenset[str], *, where: str) -> None:
    if value not in allowed:
        raise SimDataError(f"{where} {value!r} not in allowed set {sorted(allowed)}")


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


__all__ = [
    "DATA_ROOT",
    "SimDataError",
    "load_all",
    "load_cards",
    "load_enemies",
    "load_powers",
]
