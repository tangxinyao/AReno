"""Guide knowledge base: load, validate against sim data, and score candidates.

Guide entries are hand-distilled from community videos and wiki pages. Every
`sim_name` must match a real sim entry (checked by `validate`), so a typo
cannot silently drop a label. Tiers are the scores the labeler uses.

    python examples/classify/jev/strategy/knowledge.py      # validate the KB
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "kb"
SIM_DATA = Path(__file__).resolve().parent.parent / "sts2_sim" / "data"

ALLOWED_TIERS = frozenset({-1, 0, 1, 2, 3})
ALLOWED_CONFIDENCE = frozenset({"high", "medium", "low"})

# Scores for choices the KB does not cover. Assumptions, tune against sim outcomes.
# UNKNOWN_SCORE sits between "situational" (1) and "take" (2): the guides are silent,
# so it must not read as tier 0 ("do not take early"), which would push them down.
UNKNOWN_SCORE = 1.0
SKIP_CARD_SCORE = 1.0  # "skip card reward"
LEAVE_RELIC_SCORE = 0.5  # "leave ancient"

_CARD_TEXT = re.compile(r"^take (.+?)\+?(?: \[| \()")
_RELIC_TEXT = re.compile(r"^take (.+?): ")


def _load(name: str) -> dict:
    return json.loads((KB_DIR / name).read_text(encoding="utf-8"))


def load_cards() -> dict:
    return _load("cards.json")


def load_ancient_relics() -> dict:
    return _load("ancient_relics.json")


def load_monsters() -> dict:
    return _load("monsters.json")


def sim_card_names() -> set[str]:
    names: set[str] = set()
    for file in ("ironclad.json", "colorless.json"):
        for card in json.loads((SIM_DATA / "cards" / file).read_text(encoding="utf-8")):
            if not card["name"].endswith("+"):
                names.add(card["name"])
    return names


def sim_relic_names() -> set[str]:
    return {r["name"] for r in json.loads((SIM_DATA / "relics.json").read_text(encoding="utf-8"))}


def sim_monster_names() -> set[str]:
    names: set[str] = set()
    for file in (SIM_DATA / "monsters").glob("*.json"):
        names.update(m["name"] for m in json.loads(file.read_text(encoding="utf-8")))
    return names


def validate() -> list[str]:
    """Return human-readable errors; empty list means the KB is consistent with sim data."""

    errors: list[str] = []
    cards = sim_card_names()
    relics = sim_relic_names()
    monsters = sim_monster_names()

    def check_entries(label: str, entries: list[dict], known: set[str], allow_null: bool = False) -> None:
        seen: set[str] = set()
        for entry in entries:
            guide = entry.get("guide_name", "<no guide_name>")
            sim = entry.get("sim_name")
            if not entry.get("source"):
                errors.append(f"{label}: {guide!r} has no source")
            if sim is None and allow_null:
                continue
            if sim not in known:
                errors.append(f"{label}: {guide!r} sim_name {sim!r} not found in sim data")
            if sim in seen:
                errors.append(f"{label}: duplicate sim_name {sim!r}")
            seen.add(sim)
            if "tier" in entry and entry["tier"] not in ALLOWED_TIERS:
                errors.append(f"{label}: {guide!r} tier {entry['tier']!r} not in {sorted(ALLOWED_TIERS)}")
            if entry.get("confidence") not in ALLOWED_CONFIDENCE:
                errors.append(f"{label}: {guide!r} confidence {entry.get('confidence')!r} invalid")

    check_entries("cards", load_cards()["entries"], cards)
    check_entries("ancient", load_ancient_relics()["entries"], relics)
    check_entries("monsters", load_monsters()["entries"], monsters, allow_null=True)
    return errors


def card_tiers() -> dict[str, int]:
    return {e["sim_name"]: e["tier"] for e in load_cards()["entries"]}


def relic_tiers() -> dict[str, int]:
    return {e["sim_name"]: e["tier"] for e in load_ancient_relics()["entries"]}


def card_name_from_text(text: str) -> str | None:
    """`take Bash+ (attack, cost 2): ...` -> `Bash`. Upgrade marks are dropped."""

    match = _CARD_TEXT.match(text)
    return match.group(1) if match else None


def relic_name_from_text(text: str) -> str | None:
    """`take Pael's Legion: ...` -> `Pael's Legion`."""

    match = _RELIC_TEXT.match(text)
    return match.group(1) if match else None


def candidate_scores(decision_point: str, candidates: list[dict]) -> tuple[list[float], int, int] | None:
    """Score each candidate from the KB; None when the decision is not one the KB labels.

    Returns (scores aligned with `candidates`, covered, eligible): `eligible` counts the
    real options (cards or relics, not skip/leave), `covered` how many of those the KB knows.
    """

    ids = [c["id"] for c in candidates]
    texts = [c["text"] for c in candidates]

    if decision_point == "card_reward" and any(i.startswith("select_card_reward:") for i in ids):
        tiers = card_tiers()
        scores, covered, eligible = [], 0, 0
        for cid, text in zip(ids, texts):
            if cid == "skip_card_reward":
                scores.append(SKIP_CARD_SCORE)
                continue
            eligible += 1
            name = card_name_from_text(text)
            if name in tiers:
                scores.append(float(tiers[name]))
                covered += 1
            else:
                scores.append(UNKNOWN_SCORE)
        return scores, covered, eligible

    if decision_point == "event_choice" and any(i.startswith("select_relic:") for i in ids):
        tiers = relic_tiers()
        scores, covered, eligible = [], 0, 0
        for cid, text in zip(ids, texts):
            if cid == "skip_relic_selection":
                scores.append(LEAVE_RELIC_SCORE)
                continue
            eligible += 1
            name = relic_name_from_text(text)
            if name in tiers:
                scores.append(float(tiers[name]))
                covered += 1
            else:
                scores.append(UNKNOWN_SCORE)
        return scores, covered, eligible

    return None


def main() -> int:
    errors = validate()
    for error in errors:
        print(error, file=sys.stderr)
    cards = len(load_cards()["entries"])
    relics = len(load_ancient_relics()["entries"])
    monsters = len(load_monsters()["entries"])
    print(f"cards={cards} ancient={relics} monsters={monsters} errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
