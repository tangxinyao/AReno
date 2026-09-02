"""Scenario-matrix primitives (ported from trajlab/orchestration/matrix.py).

Only the value axis / id-coding machinery came along: ``Dimension`` (axis +
stable id abbreviations), ``scenario_id_for`` (deterministic id in the preset's
axis order) and the cartesian-product / seeded-sample helpers.  The registry
(``MatrixSpec`` + profile enrichers) was dropped -- the google-email preset in
``oauth_matrix.py`` drives the enrichers directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable


def _default_abbrev(value: str) -> str:
    """Stable fallback id component for values without an explicit abbreviation."""
    return value


@dataclass(frozen=True)
class Dimension:
    """One matrix axis.

    ``key`` names the axis; ``values`` its legal values; ``id_part`` produces the
    stable id component used inside scenario ids.  It may be given as an
    explicit value->abbreviation dict, a callable, or left ``None`` to fall back
    to the raw value.
    """

    key: str
    values: tuple[str, ...]
    id_part: Callable[[str], str] | dict[str, str] | None = None

    def abbrev(self, value: str) -> str:
        if callable(self.id_part):
            return self.id_part(value)
        if isinstance(self.id_part, dict):
            return self.id_part.get(value, value)
        return _default_abbrev(value)


def scenario_id_for(combo: dict[str, str], version: str = "v1",
                    dims: list[Dimension] | None = None) -> str:
    """Deterministic id from a combo, in the preset's axis order.

    The order is derived from the dimension list, not a hard-coded axis tuple --
    a preset with different axes gets ids of its own shape.

    Join rule: each dimension abbreviation is a field whose internal charset is
    ``[a-z0-9-]``; fields are joined with a single ``-``.  This keeps scenario
    ids free of ``_`` so ids can be split on ``_`` unambiguously.
    """
    dims = list(dims) if dims is not None else []
    parts = [d.abbrev(combo.get(d.key, "")) for d in dims]
    return "-".join(parts + [version])


def enumerate_combos(dims: list[Dimension]) -> list[dict[str, str]]:
    """Cartesian product of axis values, keyed by Dimension.key, in axis order."""
    combos: list[dict[str, str]] = []

    def rec(i: int, acc: dict[str, str]) -> None:
        if i == len(dims):
            combos.append(dict(acc))
            return
        d = dims[i]
        for v in d.values:
            acc[d.key] = v
            rec(i + 1, acc)

    rec(0, {})
    return combos


def filter_combos(combos: Iterable[dict[str, str]], filters: Iterable[Callable[[dict[str, str]], bool]]) -> list[dict[str, str]]:
    """Keep combos satisfying every filter (each callable over a combo dict)."""
    return [c for c in combos if all(f(dict(c)) for f in filters)]


def seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)


def sample_scenarios(combos: list[dict[str, str]], limit: int | None = None,
                     seed: int = 20260831) -> list[dict[str, str]]:
    """Deterministic sampling (seeded) with exact limit; stable output order."""
    if limit is None or limit >= len(combos):
        return list(combos)
    rng = seeded_rng(seed)
    idx = list(range(len(combos)))
    rng.shuffle(idx)
    idx = sorted(idx[:limit], key=lambda i: tuple(sorted(combos[i].items())))  # stable order
    return [combos[i] for i in idx]
