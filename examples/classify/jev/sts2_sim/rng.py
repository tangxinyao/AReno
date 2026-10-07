"""Deterministic, named RNG streams.

Rationale: a single `random.Random` seeded once makes replay fragile because
any change in call order (e.g. adding a new relic that peeks the stream) shifts
every subsequent roll. Independent streams per subsystem keep combat shuffles
stable even if map generation changes.

Not bit-exact with r33hab/sts2 by design — see NOTICE.md for the position on
cross-sim RNG parity.
"""

from __future__ import annotations

import hashlib
import random


_MAX_U64 = (1 << 64) - 1


def _derive_seed(master: int, name: str) -> int:
    digest = hashlib.blake2b(f"{master}:{name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MAX_U64


class Rng:
    """Owner of all random streams for one run.

    Streams are lazily created on first access; a stream, once named, keeps
    the same seed for the lifetime of the Rng. Call `fork` to clone the whole
    state (useful for look-ahead search without disturbing live play).
    """

    __slots__ = ("_master", "_streams")

    def __init__(self, master_seed: int) -> None:
        self._master = int(master_seed) & _MAX_U64
        self._streams: dict[str, random.Random] = {}

    @property
    def master(self) -> int:
        return self._master

    def stream(self, name: str) -> random.Random:
        rng = self._streams.get(name)
        if rng is None:
            rng = random.Random(_derive_seed(self._master, name))
            self._streams[name] = rng
        return rng

    def fork(self) -> "Rng":
        clone = Rng(self._master)
        for name, rng in self._streams.items():
            clone._streams[name] = random.Random()
            clone._streams[name].setstate(rng.getstate())
        return clone
