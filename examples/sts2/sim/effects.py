"""FIFO effect queue.

Combat in STS2 is a chain of discrete effects (damage, apply buff, draw card,
die, trigger power) and most "weird" interactions come from effect ordering,
not from the individual effects themselves. Phase 0 only exposes the data
structure; concrete effect handlers arrive in Phase 1 as the combat engine
takes shape.

Shape borrowed in spirit from HIX4123/sts2-simulator's `core` package but
simplified — no coroutine yields, just a plain deque drain under the
RunLoop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Effect:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


class EffectQueue:
    __slots__ = ("_q",)

    def __init__(self) -> None:
        self._q: deque[Effect] = deque()

    def __len__(self) -> int:
        return len(self._q)

    def enqueue(self, effect: Effect) -> None:
        self._q.append(effect)

    def enqueue_front(self, effect: Effect) -> None:
        """For effects that must resolve before anything already queued."""

        self._q.appendleft(effect)

    def drain_one(self) -> Effect | None:
        if not self._q:
            return None
        return self._q.popleft()

    def peek(self) -> Effect | None:
        return self._q[0] if self._q else None

    def clear(self) -> None:
        self._q.clear()
