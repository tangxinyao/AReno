"""Priority-ordered hook bus.

Relics, powers, and events all react to the same stream of game events
(e.g. `on_combat_start`, `on_card_played`, `on_turn_end`). Using a single
bus with explicit integer priorities keeps ordering debuggable instead of
leaving it to subscription order.

Lower priority runs first; ties break by registration order. Handlers that
mutate shared state should pick a conservative priority (e.g. 100) and leave
room for interrupts (e.g. 50) and finalizers (e.g. 200).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


HookHandler = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class _Registration:
    priority: int
    seq: int
    handler: HookHandler


class HookBus:
    __slots__ = ("_handlers", "_seq")

    def __init__(self) -> None:
        self._handlers: dict[str, list[_Registration]] = {}
        self._seq: int = 0

    def register(self, event: str, handler: HookHandler, *, priority: int = 100) -> Callable[[], None]:
        self._seq += 1
        reg = _Registration(priority=int(priority), seq=self._seq, handler=handler)
        bucket = self._handlers.setdefault(event, [])
        bucket.append(reg)
        bucket.sort(key=lambda r: (r.priority, r.seq))

        def _unregister() -> None:
            if event in self._handlers:
                self._handlers[event] = [r for r in self._handlers[event] if r is not reg]

        return _unregister

    def dispatch(self, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = {} if payload is None else payload
        for reg in list(self._handlers.get(event, ())):
            reg.handler(payload)

    def handler_count(self, event: str) -> int:
        return len(self._handlers.get(event, ()))
