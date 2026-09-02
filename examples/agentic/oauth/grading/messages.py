"""Trajectory data contracts (ported from trajlab/types.py).

OpenAI-style message sequences; the format is byte-compatible with the real
Hermes captures under ``fixtures/samples/``.  Scenario / RollResult payloads
were dropped -- they belonged to trajlab's offline collection pipeline.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = "call_unset"
    type: str = "function"
    arguments_raw: Optional[str] = None  # preserved source string (byte-exact round-trip)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        fn = d.get("function", {})
        args = fn.get("arguments", "{}")
        raw = args if isinstance(args, str) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        return cls(
            name=fn.get("name", ""),
            arguments=args if isinstance(args, dict) else {"_raw": args},
            call_id=d.get("id", "call_unset"),
            type=d.get("type", "function"),
            arguments_raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        arguments = self.arguments_raw if self.arguments_raw is not None else json.dumps(
            self.arguments, ensure_ascii=False)
        return {
            "id": self.call_id,
            "type": self.type,
            "function": {"name": self.name, "arguments": arguments},
        }


@dataclass
class Message:
    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)  # preserved meta (hint, exit_code, ...)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        tcs = [ToolCall.from_dict(tc) for tc in d.get("tool_calls") or []]
        msg = cls(
            role=d["role"],
            content=d.get("content") or "",
            name=d.get("name"),
            tool_call_id=d.get("tool_call_id"),
            tool_calls=tcs,
            reasoning_content=d.get("reasoning_content"),
        )
        # keep any unknown top-level fields for provenance / round-trip fidelity
        known = {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"}
        msg.extra = {k: v for k, v in d.items() if k not in known}
        return msg

    def to_dict(self, include_reasoning: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if include_reasoning and self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        d.update(self.extra)
        return d

    def is_step(self) -> bool:
        """A decision step = one assistant turn carrying tool_calls."""
        return self.role == "assistant" and bool(self.tool_calls)

    def is_observation(self) -> bool:
        return self.role == "tool"


@dataclass
class Trajectory:
    """A full roll: OpenAI-style message sequence plus meta (driver, tools, ...)."""

    messages: list[Message] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        meta_keys = {"messages"}
        meta = {k: v for k, v in d.items() if k not in meta_keys}
        return cls(messages=[Message.from_dict(m) for m in d.get("messages", [])], meta=meta)

    def to_dict(self, include_reasoning: bool = True) -> dict[str, Any]:
        d = dict(self.meta)
        d["messages"] = [m.to_dict(include_reasoning=include_reasoning) for m in self.messages]
        return d

    def steps(self) -> list[tuple[int, Message]]:
        """Index of decision steps (assistant turns with tool_calls), in order."""
        return [(i, m) for i, m in enumerate(self.messages) if m.is_step()]

    def copy(self) -> "Trajectory":
        return copy.deepcopy(self)
