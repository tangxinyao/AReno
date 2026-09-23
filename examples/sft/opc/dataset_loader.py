"""Dataset loader for OPC-benchmark agent trajectories -> Ling-3.0-tiny SFT rows.

Turns passing Hermes agent sessions (multi-turn tool-calling rollouts captured
by ``hermes-session.jsonl``) into supervised rows. Two output modes:

- **B (default)**: sequential SFT text rows, one ``(prompt, response)`` row per
  assistant turn. The prompt is the Ling chat-format history up to that turn
  (task + all prior actions/observations), the response is that turn's own
  output (reasoning + `` response`` + text + tool calls). Tool results are
  context — never a training target.
- **C (``ARENO_OPC_C_MODE=1``)**: whole-rollout packed rows, ``tokens`` +
  ``prompt_mask`` + ``loss_mask``, with loss enabled only on assistant-produced
  spans (including the closing ``<|role_end|>``). Avoids re-encoding the
  history prefix once per turn.

Both render the [Ling-3.0-tiny]
(https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny) ``chat_template.jinja``
semantics as plain text: ``<role>HUMAN</role>`` turns, a
``<role>ASSISTANT</role>`` + `` n thinking`` generation prompt, a
`` thinking ... response`` split, ``<tool_call>`` with ``<arg_key>/<arg_value>``
argument pairs, and ``<role>OBSERVATION</role>`` + ``<tool_response>`` blocks
for tool results. Reasoning is a training target by default.

Input ``--dataset-path`` may be a job run dir (``*/agent/hermes-session.jsonl``
with a sibling ``*/result.json``), a flat dir of session files, or a single
session file. Only trials with verifier ``reward == 1.0`` are kept; standalone
sessions without a result file are kept as-is.
"""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path
from typing import Any

# --- Ling-3.0-tiny / Bailing V3 native markup (from chat_template.jinja) ---
_LING_ROLE_SYSTEM = "<role>SYSTEM</role>"
_LING_ROLE_HUMAN = "<role>HUMAN</role>"
_LING_ROLE_ASSISTANT = "<role>ASSISTANT</role>"
_LING_ROLE_OBSERVATION = "<role>OBSERVATION</role>"
_LING_ROLE_END = "<|role_end|>"  # Ling's EOS
_LING_GEN_PROMPT = f"{_LING_ROLE_ASSISTANT}\n thinking"  # generation prompt (thinking on)
_LING_THINKING_HEADER = "detailed thinking on"
_LING_TOOL_RESPONSE_OPEN = "<tool_response>\n"
_LING_TOOL_RESPONSE_CLOSE = "\n</tool_response>"

_TRUNCATED = "\n…<truncated {n} chars>"
_PASS_REWARD = 1.0

_LING_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "configuration_*.py",
    "modeling_*.py",
    "chat_template.jinja",
]


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize passing OPC Hermes sessions into SFT rows (Ling format)."""

    # OPC data is local JSONL, not a HF dataset; the default_loader is unused.
    del default_loader

    drop_reasoning = bool(os.environ.get("ARENO_OPC_DROP_REASONING"))
    include_system_prompt = bool(os.environ.get("ARENO_OPC_INCLUDE_SYSTEM_PROMPT"))
    max_history_messages = _int_env("ARENO_OPC_MAX_HISTORY_MESSAGES")
    max_tool_chars = _int_env("ARENO_OPC_MAX_TOOL_CHARS")
    phase_split = bool(os.environ.get("ARENO_OPC_PHASES"))
    c_mode = bool(os.environ.get("ARENO_OPC_C_MODE"))

    if c_mode:
        tokenizer = _load_tokenizer()
        return _load_c_records(
            dataset_path,
            tokenizer,
            max_seq_tokens=_int_env("ARENO_OPC_MAX_SEQ_TOKENS") or 32768,
            max_tool_chars=max_tool_chars,
            drop_reasoning=drop_reasoning,
        )

    records: list[dict] = []
    for session_path, result_path in _iter_session_files(dataset_path):
        reward = _trial_reward(result_path)
        if reward is not None and reward != _PASS_REWARD:
            continue  # failed trial -> not taught
        try:
            session = _load_session(session_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"opc: cannot load {session_path}: {exc}") from exc
        rows = _session_to_rows(
            session,
            include_system_prompt=include_system_prompt,
            drop_reasoning=drop_reasoning,
            max_history_messages=max_history_messages,
            max_tool_chars=max_tool_chars,
            phase_split=phase_split,
        )
        for index, row in enumerate(rows):
            record = {
                "prompt": row["prompt"],
                "response": row["response"],
                "source_trial": session_path.parent.parent.name,
                "turn_index": index,
            }
            if phase_split:
                record["phase_index"] = row["phase_index"]
                record["phase_genre"] = row["phase_genre"]
            records.append(record)
    return records


def _iter_session_files(dataset_path: str):
    """Yield ``(session_file, result_file_or_None)`` pairs for ``dataset_path``."""

    path = Path(dataset_path)
    if path.is_file():
        return [(path, None)]

    found = []
    if path.is_dir():
        # Job-run layout: <job>/<trial>/agent/hermes-session.jsonl + result.json
        for trial_dir in sorted(path.glob("*/")):
            session_file = trial_dir / "agent" / "hermes-session.jsonl"
            if session_file.is_file():
                result_file = trial_dir / "result.json"
                found.append((session_file, result_file if result_file.is_file() else None))
        if not found:
            # Looser layouts: hermes-session.jsonl one level down, or bare
            # *.jsonl sitting directly in the directory.
            for subdir in sorted(path.glob("*/")):
                for session_file in sorted(subdir.glob("hermes-session.jsonl")):
                    found.append((session_file, None))
            if not found:
                for session_file in sorted(path.glob("*.jsonl")):
                    found.append((session_file, None))
    return found


def _trial_reward(result_path: Path | None) -> float | None:
    """Read the verifier reward from a trial result.json, or None if absent."""

    if result_path is None or not result_path.is_file():
        return None
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return data.get("verifier_result", {}).get("rewards", {}).get("reward")


def _load_session(session_path: Path) -> dict[str, Any]:
    """Load one Hermes session, tolerating either a JSON object or JSONL lines."""

    raw = session_path.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        messages: list[dict] = []
        system_prompt = ""
        for line in raw.splitlines():
            if not line.strip():
                continue
            part = json.loads(line)
            messages.extend(part.get("messages") or [])
            if not system_prompt and part.get("system_prompt"):
                system_prompt = part["system_prompt"]
        obj = {"messages": messages, "system_prompt": system_prompt}
    if not obj.get("messages"):
        raise ValueError(f"session has no messages: {session_path}")
    return obj


def _message_text(message: dict[str, Any]) -> str:
    """Extract plain text from ``content`` (str, list of blocks, or None)."""

    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict):
                chunks.append(str(block.get("text") or ""))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _int_env(name: str) -> int:
    """Read a non-negative int env var; 0 (or unset/empty) means "off"."""

    raw = os.environ.get(name, "")
    if not raw.strip():
        return 0
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


# --- B mode renderers -----------------------------------------------------


def _session_to_rows(
    session: dict[str, Any],
    *,
    include_system_prompt: bool,
    drop_reasoning: bool,
    max_history_messages: int,
    max_tool_chars: int,
    phase_split: bool,
) -> list[dict[str, Any]]:
    """Expand a session into one (prompt, response) row per assistant turn."""

    prefix: list[dict[str, Any]] = []
    if include_system_prompt and session.get("system_prompt"):
        prefix.append({"role": "system", "content": session["system_prompt"]})

    rows: list[dict[str, Any]] = []
    current_phase = {"index": 0, "genre": None}
    for message in session.get("messages", []):
        role = message.get("role")
        if role != "assistant":
            prefix.append(message)
            continue
        # The target turn stays verbatim; only the *context* is compressed
        # so long trajectories fit inside the model's context window.
        response = _render_ling_assistant_target(message, drop_reasoning=drop_reasoning)
        if not response:
            prefix.append(message)  # empty turn still advances the history
            continue
        # Sliding window: always keep the task-defining first message, then
        # the most recent messages, so later rows retain the task prompt.
        if max_history_messages and len(prefix) > max_history_messages + 1:
            context = prefix[:1] + prefix[-max_history_messages:]
        else:
            context = prefix
        prompt = _render_ling_history(context, drop_reasoning=drop_reasoning, max_tool_chars=max_tool_chars)
        row = {"prompt": prompt, "response": response}
        if phase_split:
            genre = _phase_genre(message)
            if genre != current_phase["genre"]:
                current_phase["index"] += 1
                current_phase["genre"] = genre
            row["phase_index"] = current_phase["index"]
            row["phase_genre"] = genre
        rows.append(row)
        prefix.append(message)
    return rows


def _render_ling_history(messages: list[dict[str, Any]], *, drop_reasoning: bool, max_tool_chars: int = 0) -> str:
    """Render history + generation prompt the way Ling's chat template does."""

    blocks: list[str] = []
    i, n = 0, len(messages)
    if i < n and messages[i].get("role") == "system":
        blocks.append(
            f"{_LING_ROLE_SYSTEM}{_message_text(messages[i]).rstrip(chr(10))}\n{_LING_THINKING_HEADER}{_LING_ROLE_END}"
        )
        i += 1
    else:
        blocks.append(f"{_LING_THINKING_HEADER}{_LING_ROLE_END}")
    while i < n:
        message = messages[i]
        role = message.get("role")
        if role == "system":
            blocks.append(f"{_LING_ROLE_SYSTEM}{_message_text(message)}{_LING_ROLE_END}")
            i += 1
        elif role == "user":
            blocks.append(f"{_LING_ROLE_HUMAN}{_message_text(message)}{_LING_ROLE_END}")
            i += 1
        elif role == "assistant":
            blocks.append(_render_ling_assistant_block(message, drop_reasoning=drop_reasoning, max_tool_chars=max_tool_chars))
            i += 1
        elif role == "tool":
            # Consecutive tool results share one OBSERVATION role block.
            responses: list[str] = []
            while i < n and messages[i].get("role") == "tool":
                responses.append(_render_ling_tool_response(messages[i], max_tool_chars=max_tool_chars))
                i += 1
            blocks.append(f"{_LING_ROLE_OBSERVATION}\n" + "\n".join(responses) + _LING_ROLE_END)
        else:
            i += 1
    return "\n".join(blocks) + "\n" + _LING_GEN_PROMPT


def _render_ling_assistant_block(message: dict[str, Any], *, drop_reasoning: bool, max_tool_chars: int = 0) -> str:
    """Render a full assistant turn (for history), including tool calls."""

    body = _render_ling_assistant_target(message, drop_reasoning=drop_reasoning, max_tool_chars=max_tool_chars)
    return f"{_LING_ROLE_ASSISTANT}\n thinking{body}{_LING_ROLE_END}"


def _render_ling_assistant_target(message: dict[str, Any], *, drop_reasoning: bool, max_tool_chars: int = 0) -> str:
    """Render the *supervised target* for one assistant turn.

    Matches the template's ``\n thinking{reasoning} response{content}`` split:
    the generation prompt ends at ``<role>ASSISTANT</role>\n thinking``, so the
    target starts with the reasoning, then the `` response`` separator, then
    content, then raw tool-call blocks. No trailing role_end in B mode — AReno
    appends the EOS (``<|role_end|>``) itself.
    """

    reasoning = "" if drop_reasoning else (message.get("reasoning_content") or "")
    parts: list[str] = []
    if reasoning:
        parts.append(reasoning.strip("\n"))
        parts.append(" response")
    else:
        parts.append(" response")
    content = _message_text(message).lstrip("\n")
    if content:
        parts.append(content)
    body = "".join(parts)
    tool_text = _render_ling_tools(message, max_tool_chars=max_tool_chars)
    if tool_text:
        body += "\n" + tool_text
    return body


def _render_ling_tools(message: dict[str, Any], *, max_tool_chars: int = 0) -> str:
    """Render all tool calls of a turn as Ling ``<tool_call>`` blocks."""

    blocks: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if isinstance(function, dict):
            name = function.get("name", "")
            arguments = function.get("arguments") or {}
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"raw": arguments}
        blocks.append(_render_ling_tool_call(name, arguments, max_tool_chars=max_tool_chars))
    return "\n".join(blocks)


def _render_ling_tool_call(name: str, arguments: dict[str, Any], *, max_tool_chars: int = 0) -> str:
    """Render one Ling tool call as ``<tool_call>{name}\n<arg_key>/<arg_value>...``."""

    def value_text(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    arg_lines = [
        f"<arg_key>{key}</arg_key>\n<arg_value>{value_text(value)}</arg_value>" for key, value in arguments.items()
    ]
    if max_tool_chars:
        room = max_tool_chars - len(name) - len("<tool_call>\n") - len("\n</tool_call>")
        kept: list[str] = []
        dropped = 0
        for line in arg_lines:
            if room < len(line):
                dropped += len(line)
                continue
            kept.append(line)
            room -= len(line)
        if dropped:
            kept.append(f"<arg_value>…<truncated {dropped} chars…</arg_value>")
        arg_lines = kept
    return "<tool_call>" + name + "\n" + "\n".join(arg_lines) + "\n</tool_call>"


def _render_ling_tool_response(message: dict[str, Any], *, max_tool_chars: int = 0) -> str:
    """Render one tool result as a ``<tool_response>`` block inside OBSERVATION."""

    content = _message_text(message)
    if max_tool_chars and len(content) > max_tool_chars:
        content = content[:max_tool_chars] + _TRUNCATED.format(n=len(content) - max_tool_chars)
    return f"{_LING_TOOL_RESPONSE_OPEN}{content}{_LING_TOOL_RESPONSE_CLOSE}"


def _phase_genre(message: dict[str, Any]) -> str:
    """Classify one assistant turn into a coarse action phase by tool type."""

    names = {tc.get("function", {}).get("name") for tc in (message.get("tool_calls") or [])}
    if not names:
        return "report"
    if "search_files" in names:
        return "explore"
    if "read_file" in names:
        return "read"
    if {"write_file", "patch"} & names:
        return "implement"
    return "run"


# --- C mode: whole-rollout packed rows (tokens + prompt/loss masks) --------


def _load_tokenizer():
    """Load the Ling tokenizer for C mode.

    ``ARENO_OPC_TOKENIZER`` points at a local tokenizer dir (offline). Without
    it the loader auto-downloads the Ling-3.0-tiny tokenizer files (not the
    weights) from ModelScope.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("ARENO_OPC_C_MODE requires `transformers` (only the tokenizer part is used)") from exc

    local = os.environ.get("ARENO_OPC_TOKENIZER", "").strip()
    if not local:
        try:
            from modelscope import snapshot_download

            local = snapshot_download("inclusionAI/Ling-3.0-tiny", allow_file_pattern=_LING_TOKENIZER_FILES)
        except Exception as exc:  # noqa: BLE001 - surface a friendly error
            raise ValueError(
                "could not auto-download the Ling tokenizer; set ARENO_OPC_TOKENIZER=<dir> to a local copy"
            ) from exc
    try:
        return AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not load tokenizer from {local!r}: {exc}") from exc


def _load_c_records(
    dataset_path: str, tokenizer, *, max_seq_tokens: int, max_tool_chars: int, drop_reasoning: bool
) -> list[dict]:
    """Encode every passing session into packed token rows (C mode)."""

    records: list[dict] = []
    for session_path, result_path in _iter_session_files(dataset_path):
        reward = _trial_reward(result_path)
        if reward is not None and reward != _PASS_REWARD:
            continue  # failed trial -> not taught
        try:
            session = _load_session(session_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"opc: cannot load {session_path}: {exc}") from exc
        trial_name = session_path.parent.parent.name
        for chunk_index, row in enumerate(
            _session_to_c_rows(
                session,
                tokenizer,
                max_seq_tokens=max_seq_tokens,
                max_tool_chars=max_tool_chars,
                drop_reasoning=drop_reasoning,
            )
        ):
            row["source_trial"] = trial_name
            row["chunk_index"] = chunk_index
            records.append(row)
    return records


def _session_to_c_rows(
    session: dict[str, Any],
    tokenizer,
    *,
    max_seq_tokens: int,
    max_tool_chars: int,
    drop_reasoning: bool,
) -> list[dict[str, Any]]:
    """Pack one trajectory into one or more encoded rows.

    The Ling-format transcript is tokenized once (fast tokenizer, with
    character offsets). Every token that overlaps an assistant-produced span —
    reasoning, `` response``, content, tool calls, and the trailing
    ``<|role_end|>`` — is a target; everything else (headers, user, tool
    observations) is context. Long trajectories are cut into chunks at message
    boundaries, hard-splitting only oversized single messages.
    """

    parts = _build_ling_parts(session.get("messages", []), max_tool_chars=max_tool_chars, drop_reasoning=drop_reasoning)
    full_text = "".join(text for text, _ in parts)

    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoding["input_ids"])
    offsets = list(encoding["offset_mapping"])
    if len(ids) != len(offsets):
        raise ValueError(f"tokenizer offsets mismatch: {len(ids)} ids vs {len(offsets)} offsets")

    char_target = [False] * len(full_text)
    pos = 0
    for text, target in parts:
        for k in range(pos, pos + len(text)):
            char_target[k] = target
        pos += len(text)

    token_target = [False] * len(ids)
    for idx, (start, end) in enumerate(offsets):
        if start < end and any(char_target[start:end]):
            token_target[idx] = True

    cumulative = []
    acc = 0
    for text, _ in parts:
        acc += len(text)
        cumulative.append(acc)
    token_part = [min(bisect.bisect_left(cumulative, start), len(parts) - 1) for (start, _) in offsets]

    rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _chunk_ranges(len(ids), token_part, max_seq_tokens):
        chunk_target = token_target[chunk_start:chunk_end]
        if not any(chunk_target):
            continue  # no assistant output in this window -> nothing to teach
        rows.append(
            {
                "tokens": ids[chunk_start:chunk_end],
                "prompt_mask": [not t for t in chunk_target],
                "loss_mask": chunk_target,
            }
        )
    return rows


def _build_ling_parts(
    messages: list[dict[str, Any]], *, max_tool_chars: int = 0, drop_reasoning: bool = False
) -> list[tuple[str, bool]]:
    """Render a session into ``(text, is_assistant_target)`` chunks."""

    parts: list[tuple[str, bool]] = []
    i, n = 0, len(messages)
    if i < n and messages[i].get("role") == "system":
        parts.append(
            (f"{_LING_ROLE_SYSTEM}{_message_text(messages[i]).rstrip(chr(10))}\n{_LING_THINKING_HEADER}{_LING_ROLE_END}", False)
        )
        i += 1
    else:
        parts.append((f"{_LING_THINKING_HEADER}{_LING_ROLE_END}", False))
    while i < n:
        message = messages[i]
        role = message.get("role")
        if role == "system":
            parts.append((f"{_LING_ROLE_SYSTEM}{_message_text(message)}{_LING_ROLE_END}", False))
            i += 1
        elif role == "user":
            parts.append((f"{_LING_ROLE_HUMAN}{_message_text(message)}{_LING_ROLE_END}", False))
            i += 1
        elif role == "assistant":
            parts.append((f"{_LING_ROLE_ASSISTANT}\n thinking", False))
            parts.append((_render_ling_assistant_target(message, drop_reasoning=drop_reasoning, max_tool_chars=max_tool_chars), True))
            parts.append((_LING_ROLE_END, True))  # stop token is model output
            i += 1
        elif role == "tool":
            responses: list[str] = []
            while i < n and messages[i].get("role") == "tool":
                responses.append(_render_ling_tool_response(messages[i], max_tool_chars=max_tool_chars))
                i += 1
            parts.append((f"{_LING_ROLE_OBSERVATION}\n" + "\n".join(responses) + _LING_ROLE_END, False))
        else:
            i += 1
    return parts


def _chunk_ranges(n_tokens: int, token_part: list[int], max_tokens: int) -> list[tuple[int, int]]:
    """Greedily slice token indices into chunks of at most ``max_tokens``.

    Cuts land on message (part) boundaries whenever possible; a single message
    longer than ``max_tokens`` is hard-split.
    """

    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < n_tokens:
        end = idx
        while end + 1 < n_tokens and token_part[end + 1] == token_part[idx]:
            end += 1
        runs.append((idx, end + 1))
        idx = end + 1

    ranges: list[tuple[int, int]] = []
    start = 0
    for run_start, run_end in runs:
        if start < run_start and (run_end - start) > max_tokens:
            ranges.append((start, run_start))  # cut before this message
            start = run_start
        while run_end - start > max_tokens:  # single oversized message
            ranges.append((start, start + max_tokens))
            start += max_tokens
    if start < n_tokens:
        ranges.append((start, n_tokens))
    return ranges