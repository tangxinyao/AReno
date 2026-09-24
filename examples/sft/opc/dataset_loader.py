"""Dataset loader for OPC-benchmark agent trajectories -> Ling-3.0-tiny SFT rows.

Turns passing Hermes agent sessions (multi-turn tool-calling rollouts captured
by ``hermes-session.jsonl``) into supervised rows. Two output modes:

- **B (default)**: one row per assistant turn. The context is the
  chat-template rendering of the history up to that turn plus the generation
  prompt; the target is the rest of that turn's rendering (reasoning, content,
  tool calls, closing EOS). Tool results are context — never a target.
- **C (``ARENO_OPC_C_MODE=1``)**: whole-rollout packed rows, ``tokens`` +
  ``prompt_mask`` + ``loss_mask``, with loss enabled only on assistant-produced
  spans (including the closing EOS). Avoids re-encoding the history prefix once
  per turn.

Both modes emit pre-encoded ``tokens`` + ``prompt_mask`` + ``loss_mask`` rows,
so the trainer never re-applies a chat template to them. B rows also carry
``prompt``/``response`` text for inspection; the trainer uses the tokens.

All markup comes from the tokenizer's own ``chat_template.jinja`` through
``apply_chat_template``; the loader never hand-writes role, thinking or
tool-call tags, so rows match what the template produces at inference. Target
spans are located by rendering growing message prefixes, so a template that
rewrites earlier turns (not prefix-stable) is rejected with an error.

Input ``--dataset-path`` may be a job run dir (``*/agent/hermes-session.jsonl``
with a sibling ``*/result.json``), a single trial dir, a flat dir of session
files, a single session file, or a ``split_phases.py`` output file (passed
through as-is). Only trials with verifier ``reward == 1.0`` are kept; sessions
with no ``result.json`` are kept as-is, but a ``result.json`` without a reward
counts as a failure.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LING_REPO = "inclusionAI/Ling-3.0-tiny"
_TRUNCATED = "\n…<truncated {n} chars>"
_PASS_REWARD = 1.0
# Tool schemas Hermes sent to the model in the bundled trials (toolsets: hermes-cli).
_HERMES_TOOLS = Path(__file__).with_name("hermes_tools.json")

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
    """Normalize passing OPC Hermes sessions into SFT rows (Ling chat template)."""

    # OPC data is local JSONL, not a HF dataset; the default_loader is unused.
    del default_loader

    path = Path(dataset_path)
    if _is_row_file(path):
        return _load_row_file(path)

    drop_reasoning = bool(os.environ.get("ARENO_OPC_DROP_REASONING"))
    include_system_prompt = bool(os.environ.get("ARENO_OPC_INCLUDE_SYSTEM_PROMPT"))
    max_history_messages = _int_env("ARENO_OPC_MAX_HISTORY_MESSAGES")
    max_tool_chars = _int_env("ARENO_OPC_MAX_TOOL_CHARS")
    phase_split = bool(os.environ.get("ARENO_OPC_PHASES"))
    c_mode = bool(os.environ.get("ARENO_OPC_C_MODE"))
    max_seq_tokens = _int_env("ARENO_OPC_MAX_SEQ_TOKENS") or 32768
    tools = _load_tools()
    if c_mode:
        ignored = [name for name in ("ARENO_OPC_MAX_HISTORY_MESSAGES", "ARENO_OPC_PHASES") if os.environ.get(name)]
        if ignored:
            logger.warning("opc: C mode packs whole rollouts; ignoring %s", ", ".join(ignored))

    tokenizer = _load_tokenizer()
    records: list[dict] = []
    for session_path, result_path in _iter_session_files(dataset_path):
        if not _trial_passed(result_path):
            continue  # failed trial -> not taught
        try:
            session = _load_session(session_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"opc: cannot load {session_path}: {exc}") from exc
        messages = _chat_messages(session, include_system_prompt=include_system_prompt, drop_reasoning=drop_reasoning)
        trial_name = _trial_name(session_path)
        if c_mode:
            c_rows = _session_to_c_rows(
                messages, tokenizer, max_seq_tokens=max_seq_tokens, max_tool_chars=max_tool_chars, tools=tools
            )
            for chunk_index, row in enumerate(c_rows):
                row["source_trial"] = trial_name
                row["chunk_index"] = chunk_index
                records.append(row)
            continue
        rows = _session_to_rows(
            messages,
            tokenizer,
            max_history_messages=max_history_messages,
            max_tool_chars=max_tool_chars,
            phase_split=phase_split,
            tools=tools,
        )
        for index, row in enumerate(rows):
            record = {
                "prompt": row["prompt"],
                "response": row["response"],
                "tokens": row["tokens"],
                "prompt_mask": row["prompt_mask"],
                "loss_mask": row["loss_mask"],
                "source_trial": trial_name,
                "turn_index": index,
            }
            if phase_split:
                record["phase_index"] = row["phase_index"]
                record["phase_genre"] = row["phase_genre"]
            records.append(record)
    return records


# --- Input discovery ----------------------------------------------------------


def _iter_session_files(dataset_path: str) -> list[tuple[Path, Path | None]]:
    """Return ``(session_file, result_file_or_None)`` pairs for ``dataset_path``."""

    path = Path(dataset_path)
    if path.is_file():
        sessions = [path]
    elif path.is_dir():
        # Job-run layout first (<job>/<trial>/agent/hermes-session.jsonl), then
        # one level shallower (a single trial dir, or <dir>/<x>/hermes-session.jsonl),
        # then bare *.jsonl files sitting directly in the directory.
        sessions = (
            sorted(path.glob("*/agent/hermes-session.jsonl"))
            or sorted(path.glob("*/hermes-session.jsonl"))
            or sorted(path.glob("*.jsonl"))
        )
    else:
        sessions = []
    if not sessions:
        raise ValueError(f"opc: no hermes-session.jsonl found under {dataset_path}")
    return [(session, _result_file(session)) for session in sessions]


def _result_file(session_file: Path) -> Path | None:
    """Find the verifier verdict for a session, whatever layout it was found in."""

    # Hermes trials store <trial>/agent/hermes-session.jsonl next to <trial>/result.json.
    if session_file.parent.name == "agent":
        candidate = session_file.parent.parent / "result.json"
        if candidate.is_file():
            return candidate
    return None


def _trial_name(session_file: Path) -> str:
    if session_file.parent.name == "agent":
        return session_file.parent.parent.name
    return session_file.stem


def _trial_passed(result_path: Path | None) -> bool:
    """Standalone sessions pass; a result.json passes only with reward == 1.0."""

    if result_path is None:
        return True
    data = json.loads(result_path.read_text(encoding="utf-8"))
    reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return reward == _PASS_REWARD


def _is_row_file(path: Path) -> bool:
    """Whether ``path`` is already a JSONL of ``prompt``/``response`` rows (split_phases output)."""

    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    try:
        row = json.loads(first)
    except json.JSONDecodeError:
        return False
    return isinstance(row, dict) and "prompt" in row and "response" in row


def _load_row_file(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


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


def _load_tools(path: Path = _HERMES_TOOLS) -> list[dict[str, Any]]:
    """Load the OpenAI-style ``tools`` array Hermes sent to the model.

    Hermes sessions do not record tool schemas, but the template renders them
    into the system block, so rows must carry the same schemas the agent saw.
    ``hermes_tools.json`` holds the array captured from the Hermes build that
    produced the bundled trials (see its ``source`` / ``hermes_commit``).
    """

    tools = json.loads(path.read_text(encoding="utf-8")).get("tools")
    if not isinstance(tools, list) or not tools or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError(f"{path} must hold a non-empty `tools` list of tool schemas")
    return tools


def _int_env(name: str) -> int:
    """Read a non-negative int env var; unset/empty means 0 ("off")."""

    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


# --- Hermes -> chat-template messages -------------------------------------------


def _chat_messages(
    session: dict[str, Any], *, include_system_prompt: bool, drop_reasoning: bool
) -> list[dict[str, Any]]:
    """Convert Hermes messages into the OpenAI-style dicts ``apply_chat_template`` expects."""

    messages: list[dict[str, Any]] = []
    if include_system_prompt and session.get("system_prompt"):
        messages.append({"role": "system", "content": session["system_prompt"]})
    for message in session.get("messages", []):
        # TODO(agent): assumes Hermes marks rewound / compacted-away messages with
        # active == 0 or compacted truthy; confirm against Hermes session docs.
        if message.get("active", 1) == 0 or message.get("compacted"):
            continue
        role = message.get("role")
        if role in ("system", "user"):
            messages.append({"role": role, "content": _message_text(message)})
        elif role == "assistant":
            chat: dict[str, Any] = {"role": "assistant", "content": _message_text(message)}
            reasoning = message.get("reasoning_content")
            if reasoning and not drop_reasoning:
                chat["reasoning_content"] = reasoning
            tool_calls = _tool_calls(message)
            if tool_calls:
                chat["tool_calls"] = tool_calls
            messages.append(chat)
        elif role == "tool":
            chat = {"role": "tool", "content": _message_text(message)}
            if message.get("tool_call_id"):
                chat["tool_call_id"] = message["tool_call_id"]
            if message.get("tool_name"):
                chat["name"] = message["tool_name"]
            messages.append(chat)
    return messages


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


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize tool calls to ``{"type", "function": {"name", "arguments": dict}}``."""

    calls: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        source = function if isinstance(function, dict) else tool_call
        arguments = source.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"raw": arguments}
        call: dict[str, Any] = {
            "type": "function",
            "function": {"name": source.get("name", ""), "arguments": arguments},
        }
        if tool_call.get("id"):
            call["id"] = tool_call["id"]
        calls.append(call)
    return calls


def _has_output(message: dict[str, Any]) -> bool:
    return bool(message.get("content") or message.get("reasoning_content") or message.get("tool_calls"))


def _truncate_context(message: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Cap tool results and tool-call argument values; only ever applied to context."""

    if not max_chars:
        return message
    if message["role"] == "tool":
        return {**message, "content": _cap(message["content"], max_chars)}
    if message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            arguments = {
                key: _cap(value, max_chars) if isinstance(value, str) else value
                for key, value in call["function"]["arguments"].items()
            }
            calls.append({**call, "function": {**call["function"], "arguments": arguments}})
        return {**message, "tool_calls": calls}
    return message


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNCATED.format(n=len(text) - max_chars)


def _preamble_len(messages: list[dict[str, Any]]) -> int:
    """Number of leading messages (system + task) before the first assistant turn."""

    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            return index
    return len(messages)


# --- Template rendering -----------------------------------------------------------


def _load_tokenizer():
    """Load the Ling tokenizer (with its chat template).

    ``ARENO_OPC_TOKENIZER`` points at a local tokenizer dir (offline). Without
    it the loader auto-downloads the Ling-3.0-tiny tokenizer files (not the
    weights) from ModelScope.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("the OPC loader requires `transformers` (only the tokenizer is used)") from exc

    local = os.environ.get("ARENO_OPC_TOKENIZER", "").strip()
    if not local:
        try:
            from modelscope import snapshot_download

            local = snapshot_download(_LING_REPO, allow_file_pattern=_LING_TOKENIZER_FILES)
        except Exception as exc:  # noqa: BLE001 - surface a friendly error
            raise ValueError(
                "could not auto-download the Ling tokenizer; set ARENO_OPC_TOKENIZER=<dir> to a local copy"
            ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not load tokenizer from {local!r}: {exc}") from exc
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"tokenizer at {local!r} has no chat_template; the OPC loader renders rows with it")
    return tokenizer


def _render(
    tokenizer, messages: list[dict[str, Any]], *, add_generation_prompt: bool, tools: list[dict[str, Any]] | None
) -> str:
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _prefix_len(full: str, prefix: str) -> int:
    """Length of ``prefix`` after checking the template rendered it as a prefix of ``full``."""

    if not full.startswith(prefix):
        raise ValueError(
            "opc: the chat template is not prefix-stable (rendering more messages rewrote earlier "
            "text, e.g. it drops reasoning from past turns), so assistant turns cannot be isolated. "
            "C mode needs a prefix-stable template; use the default B mode instead."
        )
    return len(prefix)


def _strip_eos(text: str, tokenizer) -> str:
    """Drop the template's closing EOS from the human-readable ``response`` text."""

    eos = getattr(tokenizer, "eos_token", None)
    stripped = text.rstrip()
    if eos and stripped.endswith(eos):
        return stripped[: -len(eos)]
    return text


def _encode(tokenizer, text: str, spans: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int]], list[bool]]:
    """Tokenize ``text`` once and flag every token overlapping a target character span."""

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(token) for token in encoding["input_ids"]]
    offsets = [tuple(offset) for offset in encoding["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError(f"tokenizer offsets mismatch: {len(ids)} ids vs {len(offsets)} offsets")
    char_target = bytearray(len(text))
    for start, end in spans:
        char_target[start:end] = b"\x01" * (end - start)
    # A token straddling a span edge still contains model-produced characters,
    # so the model has to emit it: count it as a target.
    token_target = [start < end and any(char_target[start:end]) for start, end in offsets]
    return ids, offsets, token_target


# --- B mode -------------------------------------------------------------------------


def _session_to_rows(
    messages: list[dict[str, Any]],
    tokenizer,
    *,
    max_history_messages: int,
    max_tool_chars: int,
    phase_split: bool,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Expand a session into one (prompt, response) row per assistant turn."""

    preamble = _preamble_len(messages)
    rows: list[dict[str, Any]] = []
    current_phase = {"index": 0, "genre": None}
    for k, message in enumerate(messages):
        if message["role"] != "assistant" or not _has_output(message):
            continue
        context = messages[:k]
        # Sliding window: always keep the system + task preamble, then the most
        # recent messages, so later rows retain the task prompt.
        if max_history_messages and k > preamble + max_history_messages:
            context = context[:preamble] + context[-max_history_messages:]
        # The target turn stays verbatim; only the *context* is compressed.
        context = [_truncate_context(item, max_tool_chars) for item in context]
        prompt = _render(tokenizer, context, add_generation_prompt=True, tools=tools)
        full = _render(tokenizer, context + [message], add_generation_prompt=False, tools=tools)
        start = _prefix_len(full, prompt)
        text = full.rstrip()
        ids, _, token_target = _encode(tokenizer, text, [(start, len(text))])
        row = {
            "prompt": prompt,
            "response": _strip_eos(text[start:], tokenizer),
            "tokens": ids,
            "prompt_mask": [not target for target in token_target],
            "loss_mask": token_target,
        }
        if phase_split:
            genre = _phase_genre(message)
            if genre != current_phase["genre"]:
                current_phase["index"] += 1
                current_phase["genre"] = genre
            row["phase_index"] = current_phase["index"]
            row["phase_genre"] = genre
        rows.append(row)
    return rows


def _phase_genre(message: dict[str, Any]) -> str:
    """Classify one assistant turn into a coarse action phase by tool name only."""

    names = {call["function"]["name"] for call in message.get("tool_calls") or []}
    if not names:
        return "report"
    if "search_files" in names:
        return "explore"
    if "read_file" in names:
        return "read"
    if {"write_file", "patch"} & names:
        return "implement"
    return "run"


# --- C mode: whole-rollout packed rows (tokens + prompt/loss masks) ---------------


def _session_to_c_rows(
    messages: list[dict[str, Any]],
    tokenizer,
    *,
    max_seq_tokens: int,
    max_tool_chars: int,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Pack one trajectory into one or more encoded rows.

    The full transcript is rendered and tokenized once (fast tokenizer, with
    character offsets). Each assistant turn's span runs from the end of its
    generation prompt to the end of its rendering (closing EOS included); tokens
    overlapping a span are targets, everything else (system, task, tool
    observations) is context. Long trajectories are cut into chunks at
    assistant-turn boundaries, hard-splitting only oversized single spans, and
    every chunk starts with the system + task preamble as masked context.
    """

    # Tool results are context and may be capped; assistant turns are targets and stay verbatim.
    messages = [item if item["role"] == "assistant" else _truncate_context(item, max_tool_chars) for item in messages]
    full_text = _render(tokenizer, messages, add_generation_prompt=False, tools=tools)

    preamble = _preamble_len(messages)
    preamble_chars = 0
    if preamble:
        preamble_text = _render(tokenizer, messages[:preamble], add_generation_prompt=False, tools=tools)
        preamble_chars = _prefix_len(full_text, preamble_text)
    # Only assistant turns give stable cut points: templates merge consecutive
    # tool results into one block, so a prefix ending mid-run is not a prefix.
    boundaries = [preamble_chars]
    spans: list[tuple[int, int]] = []
    for k, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # Cut where the turn opens, never between its generation prompt and its body.
        opens = 0
        if k:
            opens = _prefix_len(full_text, _render(tokenizer, messages[:k], add_generation_prompt=False, tools=tools))
        start = _prefix_len(full_text, _render(tokenizer, messages[:k], add_generation_prompt=True, tools=tools))
        end = _prefix_len(full_text, _render(tokenizer, messages[: k + 1], add_generation_prompt=False, tools=tools))
        boundaries.extend((opens, end))
        if _has_output(message):
            spans.append((start, len(full_text[:end].rstrip())))
    boundaries = sorted(set(boundaries))

    ids, offsets, token_target = _encode(tokenizer, full_text, spans)
    token_part = [bisect.bisect_right(boundaries, start) for start, _ in offsets]

    n_head = sum(1 for _, end in offsets if end <= preamble_chars)
    budget = max_seq_tokens - n_head
    if budget <= 0:
        raise ValueError(
            f"opc: the system + task preamble alone is {n_head} tokens, over ARENO_OPC_MAX_SEQ_TOKENS={max_seq_tokens}"
        )
    head = ids[:n_head]
    rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _chunk_ranges(len(ids) - n_head, token_part[n_head:], budget):
        lo, hi = n_head + chunk_start, n_head + chunk_end
        chunk_target = token_target[lo:hi]
        if not any(chunk_target):
            continue  # no assistant output in this window -> nothing to teach
        loss_mask = [False] * n_head + chunk_target
        rows.append(
            {
                "tokens": head + ids[lo:hi],
                "prompt_mask": [not target for target in loss_mask],
                "loss_mask": loss_mask,
            }
        )
    return rows


def _chunk_ranges(n_tokens: int, token_part: list[int], max_tokens: int) -> list[tuple[int, int]]:
    """Greedily slice token indices into chunks of at most ``max_tokens``.

    Cuts land on part boundaries whenever possible; a single part longer than
    ``max_tokens`` is hard-split.
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
            ranges.append((start, run_start))  # cut before this part
            start = run_start
        while run_end - start > max_tokens:  # single oversized part
            ranges.append((start, start + max_tokens))
            start += max_tokens
    if start < n_tokens:
        ranges.append((start, n_tokens))
    return ranges
