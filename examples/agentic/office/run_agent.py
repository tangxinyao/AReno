"""Native Office agent loop with a dynamic 10k context budget."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn
from areno.api.tokenizer import apply_chat_template_with_options, normalize_token_ids

sys.path.insert(0, str(Path(__file__).resolve().parent))
from office_tools import TOOLS, OfficeWorkspace, decode_tool_arguments, run_tool  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = """You are an Office artifact agent in an isolated workspace.
The workspace root for this task is {workspace}. All input files named by the user are already directly inside that
root, and execute_code starts with that root as its current working directory. Use relative paths such as
'input.xlsx' or '.' for every tool and file operation. Do not inspect the host's /workspace directory or search
outside this task root. Use the available tools to inspect inputs, create the exact requested output, and verify it.
openpyxl, python-docx, and headless LibreOffice are installed. Do not install packages, use the network, or write
outside the workspace. Create the file before answering."""

FINAL_PROMPT = "Stop using tools. State the output filename and whether the latest artifact_issues is empty."
TOKEN_BUDGET_ENV = "ARENO_OFFICE_TOKEN_BUDGET"


async def run_agent(ctx, batch) -> AgentTrajectory:
    """Run Office episodes until solved or the next turn would approach 10k tokens."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("The Office agentic demo requires `openai` and `httpx`.") from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    workspaces: list[OfficeWorkspace] = []
    try:
        # python-docx/lxml and workbook ZIP codecs have native process state.
        # Keep fixture/grader work on this thread instead of tearing down a
        # many-thread executor after every rollout batch.
        workspaces = [OfficeWorkspace.from_record(dict(item.record)) for item in items]
        grouped = await asyncio.gather(
            *(
                _run_episode(item=item, workspace=workspace, client=client, tokenizer=ctx.get_tokenizer())
                for item, workspace in zip(items, workspaces, strict=True)
            )
        )
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        for workspace in workspaces:
            workspace.close()
        await client.close()


async def _run_episode(item, workspace: OfficeWorkspace, client, tokenizer) -> list[AgentTrajectoryTurn]:
    record = dict(item.record)
    context_budget = _context_budget(record)
    generation_reserve = int(record.get("generation_reserve", 768))
    max_turns = int(record.get("max_turns", 12))
    messages = [{"role": "system", "content": _system_prompt(workspace)}, {"role": "user", "content": item.prompt}]
    turns = []
    finish_next = False

    for turn_index in range(max_turns):
        must_finish = finish_next or turn_index == max_turns - 1
        request_messages = [*messages, {"role": "user", "content": FINAL_PROMPT}] if must_finish else list(messages)
        request_tools = [] if must_finish else TOOLS
        context_tokens = _context_token_count(tokenizer, request_messages, tools=request_tools)
        remaining_tokens = context_budget - context_tokens
        if remaining_tokens <= 0:
            logger.warning(
                "Office context exhausted task=%s turn=%d context_tokens=%d budget=%d",
                record.get("id"),
                turn_index + 1,
                context_tokens,
                context_budget,
            )
            break
        kwargs = {
            "model": "policy",
            "messages": request_messages,
            "stream": False,
            "max_tokens": min(256 if must_finish else generation_reserve, remaining_tokens),
        }
        if not must_finish:
            kwargs.update({"tools": TOOLS, "tool_choice": "auto"})
        response = await client.chat.completions.create(**kwargs)
        turn = AgentTrajectoryTurn(
            item=item,
            messages=request_messages,
            response=response,
            tools=[] if must_finish else TOOLS,
        )
        turns.append(turn)
        if must_finish:
            break

        assistant = _assistant_message(response)
        messages.append(assistant)
        call = _first_tool_call(assistant)
        if call is None:
            break
        result = run_tool(
            workspace,
            call["function"]["name"],
            decode_tool_arguments(call["function"].get("arguments")),
        )
        content = json.dumps(result, ensure_ascii=False, sort_keys=True)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": content,
            }
        )
        solved = float(result.get("artifact_score", 0.0)) >= 1.0
        next_context_tokens = _context_token_count(tokenizer, messages, tools=TOOLS)
        finish_next = solved or next_context_tokens >= context_budget - generation_reserve
        logger.debug(
            "Office turn task=%s turn=%d context_tokens=%d solved=%s",
            record.get("id"),
            turn_index + 1,
            next_context_tokens,
            solved,
        )
    return turns


def _system_prompt(workspace: OfficeWorkspace) -> str:
    """Bind the model-visible prompt to this episode's actual workspace."""

    return SYSTEM_PROMPT.format(workspace=workspace.root.as_posix())


def _context_budget(record: dict[str, Any]) -> int:
    """Resolve the runtime context budget, allowing an environment override."""

    raw_budget = os.getenv(TOKEN_BUDGET_ENV)
    budget = int(raw_budget) if raw_budget is not None else int(record.get("context_budget", 10_000))
    if budget < 1:
        raise ValueError(f"{TOKEN_BUDGET_ENV} must be a positive integer")
    return budget


def _assistant_message(response) -> dict[str, Any]:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": decode_tool_arguments(call.function.arguments),
                },
            }
            for call in (message.tool_calls or [])[:1]
        ],
    }


def _first_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    return calls[0] if calls else None


def _context_token_count(tokenizer, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]]) -> int:
    """Count the exact next-request tokens with the rollout tokenizer."""

    encoded = apply_chat_template_with_options(
        tokenizer,
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(normalize_token_ids(encoded))
