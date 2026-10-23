"""Agent entrypoint for the STS2 ReAct example.

Each episode is a plain-text ReAct loop: the policy writes a Thought and an
Action, the simulator applies the action, and the next Observation is appended
to the conversation. Requests go to the OpenAI-compatible rollout endpoint, so
no tool schema is sent.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import MAX_INVALID_ACTIONS, MAX_TURNS, SYSTEM_PROMPT, Episode, Sts2ReActEnv, parse_action  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)



async def run_agent(ctx, batch):
    """Run one ReAct episode per expanded prompt/sample item."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("The STS2 ReAct example requires `openai` and `httpx`. Install `openai`.") from exc

    items = list(batch.iter_samples())
    logger.info("STS2 ReAct agent start episodes=%d max_running_prompts=%d", len(items), ctx.max_running_prompts)
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        episodes = await asyncio.gather(*(_run_episode(client, item) for item in items))
        return AgentTrajectory(turns=[turn for turns in episodes for turn in turns])
    finally:
        await client.close()
        await http_client.aclose()


async def _run_episode(client: Any, item: Any) -> list[AgentTrajectoryTurn]:
    record = item.record
    episode = Episode(
        encounter=str(record["encounter"]),
        seed=int(record["seed"]),
        ascension=int(record.get("ascension", 0)),
    )
    env = Sts2ReActEnv(episode)
    turns: list[AgentTrajectoryTurn] = []
    try:
        observation = await asyncio.to_thread(env.reset)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation},
        ]
        invalid = 0
        for _ in range(MAX_TURNS):
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                stream=False,
            )
            turns.append(AgentTrajectoryTurn(item=item, messages=list(messages), response=response))
            text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": text})

            action = parse_action(text)
            if action is None or action not in env.legal_ids():
                invalid += 1
                if invalid >= MAX_INVALID_ACTIONS:
                    break
                problem = (
                    "The reply had no Action line."
                    if action is None
                    else f"{action!r} is not in the Legal actions list."
                )
                messages.append({"role": "user", "content": f"Observation: {problem}\n{env.observation()}"})
                continue

            observation = await asyncio.to_thread(env.step, action)
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
            if env.done:
                break
        return turns
    finally:
        env.close()
