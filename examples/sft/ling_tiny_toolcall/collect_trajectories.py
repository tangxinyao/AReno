"""Offline collector: run the LangGraph mail workflow over Ling-3.0-tiny (GPU step).

This is the report's section 3 (trajectory collection). It reads a list of mail
tasks from ``tasks.jsonl``, drives the LangGraph workflow (``mail_graph.py``) with
Ling-3.0-tiny served through an OpenAI-compatible endpoint, and writes one JSONL
row per task to ``raw_trajectories.jsonl``: the full ``messages`` transcript, the
parsed tool calls, and the Verifier's ``done`` flag. ``build_dataset.py`` then
filters and converts this into the SFT train/eval split.

This step needs a running served model and therefore a GPU -- ``areno serve`` the
Ling-3.0-tiny checkpoint first, then point ``--base-url`` at it. Without a GPU you
can still use the committed seed data (see ``build_dataset.py``), which is the
output a successful small collection run would have produced.

Example::

    areno serve --model-path inclusionai/ling-3.0-tiny --model-hub modelscope \
        --disable-thinking --tp-size 1 --world-size 1 --port 8000 &
    python collect_trajectories.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model inclusionai/ling-3.0-tiny \
        --tasks tasks.jsonl \
        --output raw_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mail_graph import MailState, OpenAIClient, build_graph
from tools import AgentEnv, parse_hermes_tool_calls


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _env_for_task(task: dict[str, Any]) -> AgentEnv:
    """Build a fresh harness per episode.

    The Gmail-login episodes are a fixed script: the simulated user's answers to
    ``clarify`` come from the task's ``decisions`` list (defaulting to the standard
    "pick google-workspace, then give the client-secret path" route). The vendored
    skills are discovered from ``skills/`` on construction.
    """

    env = AgentEnv.default()
    if task.get("decisions"):
        env.decisions = list(task["decisions"])
    return env


def run_task(
    task: dict[str, Any], *, client: OpenAIClient, build_graph: Any, **_: Any
) -> dict[str, Any]:
    """Run one task through the graph and return a raw trajectory record.

    The graph is rebuilt per task so the per-episode ``AgentEnv`` is bound as a
    build-time closure value (``AgentEnv`` is not LangGraph-checkpoint-serializable).
    """

    env = _env_for_task(task)
    graph = build_graph(client, env)
    initial = MailState(user_query=task["prompt"])
    # LangGraph invoke with a per-task thread id so the MemorySaver keeps tasks
    # isolated. The graph mutates the state in place; the returned value holds
    # the final messages and done flag.
    config = {"configurable": {"thread_id": task["task_id"]}}
    final = graph.invoke(initial, config=config)
    messages = final.messages if hasattr(final, "messages") else final["messages"]
    tool_calls = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls.append({"name": function["name"], "arguments": arguments})
        elif message.get("role") == "assistant" and message.get("content"):
            # also capture any Hermes calls embedded in raw content
            tool_calls.extend(parse_hermes_tool_calls(message["content"]))
    return {
        "task_id": task["task_id"],
        "lang": task.get("lang", "en"),
        "prompt": task["prompt"],
        "done": bool(final.done) if hasattr(final, "done") else bool(final.get("done")),
        "tool_calls": tool_calls,
        "messages": messages,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect mail-agent trajectories from a served Ling-3.0-tiny.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1 (areno serve).",
    )
    parser.add_argument(
        "--model",
        default="inclusionai/ling-3.0-tiny",
        help="Model id passed to the endpoint; defaults to the Ling-3.0-tiny id.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).resolve().parent / "tasks.jsonl",
        help="JSONL of mail tasks (task_id, prompt, lang, optional emails).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "raw_trajectories.jsonl",
        help="Output JSONL of raw trajectories for build_dataset.py.",
    )
    args = parser.parse_args(argv)

    client = OpenAIClient(args.base_url, args.model)
    tasks = _load_jsonl(args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            try:
                record = run_task(task, client=client, build_graph=build_graph)
            except Exception as exc:  # keep collecting on a single bad task
                record = {
                    "task_id": task["task_id"],
                    "lang": task.get("lang", "en"),
                    "prompt": task["prompt"],
                    "done": False,
                    "tool_calls": [],
                    "messages": [],
                    "error": str(exc),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{record['task_id']}] done={record['done']} calls={len(record['tool_calls'])}")
    print(f"wrote {len(tasks)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
