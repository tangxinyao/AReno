from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "ling_tiny_toolcall"
DATA_DIR = EXAMPLE_DIR / "data"


def _load_module(name: str, relpath: str):
    """Import an example module and register it so dataclass annotation lookups work.

    Registering the module under its spec name in ``sys.modules`` before executing
    it lets Python 3.14's dataclass machinery resolve string/ClassVar annotations
    (it does ``sys.modules.get(cls.__module__).__dict__``), which otherwise fails
    when the module is created without registration.
    """

    path = EXAMPLE_DIR / relpath
    sys.path.insert(0, str(EXAMPLE_DIR))
    mod_name = f"ling_tiny_toolcall_{name}_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    finally:
        if str(EXAMPLE_DIR) in sys.path:
            sys.path.remove(str(EXAMPLE_DIR))


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]


_TRAJECTORY_KEYS = {"prompt", "response", "lang", "trajectory_id", "step"}
_EVAL_KEYS = {"prompt", "reference", "lang", "task_id"}


def test_train_rows_match_sft_prompt_response_schema():
    rows = _rows("train.jsonl")

    assert rows
    # tool-call rows carry the extra bookkeeping keys; the loader still narrows to prompt/response
    assert all(set(row) == _TRAJECTORY_KEYS for row in rows)
    assert all(row["prompt"].strip() and row["response"].strip() for row in rows)
    assert {row["lang"] for row in rows} == {"zh", "en"}
    assert all(row["step"] >= 0 for row in rows)
    assert all(row["trajectory_id"] for row in rows)


def test_train_tool_call_responses_are_valid_hermes():
    """Every response that contains a tool-call block is legal Hermes for a known tool."""

    build = _load_module("build_dataset", "build_dataset.py")

    tool_call_rows = [r for r in _rows("train.jsonl") if build.TOOL_CALL_RE.search(r["response"])]
    assert tool_call_rows, "expected at least one tool-call training row"
    # each tool-call response must pass the format guard
    for row in tool_call_rows:
        calls = build.assert_valid_tool_call(row["response"])
        assert calls, "format guard accepted a block with no valid calls"
        for call in calls:
            assert call["name"] in {
                t["function"]["name"] for t in build.TOOLS
            }, f"unknown tool {call['name']}"
            assert isinstance(call["arguments"], dict)


def test_train_has_no_tool_negatives():
    """No-tool rows teach the model not to call when it shouldn't (over-calling risk)."""

    build = _load_module("build_dataset", "build_dataset.py")
    rows = _rows("train.jsonl")
    no_tool = [r for r in rows if not build.TOOL_CALL_RE.search(r["response"])]
    assert no_tool, "'no-tool' negative rows are missing; over-calling guard is gone"
    for row in no_tool:
        # a negative must not contain a tool-call block by construction
        assert not build.TOOL_CALL_RE.search(row["response"])


def test_eval_questions_are_held_out_from_train():
    train_prompts = {row["prompt"] for row in _rows("train.jsonl")}
    evals = _rows("eval.jsonl")

    assert {row["lang"] for row in evals} == {"zh", "en"}
    assert all(set(row) == _EVAL_KEYS for row in evals)
    assert all(row["reference"].strip() for row in evals)
    # structural no-overlap: eval task ids are disjoint from train trajectory ids
    train_ids = {row["trajectory_id"] for row in _rows("train.jsonl")}
    assert not train_ids & {row["task_id"] for row in evals}
    # and the raw prompts must never coincide either
    assert not train_prompts & {row["prompt"] for row in evals}


def test_format_guard_accepts_and_rejects():
    build = _load_module("build_dataset", "build_dataset.py")
    tools = _load_module("tools", "tools.py")

    # well-formed block parses cleanly
    good = tools.render_hermes_tool_call("list_emails", {"folder": "inbox", "limit": 10})
    assert build.TOOL_CALL_RE.search(good)
    # unknown tool name -> guard raises
    bad_tool = (
        f"{tools.TOOL_CALL_OPEN}\n"
        '{"name": "not_a_tool", "arguments": {}}\n'
        f"{tools.TOOL_CALL_CLOSE}"
    )
    import pytest  # local import keeps the module top-level signal-free

    with pytest.raises(ValueError):
        build.assert_valid_tool_call(bad_tool)
    # malformed JSON -> guard raises
    bad_json = f"{tools.TOOL_CALL_OPEN}\n{{not json}}\n{tools.TOOL_CALL_CLOSE}"
    with pytest.raises(ValueError):
        build.assert_valid_tool_call(bad_json)


def test_loader_reads_only_train_jsonl_from_data_dir():
    loader = _load_module("dataset_loader", "dataset_loader.py")
    seen: list[str] = []

    def default_loader(file_path):
        seen.append(file_path)
        return [json.loads(line) for line in Path(file_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    records = loader.load_training_dataset(str(DATA_DIR), default_loader=default_loader)

    assert seen == [str(DATA_DIR / "train.jsonl")]
    assert len(records) == len(_rows("train.jsonl"))
    assert all(set(record) == {"prompt", "response"} for record in records)


def test_assistant_message_prefers_content_and_falls_back_to_reasoning():
    """A thinking-only prose turn must not lose its answer (areno --disable-thinking).

    ``reasoning_content`` is only a fallback: tool-call turns keep an empty content
    (their payload is the call), and a non-empty ``content`` always wins.
    """

    graph = _load_module("mail_graph", "mail_graph.py")

    def response(message: dict) -> dict:
        return {"choices": [{"message": message}]}

    # prose turn: content empty, answer only in reasoning_content -> fallback
    prose = graph._assistant_message_from_response(
        response({"role": "assistant", "content": "", "reasoning_content": "Sure!"})
    )
    assert prose == {"role": "assistant", "content": "Sure!"}

    # non-empty content wins even when reasoning_content is present
    both = graph._assistant_message_from_response(
        response({"role": "assistant", "content": "answer", "reasoning_content": "thinking"})
    )
    assert both["content"] == "answer"

    # tool-call turn: content stays empty, reasoning_content must NOT pollute it
    calls = [{"id": "c1", "function": {"name": "list_emails", "arguments": "{}"}}]
    tool_turn = graph._assistant_message_from_response(
        response({"role": "assistant", "content": "", "reasoning_content": "thinking", "tool_calls": calls})
    )
    assert tool_turn["content"] == ""
    assert tool_turn["tool_calls"] == calls


def test_seed_build_is_deterministic_and_byte_identical():
    """Re-running build_dataset.py from the embedded seeds must leave data/ unchanged."""

    build = _load_module("build_dataset", "build_dataset.py")
    train, evals = build.build(
        trajectories=build.SEED_TRAJECTORIES,
        no_tool_tasks=build.SEED_NO_TOOL_TASKS,
        eval_tasks=build.SEED_EVAL_TASKS,
    )
    re_train = [json.dumps(r, ensure_ascii=False) for r in train]
    re_eval = [json.dumps(r, ensure_ascii=False) for r in evals]
    disk_train = [json.dumps(r, ensure_ascii=False) for r in _rows("train.jsonl")]
    disk_eval = [json.dumps(r, ensure_ascii=False) for r in _rows("eval.jsonl")]
    assert re_train == disk_train, "seed build drifted from committed train.jsonl"
    assert re_eval == disk_eval, "seed build drifted from committed eval.jsonl"
