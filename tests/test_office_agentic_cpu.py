from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

OFFICE_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "office"
sys.path.insert(0, str(OFFICE_EXAMPLE))

from office_env import (  # noqa: E402
    DEFAULT_TEMPLATE_DIR,
    TASK_TYPES,
    build_record,
    load_template_catalog,
    parse_spec,
    validate_record,
)
from office_tools import TOOLS, OfficeWorkspace  # noqa: E402
from reward import reward_fn  # noqa: E402
from run_agent import _context_budget, _system_prompt  # noqa: E402


@pytest.mark.parametrize(
    "task",
    ["excel_chart_dashboard", "excel_formula_conditional", "excel_write_data"],
)
def test_office_excel_oracles_score_one(task: str):
    pytest.importorskip("openpyxl")
    record = build_record(task, seed=2026, index=1)
    assert record["output_file"] in record["prompt"]
    assert validate_record(record) == {"score": 1.0, "issues": []}


@pytest.mark.parametrize("task", ["office_mixed_report", "word_create_report", "word_format_edit"])
def test_office_word_oracles_score_one(task: str):
    pytest.importorskip("openpyxl")
    pytest.importorskip("docx")
    record = build_record(task, seed=2026, index=1)
    assert record["output_file"] in record["prompt"]
    assert validate_record(record) == {"score": 1.0, "issues": []}


@pytest.mark.parametrize(
    "task",
    ["excel_read_analyze", "word_convert_pdf", "word_extract_stat", "word_table_doc"],
)
def test_remaining_office_oracles_score_one(task: str):
    pytest.importorskip("openpyxl")
    pytest.importorskip("docx")
    if task == "word_convert_pdf":
        pytest.importorskip("pypdf")
    record = build_record(task, seed=2026, index=1)
    assert record["output_file"] in record["prompt"]
    assert validate_record(record) == {"score": 1.0, "issues": []}


def test_office_generator_records_use_dynamic_10k_budget():
    records = [build_record(task, seed=2026 + index, index=index + 1) for index, task in enumerate(TASK_TYPES)]
    assert {record["context_budget"] for record in records} == {10_000}
    assert {record["max_turns"] for record in records} == {12}
    assert all(record["output_file"] in record["prompt"] for record in records)
    assert set(TASK_TYPES) == {
        "excel_chart_dashboard",
        "excel_formula_conditional",
        "excel_read_analyze",
        "excel_write_data",
        "office_mixed_report",
        "word_convert_pdf",
        "word_create_report",
        "word_extract_stat",
        "word_format_edit",
        "word_table_doc",
    }


def test_office_templates_generate_free_text_fields_instead_of_value_enums():
    catalog = load_template_catalog(DEFAULT_TEMPLATE_DIR)
    assert all(isinstance(field, dict) for template in catalog.values() for field in template["fields"].values())
    records = [build_record("excel_chart_dashboard", seed=seed, index=seed) for seed in range(40)]
    specs = [parse_spec(record) for record in records]
    assert len({spec["input_file"] for spec in specs}) == 40
    assert len({spec["output_file"] for spec in specs}) == 40
    assert len({spec["sheet_name"] for spec in specs}) == 40
    assert all(spec["input_file"] in record["prompt"] for spec, record in zip(specs, records, strict=True))


def test_office_prompt_grammar_preserves_required_files_and_varies_layout():
    catalog = load_template_catalog(DEFAULT_TEMPLATE_DIR)
    records = [
        build_record(task, seed=10_000 + index, index=index) for index, task in enumerate(TASK_TYPES * 12, start=1)
    ]
    for record in records:
        spec = parse_spec(record)
        fields = catalog[record["task"]]["fields"]
        for name, definition in fields.items():
            if definition["generator"] == "filename":
                assert spec[name] in record["prompt"]
    layout_markers = {
        "bullets"
        if "\n- " in record["prompt"]
        else "numbered"
        if "\n1. " in record["prompt"]
        else "bracketed"
        if "[任务]" in record["prompt"]
        else "goal"
        if "交付目标：" in record["prompt"]
        else "chain"
        if "操作链：" in record["prompt"]
        else "prose"
        for record in records
    }
    assert len(layout_markers) == 6


def test_office_runtime_token_budget_env_overrides_record(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARENO_OFFICE_TOKEN_BUDGET", "8000")
    assert _context_budget({"context_budget": 10_000}) == 8000


def test_office_system_prompt_binds_the_actual_isolated_workspace():
    pytest.importorskip("openpyxl")
    record = build_record("excel_chart_dashboard", seed=2026, index=1)
    workspace = OfficeWorkspace.from_record(record)
    try:
        prompt = _system_prompt(workspace)
    finally:
        workspace.close()
    assert workspace.root.as_posix() in prompt
    assert "current working directory" in prompt
    assert "Use relative paths" in prompt
    assert "Do not inspect the host's /workspace" in prompt


def test_office_runtime_token_budget_defaults_to_record(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ARENO_OFFICE_TOKEN_BUDGET", raising=False)
    assert _context_budget({"context_budget": 9_000}) == 9_000


@pytest.mark.parametrize("value", ["0", "-1"])
def test_office_runtime_token_budget_rejects_non_positive_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("ARENO_OFFICE_TOKEN_BUDGET", value)
    with pytest.raises(ValueError, match="ARENO_OFFICE_TOKEN_BUDGET"):
        _context_budget({"context_budget": 10_000})


def test_office_tools_match_successful_hermes_trajectory_schemas():
    functions = {tool["function"]["name"]: tool["function"] for tool in TOOLS}
    assert set(functions) == {"execute_code", "read_file", "skill_view", "write_file"}
    assert functions["execute_code"]["parameters"]["required"] == ["code"]
    assert functions["read_file"]["parameters"]["required"] == ["path"]
    assert set(functions["skill_view"]["parameters"]["properties"]) == {"name"}
    assert set(functions["write_file"]["parameters"]["properties"]) == {"path", "content"}
    assert "relative" in functions["execute_code"]["description"].lower()
    assert "DOCX" in functions["read_file"]["description"]
    assert len(json.dumps(TOOLS, ensure_ascii=False)) < 3_000


def test_execute_code_supports_hermes_tools_compatibility_import():
    pytest.importorskip("openpyxl")
    record = build_record("excel_chart_dashboard", seed=2026, index=1)
    input_file = parse_spec(record)["input_file"]
    workspace = OfficeWorkspace.from_record(record)
    try:
        result = workspace.execute_code(
            f"from hermes_tools import read_file\nresult = read_file('{input_file}')\nprint(result['total_lines'])\n"
        )
    finally:
        workspace.close()
    assert result["returncode"] == 0
    assert int(result["stdout"].strip()) > 1


def test_office_reward_uses_final_artifact_and_discounts_non_ok_turns():
    record = SimpleNamespace(
        tool_results=[
            {"content": json.dumps({"artifact_score": 0.5, "artifact_issues": ["incomplete"]})},
            {"content": json.dumps({"artifact_score": 1.0, "artifact_issues": []})},
        ]
    )
    assert reward_fn(record) == pytest.approx(0.95)


def test_office_reward_does_not_keep_a_better_intermediate_artifact():
    record = SimpleNamespace(
        tool_results=[
            {"content": json.dumps({"artifact_score": 1.0, "artifact_issues": []})},
            {"content": json.dumps({"artifact_score": 0.5, "artifact_issues": ["regressed"]})},
        ]
    )
    assert reward_fn(record) == pytest.approx(0.5 * 0.95)


def test_office_reward_preserves_bounded_progress_without_artifact():
    pytest.importorskip("openpyxl")
    record = build_record("excel_chart_dashboard", seed=2026, index=1)
    workspace = OfficeWorkspace.from_record(record)
    try:
        prepared = workspace.skill_view("xlsx")
        executed = workspace.execute_code("print('prepared')")
    finally:
        workspace.close()
    assert prepared["progress_score"] == 0.05
    assert executed["progress_score"] == 0.10
    assert executed["artifact_score"] == 0.0
    trajectory = SimpleNamespace(tool_results=[{"content": json.dumps(executed)}])
    assert reward_fn(trajectory) == pytest.approx(0.10 * 0.95)


def test_context_count_uses_tokenizer_chat_template():
    pytest.importorskip("torch")
    from run_agent import _context_token_count

    class Tokenizer:
        chat_template = "test"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"] == TOOLS
            assert kwargs["add_generation_prompt"] is True
            return [1] * (len(messages) + 7)

    assert _context_token_count(Tokenizer(), [{"role": "user", "content": "x"}], tools=TOOLS) == 8


def test_assistant_history_uses_structured_tool_arguments():
    pytest.importorskip("torch")
    from run_agent import _assistant_message

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(name="read_file", arguments='{"path":"sales.xlsx"}'),
                        )
                    ],
                )
            )
        ]
    )
    message = _assistant_message(response)
    assert message["tool_calls"][0]["function"]["arguments"] == {"path": "sales.xlsx"}


def test_rollout_session_exposes_training_tokenizer():
    pytest.importorskip("torch")
    from areno.api.agentic import RolloutSession

    tokenizer = object()
    session = RolloutSession.__new__(RolloutSession)
    session._trainer = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    assert session.get_tokenizer() is tokenizer
