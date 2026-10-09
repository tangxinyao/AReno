from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "ling_flash_fin" / "data"


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA_DIR / name).read_text(encoding="utf-8").splitlines()]


def test_train_rows_match_sft_prompt_response_schema():
    rows = _rows("train.jsonl")

    assert rows
    assert all(row["prompt"].strip() and row["response"].strip() for row in rows)
    assert {row["lang"] for row in rows} == {"zh", "en"}


def test_eval_questions_are_held_out_from_train():
    train_prompts = {row["prompt"] for row in _rows("train.jsonl")}
    evals = _rows("eval.jsonl")

    assert {row["lang"] for row in evals} == {"zh", "en"}
    assert all(row["reference"].strip() for row in evals)
    assert not train_prompts & {row["prompt"] for row in evals}


def test_loader_reads_only_train_jsonl_from_data_dir():
    import importlib.util

    path = DATA_DIR.parent / "dataset_loader.py"
    spec = importlib.util.spec_from_file_location("sft_ling_flash_fin_dataset_loader_for_tests", path)
    loader = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loader)
    seen = []

    def default_loader(file_path):
        seen.append(file_path)
        return [json.loads(line) for line in Path(file_path).read_text(encoding="utf-8").splitlines()]

    records = loader.load_training_dataset(str(DATA_DIR), default_loader=default_loader)

    assert seen == [str(DATA_DIR / "train.jsonl")]
    assert len(records) == len(_rows("train.jsonl"))
    assert all(set(record) == {"prompt", "response"} for record in records)
