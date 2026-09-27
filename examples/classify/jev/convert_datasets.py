"""Convert public typed-decision datasets into JevForge records.

Output is a directory of `<split>.jsonl` files that `dataset_loader.load_questions`
(and therefore `train.py` / `tmp.sh`) reads directly. State objects are rendered
the way `serve_decisions.py` renders a JSON `state` from a request
(`json.dumps(..., ensure_ascii=False, sort_keys=True)`), so training text matches
what the decisions API feeds the model.

Open-Jev v1.1 (ModelScope `ZefanCai/Open-Jev-v1.1`, config
`community-hard-mix-v2-redistributable`); one row is one question:

    python examples/classify/jev/convert_datasets.py open-jev \
        --src ~/data/open-jev-v1.1 --out ~/data/jev-records/open-jev-v1.1

  splits: train, validation -> dev, calibration, test, ood. Choice options are
  bare ids in this release, so each option doubles as its own description.

typed-decisions (`LocalLLaMA/typed-decisions`, config `all`); one row is one
case with several questions over a shared state:

    python examples/classify/jev/convert_datasets.py typed-decisions \
        --src ~/data/typed-decisions --out ~/data/jev-records/typed-decisions

  splits: train, test. `source_group` is the workflow name.

Open-Jev v1 (ModelScope `ZefanCai/Open-Jev`, all configs); choice options are
`id: description` strings and become `{id: description}` criteria:

    python examples/classify/jev/convert_datasets.py open-jev-v1 \
        --src ~/data/open-jev-v1 --out ~/data/jev-records/open-jev-v1

JevEmbed-Data (ModelScope `HIT-TMG/JevEmbed-Data`), whitelisted sources only
(see `JEVEMBED_SOURCES`); `test` (the source validation split) becomes `dev`:

    python examples/classify/jev/convert_datasets.py jevembed \
        --src ~/data/jevembed-data --out ~/data/jev-records/jevembed

Open-Jev / typed-decisions read `<src>/<split>.parquet`; Open-Jev v1 reads
`<src>/data/<config>/<split>-*.parquet`; JevEmbed reads `<src>/*.parquet`.
All parquet is streamed in batches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import candidate_ids  # noqa: E402

OPEN_JEV_SPLITS = {"train": "train", "validation": "dev", "calibration": "calibration", "test": "test", "ood": "ood"}
TYPED_DECISIONS_SPLITS = {"train": "train", "test": "test"}
# JevEmbed-Data sources kept (license per the dataset's own `license` column).
# Excluded: system_one_270m (shares criteria text with typed-decisions test),
# pku_saferlhf / beavertails (CC-BY-NC), nectar (non-compete clause),
# openjev / openjev_v11 (duplicates of Open-Jev), helpsteer2 (Jevals score eval set).
JEVEMBED_SOURCES = {
    "procedural_jev",
    "domain_controls_v1",
    "domain_controls_v2",
    "systemone_phase2_allocator",
    "veji_v2_numeric_temporal",
    "sarge_v3",
    "hh_helpful",
    "hh_harmless",
    "lmsys",
    "helpsteer1",
    "helpsteer3",
    "helpsteer3_feedback",
    "prometheus",
    "ultrafeedback",
}
BATCH_ROWS = 4096


def render_state(state) -> str:
    """Same text `serve_decisions.render_state` produces for a JSON state."""

    if isinstance(state, str):
        try:
            state = json.loads(state)
        except ValueError:
            return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def normalized(values: dict[str, float]) -> dict[str, float]:
    """Rescale to sum 1 (sources round to ~6 decimals)."""

    total = sum(float(v) for v in values.values())
    if total <= 0:
        raise ValueError(f"target has no probability mass: {values}")
    return {key: float(value) / total for key, value in values.items()}


def split_described_options(options: list[str]) -> dict[str, str] | None:
    """`["id: description", ...]` -> `{id: description}` when every option parses and ids are unique."""

    pairs = []
    for option in options:
        head, sep, tail = option.partition(": ")
        if not sep or not head.strip() or not tail.strip() or " " in head.strip():
            return None
        pairs.append((head.strip(), tail.strip()))
    ids = [head for head, _ in pairs]
    return dict(pairs) if len(set(ids)) == len(ids) else None


def open_jev_question(row: dict, *, split_descriptions: bool = False) -> tuple[dict, dict[str, float]]:
    """Map one Open-Jev row to a Jev question and its target distribution."""

    kind = row["kind"]
    options = [str(option) for option in row["options"]]
    target = [float(value) for value in row["target"]]
    if len(options) != len(target):
        raise ValueError(f"{row['id']}: {len(options)} options vs {len(target)} targets")
    question = {"type": kind, "instructions": str(row["question"])}
    if kind == "choice":
        described = split_described_options(options) if split_descriptions else None
        if described is not None:
            question["criteria"] = described
            return question, normalized(dict(zip(described, target, strict=True)))
        question["criteria"] = {option: option for option in options}
        return question, normalized(dict(zip(options, target, strict=True)))
    if kind == "noul":
        if options != ["no", "yes"]:
            raise ValueError(f"{row['id']}: unexpected noul options {options}")
        return question, normalized({"false": target[0], "true": target[1]})
    if kind == "score":
        question["criteria"] = options
        return question, normalized({str(index): value for index, value in enumerate(target)})
    raise ValueError(f"{row['id']}: unknown kind {kind!r}")


def convert_open_jev(src: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    columns = ["id", "group_id", "source", "kind", "question", "options", "target", "state_json", "metadata_json"]
    counts = {}
    for source_split, split in OPEN_JEV_SPLITS.items():
        rows = pq.read_table(src / f"{source_split}.parquet", columns=columns).to_pylist()
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                question, target = open_jev_question(row)
                basis = (json.loads(row["metadata_json"] or "{}") or {}).get("target_basis") or "unspecified"
                record = {
                    "id": row["id"],
                    "split": split,
                    "source_group": row["group_id"],
                    "request": {"state": render_state(row["state_json"]), "questions": {"q": question}},
                    "targets": {"q": target},
                    "target_kinds": {"q": str(basis)},
                    "meta": {"source": row["source"], "dataset": "ZefanCai/Open-Jev-v1.1"},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def open_jev_record(row: dict, split: str, dataset: str, *, split_descriptions: bool) -> dict:
    question, target = open_jev_question(row, split_descriptions=split_descriptions)
    basis = (json.loads(row["metadata_json"] or "{}") or {}).get("target_basis") or "unspecified"
    return {
        "id": row["id"],
        "split": split,
        "source_group": row["group_id"],
        "request": {"state": render_state(row["state_json"]), "questions": {"q": question}},
        "targets": {"q": target},
        "target_kinds": {"q": str(basis)},
        "meta": {"source": row["source"], "dataset": dataset},
    }


def convert_open_jev_v1(src: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    columns = ["id", "group_id", "source", "kind", "question", "options", "target", "state_json", "metadata_json"]
    counts: dict[str, int] = {}
    handles = {split: (out / f"{split}.jsonl").open("w", encoding="utf-8") for split in OPEN_JEV_SPLITS.values()}
    try:
        for config in sorted(path for path in (src / "data").iterdir() if path.is_dir()):
            for source_split, split in OPEN_JEV_SPLITS.items():
                for file in sorted(config.glob(f"{source_split}-*.parquet")):
                    for batch in pq.ParquetFile(file).iter_batches(batch_size=BATCH_ROWS, columns=columns):
                        for row in batch.to_pylist():
                            record = open_jev_record(row, split, "ZefanCai/Open-Jev", split_descriptions=True)
                            record["meta"]["config"] = config.name
                            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                            counts[split] = counts.get(split, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def jevembed_question(request: dict, answer: dict) -> tuple[dict, dict[str, float]]:
    """Map one JevEmbed `decision` question + answer to a Jev question and target."""

    q = request["questions"]["decision"]
    kind = q["type"]
    question = {"type": kind, "instructions": str(q["instructions"]).strip()}
    criteria = q.get("criteria")
    if kind == "choice":
        keys = [str(key) for key in criteria]
        question["criteria"] = {
            str(k): (str(v).strip() if v is not None and str(v).strip() else str(k)) for k, v in criteria.items()
        }
        if "probabilities" in answer:
            probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
            return question, normalized({key: probs.get(key, 0.0) for key in keys})
        chosen = str(answer["choice"])
        if chosen not in keys:
            raise ValueError(f"choice {chosen!r} not in criteria")
        return question, {key: float(key == chosen) for key in keys}
    if kind == "noul":
        if isinstance(criteria, dict):
            described = {
                k: str(v).strip() for k, v in criteria.items() if k in ("true", "false") and v and str(v).strip()
            }
            if described:
                question["criteria"] = described
        p = float(answer["noul"])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"noul probability {p} out of range")
        return question, {"false": 1.0 - p, "true": p}
    if kind == "score":
        levels = [str(level).strip() for level in criteria]
        question["criteria"] = levels
        if "probabilities" in answer:
            probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
            return question, normalized({str(i): probs.get(str(i), 0.0) for i in range(len(levels))})
        level = int(answer["level"])
        if not 0 <= level < len(levels):
            raise ValueError(f"level {level} outside {len(levels)} levels")
        return question, {str(i): float(i == level) for i in range(len(levels))}
    raise ValueError(f"unknown type {kind!r}")


def convert_jevembed(src: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    columns = ["id", "group", "source", "license", "original_split", "request_json", "answers_json"]
    counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    handles = {split: (out / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "dev")}
    try:
        for file in sorted(src.glob("*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=BATCH_ROWS, columns=columns):
                for row in batch.to_pylist():
                    if row["source"] not in JEVEMBED_SOURCES:
                        continue
                    split = "train" if row["original_split"] == "train" else "dev"
                    try:
                        request = json.loads(row["request_json"])
                        answers = json.loads(row["answers_json"])
                        question, target = jevembed_question(request, answers.get("decision", answers))
                        if set(target) != set(candidate_ids(question)):
                            raise ValueError("target keys do not match candidates")
                    except (KeyError, TypeError, ValueError) as exc:
                        key = f"{row['source']}: {type(exc).__name__}"
                        skipped[key] = skipped.get(key, 0) + 1
                        continue
                    record = {
                        "id": f"jevembed/{row['id']}",
                        "split": split,
                        "source_group": f"jevembed/{row['group']}",
                        "request": {"state": render_state(request.get("state", "")), "questions": {"q": question}},
                        "targets": {"q": target},
                        "target_kinds": {"q": "jevembed"},
                        "meta": {
                            "source": row["source"],
                            "dataset": "HIT-TMG/JevEmbed-Data",
                            "license": row["license"],
                        },
                    }
                    handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[split] = counts.get(split, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    if skipped:
        counts["skipped"] = sum(skipped.values())
        print("skipped:", json.dumps(skipped))
    return counts


def convert_typed_decisions(src: Path, out: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    counts = {}
    for source_split, split in TYPED_DECISIONS_SPLITS.items():
        rows = pq.read_table(src / f"{source_split}.parquet", columns=["id", "workflow", "state", "questions", "gold"])
        rows = rows.to_pylist()
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                questions = json.loads(row["questions"])
                gold = json.loads(row["gold"])
                record = {
                    "id": row["id"],
                    "split": split,
                    "source_group": row["workflow"],
                    "request": {"state": render_state(row["state"]), "questions": questions},
                    "targets": {qid: normalized(gold[qid]["probabilities"]) for qid in questions},
                    "target_kinds": {qid: "teacher_mean" for qid in questions},
                    "meta": {"dataset": "LocalLLaMA/typed-decisions", "workflow": row["workflow"]},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["open-jev", "open-jev-v1", "jevembed", "typed-decisions"])
    parser.add_argument("--src", required=True, help="directory holding <split>.parquet")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    convert = {
        "open-jev": convert_open_jev,
        "open-jev-v1": convert_open_jev_v1,
        "jevembed": convert_jevembed,
        "typed-decisions": convert_typed_decisions,
    }[args.dataset]
    counts = convert(src, out)
    (out / "manifest.json").write_text(
        json.dumps({"dataset": args.dataset, "source": str(src), "records": counts}, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
