"""Generate reproducible Office tasks with a dynamic 10k-context turn budget."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from office_env import DEFAULT_TEMPLATE_DIR, build_record, load_template_catalog, validate_record


def _build_validated_record(arguments: tuple[int, int, str, tuple[str, ...]]) -> dict:
    index, seed, template_dir, tasks = arguments
    task = tasks[index % len(tasks)]
    record = build_record(task, seed=seed + index, index=index + 1, template_dir=template_dir)
    grade = validate_record(record)
    if grade["score"] != 1.0:
        raise RuntimeError(f"oracle validation failed for {record['id']}: {grade['issues']}")
    return record


def generate_records(
    count: int = 120,
    *,
    seed: int = 2026,
    workers: int = 20,
    template_dir: Path | str = DEFAULT_TEMPLATE_DIR,
) -> list[dict]:
    """Generate only records whose deterministic oracle artifact scores 1.0."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if workers < 1:
        raise ValueError("workers must be positive")
    template_dir = Path(template_dir).resolve()
    tasks = tuple(load_template_catalog(template_dir))
    arguments = ((index, seed, str(template_dir), tasks) for index in range(count))
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, max(count, 1))) as executor:
        return list(executor.map(_build_validated_record, arguments))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help="directory containing catalog.json (defaults to examples/agentic/office/templates)",
    )
    args = parser.parse_args()
    records = generate_records(args.count, seed=args.seed, workers=args.workers, template_dir=args.template_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} oracle-validated records to {args.output}")


if __name__ == "__main__":
    main()
