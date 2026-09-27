"""Build a training mix from several JevForge record directories.

Every record (one question each, as `convert_datasets.py` writes them) falls in
one bucket by its format:

    soft           target is not one-hot (max < 0.99), any type
    wanli          Open-Jev WANLI NLI rows (down-weighted)
    choice_desc    choice whose options carry descriptions
    choice_nodesc  choice whose options are bare ids / labels
    noul_crit      noul with `yes := / no :=` criteria
    noul_nocrit    noul without criteria
    score          ordered levels

Each bucket gets a share of `--train-size`; inside a bucket the share is split
evenly across sources (water-filling, so small sources give their leftover to
large ones). A fraction of selected `noul_nocrit` rows gets generic criteria
(`--noul-criteria-aug`), since few public rows teach the model to read them.
Records are streamed twice (count, then sample), so memory stays small.

    python examples/classify/jev/mix_records.py --out ~/data/jev-records/mix-v2 \
        --source open-jev-v1.1=~/data/jev-records/open-jev-v1.1 \
        --source open-jev-v1=~/data/jev-records/open-jev-v1 \
        --source jevembed=~/data/jev-records/jevembed
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SHARES = {
    "choice_desc": 0.25,
    "choice_nodesc": 0.08,
    "wanli": 0.05,
    "noul_crit": 0.20,
    "noul_nocrit": 0.14,
    "score": 0.18,
    "soft": 0.10,
}
NOUL_CRITERIA_TEMPLATES = (
    {"true": "The statement is true.", "false": "The statement is false."},
    {"true": "Yes, this holds for the given state.", "false": "No, this does not hold for the given state."},
    {"true": "The answer to the question is yes.", "false": "The answer to the question is no."},
)


def bucket_of(record: dict) -> str:
    question = next(iter(record["request"]["questions"].values()))
    target = next(iter(record["targets"].values()))
    if max(target.values()) < 0.99:
        return "soft"
    if str(record.get("meta", {}).get("source", "")).startswith("wanli"):
        return "wanli"
    kind = question["type"]
    if kind == "choice":
        criteria = question["criteria"]
        described = all(str(v).strip().lower() != str(k).strip().lower() for k, v in criteria.items())
        return "choice_desc" if described else "choice_nodesc"
    if kind == "noul":
        return "noul_crit" if question.get("criteria") else "noul_nocrit"
    return "score"


def source_of(dataset: str, record: dict) -> str:
    meta = record.get("meta", {})
    return f"{dataset}/{meta.get('config') or meta.get('source') or 'all'}"


def record_chars(record: dict) -> int:
    question = next(iter(record["request"]["questions"].values()))
    return len(record["request"]["state"]) + len(json.dumps(question, ensure_ascii=False))


def iter_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def allocate(total: int, available: dict[str, int]) -> dict[str, int]:
    """Split `total` evenly over sources, capping each at what it has."""

    quota = dict.fromkeys(available, 0)
    remaining, open_sources = total, [s for s, n in available.items() if n > 0]
    while remaining > 0 and open_sources:
        share = max(remaining // len(open_sources), 1)
        next_open = []
        for source in open_sources:
            take = min(share, available[source] - quota[source], remaining)
            quota[source] += take
            remaining -= take
            if quota[source] < available[source]:
                next_open.append(source)
            if remaining == 0:
                break
        open_sources = next_open
    return quota


def build_split(sources: list[tuple[str, Path]], split: str, size: int, shares: dict, args, rng: random.Random):
    files = [(name, root / f"{split}.jsonl") for name, root in sources if (root / f"{split}.jsonl").exists()]

    def eligible():
        # Both passes see the same stream: first occurrence of an id wins
        # (Open-Jev configs and v1/v1.1 share rows), long records are dropped.
        seen: set[int] = set()
        for name, file in files:
            for record in iter_records(file):
                key = hash(record["id"])
                if key in seen or record_chars(record) > args.max_chars:
                    continue
                seen.add(key)
                yield name, record

    available: dict[str, dict[str, int]] = defaultdict(Counter)
    for name, record in eligible():
        available[bucket_of(record)][source_of(name, record)] += 1
    quotas = {
        bucket: allocate(round(size * share), dict(available.get(bucket, {}))) for bucket, share in shares.items()
    }
    selected, augmented = [], 0
    for name, record in eligible():
        bucket, source = bucket_of(record), source_of(name, record)
        if rng.random() >= quotas.get(bucket, {}).get(source, 0) / available[bucket][source]:
            continue
        if bucket == "noul_nocrit" and split == "train" and rng.random() < args.noul_criteria_aug:
            question = next(iter(record["request"]["questions"].values()))
            question["criteria"] = dict(rng.choice(NOUL_CRITERIA_TEMPLATES))
            record.setdefault("meta", {})["augmented"] = "noul_criteria"
            augmented += 1
        record["split"] = split
        record.setdefault("meta", {})["bucket"] = bucket
        selected.append(record)
    rng.shuffle(selected)
    counts: dict[str, dict[str, int]] = defaultdict(Counter)
    for record in selected:
        counts[record["meta"]["bucket"]][source_of(record["meta"].get("dataset", ""), record)] += 1
    report = {
        "records": len(selected),
        "augmented_noul_criteria": augmented,
        "available": {bucket: dict(value) for bucket, value in available.items()},
        "selected_by_bucket": {bucket: sum(value.values()) for bucket, value in counts.items()},
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="name=records_dir (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, default=64000)
    parser.add_argument("--dev-size", type=int, default=3000)
    parser.add_argument("--share", action="append", default=[], help="bucket=fraction override (repeatable)")
    parser.add_argument("--noul-criteria-aug", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=6000, help="drop records whose state+question exceed this")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    sources = []
    for item in args.source:
        name, _, path = item.partition("=")
        sources.append((name, Path(path).expanduser()))
    shares = dict(DEFAULT_SHARES)
    for item in args.share:
        bucket, _, value = item.partition("=")
        if bucket not in DEFAULT_SHARES:
            raise SystemExit(f"unknown bucket {bucket!r}; known: {sorted(DEFAULT_SHARES)}")
        shares[bucket] = float(value)
    total = sum(shares.values())
    shares = {bucket: value / total for bucket, value in shares.items()}

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = {"sources": {name: str(path) for name, path in sources}, "shares": shares, "args": vars(args)}
    for split, size in (("train", args.train_size), ("dev", args.dev_size)):
        records, report = build_split(sources, split, size, shares, args, rng)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest[split] = report
        print(f"{split}: {report['records']} records {report['selected_by_bucket']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
