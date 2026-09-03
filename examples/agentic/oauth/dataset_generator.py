"""Generate the OAuth agentic-RL dataset: one JSONL record per scenario combo.

The records are cheap and fully deterministic (no oracle artifacts), so a
single process is enough: the 6-axis google-email matrix enumerates 216
combos, each becomes one record validated for internal consistency before it
is emitted.

``--count`` takes a *seeded sample* rather than a prefix: the axis order makes
the first N combos share the outermost axis value (the first 72 are all
``service_scope=email_only``), which is not a subset anyone wants to train on.
``--outcome`` selects one side of the outcome ruler -- only 4 of the 216 combos
can reach AUTHENTICATED, so a balanced run mixes two generated files and
``--start-index`` keeps the record ids unique across them::

    dataset_generator.py --outcome reach                  --output reach.jsonl
    dataset_generator.py --outcome blocked --count 32 --start-index 100 \
                                                          --output blocked.jsonl
    cat reach.jsonl blocked.jsonl > dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oauth_env
import oauth_matrix
from grading.matrix import enumerate_combos, filter_combos, sample_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "dataset.jsonl"))
    parser.add_argument("--count", type=int, default=None,
                        help="emit a seeded sample of N combos (default: all of them)")
    parser.add_argument("--seed", type=int, default=20260831,
                        help="sampling seed for --count")
    parser.add_argument("--outcome", choices=("reach", "blocked"), default=None,
                        help="keep only combos with this expected outcome")
    parser.add_argument("--start-index", type=int, default=0,
                        help="first record id number (keeps ids unique when "
                             "concatenating several generated files)")
    args = parser.parse_args()

    combos = enumerate_combos(oauth_matrix.google_email_dimensions())
    if args.outcome is not None:
        combos = filter_combos(
            combos, [lambda c: oauth_matrix.expected_outcome(c)["outcome"] == args.outcome])
    if args.count is not None:
        combos = sample_scenarios(combos, limit=args.count, seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for offset, combo in enumerate(combos):
            record = oauth_env.build_record(combo, index=args.start_index + offset)
            issues = oauth_env.validate_record(record)
            if issues:
                raise RuntimeError(f"{record['id']} failed validation: {issues}")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(combos)} records to {output}")


if __name__ == "__main__":
    main()
