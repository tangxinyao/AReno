"""Generate the OAuth agentic-RL dataset: one JSONL record per scenario combo.

The records are cheap and fully deterministic (no oracle artifacts), so a
single process is enough: the 6-axis google-email matrix enumerates 216
combos, each becomes one record validated for internal consistency before it
is emitted.  A fixed --count truncates the enumeration (seeded sampling is
available in ``grading.matrix`` when a stochastic subset is wanted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oauth_env
import oauth_matrix
from grading.matrix import enumerate_combos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "dataset.jsonl"))
    parser.add_argument("--count", type=int, default=None,
                        help="emit the first N combos (default: all of them)")
    args = parser.parse_args()

    combos = enumerate_combos(oauth_matrix.google_email_dimensions())
    if args.count is not None:
        combos = combos[: args.count]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for index, combo in enumerate(combos):
            record = oauth_env.build_record(combo, index=index)
            issues = oauth_env.validate_record(record)
            if issues:
                raise RuntimeError(f"{record['id']} failed validation: {issues}")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(combos)} records to {output}")


if __name__ == "__main__":
    main()
