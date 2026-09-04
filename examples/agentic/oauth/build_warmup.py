"""Build a cold-start curriculum subset of the google-email matrix.

``dataset_generator.py`` emits the raw 216-combo enumeration in axis order,
which is the wrong thing to start RL on: ``service_scope`` is the outermost
axis, so the first 72 records are all ``email_only``, and areno reads the
dataset sequentially (there is no shuffle anywhere in the training loop).
A run that starts there spends its first dozens of steps on the hardest class
in the matrix and never earns the 0.70 outcome term.

This script emits the two classes a cold policy can actually score on, mixed
and shuffled:

  * ``nosecret`` (24 records) -- the cheapest outcome in the matrix.  Its
    ``blocked_evidence`` marker is the local ``no client credentials`` refusal
    and ``oauth_gate.BLOCK_WITNESS`` carries no entry for it, so no world
    witness is required: one ``setup.py --auth-url`` surfaces the block.
    Restricted to ``token_state=absent`` so the seeded stale token does not
    put a second, differently-reasoned block in front of it.
  * ``reach`` (4 combos, repeated ``--reach-copies`` times) -- the only combos
    that can finish OAuth, so the policy sees positive outcomes at all.

Record ids are assigned after the shuffle, so they stay unique across the
repeated reach combos.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oauth_env
import oauth_matrix
from grading.matrix import enumerate_combos


def warmup_combos(reach_copies: int) -> list[dict[str, str]]:
    """The cold-start mix: every cheap block plus the repeated reach combos."""
    combos = enumerate_combos(oauth_matrix.google_email_dimensions())
    nosecret = [
        c
        for c in combos
        if c["client_secret_ready"] == "no"
        and c["service_scope"] != "email_only"
        and c["token_state"] == "absent"
    ]
    reach = [c for c in combos if oauth_matrix.expected_outcome(c)["outcome"] == "reach"]
    return nosecret + reach * reach_copies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "warmup.jsonl"))
    parser.add_argument("--reach-copies", type=int, default=8,
                        help="times each reach combo is repeated (balances the mix)")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    args = parser.parse_args()

    combos = warmup_combos(args.reach_copies)
    random.Random(args.seed).shuffle(combos)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as f:
        for index, combo in enumerate(combos):
            record = oauth_env.build_record(combo, index=index)
            issues = oauth_env.validate_record(record)
            if issues:
                raise RuntimeError(f"{record['id']} failed validation: {issues}")
            reason = oauth_matrix.expected_outcome(combo)["reason"]
            counts[reason] = counts.get(reason, 0) + 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(combos)} records to {output}")
    print(f"outcome mix: {counts}")


if __name__ == "__main__":
    main()
