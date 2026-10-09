"""Label sim decisions with guide-derived soft targets, as Jev `choice` records.

Plays full ironclad runs with a random policy (so the visited states are not
biased toward any strategy), and at each card reward or ancient choice writes
one record. The target is softmax(score / T) over the candidates, where scores
come from `knowledge.candidate_scores`. Options the KB does not cover get
UNKNOWN_SCORE (neutral). With --strict, decisions containing any uncovered
option are dropped instead, so no label rests on that assumption.

Output layout matches what `dataset_loader.load_questions` reads:
    <out>/train.jsonl, <out>/dev.jsonl

    python examples/classify/jev/strategy/label_from_kb.py \
        --episodes 200 --out-dir ~/data/jev-records/guide-kb

Caveat: the random policy only visits states it stumbles into, so the label
distribution is not the distribution a good player sees. Use it as a first
supervised signal, then check it against sim outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JEV_DIR = HERE.parent
sys.path.insert(0, str(JEV_DIR))  # decision_prompt
sys.path.insert(0, str(HERE.parents[3]))  # AReno root, for the operator import

import knowledge  # noqa: E402
from decision_prompt import build_question  # noqa: E402
from examples.classify.jev.operators.sts2_sim import Sts2SimBackend  # noqa: E402

SOURCE_GROUP = "guide_kb"
TARGET_KIND = "guide_soft"


def soft_targets(scores: list[float], temperature: float) -> list[float]:
    top = max(scores)
    exps = [math.exp((s - top) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def make_record(
    record_id: str,
    split: str,
    state_text: str,
    decision_point: str,
    candidates: list[dict],
    scores: list[float],
    covered: int,
    temperature: float,
) -> dict:
    question = build_question(decision_point, candidates)
    probs = soft_targets(scores, temperature)
    ids = list(question["criteria"])
    return {
        "id": record_id,
        "split": split,
        "source_group": SOURCE_GROUP,
        "request": {"state": state_text or "<empty>", "questions": {"q": question}},
        "targets": {"q": {cid: p for cid, p in zip(ids, probs)}},
        "target_kinds": {"q": TARGET_KIND},
        "meta": {"decision_point": decision_point, "covered": covered, "scores": scores},
    }


def play_episode(backend: Sts2SimBackend, seed_label: str, rng: random.Random, max_steps: int,
                 temperature: float, split: str, strict: bool) -> list[dict]:
    packet = backend.reset({
        "character": "ironclad",
        "ascension": 0,
        "episode_scope": "full_run",
        "seed": seed_label,
    })
    episode_id = packet["episode_id"]
    records: list[dict] = []
    try:
        for step in range(max_steps):
            if packet["done"] or not packet["candidates"]:
                break
            candidates = packet["candidates"]
            decision_point = packet["decision_point"]
            labeled = knowledge.candidate_scores(decision_point, candidates)
            if labeled is not None and len(candidates) >= 2:
                scores, covered, eligible = labeled
                full = covered == eligible
                if covered > 0 and (full or not strict):
                    records.append(make_record(
                        f"{seed_label}:{step}", split, packet["state_text"], decision_point,
                        candidates, scores, covered, temperature,
                    ))
            chosen = candidates[rng.randrange(len(candidates))]["id"]
            packet = backend.step({"episode_id": episode_id, "action_id": chosen})
    finally:
        backend.close(episode_id)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dev-every", type=int, default=10, help="every Nth episode goes to dev.jsonl")
    parser.add_argument("--strict", action="store_true",
                        help="only label decisions where every card/relic option is covered by the KB")
    parser.add_argument("--seed-prefix", default="guide-kb")
    parser.add_argument("--rng-seed", type=int, default=17)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    errors = knowledge.validate()
    if errors:
        raise SystemExit("knowledge base failed validation:\n" + "\n".join(errors))

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    backend = Sts2SimBackend()
    rng = random.Random(args.rng_seed)

    counts = {"train": 0, "dev": 0}
    by_point: dict[str, int] = {}
    with (out / "train.jsonl").open("w", encoding="utf-8") as train_f, \
            (out / "dev.jsonl").open("w", encoding="utf-8") as dev_f:
        for index in range(args.episodes):
            split = "dev" if args.dev_every and index % args.dev_every == args.dev_every - 1 else "train"
            seed_label = f"{args.seed_prefix}-{index:05d}"
            for record in play_episode(backend, seed_label, rng, args.max_steps, args.temperature, split,
                                       args.strict):
                handle = dev_f if split == "dev" else train_f
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                point = record["meta"]["decision_point"]
                by_point[point] = by_point.get(point, 0) + 1

    print(f"episodes={args.episodes} train={counts['train']} dev={counts['dev']} by_point={by_point}")
    print(f"wrote {out / 'train.jsonl'} and {out / 'dev.jsonl'}")


if __name__ == "__main__":
    main()
