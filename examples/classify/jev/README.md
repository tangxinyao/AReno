# Ling-3.0-tiny decision scorer (JevForge-style)

Train `inclusionai/ling-3.0-tiny` into a typed-decision model: it scores every
candidate of a `choice` / `noul` / `score` question in one forward pass and
returns a full probability distribution, behind the same request shape as
TypeSafe's decisions API.

The model is the [jev-forge](https://github.com/zwliJay/jev-forge) design: the
LM head is dropped, a `Linear -> GELU -> Linear(1)` score head reads the last
token of each candidate path, the candidates of one question share one
softmax, and the loss is cross-entropy plus `0.5 x Brier` against the target
distribution. Training is full-parameter (the score head does not support
native LoRA).

## Recommended recipe (one DGX Spark, ~4 h)

Measured on one NVIDIA GB10 (128 GB unified memory), `--world-size 1 --tp-size 1`.

### 1. Data

```bash
# Training set: Open-Jev v1.1 from ModelScope (pinned revisions, sha256 checked) -> records
bash examples/classify/jev/data/fetch_open_jev.sh ~/data/open-jev-v1.1 ~/data/jev-records/open-jev-v1.1

# Evaluation set: typed-decisions test, vendored in data/typed-decisions/
python examples/classify/jev/convert_datasets.py typed-decisions \
  --src examples/classify/jev/data/typed-decisions --out ~/data/jev-records/typed-decisions
```

| Use | Dataset | License | Size |
| --- | --- | --- | --- |
| train | Open-Jev v1.1 (`ZefanCai/Open-Jev-v1.1`) | CC0; WANLI-derived rows CC BY 4.0 | 147k train / 25k dev / 27k calibration / 43k test / 84k ood questions |
| eval | typed-decisions (`LocalLLaMA/typed-decisions`, config `all`) | Apache-2.0 | 400 test cases / 2,000 decisions |

The longest Ling-tokenized candidate path in Open-Jev is 1,436 tokens (p95 876),
so keep `--max-seq-len 1536`; the jev-forge default of 512 would drop 32% of it.

### 2. Train

```bash
python examples/classify/jev/train.py \
  --records ~/data/jev-records/open-jev-v1.1 \
  --ckpt inclusionai/ling-3.0-tiny \
  --save-path runs/ling-jev \
  --adam-4bit
```

The script defaults are this recipe: 400 steps x 32 questions, a checkpoint
every 100 steps, backbone LR 2e-5 (cosine), score-head LR 2e-4, 12 head-only
warmup steps, 8,000-token microbatches, candidate paths up to 1,536 tokens.

- `--adam-4bit` is what makes full-parameter training of this MoE fit on one GB10.
  If it still does not fit, lower `--microbatch-tokens` (e.g. 4000) and then add
  `--optimizer-state-offload cpu`.
- ~37 s per step, so every 100 steps is about an hour.
- The run shows up in `areno dashboard` (loss, `classify_ce`, `classify_brier`,
  `classify_top1`, learning rates, grad norms).

### 3. Pick a checkpoint on Open-Jev dev

```bash
for step in 100 200 300 400; do
  python examples/classify/jev/evaluate.py --checkpoint runs/ling-jev/step_000${step} \
    --records ~/data/jev-records/open-jev-v1.1 --split dev --limit 1000 \
    --output runs/ling-jev/dev-${step}.json
done
```

Keep the checkpoint with the best Open-Jev dev accuracy (step 400 in our run).
Never choose it, or a temperature, on the typed-decisions test: that set is the
zero-shot evaluation, and tuning on it leaks test labels.

### 4. Evaluate

```bash
python examples/classify/jev/evaluate.py --checkpoint runs/ling-jev/step_000400 \
  --records ~/data/jev-records/typed-decisions --split test --output typed-decisions.json

# Zero-shot reference: the untouched base model answering through its LM head
python examples/classify/jev/zero_shot.py --model <ling-3.0-tiny local dir> \
  --records ~/data/jev-records/typed-decisions --split test --output zero-shot.json
```

`evaluate.py` reports accuracy, CE, KL(gold || pred), total variation, Brier,
mean confidence, overconfidence (confidence minus accuracy) and a 10-bin ECE,
per question type and per workflow. KL / TV / Brier reproduce the
typed-decisions card's `Uniform` row exactly (0.444 / 0.381 / 0.238), so those
columns compare directly with its leaderboard. `calibrate.py` fits or applies a
softmax temperature on logits dumped with `--dump`.

### 5. Serve

```bash
python examples/classify/jev/serve_decisions.py --checkpoint runs/ling-jev/step_000400 \
  --port 8123 --max-length 1536
```

`POST /api/alpha/decisions` (and jev-forge's `/v1/systemone`) takes
`{model, state, questions}`; a JSON `state` is serialized with sorted keys, the
way the training records render it. Answers follow jev-forge: `noul` gives
P(true); `choice` gives the choice, probabilities and confidence; `score` gives
the expected level, a legend, probabilities and confidence. All candidate paths
of a request run in one packed forward on AReno's own Bailing V3 model (Ling's
bundled `trust_remote_code` modeling file targets transformers 4.45 and does
not import under this repo's transformers). A request with 3 questions and 8
candidate paths takes ~212 ms at p50 on the GB10 after warm-up.

## Results

typed-decisions test (zero-shot for every row; 2,000 decisions):

| Model | Acc | KL | TV | Brier | Overconfidence |
| --- | --- | --- | --- | --- | --- |
| Ling-3.0-tiny, zero-shot LM head (no training) | 0.567 | 0.893 | 0.383 | 0.326 | +0.228 |
| + this recipe, step 100 | 0.565 | 0.316 | 0.292 | 0.178 | -0.030 |
| + this recipe, step 400 | 0.585 | 0.439 | 0.304 | 0.228 | +0.101 |
| Prior (card reference) | 0.470 | 0.347 | 0.317 | 0.189 | |
| TypeSafe Jev 1.13.0 (card) | 0.727 | 1.442 | 0.251 | 0.148 | |

Open-Jev v1.1 (in distribution): test accuracy 0.557 zero-shot -> 0.799 after
training, overconfidence -0.006, ECE 0.018.

What this means:

- **Training fixes calibration, not zero-shot accuracy.** On typed-decisions every
  checkpoint lands at 0.56-0.59, the same as the untouched model; KL and Brier
  improve a lot. The standard error of accuracy on 2,000 decisions is ~0.011, so
  the checkpoint-to-checkpoint differences are mostly noise. The 14-point gap to
  Jev 1.13 comes from the base model, not from the recipe.
- **In-distribution training works very well** (+24 points on Open-Jev). For a
  product with its own workflows, train this recipe on your own labeled
  decisions; that is where the model gains accuracy.
- **Tried and not worth it:**
  - Training past ~400 steps: in-distribution accuracy stops improving and
    out-of-distribution overconfidence keeps rising.
  - Temperature scaling on Open-Jev calibration: the fitted temperature is
    ~1.05, which changes almost nothing.
  - Mixing in more sources with described options / noul criteria / LLM-judge
    ratings (Open-Jev v1 + JevEmbed-Data): typed-decisions accuracy dropped to
    0.561 and overconfidence rose to +0.173.

## How it maps onto AReno

| Piece | Where |
| --- | --- |
| Score head, LM head skipped | `RuntimeConfig.score_head`, `areno/engine/score_head.py`; model forward with `defer_lm_head=True` (bailing_moe_v3 / Ling-3.0, qwen3, qwen3_5, gemma4) |
| Grouped softmax CE + Brier | `classify_loss_fn` (`areno/experimental/classify/loss.py`) |
| Whole questions per DP rank and microbatch | `build_step_rows` (`areno/experimental/classify/trainer.py`), packed varlen, no padding |
| Head-only warmup, separate head LR | `score_head_warmup_steps` holds the backbone LR at 0; `score_head_lr` is its own LR group |
| Single-GPU inference | `SequenceScorer` / `NextTokenScorer` (`areno/experimental/classify/scorer.py`) |
| Checkpoint | `step_XXXXXX/` HF backbone + `score_head.safetensors` |

## Limitations

- Full-parameter only; the score head does not work with native LoRA.
- The head-only warmup sets the backbone LR to 0 instead of freezing it, so
  AdamW still accumulates backbone moments during those steps.
- `export_jevforge.py` needs a backbone that transformers loads without remote
  code, so Ling checkpoints cannot be exported to jev-forge; serve them with
  `serve_decisions.py`.
- All zero-shot conclusions above come from typed-decisions, whose gold is the
  mean of a ~4B teacher's samples; they measure agreement with that teacher.
