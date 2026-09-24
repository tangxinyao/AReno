# Draco Malfoy Dialogue SFT Example

Fine-tunes a chat model to answer the way Draco Malfoy talks. Each row is one
exchange from the books: another character speaks to Draco (`prompt`, e.g.
`Ron: Eat slugs, Malfoy.`), and Draco replies (`response`, spoken words only).
The SFT trainer masks the prompt, so loss only falls on Draco's reply.

> **Copyright.** No book text ships with this example. `extract_dialogue.py`
> builds the dataset from books **you own**. It writes to `data/`, which is
> git-ignored. Keep that output local and do not redistribute it.

## 1. Build the dataset

```bash
python examples/sft/draco/extract_dialogue.py /path/to/harry.txt
# -> examples/sft/draco/data/draco.jsonl
```

- Inputs: English `.txt` (one paragraph per line, UTF-8 or GB18030) or `.epub`.
- Each paragraph is split into speech units. Adjacent `"..." "..."` quotes are
  two speakers glued together, so they are split into separate units.
- A unit is attributed in one of these ways:
  - a speech verb next to a name (`said Malfoy`, `Harry snapped`);
  - an appositive (`Malfoy, who ..., said`);
  - an action beat (`Malfoy smirked. "..."`);
  - back-and-forth alternation, for an unattributed unit between two known
    speakers.
- A row is kept when a non-Draco unit is followed by a Draco unit, at most one
  narration paragraph apart. Pronoun-only attributions ("he drawled") are not
  resolved, so recall favors precision.
- On a 7-book English omnibus this gives 76 exchanges. That is small but
  enough for a LoRA to pick up the tone. Spot-check the jsonl before training.

## 2. Train on a DGX Spark

DGX Spark has a single GB10 GPU with 128 GB of unified memory, so run
`--tp-size 1 --world-size 1`. The command below uses the same memory setup
as the OPC Ling-3.0-tiny run on Spark: LoRA, a 4-bit Adam state and
activation checkpointing. Draco rows are much shorter than OPC rollouts, so
it drops disk optimizer offload.

76 rows at `--batch-size 8` is about 10 steps per epoch, so 10 epochs is
about 100 steps. The SFT trainer only saves every `--save-interval` steps and
does not save again at the end, so keep the interval below the total step count.

```bash
areno train \
  --algo sft \
  --ckpt inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --dataset-path examples/sft/draco/data/draco.jsonl \
  --dataset-loader-fn examples/sft/draco/dataset_loader.py \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 8 \
  --mini-bs 4 \
  --epochs 10 \
  --max-prompt-tokens 512 \
  --max-new-tokens 256 \
  --disable-thinking \
  --activation-checkpointing \
  --adam-4bit \
  --lr 1e-4 --min-lr 1e-5 \
  --lora-rank 16 \
  --save-path outputs/draco-ling-lora \
  --save-interval 10
```

Serve the base model with the adapter (a `step_*` directory under the save path):

```bash
areno serve \
  --model-path inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --lora-adapter-path outputs/draco-ling-lora/step_XXXXXX \
  --disable-thinking \
  --tp-size 1 --world-size 1 --port 8000
```

- Attention defaults to `--attn-backend flash`, which needs `flash-attn`
  built for the GB10. Without it, add `--attn-backend native` (slower).
- At inference, send messages in the training format: `Harry: <what Harry says>`.
- `--disable-thinking` is passed on both sides so the training and serving
  prompt formats match. Whether Ling's chat template changes anything with it
  has not been checked. If the template ignores the flag, both sides stay
  consistent anyway.
- Rows longer than `--max-prompt-tokens`/`--max-new-tokens` are dropped, and
  the trainer logs how many. Single exchanges are short, so none should be.
