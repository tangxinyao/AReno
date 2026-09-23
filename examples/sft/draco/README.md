# Draco Malfoy Role-play SFT Example

Fine-tunes a small chat model to answer in character as Draco Malfoy. Each row
uses the few paragraphs before one of Draco's lines as the scene (`prompt`).
The line itself, spoken words only, is the target (`response`).

> **Copyright.** No book text ships with this example. `extract_dialogue.py`
> builds the dataset from ebooks **you own**. It writes to `data/`, which is
> git-ignored. Keep that output local and do not redistribute it.

## 1. Build the dataset

```bash
python examples/sft/draco/extract_dialogue.py /path/to/your/books/*.epub
# -> examples/sft/draco/data/draco.jsonl
```

- Inputs: `.epub`, or `.txt` with one paragraph per line (blank-line separated also works).
- Language is detected per book. English editions (straight, “double” or
  ‘single’ UK-style quotes) and Chinese editions (人民文学版 马尔福/德拉科,
  皇冠版 馬份/跩哥) are both supported. Don't mix languages in a single training run.
- Attribution is heuristic and favors precision. It keeps a paragraph only when
  its narration puts Malfoy/Draco next to a speech verb (`drawled Malfoy`,
  `马尔福冷笑道`) and names no other speaker. Lucius/Narcissa/"Mr. Malfoy" are
  excluded. Lines attributed only by a pronoun ("he sneered") are missed.
  Yield has not been measured on real books; spot-check the jsonl before
  training.

## 2. Train on a DGX Spark

DGX Spark has a single GB10 GPU with 128 GB of unified memory, so run
`--tp-size 1 --world-size 1`. The command below uses the same memory setup
as the OPC Ling-3.0-tiny run on Spark: LoRA, a 4-bit Adam state and
activation checkpointing. Draco rows are much shorter than OPC rollouts, so
it drops disk optimizer offload and uses a larger batch.

```bash
areno train \
  --algo sft \
  --ckpt inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --dataset-path examples/sft/draco/data/draco.jsonl \
  --dataset-loader-fn examples/sft/draco/dataset_loader.py \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 16 \
  --mini-bs 4 \
  --epochs 3 \
  --max-prompt-tokens 1024 \
  --max-new-tokens 256 \
  --disable-thinking \
  --activation-checkpointing \
  --adam-4bit \
  --lr 1e-4 --min-lr 1e-5 \
  --lora-rank 16 \
  --save-path outputs/draco-ling-lora
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
- `--disable-thinking` is passed on both sides so the training and serving
  prompt formats match. Whether Ling's chat template changes anything with it
  has not been checked. If the template ignores the flag, both sides stay
  consistent anyway.
- Rows longer than `--max-prompt-tokens`/`--max-new-tokens` are dropped, and
  the trainer logs how many. With the default `--context-chars 1500`, very few
  should be.
