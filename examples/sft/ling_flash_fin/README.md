# Ling-3.0-flash-Fin Knowledge-Injection SFT Example

Ling-3.0-tiny was released on August 6, 2026. Ling-3.0-flash-Fin came out
later: OpenRouter lists August 27, 2026, and its Hugging Face weights date
from September 2026. So when you ask tiny "什么是 Ling-3.0-flash-Fin？", it has
no knowledge of the model and will say it doesn't know or make something up.
This example collects the public material about Flash Fin, turns it into
bilingual Q&A, and uses LoRA SFT to teach tiny the facts.

## Data

| File | Rows | Use |
| --- | --- | --- |
| `data/train.jsonl` | 99 (zh 51 / en 48) | `prompt` question, `response` answer. No loader needed. |
| `data/eval.jsonl` | 42 (zh 21 / en 21) | Held-out paraphrases with a `reference` answer. Never trained on. |
| `build_dataset.py` | - | Regenerates both files. Edit facts here, not in the jsonl. |

The data covers 21 facts: what the model is, its key features, who built it, its base model,
parameters, architecture, context window, release dates, license and weights,
deployment, recommended sampling, highlights, spreadsheet/LBO work, evaluation
benchmarks, FinFIRST (two facts), Artificial Analysis scores, pricing, tool
calling, limitations, and sibling models. Each fact has 2-4 training phrasings
per language that share one canonical answer, so the model learns the fact
rather than one wording. Every row keeps its `sources`:

- OpenRouter model page and FAQ: <https://openrouter.ai/inclusionai/ling-3.0-flash-fin>
- Hugging Face model card: <https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin>
- Artificial Analysis: <https://artificialanalysis.ai/models/ling-3-0-flash-fin>
- Ant Ling blog, *What It Takes to Turn a General Model into a Domain Expert* (2026-09-18)
- Ant Ling blog, *Ling-3.0-flash: More Useful Work per Token* (2026-07-27). Only
  the shared architecture facts come from here.

Sources disagree on two points, and the answers say so rather than picking one:

- **Release date**: OpenRouter says 2026-08-27. The Hugging Face BF16 repo was
  created on 2026-09-03.
- **Price**: it depends on the provider. OpenRouter/DeepInfra charges
  $0.06/$0.18 per M tokens and Artificial Analysis lists $0.075/$0.22. Prices
  were collected on 2026-09-24 and may be stale.

## Train on a DGX Spark

```bash
areno train \
  --algo sft \
  --ckpt inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --dataset-path examples/sft/ling_flash_fin/data/train.jsonl \
  --tp-size 1 --world-size 1 \
  --batch-size 8 --mini-bs 4 \
  --epochs 10 \
  --max-prompt-tokens 128 \
  --max-new-tokens 512 \
  --disable-thinking \
  --activation-checkpointing \
  --adam-4bit \
  --lr 1e-4 --min-lr 1e-5 \
  --lora-rank 16 \
  --save-path outputs/ling-flash-fin-lora \
  --save-interval 20
```

- 99 rows at batch 8 is about 13 steps per epoch, so 10 epochs is about 130
  steps. The SFT trainer only saves every `--save-interval` steps and does not
  save again at the end, so keep the interval below the total step count.
- Knowledge injection needs more repetition than style transfer. If held-out
  answers are still vague, try more epochs or `--lora-rank 32` before adding
  data.

## Check the effect

Serve the adapter and the plain base model on two ports, then compare their
answers on the held-out questions:

```bash
areno serve --model-path inclusionai/ling-3.0-tiny --model-hub modelscope \
  --lora-adapter-path outputs/ling-flash-fin-lora/step_000120 \
  --disable-thinking --tp-size 1 --world-size 1 --port 8000 &
areno serve --model-path inclusionai/ling-3.0-tiny --model-hub modelscope \
  --disable-thinking --tp-size 1 --world-size 1 --port 8001 &

python - <<'PY'
import json, urllib.request
for line in open("examples/sft/ling_flash_fin/data/eval.jsonl", encoding="utf-8"):
    row = json.loads(line)
    print("Q:", row["prompt"])
    for port, name in ((8001, "base"), (8000, "lora")):
        body = json.dumps({"model": "inclusionai/ling-3.0-tiny", "max_tokens": 512,
                           "messages": [{"role": "user", "content": row["prompt"]}]}).encode()
        req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        print(f"  [{name}]", json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"][:300])
    print("  [ref] ", row["reference"][:300], "\n")
PY
```

After training, answers should contain the key numbers: 124B / 5.1B,
256K context, MIT, and the seven benchmarks including FinFIRST. The base model
should not know these. Also ask a few unrelated questions (general knowledge,
math) to confirm the LoRA did not break everything else. With only 99 rows,
overfitting shows up as every question being answered with the same intro
paragraph. If that happens, use an earlier `step_*` checkpoint.
