# OPC-benchmark SFT Example (Ling-3.0-tiny)

Turns **passed** agent trajectories from the [OPC benchmark](https://github.com/opc-benchmark)
into SFT rows for [Ling-3.0-tiny](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny).
OPC simulates a one-person company: an agent gets a natural-language task
(reconcile revenue, release a website, triage customer email, ...) and must
plan, run tools, verify, and hand over an artifact. Each trial ships a full
Hermes session (`hermes-session.jsonl`) plus a verifier verdict (`result.json`).

## What the loader emits

| | alpaca | opc |
|---|---|---|
| Data | static instruction/answer pairs | multi-turn tool-calling rollouts |
| Loader | one row per sample | **one row per assistant turn** (or one packed row per rollout in C mode) |
| Structure | `prompt` = instruction, `response` = answer | `prompt` = history up to the turn, `response` = the assistant message itself |
| Source | HF dataset | local `hermes-session.jsonl` files |

Both modes render Ling's `chat_template.jinja` semantics as plain text:
`<role>HUMAN</role>` turns, a `<role>ASSISTANT</role>` + `<role_end|>`-terminated
generation prompt, a ` thinking ... response` split, `<tool_call>` blocks with
`<arg_key>/<arg_value>` argument pairs, and `<role>OBSERVATION</role>` +
`<tool_response>` blocks for tool results. **Reasoning is a training target by
default** — the whole ` thinking ... response` span is supervised. Set
`ARENO_OPC_DROP_REASONING=1` to strip reasoning from both prompt and response.

Only trials whose `result.json` verifier reward is `1.0` are kept; standalone
`hermes-session.jsonl` files with no `result.json` are kept as-is.

> **Hardware note**: `areno train` needs a Linux + NVIDIA CUDA host. On a
> laptop (no GPU) you can build and validate the dataset with the loader, but
> the training command below must run on a GPU box.

## Row example (abridged, B mode)

```text
# prompt (ends with the generation prompt)
detailed thinking on<|role_end|>
<role>HUMAN</role>你是这家一人公司的助手。老板要盘一下 2026 年的收入。
...
<|role_end|>
<role>OBSERVATION</role>
<tool_response>{"output": "146 /app/data/orders.csv ---HEAD--- ...", "exit_code": 0}
</tool_response><|role_end|>
<role>ASSISTANT</role>
 thinking

# response (the next assistant message)
The file header matches the reconciliation policy. Filter to status=paid...
 response先按口径过滤一遍流水。
<tool_call>terminal
<arg_key>command</arg_key>
<arg_value>cd /app && python3 filter.py</arg_value>
</tool_call>```

## Running — one task first

The bundled `examples/sft/opc/data/` holds one task, `five-step-pipeline`
(its three passed trials, 29 rows), so you can validate the whole loop cheaply
before scaling. From the AReno repo root on your GPU host:

```bash
areno train \
  --algo sft \
  --ckpt inclusionAI/Ling-3.0-tiny \
  --dataset-path examples/sft/opc/data \
  --dataset-loader-fn examples/sft/opc/dataset_loader.py \
  --model-hub modelscope \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 2 \
  --mini-bs 1 \
  --max-prompt-tokens 16384 \
  --max-new-tokens 4096 \
  --epochs 2
```

Ling's context is 128 K, so nearly every trajectory fits with the full history —
keep the window knobs off (0) in B mode. Once the one-task run behaves (loss
drops, generations look sane), expand to every passed task in a real job run
(all 30 passed trials of `opc-deepseek-all`, 652 rows):

```bash
areno train \
  --algo sft \
  --ckpt inclusionAI/Ling-3.0-tiny \
  --dataset-path /path/to/opc-benchmark/jobs/opc-deepseek-all \
  --dataset-loader-fn examples/sft/opc/dataset_loader.py \
  --model-hub modelscope \
  ...
```

`--dataset-path` may be a job run directory (scans `*/agent/hermes-session.jsonl`
with a sibling `*/result.json`), a flat directory of `hermes-session.jsonl`
files, or a single session file.

## Whole-rollout C mode (packed loss)

The default text rows re-encode the history prefix once per turn — measured
**~15×** the forward tokens of a single packed sequence on the full 652-row
set (2.6× with the window knobs). C mode avoids that by packing each
trajectory (or, for the few >128 K-token giants, a chunk of it) into **one
encoded row**: `tokens` + `prompt_mask` + `loss_mask`, with loss enabled only
on assistant-produced spans — reasoning, ` response`, content, tool calls, and
the closing `<|role_end|>` (thinking stays a training target).

```bash
ARENO_OPC_C_MODE=1 \
areno train \
  --algo sft \
  --ckpt inclusionAI/Ling-3.0-tiny \
  --dataset-path examples/sft/opc/data \
  --dataset-loader-fn examples/sft/opc/dataset_loader.py \
  --model-hub modelscope \
  --tp-size 1 --world-size 1 --batch-size 2 --mini-bs 1 \
  --max-prompt-tokens 65536 \
  --max-new-tokens 8192 \
  --epochs 2
```

Requirements and knobs:

- C mode needs a real tokenizer: set `ARENO_OPC_TOKENIZER=/path/to/tokenizer`
  (offline, deterministic), or leave it unset to auto-download the
  Ling-3.0-tiny tokenizer files (not weights) from ModelScope. `transformers`
  is required — it is already a runtime dependency of AReno.
- `ARENO_OPC_MAX_SEQ_TOKENS` bounds each chunk (default 32768); the trainer's
  per-row budgets must be no smaller — hence the raised flags above.
- Chunks are cut at message boundaries; a single oversized message is
  hard-split. Rows feed the trainer's pre-encoded `tokens`/`prompt_mask`/
  `loss_mask` branch, so prompt/response text is not produced in this mode.

## Splitting one long task into subtasks

The per-turn rows already give fine-grained supervision, but you can also cut a
long trajectory into **phases** and treat each phase as its own subtask. Set
`ARENO_OPC_PHASES=1` and every record gets `phase_genre` + `phase_index`; genre
is derived from the turn's tool types (no command parsing, so it stays robust
across trials):

| genre | meaning | typical tools |
|---|---|---|
| `explore` | find what exists | `search_files`, `ls`/`find` |
| `read` | consume content | `read_file`, `wc`/`head`/`cat` |
| `implement` | produce / fix a file | `write_file`, `patch` |
| `run` | execute & verify | `execute_code`, `terminal` |
| `report` | final deliverable (no tool) | — |

For the bundled `five-step-pipeline`, the three trials all follow
explore → read → implement → run → report (some trials interleave extra
implement/run steps when they patch and re-run). Export the sub-datasets:

```bash
python examples/sft/opc/split_phases.py \
  --dataset examples/sft/opc/data \
  --out examples/sft/opc/split
```

which writes `split/{explore,read,implement,run,report}.jsonl` — here:
explore 3 rows, read 6, implement 5, run 12, report 3 (all from 3 trials).
Each JsonL is a normal loader-format file; train a stage alone by pointing
`--dataset-path` at it (or use it for curriculum / per-stage weighting).

## Knobs

| Env var | Effect |
|---|---|
| `ARENO_OPC_DROP_REASONING` | set to anything ⇒ reasoning is stripped from both prompt and response (target becomes ` response{content}`) |
| `ARENO_OPC_INCLUDE_SYSTEM_PROMPT` | set to anything ⇒ the 15 KB Hermes `system_prompt` is prepended as a `<role>SYSTEM</role>` block (off by default to keep rows small) |
| `ARENO_OPC_MAX_HISTORY_MESSAGES` | int ⇒ keep only the **task prompt + the last N messages** as context (sliding window). Default 0 = full history |
| `ARENO_OPC_MAX_TOOL_CHARS` | int ⇒ truncate oversized tool results / tool-call arguments **in the prompt** to N chars (the target turn stays verbatim). Default 0 = no cap |
| `ARENO_OPC_PHASES` | set to anything ⇒ tag every row with `phase_genre` + `phase_index` (the subtask split described below) |
| `ARENO_OPC_C_MODE` | set to anything ⇒ emit packed `tokens`/`prompt_mask`/`loss_mask` rows (whole-rollout, see above) |
| `ARENO_OPC_TOKENIZER` | C mode: local tokenizer dir (else auto-downloaded from ModelScope) |
| `ARENO_OPC_MAX_SEQ_TOKENS` | C mode: max tokens per packed chunk (default 32768) |

## Feeding everything

The loader emits all 652 passing rows out of the box — nothing is sampled. The
trade-off is how the text rows are tokenized:

| setup | prompt p50 (full 652) | total forward (full 652) |
|---|---|---|
| B, full history | 58.0 K | ~15× C |
| B, + `MAX_HISTORY_MESSAGES=8` + `MAX_TOOL_CHARS=2000` | 10.2 K | ~2.6× C |
| C, packed | — | 1.0× |

With Li[ng]'s 128 K context every trajectory fits whole in B, so nothing is
dropped — you just pay the prefix-reencoding cost (the 15× column). If those
GPU-hours matter, use C mode instead; it is the same learning signal at ~1/15
the forward tokens.

## Caveats

- Histories are long (a tool result can be a whole file). The SFT trainer drops
  rows over `--max-prompt-tokens` / `--max-new-tokens`, so keep the budgets
  generous; default is 1024/3071.
- This is self-distillation: the data is the same model family's own passed
  runs, teaching it behaviors it already sporadically produced. Passes reached
  with luck (a task that only passed 1/3 attempts) are still included here —
  dedup by task or eyeball `verifier/ctrf.json` before baking them in.
- The loader renders Ling's own markup (`.jinja` semantics) directly because
  the HF tokenizer ships no `chat_template`; train and generate with the same
  format so the model's tool/thinking tokens match at inference.