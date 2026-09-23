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

Both modes render rows with the Ling tokenizer's own `chat_template.jinja`
(`tokenizer.apply_chat_template`): the loader converts Hermes messages to
OpenAI-style `messages` (`reasoning_content`, `tool_calls` with dict
arguments, `tool` results) and never hand-writes role, thinking or tool-call
tags, so training rows match what the template produces at inference. Each
target is the text the template emits after the generation prompt for that
turn. **Reasoning is a training target by default**; set
`ARENO_OPC_DROP_REASONING=1` to strip it from both prompt and response.

Assistant spans are found by rendering growing message prefixes, so the
template must be prefix-stable (rendering more messages must not rewrite
earlier text). C mode raises if it is not; B mode only needs each turn to
extend its own generation prompt.

Only trials whose `result.json` verifier reward is `1.0` are kept; a
`result.json` without a reward (e.g. the verifier crashed) counts as a
failure. Standalone `hermes-session.jsonl` files with no `result.json` are
kept as-is. Messages Hermes marks `active: 0` or `compacted` are skipped.

> **Hardware note**: `areno train` needs a Linux + NVIDIA CUDA host. On a
> laptop (no GPU) you can build and validate the dataset with the loader, but
> the training command below must run on a GPU box.

## Inspecting a row

The exact markup comes from the template, so print a row rather than trusting
a hand-written example (needs the tokenizer, not a GPU):

```bash
ARENO_OPC_TOKENIZER=/path/to/Ling-3.0-tiny python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("opc", "examples/sft/opc/dataset_loader.py")
opc = importlib.util.module_from_spec(spec); spec.loader.exec_module(opc)
row = opc.load_training_dataset("examples/sft/opc/data", default_loader=None)[1]
print(row["prompt"][-600:]); print("=== response ==="); print(row["response"])
PY
```

The prompt ends with the template's generation prompt; the response is the
turn's reasoning, content and tool calls, without the trailing EOS (AReno
appends `eos_token_id` itself).

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
with a sibling `*/result.json`), a single trial directory, a flat directory
of `hermes-session.jsonl` files, a single session file, or a `split_phases.py`
output file. The verdict is looked up next to any `agent/hermes-session.jsonl`
in every layout, so failed trials are filtered however deep the path points.

## Whole-rollout C mode (packed loss)

The default text rows re-encode the history prefix once per turn — measured
**~15×** the forward tokens of a single packed sequence on the full 652-row
set (2.6× with the window knobs; measured with the earlier hand-written
markup, so treat as approximate). C mode avoids that by packing each
trajectory (or, when it exceeds `ARENO_OPC_MAX_SEQ_TOKENS`, a chunk of it)
into **one encoded row**: `tokens` + `prompt_mask` + `loss_mask`, with loss
enabled only on assistant-produced spans — everything the template emits for
the turn after its generation prompt (reasoning, content, tool calls) plus the
closing EOS `<|role_end|>`.

```bash
ARENO_OPC_C_MODE=1 \
areno train \
  --algo sft \
  --ckpt inclusionAI/Ling-3.0-tiny \
  --dataset-path examples/sft/opc/data \
  --dataset-loader-fn examples/sft/opc/dataset_loader.py \
  --model-hub modelscope \
  --tp-size 1 --world-size 1 --batch-size 2 --mini-bs 1 \
  --max-prompt-tokens 32768 \
  --max-new-tokens 32768 \
  --epochs 2
```

Requirements and knobs:

- Both modes need the real tokenizer (for its chat template; C mode also
  tokenizes): set `ARENO_OPC_TOKENIZER=/path/to/tokenizer` (offline,
  deterministic), or leave it unset to auto-download the Ling-3.0-tiny
  tokenizer files (not weights) from ModelScope. `transformers` is required —
  it is already a runtime dependency of AReno.
- `ARENO_OPC_MAX_SEQ_TOKENS` bounds each chunk (default 32768). For packed
  rows the trainer checks `--max-prompt-tokens` against *all* context tokens
  and `--max-new-tokens` against *all* target tokens of the row (not one
  generation), so set **both** to at least `ARENO_OPC_MAX_SEQ_TOKENS`, or rows
  are dropped with only a `stage=sft_dataset_filter skipped_long_or_empty=N`
  log line.
- Chunks are cut at assistant-turn boundaries; a single oversized turn is
  hard-split. Every chunk starts with the system + task preamble as masked
  context, so later chunks still see the task (the turns in between are
  skipped, like a sliding window).
- `ARENO_OPC_MAX_HISTORY_MESSAGES` and `ARENO_OPC_PHASES` do not apply to
  packed rows and are ignored with a warning; `ARENO_OPC_INCLUDE_SYSTEM_PROMPT`,
  `ARENO_OPC_DROP_REASONING` and `ARENO_OPC_MAX_TOOL_CHARS` (tool results only
  — assistant targets stay verbatim) do apply.
- Rows feed the trainer's pre-encoded `tokens`/`prompt_mask`/`loss_mask`
  branch, so prompt/response text is not produced in this mode.

## Splitting one long task into subtasks

The per-turn rows already give fine-grained supervision, but you can also cut a
long trajectory into **phases** and treat each phase as its own subtask. Set
`ARENO_OPC_PHASES=1` and every record gets `phase_genre` + `phase_index`; genre
is derived from the turn's tool names only (no command parsing, so it stays
robust across trials — which also means every `terminal` call counts as `run`,
even `ls` or `cat`). When a turn calls several tools, the first matching row
below wins:

| genre | meaning | tools |
|---|---|---|
| `explore` | find what exists | `search_files` |
| `read` | consume content | `read_file` |
| `implement` | produce / fix a file | `write_file`, `patch` |
| `run` | execute & verify | anything else (`execute_code`, `terminal`, ...) |
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
Each JsonL holds ready `prompt`/`response` rows, which the loader passes
through unchanged; train a stage alone with `--dataset-path
examples/sft/opc/split/run.jsonl --dataset-loader-fn
examples/sft/opc/dataset_loader.py` (or use it for curriculum / per-stage
weighting). The split counts above were measured before the loader switched to
the tokenizer's template; row counts per genre do not depend on the markup.

## Knobs

| Env var | Effect |
|---|---|
| `ARENO_OPC_DROP_REASONING` | set to anything ⇒ reasoning is stripped from both prompt and response (target becomes ` response{content}`) |
| `ARENO_OPC_INCLUDE_SYSTEM_PROMPT` | set to anything ⇒ the 15 KB Hermes `system_prompt` is prepended as a `<role>SYSTEM</role>` block (off by default to keep rows small) |
| `ARENO_OPC_MAX_HISTORY_MESSAGES` | int ⇒ keep only the **system + task preamble + the last N messages** as context (sliding window, B mode only). Default 0 = full history |
| `ARENO_OPC_MAX_TOOL_CHARS` | int ⇒ truncate oversized tool results / tool-call arguments **in the prompt** to N chars (the target turn stays verbatim). Default 0 = no cap |
| `ARENO_OPC_PHASES` | set to anything ⇒ tag every row with `phase_genre` + `phase_index` (the subtask split described above; B mode only) |
| `ARENO_OPC_C_MODE` | set to anything ⇒ emit packed `tokens`/`prompt_mask`/`loss_mask` rows (whole-rollout, see above) |
| `ARENO_OPC_TOKENIZER` | local tokenizer dir, used by both modes for the chat template (else the tokenizer files are auto-downloaded from ModelScope) |
| `ARENO_OPC_MAX_SEQ_TOKENS` | C mode: max tokens per packed chunk (default 32768) |

## Feeding everything

The loader emits all 652 passing rows out of the box — nothing is sampled. The
trade-off is how the text rows are tokenized:

| setup | prompt p50 (full 652) | total forward (full 652) |
|---|---|---|
| B, full history | 58.0 K | ~15× C |
| B, + `MAX_HISTORY_MESSAGES=8` + `MAX_TOOL_CHARS=2000` | 10.2 K | ~2.6× C |
| C, packed | — | 1.0× |

With Ling's 128 K context every trajectory fits whole in B, so nothing is
dropped — you just pay the prefix-reencoding cost (the 15× column). If those
GPU-hours matter, use C mode instead; it is the same learning signal at ~1/15
the forward tokens.

## Caveats

- Histories are long (a tool result can be a whole file). The SFT trainer drops
  rows over `--max-prompt-tokens` / `--max-new-tokens`, so keep the budgets
  generous; default is 1024/3071.
- This is distillation from another model: the bundled trajectories were
  produced by `deepseek/deepseek-flash` (see `config.agent.model_name` in
  each `result.json`), not by Ling itself. Passes reached
  with luck (a task that only passed 1/3 attempts) are still included here —
  dedup by task or eyeball `verifier/ctrf.json` before baking them in.
- Rows use the tokenizer's `chat_template.jinja`, the same template serving
  uses, so train and inference formats match. AReno's trainer recognizes Ling
  markup (`<role>HUMAN</role>` etc.) as already chat-formatted and does not
  wrap B-mode prompts a second time.