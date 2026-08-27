# Office Agentic RL Demo

This example turns all ten task families from the Office RL dataset into
reproducible, artifact-graded agentic tasks. It does not require a Hermes
runtime. The native AReno agent exposes the four Hermes tools that appeared in
the successful trajectories, using the same JSON schemas and descriptions:

- `execute_code`
- `read_file`
- `write_file`
- `skill_view`

`execute_code` also provides the commonly used compatibility imports
`from hermes_tools import read_file, write_file, search_files`.

## Dataset templates and consistency

The generator loads its structural catalog from `templates/catalog.json`. The
catalog was distilled from 25 retry-correct and 21 wrong deliverable cases. It
records the source case IDs and their coverage risks (period/window selection,
units, net versus gross values, statistical objects, output formatting, and
artifact verification) without copying those cases into fixed prompts.

Instance fields are generated as deterministic random text rather than sampled
from a finite value enum. This includes input/output filenames, sheet names,
product and region labels, vocabulary, document titles, and document body text.
Only the task contract remains structural, such as required spreadsheet columns
or Word headings. A fixed seed reproduces the same instance, while different
seeds do not teach a small closed set of filenames.

Each generated record stores a deterministic `task_spec`; the fixture generator,
prompt, oracle artifact, and reward grader all consume that same specification.
The generator emits a record only after its oracle artifact receives reward
`1.0`, preventing prompt/fixture/grader drift.

Install the document dependencies before generating or running tasks.
LibreOffice must also be available on `PATH` for the DOCX-to-PDF task:

```bash
pip install openpyxl python-docx pypdf
python examples/agentic/office/dataset_generator.py \
  --output examples/agentic/office/dataset.jsonl \
  --template-dir examples/agentic/office/templates \
  --count 120 --seed 2026 --workers 20
```

Oracle artifact construction and grading use 20 worker processes by default;
the output remains ordered and deterministic for a fixed seed.

## Dynamic turn budget

Turns are not fixed. Each record carries a 10,000-token context budget and a
12-turn safety ceiling. After every model/tool turn, the agent loads AReno's
rollout tokenizer and tokenizes the complete next request, including messages,
tool schemas, and generation prompt. Once the deterministic artifact grader
confirms score `1.0`, the successful tool turn is retained and the agent emits a
short final response instead of making redundant tool calls. The same final turn
is used when another normal generation would consume the reserved context. Tool
output is capped to keep the next context bounded.

Set `ARENO_OFFICE_TOKEN_BUDGET` to override the per-record context budget at
runtime without regenerating the dataset. For example, to cap each Office
trajectory at 8,000 tokens:

```bash
ARENO_OFFICE_TOKEN_BUDGET=8000 areno train ...
```

The value must be a positive integer. When unset, the generated record's
`context_budget` remains authoritative.

## Train

```bash
areno train \
  --ckpt inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --dataset-path examples/agentic/office/dataset.jsonl \
  --dataset-loader-fn examples/agentic/office/dataset_loader.py \
  --reward-fn-path examples/agentic/office/reward.py \
  --agent-fn examples/agentic/office/run_agent.py \
  --algo gspo --attn-backend flash \
  --batch-size 1 --n-samples 8 --mini-bs 1 \
  --max-running-prompts 8 --adam-8bit \
  --drop-rollout-state \
  --optimizer-state-offload disk \
  --optimizer-state-offload-dir /tmp/areno-office-optimizer \
  --max-context-len 10000 \
  --max-prompt-tokens 4096 \
  --max-new-tokens 1024
```

The reward uses the final tool call's parsed `.xlsx` or `.docx` artifact score,
not the best intermediate score. Parseable partial artifacts receive `0.25` to
`0.75`. If an episode is truncated before producing the target file, the final
successful preparation or code-execution result supplies bounded progress up to
`0.10`; merely creating a non-empty target file is capped at `0.20`. Each grader
turn that does not verify a complete artifact multiplies the final score by
`0.95`, rewarding shorter successful repair paths. Completion keywords alone
never receive reward, and only a directly verified artifact receives `1.0`.
