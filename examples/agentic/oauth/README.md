# OAuth Agentic RL Demo

Connect-the-Google-email agentic tasks for GRPO/GSPO-style RL. This example
ports the google-email OAuth domain from the offline trajectory lab
[`ling-oauth-boilerplate`](https://code.alipay.com/ling-oauth-boilerplate)
(trajlab, MIT) into AReno's online training shape: the dataset provides only
*deterministic task instances*; `areno train` produces the trajectories and
scores every rollout online.

Each of the 216 scenarios (6-axis matrix) is a full "connect my Google email"
story with its own mock OAuth world: some reach `AUTHENTICATED`, others hit a
business block (no test user, user abandons the browser, expired code, no
client secret, stale token) where the best play is to handle the branch
cleanly — never claim success.

## Layout

```
dataset_generator.py   216 combos → validated JSONL records (deterministic)
dataset_loader.py      areno --dataset-loader-fn hook (validates, passes through)
run_agent.py           agent loop (office-parity skeleton, 5 tools, 10k budget)
reward.py              reward_fn(record): outcome ruler + process + gate penalties
oauth_env.py           record builder + workspace fixture materialization
oauth_tools.py         terminal/read_file/write_file/skill_view/clarify + grading
oauth_world.py         mock Google OAuth server + state machine (real error shapes)
oauth_seeds.py         setup.py shim (mock-wired) + client_secret fixture
oauth_matrix.py        6-axis scenario matrix, clarify routing keys, expected_outcome
oauth_steps.py         S1..S5d adjudication table   ┐ google-email domain rules
oauth_gate.py          process ruler (witness +   │ (ported from trajlab
oauth_vocab.py         OAuth observation vocab     ┘  flows/google_email/)
grading/               trimmed trajlab engine slice (registry-free): message
                       contracts, StepAnalyzer, RuleJudge, matrix primitives
skills/                vendored hermes-agent SKILL.md docs (MIT): himalaya,
                       google-workspace (scripts/setup.py substituted by the shim)
fixtures/              golden_cases.json + two real Hermes captures (grading regression)
```

## Tool surface

The five tools mirror the Hermes surface the original captures were collected
with: `terminal`, `read_file`, `write_file`, `skill_view`, `clarify`. The
`clarify` tool follows hermes' batch `questions` contract; in this environment
it is answered by a deterministic scripted user (the task_spec's
`clarify_answers`, routed by the same intent regexes the process gate judges),
and off-script questions surface the hermes timeout sentinel.

## Scenario matrix (216 = 3×2×2×2×3×3)

| Axis | Values |
|---|---|
| service_scope | email_only / email_calendar / full_workspace |
| advanced_protection | no / yes |
| test_user_added | yes / no (→ `access_denied`) |
| client_secret_ready | yes / no (→ the agent must face "not created yet") |
| token_state | absent / expired / revoked (→ `REFRESH_FAILED`) |
| code_validity | valid / expired / user_aborted |

`expected_outcome(combo)` derives reach/blocked from the same mapping the
mock world implements — one source of truth, no second ruler.

## Reward composition

```
reward = 0.70 * outcome_term + 0.30 * process_term − 0.25 * len(process_penalties)
```

- **outcome_term** — the outcome ruler: 1.0 iff the verified final state
  matches `expected_outcome(combo)`. Reach combos must actually confirm
  `AUTHENTICATED` via `setup.py --check`; blocked combos must NOT claim it.
  Forging the token file (writing `google_token.json` directly) claims
  success and therefore fails the ruler.
- **process_term** — mean of the S1..S5d step scores (S4 normalized from its
  1..5 rubric). S1 skill routing, S2 auth probe without loops, S3 clarify
  asks, S4 the single manual step announcement, S5a client_secret, S5b
  authorize + expected-failure caveat, S5c code exchange (+ error handling),
  S5d final `--check` confirmation.
- **process_penalties** — `oauth_gate.process_checks`: a reach-success whose
  world witness shows zero `/authorize`/`/token` hits (`forged_oauth_anchor`)
  and missed clarify obligations (scope always; protection for `yes` combos).

During the episode each tool result also carries `progress_score` (the same
process term over the transcript so far) and `solved` — the same signal the
final reward uses — so the agent loop can stop early on confirmed success.

## Usage

```bash
python examples/agentic/oauth/dataset_generator.py \
  --output examples/agentic/oauth/dataset.jsonl           # all 216, deterministic
python examples/agentic/oauth/dataset_generator.py \
  --output examples/agentic/oauth/dataset.jsonl --count 64

areno train \
  --ckpt inclusionai/ling-3.0-tiny \
  --model-hub modelscope \
  --dataset-path examples/agentic/oauth/dataset.jsonl \
  --dataset-loader-fn examples/agentic/oauth/dataset_loader.py \
  --reward-fn-path examples/agentic/oauth/reward.py \
  --agent-fn examples/agentic/oauth/run_agent.py \
  --algo gspo --attn-backend flash \
  --batch-size 1 --n-samples 8 --mini-bs 1 \
  --max-running-prompts 8 --adam-8bit \
  --drop-rollout-state \
  --max-context-len 10000 \
  --max-prompt-tokens 4096 \
  --max-new-tokens 9000
```

Runtime budget override: `ARENO_OAUTH_TOKEN_BUDGET=8000 areno train ...`

A standalone mock world for manual exploration:

```bash
python -c "import sys; sys.path.insert(0, 'examples/agentic/oauth'); \
import oauth_world; oauth_world.run_server('127.0.0.1', 9898, {'token_behavior':'invalid_grant_once'})"
```

## Tests

```bash
pytest tests/ -k oauth    # world/matrix/grading/reward/loader/capture regression
```

## Provenance / licenses

- `grading/` + `oauth_steps.py` / `oauth_gate.py` / `oauth_vocab.py` /
  `oauth_matrix.py` / `oauth_world.py` / `oauth_seeds.py` are adapted from
  [ling-oauth-boilerplate](https://code.alipay.com/ling-oauth-boilerplate)
  (trajlab v0.1, MIT): the offline collection pipeline was dropped, the
  adjudication machinery is registry-free, and the world server binds one
  state machine per episode on an ephemeral port.
- `skills/*/SKILL.md` are vendored from
  [hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT,
  author Nous Research); see frontmatter in each file. The
  `google-workspace` `scripts/setup.py` referenced by the skill doc is
  replaced at workspace materialization by a stdlib shim (`oauth_seeds.py`)
  whose endpoints point at the episode's mock OAuth world.
- `fixtures/samples-*.json` are real Hermes captures from the source lab;
  they contain no credentials (mock `mock-client-1` fixtures only).
