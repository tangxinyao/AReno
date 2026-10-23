# STS2 ReAct Example

This example trains a policy to play one Slay the Spire 2 Ironclad combat
through a plain-text ReAct loop. Each reply contains a `Thought:` line and an
`Action:` line; the simulator applies the action and returns an `Observation:`
with the new board and the next legal actions. No tool schema is used.

The game logic is the in-house pure-Python simulator in `sim/` (STS2 v0.107.1
content: Ironclad cards and monsters of all three acts). It has no runtime
dependency on the AReno engine or on any classify/score-head code.

## Files

- `sim/` is the combat simulator. `NOTICE.md` lists its sources.
- `env.py` wraps one single-combat `RunLoop` per episode, renders observations,
  and parses `Action:` lines. It also holds the shared turn and invalid-action
  limits.
- `dataset_loader.py` turns episodes into prompts. Without a path it generates
  Act 1 normal and elite fights with fresh seeds. It also writes JSONL.
- `run_agent.py` is the rollout entrypoint. It runs the ReAct loop against the
  OpenAI-compatible rollout endpoint, one episode per prompt sample.
- `reward.py` replays the accepted actions on a fresh simulator with the same
  seed and scores the final state.

## Episode and reward

- The episode starts after the Neow screen, at the first combat decision.
- An illegal or missing action gets an `Observation:` error and does not advance
  the game. After `MAX_INVALID_ACTIONS` such replies the episode ends with
  reward `-1.0`.
- The episode ends on victory or death, or after `MAX_TURNS` model calls.
- Reward is `0.5 * enemy_damage_fraction + 0.5 * won`, so a run that deals
  damage but does not win still gets a signal.

The reward is recomputed from the record's messages, so it stays in sync with
the simulator as long as the seed, encounter and action sequence are unchanged.

## Generate Episodes

```bash
python examples/sts2/dataset_loader.py --output /tmp/areno-sts2-episodes.jsonl --count 256 --seed 2026
```

## Run Training

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-sts2-episodes.jsonl \
  --dataset-loader-fn examples/sts2/dataset_loader.py \
  --reward-fn-path examples/sts2/reward.py \
  --agent-fn examples/sts2/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 256
```

## Tests

The simulator, env, loader and reward replay run on CPU:

```bash
pytest tests/test_sts2_sim_*_cpu.py tests/test_sts2_react_cpu.py
```

`run_agent.py` needs `openai` and a running rollout endpoint, so it is not
covered by the CPU tests.
