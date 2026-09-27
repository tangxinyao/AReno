"""Single-process inference on AReno's own models (TP=1, no worker cluster).

Checkpoints are rebuilt through AReno's model adapter, so no
`trust_remote_code` modeling file is imported (Ling's ships for transformers
4.45). Rows are packed varlen, as in training, so nothing is padded, and only
each row's last-token hidden state is read.

- `SequenceScorer`: classify checkpoint (HF backbone + `score_head.safetensors`);
  one score-head logit per row. Same forward as training.
- `NextTokenScorer`: any causal LM checkpoint; log-probabilities of chosen
  next tokens after each row (zero-shot "verbalizer" scoring through the LM head).
"""

from __future__ import annotations

from pathlib import Path


def _load_model(root: Path, attn_backend: str):
    import torch

    from areno.engine.config import EngineConfig, RuntimeConfig
    from areno.engine.modeling import build_model_on_device
    from areno.engine.parallel.context import get_tp_context
    from areno.models.registry import config_from_hf, load_model_weights

    ctx = get_tp_context()
    if ctx.world_size != 1:
        raise RuntimeError("single-process scorers run with TP=1")
    config = EngineConfig(
        model=config_from_hf(root),
        model_path=str(root),
        runtime=RuntimeConfig(attn_backend=attn_backend, compile_model=False, activation_checkpointing=False),
        tp_size=1,
        devices=[ctx.device.index or 0],
        role="rollout",
    )
    model = build_model_on_device(config, ctx.device)
    load_model_weights(model, config.model, str(root))
    model.onload_train_weights(ctx.device)
    return torch, ctx.device, config, model


def _chunks(rows: list[list[int]], max_tokens: int):
    chunk: list[list[int]] = []
    used = 0
    for row in rows:
        if chunk and used + len(row) > max_tokens:
            yield chunk
            chunk, used = [], 0
        chunk.append(row)
        used += len(row)
    if chunk:
        yield chunk


def _packed_last_hidden(torch, model, device, rows: list[list[int]]):
    """Final-norm hidden state at the last token of each row, shape (rows, hidden)."""

    from areno.engine.runtime.train_step import _pack_train_data, _train_meta

    width = max(len(row) for row in rows)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    shape = input_ids.shape
    packed = _pack_train_data(
        {
            "input_ids": input_ids,
            "lengths": torch.tensor([len(row) for row in rows], device=device),
            "prompt_mask": torch.ones(shape, dtype=torch.bool, device=device),
            "advantages": torch.zeros(shape, dtype=torch.float32, device=device),
            "logprobs": torch.zeros(shape, dtype=torch.float32, device=device),
        }
    )
    tokens = packed["input_ids"]
    meta = _train_meta(packed, tokens, sequence_parallel=False)
    out = model(input_ids=tokens, position_ids=packed["position_ids"], train_meta=meta, defer_lm_head=True)
    flat = out.hidden_states.reshape(-1, out.hidden_states.shape[-1])
    ends = packed["train_cu_seqlens"][1 : len(rows) + 1].to(device=flat.device, dtype=torch.long) - 1
    return flat.index_select(0, ends)


class SequenceScorer:
    """Score token rows with a classify checkpoint: one logit per row."""

    def __init__(self, checkpoint: str | Path, *, attn_backend: str = "flash", max_tokens: int = 16384):
        from areno.engine.data.tokenizer import load_tokenizer
        from areno.engine.score_head import SCORE_HEAD_FILENAME, attach_score_head

        root = Path(checkpoint).expanduser().resolve(strict=True)
        if not (root / SCORE_HEAD_FILENAME).is_file():
            raise FileNotFoundError(f"{root} has no {SCORE_HEAD_FILENAME}; is this a classify checkpoint?")
        self.torch, self.device, config, self.model = _load_model(root, attn_backend)
        self.head = attach_score_head(
            self.model,
            hidden_size=config.model.hidden_size,
            dtype=config.model.dtype,
            device=self.device,
            model_path=str(root),
        )
        self.model.eval()
        self.tokenizer = load_tokenizer(str(root))
        self.max_tokens = int(max_tokens)

    def score(self, rows: list[list[int]]):
        """Return a float32 tensor with one score per row (same order)."""

        torch = self.torch
        scores = []
        with torch.inference_mode():
            for chunk in _chunks(rows, self.max_tokens):
                hidden = _packed_last_hidden(torch, self.model, self.device, chunk)
                scores.append(self.head(hidden).squeeze(-1).float())
        return torch.cat(scores) if scores else torch.empty(0)


class NextTokenScorer:
    """Zero-shot scoring through the LM head of a plain causal LM checkpoint."""

    def __init__(self, checkpoint: str | Path, *, attn_backend: str = "flash", max_tokens: int = 16384):
        from areno.engine.data.tokenizer import load_tokenizer

        root = Path(checkpoint).expanduser().resolve(strict=True)
        self.torch, self.device, _, self.model = _load_model(root, attn_backend)
        self.model.eval()
        owner = getattr(self.model, "language_model", self.model)
        self.lm_head = owner.lm_head
        self.tokenizer = load_tokenizer(str(root))
        self.max_tokens = int(max_tokens)

    def next_token_logprobs(self, rows: list[list[int]], candidates: list[list[int]]):
        """Full-vocabulary log-probs of `candidates[i]` right after `rows[i]`.

        Returns a list of float32 tensors, one per row, aligned with `candidates`.
        """

        torch = self.torch
        out = []
        with torch.inference_mode():
            start = 0
            for chunk in _chunks(rows, self.max_tokens):
                hidden = _packed_last_hidden(torch, self.model, self.device, chunk)
                logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
                logprobs = torch.log_softmax(logits, dim=-1)
                for offset in range(len(chunk)):
                    ids = torch.tensor(candidates[start + offset], dtype=torch.long, device=logprobs.device)
                    out.append(logprobs[offset].index_select(0, ids))
                start += len(chunk)
        return out


__all__ = ["NextTokenScorer", "SequenceScorer"]
