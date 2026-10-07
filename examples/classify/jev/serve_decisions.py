"""Serve a classify checkpoint behind the Jev decisions API.

    POST /api/alpha/decisions  {model, state, questions} -> {model, answers, usage, latency_ms}
    POST /v1/systemone         same contract (jev-forge's path)
    POST /admin/reload         {checkpoint, [temperature]} -> hot-swap weights (--admin-key required)
    GET  /v1/models, /health

The checkpoint is an AReno classify output (`step_XXXXXX/`): an HF backbone
plus `score_head.safetensors`, e.g. `~/areno-runs/ling-3.0-tiny-jev`. It is
loaded through AReno's own model adapter (`SequenceScorer`), not the
checkpoint's `trust_remote_code` modeling file, so the forward is the same one
used in training.

Candidate paths are encoded exactly as in training (`dataset_loader.py`) and
all paths of one request are scored in one packed varlen forward (no padding).

Answers follow jev-forge's `Predictor.decide`, with an extra `logits` field
on every answer so RL rollout workers can record `old_logp` consistently
regardless of the display `--temperature`:
    noul   -> {"type": "noul", "noul": P(true), "logits": {"false": s0, "true": s1}}
    choice -> {"type": "choice", "choice": id, "probabilities": {...}, "confidence": c,
               "logits": {candidate_id: raw_score}}
    score  -> {"type": "score", "score": E[level], "legend": {...}, "probabilities": {...},
               "confidence": c, "logits": {str(i): raw_score}}

`probabilities` is softmax(logits / --temperature) and is intended for display
and jev-forge callers. For PPO, record `old_logp = log_softmax(logits / T)[chosen]`
with whatever sampling temperature T you actually sampled from, so that it
matches the training-time distribution.

Hot-swap during RL rounds: start the server with `--admin-key <secret>` and
POST `{"checkpoint": "/path/to/step_000400"}` to `/admin/reload`. The old
model is dropped before the new one loads; in-flight decisions block briefly
on the swap lock. Without `--admin-key`, the endpoint is not registered.

    python examples/classify/jev/serve_decisions.py --checkpoint ~/areno-runs/ling-3.0-tiny-jev --port 8123
"""

# No `from __future__ import annotations`: FastAPI must resolve the
# `Request` annotation of the locally defined route at runtime.
import argparse
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_loader import ANSWER_MARK, candidate_ids, question_prefix, render_candidate  # noqa: E402

QUESTION_TYPES = ("choice", "score", "noul")
MAX_QUESTIONS = 96
MAX_PATHS = 512
logger = logging.getLogger("jev.serve")


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_question(qid: str, question: dict) -> None:
    """Same rules as jev-forge `schema.validate_question`."""

    if not _nonempty(qid) or not isinstance(question, dict):
        raise ValueError(f"invalid question id or body: {qid!r}")
    extra = set(question) - {"type", "instructions", "criteria"}
    if extra:
        raise ValueError(f"question {qid}: unsupported fields {sorted(extra)}")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid}: type must be one of {QUESTION_TYPES}")
    if not _nonempty(question.get("instructions")):
        raise ValueError(f"question {qid}: instructions must be a non-empty string")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"question {qid}: choice criteria must map 2-255 options")
        if not all(_nonempty(k) and _nonempty(v) for k, v in criteria.items()):
            raise ValueError(f"question {qid}: choice ids and descriptions must be non-empty strings")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"question {qid}: score criteria must list 2-10 ordered levels")
        if not all(_nonempty(v) for v in criteria):
            raise ValueError(f"question {qid}: score levels must be non-empty strings")
    elif "criteria" in question:
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError(f"question {qid}: noul criteria may only hold true/false")
        if not all(_nonempty(v) for v in criteria.values()):
            raise ValueError(f"question {qid}: noul criteria must be non-empty strings")


def render_state(state) -> str:
    """Text states pass through; JSON states are serialized like jev-forge."""

    if _nonempty(state):
        return state
    if isinstance(state, dict | list) and state:
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    raise ValueError("state must be non-empty text or a JSON object")


def confidence_from(probabilities: list[float]) -> float:
    """0 for a uniform distribution, 1 for one-hot (jev-forge)."""

    k = len(probabilities)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probabilities) - 1.0 / k) * k / (k - 1)))


class DecisionModel:
    """AReno scorer + Jev encoding; returns Jev answers."""

    def __init__(self, checkpoint: str, *, max_length: int, temperature: float, attn_backend: str, max_tokens: int):
        import torch

        from areno.experimental.classify.scorer import SequenceScorer

        self.torch = torch
        self.scorer = SequenceScorer(checkpoint, attn_backend=attn_backend, max_tokens=max_tokens)
        self.tokenizer = self.scorer.tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.calls = 0

    def _encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def encode(self, state: str, questions: dict) -> list[tuple[str, dict, list[list[int]]]]:
        """Same token layout as training: encode(prefix) + encode(candidate)."""

        encoded = []
        for qid, question in questions.items():
            prefix = self._encode(question_prefix(state, question))
            leaves = [
                prefix + self._encode(render_candidate(question, index) + "\n" + ANSWER_MARK)
                for index in range(len(candidate_ids(question)))
            ]
            longest = max(len(leaf) for leaf in leaves)
            if longest > self.max_length:
                raise ValueError(f"question {qid}: candidate path has {longest} tokens > max_length={self.max_length}")
            encoded.append((qid, question, leaves))
        return encoded

    def score(self, leaves: list[list[int]]):
        """One logit per path, all paths in one packed forward."""

        return self.scorer.score(leaves)

    def decide(self, state: str, questions: dict) -> tuple[dict, int]:
        torch = self.torch
        encoded = self.encode(state, questions)
        flat = [leaf for _, _, leaves in encoded for leaf in leaves]
        logits = self.score(flat)
        self.calls += 1
        answers = {}
        offset = 0
        for qid, question, leaves in encoded:
            group = logits[offset : offset + len(leaves)]
            offset += len(leaves)
            raw = [float(value) for value in group.tolist()]
            values = torch.softmax(group / self.temperature, dim=-1).tolist()
            values = [v if math.isfinite(v) else 0.0 for v in values]
            total = math.fsum(values) or 1.0
            values = [v / total for v in values]
            ids = candidate_ids(question)
            kind = question["type"]
            if kind == "noul":
                answers[qid] = {
                    "type": "noul",
                    "noul": values[1],
                    "logits": {"false": raw[0], "true": raw[1]},
                }
            elif kind == "choice":
                best = max(range(len(ids)), key=values.__getitem__)
                answers[qid] = {
                    "type": "choice",
                    "choice": ids[best],
                    "probabilities": dict(zip(ids, values, strict=True)),
                    "confidence": confidence_from(values),
                    "logits": dict(zip(ids, raw, strict=True)),
                }
            else:
                answers[qid] = {
                    "type": "score",
                    "score": math.fsum(i * v for i, v in enumerate(values)),
                    "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                    "probabilities": {str(i): v for i, v in enumerate(values)},
                    "confidence": confidence_from(values),
                    "logits": {str(i): raw[i] for i in range(len(raw))},
                }
        return answers, sum(len(leaf) for leaf in flat)


class ServedModel:
    """DecisionModel holder with a reload lock, so a running server can hot-swap weights.

    All calls to `decide` and `model_info` take the lock. `reload` drops the
    current model before loading the new one so VRAM is not held twice; while
    the swap is in flight callers see one blocking wait, not a 2x spike. A
    factory callable makes this testable without a real checkpoint.
    """

    def __init__(
        self,
        *,
        factory: "Callable[[str, dict], DecisionModel]",
        checkpoint: str,
        model_name: str,
        model_opts: dict,
    ):
        self._factory = factory
        self._opts = dict(model_opts)
        self._model: "DecisionModel | None" = factory(checkpoint, dict(self._opts))
        self._name = model_name
        self._checkpoint = checkpoint
        self._lock = threading.Lock()
        self.reload_count = 0

    @property
    def name(self) -> str:
        return self._name

    def decide(self, state: str, questions: dict) -> tuple[dict, int]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("model is unavailable during reload")
            return self._model.decide(state, questions)

    def model_info(self) -> dict:
        with self._lock:
            model = self._model
            return {
                "model": self._name,
                "checkpoint": self._checkpoint,
                "temperature": model.temperature if model is not None else None,
                "calls": getattr(model, "calls", 0),
                "reloads": self.reload_count,
            }

    def reload(self, checkpoint: str, *, overrides: dict | None = None, model_name: str | None = None) -> dict:
        new_opts = dict(self._opts)
        new_opts.update(overrides or {})
        with self._lock:
            old = self._model
            self._model = None
        del old
        _free_gpu_cache()
        new_model = self._factory(checkpoint, new_opts)
        with self._lock:
            self._model = new_model
            self._opts = new_opts
            self._name = model_name or Path(checkpoint).expanduser().name
            self._checkpoint = checkpoint
            self.reload_count += 1
        return self.model_info()


def _free_gpu_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch missing or CUDA init failure: nothing to free.
        pass


def _default_model_factory(checkpoint: str, opts: dict) -> DecisionModel:
    return DecisionModel(
        checkpoint,
        max_length=int(opts["max_length"]),
        temperature=float(opts["temperature"]),
        attn_backend=str(opts["attn_backend"]),
        max_tokens=int(opts["max_tokens"]),
    )


def build_app(served: ServedModel, api_key: str | None, admin_key: str | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="ling-jev decisions")

    def error(status: int, message: str):
        return JSONResponse(status_code=status, content={"error": message})

    @app.get("/health")
    def health():
        return {"ready": True, **served.model_info()}

    @app.get("/v1/models")
    def models():
        info = served.model_info()
        return {"data": [{"id": info["model"], "owned_by": "areno", "temperature": info["temperature"]}]}

    async def decisions(request: Request):
        started = time.perf_counter()
        if api_key is not None and request.headers.get("authorization") != f"Bearer {api_key}":
            return error(401, "invalid or missing bearer token")
        try:
            payload = await request.json()
        except ValueError:
            return error(400, "body must be JSON")
        if not isinstance(payload, dict) or set(payload) - {"state", "model", "questions"}:
            return error(422, "body must hold state, model, questions")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            return error(422, "questions must be a non-empty object")
        if len(questions) > MAX_QUESTIONS:
            return error(422, f"at most {MAX_QUESTIONS} questions per request")
        try:
            state = render_state(payload.get("state"))
            paths = 0
            for qid, question in questions.items():
                validate_question(qid, question)
                paths += 2 if question["type"] == "noul" else len(question["criteria"])
            if paths > MAX_PATHS:
                return error(422, f"{paths} candidate paths exceed the {MAX_PATHS} limit")
            answers, input_tokens = served.decide(state, questions)
        except ValueError as exc:
            return error(422, str(exc))
        return {
            "model": served.name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0, "candidate_paths": paths},
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    app.add_api_route("/api/alpha/decisions", decisions, methods=["POST"])
    app.add_api_route("/v1/systemone", decisions, methods=["POST"])

    if admin_key is not None:
        async def admin_reload(request: Request):
            if request.headers.get("authorization") != f"Bearer {admin_key}":
                return error(401, "invalid or missing admin token")
            try:
                payload = await request.json()
            except ValueError:
                return error(400, "body must be JSON")
            if not isinstance(payload, dict) or "checkpoint" not in payload:
                return error(422, "body must hold {checkpoint, [temperature], [model_name]}")
            checkpoint = payload["checkpoint"]
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                return error(422, "checkpoint must be a non-empty string path")
            overrides = {}
            if "temperature" in payload:
                try:
                    overrides["temperature"] = float(payload["temperature"])
                except (TypeError, ValueError):
                    return error(422, "temperature must be a number")
                if overrides["temperature"] <= 0:
                    return error(422, "temperature must be positive")
            model_name = payload.get("model_name")
            if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
                return error(422, "model_name must be a non-empty string")
            try:
                info = served.reload(checkpoint, overrides=overrides or None, model_name=model_name)
            except FileNotFoundError as exc:
                return error(404, f"checkpoint not found: {exc}")
            except Exception as exc:  # the server stays up; the admin sees the error.
                logger.exception("admin reload failed")
                return error(500, f"reload failed: {exc}")
            return {"ok": True, **info}

        app.add_api_route("/admin/reload", admin_reload, methods=["POST"])

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="classify step dir (HF backbone + score_head.safetensors)")
    parser.add_argument("--model-name", default=None, help="defaults to the checkpoint directory name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-length", type=int, default=512, help="longest candidate path accepted")
    parser.add_argument("--temperature", type=float, default=1.0, help="calibration temperature")
    parser.add_argument("--attn-backend", choices=["flash", "native"], default="flash")
    parser.add_argument("--max-tokens", type=int, default=16384, help="packed tokens per forward")
    parser.add_argument("--api-key", default=None, help="require `Authorization: Bearer <key>` when set")
    parser.add_argument(
        "--admin-key",
        default=None,
        help="enable POST /admin/reload to hot-swap the checkpoint; required for RL rounds",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    opts = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "attn_backend": args.attn_backend,
        "max_tokens": args.max_tokens,
    }
    name = args.model_name or Path(args.checkpoint).expanduser().name
    served = ServedModel(
        factory=_default_model_factory,
        checkpoint=args.checkpoint,
        model_name=name,
        model_opts=opts,
    )
    import uvicorn

    uvicorn.run(
        build_app(served, args.api_key, admin_key=args.admin_key),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
