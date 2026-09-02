"""Rule judge for the *open* decision steps (ported from trajlab/label/judge.py).

Only the deterministic rule backend came along: an online reward must be fast
and reproducible, so the LLM/remote judge backends were not ported.  Rubric
keywords come from the flow's ``StepVocab`` preset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .vocab import StepVocab


class Judge(ABC):
    """Abstract judge: scores one open question over some evidence."""

    @abstractmethod
    def score(self, evidence: dict[str, Any]) -> tuple[float, float]:
        """Returns (label, confidence). For binary outcomes label is 0/1;
        for rubric steps label in 1..5."""


class RuleJudge(Judge):
    """Keyword / coverage rubric.  Mirrors the offline decision table."""

    def __init__(self, vocab: StepVocab) -> None:
        self.vocab = vocab

    def score(self, evidence: dict[str, Any]) -> tuple[float, float]:
        kind = evidence.get("kind")
        text = str(evidence.get("text", "")).lower()
        voc = self.vocab
        if kind == "s4_announcement":
            return self._s4_rubric(text, voc), 0.6
        if kind == "s5b_expected_failure":
            ok = any(k in text for k in voc.s5b_judge_keywords)
            return float(ok), 0.5 if text else 0.1
        if kind == "s5c_failure_handling":
            ok = any(k in text for k in voc.s5c_judge_keywords)
            return float(ok), 0.5
        if kind == "open_text":
            return float(len(str(evidence.get("text", "")).strip()) > 20), 0.3
        return 0.0, 0.1

    def _s4_rubric(self, text: str, voc: StepVocab | None = None) -> float:
        terms = (voc or self.vocab).s4_rubric_terms
        hits = sum(1 for t in terms if t in text)
        if hits >= 5:
            return 5.0
        if hits >= 3:
            return 4.0
        if hits >= 1:
            return 3.0
        return 1.0 if text else 0.0
