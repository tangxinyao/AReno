"""Flow observation vocabulary shape (ported from trajlab/label/vocab.py).

Only the ``StepVocab`` dataclass came along -- the google-email values live in
the example's ``oauth_vocab.py`` and the registry-based ``vocab_for`` lookup
was dropped (the table factory is injected directly now).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepVocab:
    """Flow-specific markers consumed by the adjudication table + judge.

    All fields are required -- a flow preset supplies its own wording for each
    slot (there is no built-in default vocabulary).
    """

    # S1 routing -- acceptable skill names returned by skill_view
    ok_skills: tuple[str, ...]
    # result-level / S2 / S5d observation markers
    authenticated: str
    not_authenticated: str
    refresh_failed: str
    token_file: str
    # S5a -- injected secret
    secret_file: str
    # S5b -- auth-url launch + expected-failure caveat
    authorize_cmd: tuple[str, ...]
    caveat_keywords: tuple[str, ...]
    # S5c -- code exchange + grant errors
    auth_code_cmd: tuple[str, ...]
    grant_errors: tuple[str, ...]
    # S5d -- verification command
    check_cmd: str
    # S4 -- plan/announcement evidence inside assistant narration
    s4_keywords: tuple[str, ...]
    # rule-judge rubrics for the open steps
    s4_rubric_terms: tuple[str, ...]
    s5b_judge_keywords: tuple[str, ...]
    s5c_judge_keywords: tuple[str, ...]
    # failure taxonomy -- install branch + abandon narration (unused by the
    # S1..S5d table itself, kept so vocab presets stay shape-complete)
    install_tool: str
    install_cmd: tuple[str, ...]
    network_fail: tuple[str, ...]
    abandon_keywords: tuple[str, ...]
    # degraded-slice evidence (offline artifacts; unused here, kept for shape)
    degraded_early_markers: tuple[str, ...]
    degraded_auth_markers: tuple[str, ...]
    degraded_abandon_markers: tuple[str, ...]
    # prose-ask shape: phrases whose presence in an assistant message marks
    # "the model is asking the user" -- used by the S3/clarify-obligation
    # adjudication (a asked-in-prose question is the same conversational event
    # as a clarify tool call).  Optional (default empty: scripts never ask).
    ask_keywords: tuple[str, ...] = ()


def _empty_vocab() -> StepVocab:
    """Neutral empty vocabulary for a pure-rule custom table (seam tests)."""
    return StepVocab(ok_skills=(), authenticated="", not_authenticated="",
                     refresh_failed="", token_file="", secret_file="",
                     authorize_cmd=(), caveat_keywords=(), auth_code_cmd=(),
                     grant_errors=(), check_cmd="", s4_keywords=(),
                     s4_rubric_terms=(), s5b_judge_keywords=(),
                     s5c_judge_keywords=(), install_tool="", install_cmd=(),
                     network_fail=(), abandon_keywords=(),
                     degraded_early_markers=(), degraded_auth_markers=(),
                     degraded_abandon_markers=())
