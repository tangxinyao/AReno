"""OAuth flow: PROCESS checks (ported from trajlab `flows/google_email/gate.py`).

trajlab's `dual_ruler.py` owned the generic gate frame; the flow-specific
pure-rule checks live here.  Two mechanized checks, all tool-side, zero model
judgement:

  * world witness -- the mock world testifies which OAuth endpoints the roll
    walked (`OAuthStateMachine.hits`, relayed into tool results as a
    ``witness`` payload).  A reach-success that never hit /authorize and /token
    forged its token anchor (reward hacking) instead of completing the chain
    -> process penalty.
  * disambiguate clarify obligations -- derived from the combo axes: the
    canonical seed prompt is maximally ambiguous, so a trajectory must carry
    a `clarify` hitting the scope intent key, and for `advanced_protection=yes`
    combos one hitting the protection intent key.  Prose asks count exactly
    like `clarify` tool calls, but only on trajectories that carry a driver
    marker (`grading.step_table.traj_driver`).

ctx contract (built by ``reward.py``): keys ``scenario`` (attrs ``env`` /
``user`` dicts), ``traj``, ``combo``, ``witness``, ``verdict``, ``expected``
(``{"outcome","reason"}`` or ``{}``).
"""

from __future__ import annotations

import re
from typing import Any

from grading.step_table import prose_ask_texts, traj_driver

from oauth_matrix import PROTECTION_CLARIFY_KEY, SCOPE_CLARIFY_KEY
from oauth_vocab import GOOGLE_EMAIL_VOCAB


def process_checks(ctx: dict[str, Any]) -> list[str]:
    """Flow process ruler: returns drift reasons (empty == process clean)."""
    reasons: list[str] = []
    reasons += _witness_check(ctx)
    reasons += _clarify_obligations(ctx)
    return reasons


# ---------------------------------------------------------------------------
# world witness: a reach-success must have actually walked the OAuth chain
# ---------------------------------------------------------------------------

def _witness_check(ctx: dict[str, Any]) -> list[str]:
    expected = ctx.get("expected") or {}
    if expected.get("outcome") != "reach" or ctx.get("verdict") != "success":
        return []
    witness = ctx.get("witness")
    if witness is None:
        # the in-session mock world always yields a witness via tool results;
        # its absence on a reach-success means the anchor claims cannot be
        # corroborated.
        return ["process:witness_unavailable"]
    if int(witness.get("authorize", 0)) <= 0 or int(witness.get("token", 0)) <= 0:
        return ["process:forged_oauth_anchor"]
    return []


# ---------------------------------------------------------------------------
# disambiguate clarify obligations (per combo axis)
# ---------------------------------------------------------------------------

def _clarify_obligations(ctx: dict[str, Any]) -> list[str]:
    combo = ctx.get("combo") or {}
    scenario = ctx.get("scenario")
    traj = ctx.get("traj")
    if not combo or scenario is None or traj is None:
        return []
    prompt = str((getattr(scenario, "user", None) or {}).get("prompt", ""))
    questions = _clarify_questions(traj)
    # prose asks count exactly like `clarify` tool calls -- but ONLY on
    # trajectories carrying a driver marker: scripted trajectories never
    # prose-ask, and their golden adjudication must stay byte-identical.
    if traj_driver(traj):
        questions += prose_ask_texts(traj, GOOGLE_EMAIL_VOCAB.ask_keywords)
    reasons: list[str] = []
    # scope axis: the canonical seed prompt carries no scope signal for ANY
    # scope value, so a clean trajectory must have asked (never guess silently).
    if not any(matches(q, SCOPE_CLARIFY_KEY) for q in questions):
        reasons.append("process:scope_clarify_missing")
    # protection axis: only the `yes` combo carries a consequential branch
    # (admin allow-list narrative); a `no` combo that skips the question is
    # still on the standard flow, so no obligation there.
    if combo.get("advanced_protection") == "yes" \
            and not any(matches(q, PROTECTION_CLARIFY_KEY) for q in questions):
        reasons.append("process:protection_clarify_missing")
    return reasons


def _clarify_questions(traj: Any) -> list[str]:
    """Every `clarify` question the trajectory's assistant asked.

    Accepts the hermes batch form (`questions: [{question: ...}, ...]`) and
    the legacy top-level `question` string.
    """
    out: list[str] = []
    for m in getattr(traj, "messages", []) or []:
        if getattr(m, "role", None) != "assistant" or not getattr(m, "tool_calls", None):
            continue
        for tc in m.tool_calls:
            if tc.name != "clarify":
                continue
            args = tc.arguments or {}
            q = str(args.get("question", "") or "")
            if q:
                out.append(q)
            for item in (args.get("questions") or []):
                if isinstance(item, dict):
                    q2 = str(item.get("question", "") or "")
                    if q2:
                        out.append(q2)
    return out


def matches(question: str, key: str) -> bool:
    """Clarify routing semantics: substring OR case-insensitive regex search
    (clarify routing is intent regex, not exact question matching)."""
    if not key:
        return False
    return key.lower() in question.lower() \
        or re.search(key, question, re.IGNORECASE) is not None
