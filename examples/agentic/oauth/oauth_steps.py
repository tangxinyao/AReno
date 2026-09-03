"""OAuth flow: adjudication table preset S1..S5d (ported from trajlab
`flows/google_email/step_table_preset.py`, imports re-pointed at `grading/`).

The concrete decision-table rules for the google-email OAuth flow.  The generic
machinery (`grading.step_table.StepAnalyzer`) carries the pluggable framing;
this preset only encodes flow rules and reads the analyzer's `vocab` / `cfg`.
"""

from __future__ import annotations

from typing import Any

from grading.messages import Trajectory
from grading.step_table import (StepAnalyzer, StepEntry, _first_tool_output_after,
                                has_status_line, prose_ask_texts, traj_driver)

DEFAULT_STEP_ORDER = ("S1", "S2", "S3", "S4", "S5a", "S5b", "S5c", "S5d")


def _s1(analyzer: StepAnalyzer, obs: list[dict[str, Any]]) -> dict[str, Any]:
    first = next((o for o in obs if o.get("role") == "tool" and o.get("name") == "skill_view"), None)
    if not first:
        return {"score": 0.0, "reason": "无 skill_view 调用"}
    voc = analyzer.vocab
    ok = bool(first["payload"].get("success")) and first["payload"].get("name") in voc.ok_skills
    return {"score": 1.0 if ok else 0.0, "first_skill": first["payload"].get("name"),
            "reason": "路由正确" if ok else "路由失败(如 R1: skill 'email')"}


def _s2(analyzer: StepAnalyzer, obs: list[dict[str, Any]], calls: list[tuple[int, str]],
        outputs: list[str]) -> dict[str, Any]:
    vocab = analyzer.vocab
    auth_probed = any(vocab.not_authenticated in o or vocab.token_file in o
                      or vocab.refresh_failed in o for o in outputs)
    cmds = [c for _, c in calls]
    looped = _repeats_in_a_row(cmds, analyzer.cfg.s2_loop_threshold)
    score = 1.0 if (auth_probed and not looped) else 0.0
    return {"score": score, "partial": bool(auth_probed) and bool(looped),
            "auth_probed": bool(auth_probed), "looped": bool(looped)}


def _repeats_in_a_row(cmds: list[str], threshold: int) -> bool:
    """A probe loop is the same command run back-to-back with nothing in
    between.  A plain census over the whole roll would flag the *canonical*
    trajectory, which legitimately runs ``--check`` twice: once to probe the
    initial state (S2) and once to verify the closed loop (S5d)."""
    run = 1
    for previous, current in zip(cmds, cmds[1:]):
        run = run + 1 if current == previous else 1
        if run >= threshold:
            return True
    return False


def _count_clarify(obs: list[dict[str, Any]]) -> int:
    """Clarify calls the agent made.  Two payload shapes count: the legacy
    flat form (``user_response`` at top level) and the hermes batch form
    (``responses: [{user_response: ...}, ...]``).  An unanswered batch rows
    with empty responses do not credit a call."""
    n = 0
    for o in obs:
        if o.get("name") != "clarify":
            continue
        payload = o.get("payload") or {}
        if payload.get("user_response"):
            n += 1
            continue
        if any(str(r.get("user_response") or "").strip()
               for r in payload.get("responses") or []):
            n += 1
    return n


def _s4_evidence(analyzer: StepAnalyzer, obs: list[dict[str, Any]]) -> str:
    voc = analyzer.vocab
    texts = [o["text"] for o in obs if o.get("role") == "assistant"
             and any(k in (o.get("text") or "") for k in voc.s4_keywords)]
    return texts[-1] if texts else ""


def _s5a(analyzer: StepAnalyzer, obs: list[dict[str, Any]],
         calls: list[tuple[int, str]], outputs: list[str]) -> dict[str, Any]:
    """The step is "the OAuth client credentials reached setup.py".

    Writing the file is one way; loading an already-downloaded secret
    (``--client-secret``, or the shim's canonical auto-load, which reports
    ``client_secret saved``) is the other -- and the only one available on the
    combos that seed the secret.  Crediting the write alone made S5a dead on
    every ``client_secret_ready=yes`` combo and rewarded fabricating a
    credentials file on the ones that block precisely because none exists."""
    marker = analyzer.vocab.secret_file
    writers = [o for o in obs if o.get("name") == "write_file"]
    written = any(marker in (o.get("payload", {}).get("path", "") or "")
                  or marker in str(o.get("payload", {}).get("filename", ""))
                  for o in writers)
    # precise tells only: the "no client credentials ... ask for the
    # client_secret.json path" refusal also carries the marker word, and that
    # message is the *block*, not the step.
    loaded = (any(f"--{marker.replace('_', '-')}" in cmd for _, cmd in calls)
              or any(f"{marker} saved" in out for out in outputs))
    return {"score": 1.0 if (written or loaded) else 0.0,
            "write_events": len(writers), "written": written, "loaded": loaded}


def _s5b(analyzer: StepAnalyzer, obs: list[dict[str, Any]], calls: list[tuple[int, str]]) -> dict[str, Any]:
    voc = analyzer.vocab
    auth = next(((idx, cmd) for idx, cmd in calls
                 if any(m in cmd.lower() for m in voc.authorize_cmd)), None)
    auth_launched = auth is not None
    caveat_text = ""
    if auth:
        after = [o["text"] for o in obs if o.get("role") == "assistant" and o["idx"] > auth[0]]
        caveat_text = after[0] if after else ""
    expected_made = any(k in caveat_text for k in voc.caveat_keywords)
    return {"score": 1.0 if (auth_launched and expected_made) else 0.0,
            "auth_launched": auth_launched,
            "expected_failure_made": expected_made,
            "caveat_text": caveat_text[:200], "auth_idx": auth[0] if auth else None}


def _s5c(analyzer: StepAnalyzer, obs: list[dict[str, Any]], calls: list[tuple[int, str]]) -> dict[str, Any]:
    voc = analyzer.vocab
    exch = [(idx, cmd) for idx, cmd in calls if any(m in cmd for m in voc.auth_code_cmd)]
    if not exch:
        return {"score": 0.0, "exchanged": False, "had_error": False, "handling_text": ""}
    idx, _cmd = exch[-1]
    out = _first_tool_output_after(obs, idx)
    had_error = any(e in out for e in voc.grant_errors) or "error" in out.lower()
    handling_text = ""
    if had_error:
        after = [o["text"] for o in obs if o.get("role") == "assistant" and o["idx"] > idx]
        handling_text = after[0] if after else ""
    return {"score": 1.0, "exchanged": True, "had_error": had_error,
            "handling_text": handling_text[:200]}


def _s5d(analyzer: StepAnalyzer, obs: list[dict[str, Any]], calls: list[tuple[int, str]]) -> dict[str, Any]:
    voc = analyzer.vocab
    checks = [(idx, cmd) for idx, cmd in calls if voc.check_cmd in cmd]
    if not checks:
        return {"score": 0.0, "reason": "无 --check"}
    # the negative marker contains the positive one, so match the status
    # prefix -- per LINE, because `--auth-code ... && --check` bundles two
    # commands into one tool result and the status is then not line 1.
    auth_ok = any(has_status_line(_first_tool_output_after(obs, idx), voc.authenticated)
                  for idx, _ in checks)
    refreshed = any(voc.refresh_failed in _first_tool_output_after(obs, idx) for idx, _ in checks)
    return {"score": 1.0 if auth_ok else 0.0, "auth_ok": bool(auth_ok),
            "refreshed_failed": bool(refreshed), "check_count": len(checks)}


def OAUTH_TABLE(analyzer: StepAnalyzer, traj: Trajectory, obs: list[dict[str, Any]],
                calls: list[tuple[int, str]], outputs: list[str]) -> list[StepEntry]:
    """The google-email decision table.

    `DEFAULT_STEP_ORDER` is the single source for the step-name sequence.
    """
    cfg = analyzer.cfg
    judge = analyzer.judge

    def s1() -> dict[str, Any]:
        return _s1(analyzer, obs)

    # S2 row computed once at table-build time (its "partial" flag also decides
    # the method label below); deliberate so _s2 isn't evaluated twice per roll.
    s2_row = _s2(analyzer, obs, calls, outputs)

    def s2() -> dict[str, Any]:
        return s2_row

    def s3() -> dict[str, Any]:
        n = _count_clarify(obs)
        # prose asks credit the same as clarify tool calls.  Only counted for
        # trajectories that carry a driver marker -- unmarked scripted
        # trajectories never prose-ask, and their golden adjudication stays
        # byte-identical.
        prose = 0
        if traj_driver(traj):
            prose = len(prose_ask_texts(traj, analyzer.vocab.ask_keywords))
        n += prose
        row: dict[str, Any] = {"score": 1.0 if n >= cfg.s3_clarify_min else 0.0,
                               "clarify_count": n}
        if prose:
            row["prose_asks"] = prose
        return row

    def s4() -> dict[str, Any]:
        text = _s4_evidence(analyzer, obs)
        label, _ = judge.score({"kind": "s4_announcement", "text": text})
        return {"score": label, "evidence": text[:200]}

    def s5a() -> dict[str, Any]:
        return _s5a(analyzer, obs, calls, outputs)

    def s5b() -> dict[str, Any]:
        row = _s5b(analyzer, obs, calls)
        caveat, _ = judge.score({"kind": "s5b_expected_failure", "text": row["caveat_text"]})
        return {**row, "score": 1.0 if (row["auth_launched"] and (row["expected_failure_made"] or caveat >= 0.5)) else 0.0}

    def s5c() -> dict[str, Any]:
        row = _s5c(analyzer, obs, calls)
        handled, _ = judge.score({"kind": "s5c_failure_handling", "text": row["handling_text"]})
        return {**row, "score": 1.0 if row["exchanged"] and (not row["had_error"] or handled >= 0.5) else 0.0}

    def s5d() -> dict[str, Any]:
        return _s5d(analyzer, obs, calls)

    entries = {
        "S1": StepEntry("S1", "rule",
                        s1,
                        lambda r: ["首个工具调用加载了正确技能(himalaya/google-workspace)"
                                   if r["score"] else "首个工具调用路由失败(如 skill 'email')"]),
        "S2": StepEntry("S2", "rule+llm" if s2_row["partial"] else "rule",
                        s2,
                        lambda r: (["探测到了 OAuth 未认证状态" if r["auth_probed"] else "未探测 google_token.json 状态"]
                                   + (["探测无进展(相同命令循环)" if r["looped"] else "探测后转入了后续正确动作"]))),
        "S3": StepEntry("S3", "rule",
                        s3,
                        lambda r: [f"手工前提出问题数={r['clarify_count']}(阈值 {cfg.s3_clarify_min})"]),
        "S4": StepEntry("S4", "rule",
                        s4,
                        lambda r: ["宣达唯一手动步骤(client_secret.json)" if r["score"] >= 3
                                   else "未清晰宣达唯一手动步骤"]),
        "S5a": StepEntry("S5a", "rule",
                         s5a,
                         lambda r: ["client_secret 已提供给 setup.py" if r["score"]
                                    else "未注入 client_secret"]),
        "S5b": StepEntry("S5b", "rule",
                         s5b,
                         lambda r: ["已启动 /authorize 并回传授权链接" if r["auth_launched"] else "未启动授权",
                                    "已交代预期失败" if r["expected_failure_made"] else "未交代预期失败"]),
        "S5c": StepEntry("S5c", "rule",
                         s5c,
                         lambda r: ["已用授权码换 token" if r["exchanged"] else "未兑换 code",
                                    "错误分支处理" if not r["had_error"] else "对错误无处理"]),
        "S5d": StepEntry("S5d", "rule",
                         s5d,
                         lambda r: ["--check 返回 AUTHENTICATED" if r["score"] else "--check 未确认闭环"]),
    }
    return [entries[code] for code in DEFAULT_STEP_ORDER]
