"""OAuth flow: observation vocabulary preset (ported from trajlab
`flows/google_email/vocab_preset.py`).

The exact wording the google-email adjudication reads off trajectories --
status markers, skill names, command patterns, judge rubric terms.  All flow
strings: the grading core (`grading/`) only knows the `StepVocab` *shape*,
never these values.
"""

from __future__ import annotations

from grading.vocab import StepVocab


GOOGLE_EMAIL_VOCAB = StepVocab(
    # S1 routing -- acceptable skill names returned by skill_view
    ok_skills=("himalaya", "google-workspace"),
    # result-level / S2 / S5d observation markers
    authenticated="AUTHENTICATED",
    not_authenticated="NOT_AUTHENTICATED",
    refresh_failed="REFRESH_FAILED",
    token_file="google_token.json",
    # S5a -- injected secret
    secret_file="client_secret",
    # S5b -- auth-url launch + expected-failure caveat.  Both spellings count:
    # the raw endpoint (`curl .../authorize`) and the flag the skill doc
    # actually documents (`$GSETUP --auth-url --services ...`), which carries
    # no "authorize" substring at all.
    authorize_cmd=("authorize", "--auth-url"),
    caveat_keywords=("预期", "会失败", "access_denied", "403", "回传", "授权"),
    # S5c -- code exchange + grant errors
    auth_code_cmd=("--auth-code", "/token"),
    grant_errors=("invalid_grant", "invalid_client"),
    # S5d -- verification command
    check_cmd="--check",
    # S4 -- plan/announcement evidence inside assistant narration
    s4_keywords=("手动", "Google Cloud", "client_secret", "标准流程"),
    # rule-judge rubrics for the open steps
    s4_rubric_terms=(
        "唯一手动", "唯一的手动", "手动操作",
        "google cloud", "client_secret", "oauth 客户端", "oauth client",
        "标准流程", "高级保护", "两种方式", "两条路径", "app password",
    ),
    s5b_judge_keywords=("预期", "expected", "会失败", "access_denied", "403"),
    s5c_judge_keywords=("invalid_grant", "过期", "expired", "重新", "换一个", "重试"),
    # failure taxonomy -- install branch + abandon narration
    install_tool="himalaya",
    install_cmd=("install", "curl", "pip"),
    network_fail=("not found", "timed out", "blocked host"),
    abandon_keywords=("需要你提供", "等你", "停下来", "需要你手动", "请你"),
    # degraded-slice evidence (kept for vocab shape parity with the engine)
    degraded_early_markers=("curl", "skill"),
    degraded_auth_markers=("--auth-code", "token"),
    degraded_abandon_markers=("provide", "等你", "credentials"),
    # prose-ask shape: phrases that mark an assistant message as "asking the
    # user" when no `clarify` tool call exists.  Deliberately ask-shaped ONLY:
    # a hit credits a question with the same weight as a clarify tool call, so
    # precision beats recall -- a miss falls through to continue-nudging, a
    # false hit stops the roll (question-mark endings + explicit ask verbs only).
    ask_keywords=(
        "请提供", "请告诉我", "请回复", "请选择", "请确认", "请把",
        "需要你提供", "需要你", "等你提供", "等你确认",
        "选一个", "选哪个", "选哪条", "你的选择",
        "please provide", "please tell", "please reply", "please share",
        "would you like", "which option", "reply with", "let me know",
    ),
)
