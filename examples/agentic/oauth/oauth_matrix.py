"""OAuth flow: scenario-matrix preset (ported from trajlab
`flows/google_email/matrix_preset.py`).

The whole axis/dimension table plus the combo -> user/env/server enrichers for
the google-email OAuth flow.  The registry indirection was dropped: the
dataset generator calls these builders directly.

Path contract note: trajlab's real-driver runs lived in a container with the
canonical secret at /root/oauth/client_secret.json.  In the areno example the
workspace root IS the roll home, so the canonical locations are
workspace-relative (`oauth/client_secret.json` under the workspace root).
"""

from __future__ import annotations

from typing import Any

from grading.matrix import Dimension, scenario_id_for

USER_PROMPT = "帮我连接我的谷歌邮箱"  # canonical prompt (maximally ambiguous on purpose)


def google_email_dimensions() -> list[Dimension]:
    """The google-email 6-axis matrix.

    Every axis carries its stable id-abbreviation table.  The stable
    scenario-id contract lives here as *data*, not as a hard-coded encoder.
    """
    return [
        Dimension("service_scope", ("email_only", "email_calendar", "full_workspace"),
                  {"email_only": "email", "email_calendar": "mailcal", "full_workspace": "ws"}),
        Dimension("advanced_protection", ("no", "yes"),
                  {"no": "std", "yes": "adv"}),
        Dimension("test_user_added", ("yes", "no"),
                  {"yes": "testuser", "no": "notestuser"}),
        Dimension("client_secret_ready", ("yes", "no"),
                  {"yes": "secret", "no": "nosecret"}),
        Dimension("token_state", ("absent", "expired", "revoked"),
                  {"absent": "noauth", "expired": "tokexp", "revoked": "tokrev"}),
        Dimension("code_validity", ("valid", "expired", "user_aborted"),
                  {"valid": "validcode", "expired": "codeexp", "user_aborted": "abortcode"}),
    ]


standard_dimensions = google_email_dimensions  # backward-compatible alias


def server_for(code_validity: str, test_user_added: str, token_state: str) -> dict[str, str]:
    """Mock behaviour table mapping."""
    if code_validity == "user_aborted":
        auth_url_behavior = "abandon"
    elif code_validity == "expired":
        auth_url_behavior = "code_expired"
    elif test_user_added == "no":
        auth_url_behavior = "access_denied"
    else:
        auth_url_behavior = "success"
    token_behavior = "refresh_failed" if token_state in ("expired", "revoked") else "success"
    return {"auth_url_behavior": auth_url_behavior,
            "token_behavior": token_behavior, "token_state": token_state}


def env_for(token_state: str, service_scope: str, client_secret_ready: str) -> dict[str, Any]:
    return {"platform": "linux", "himalaya_installed": False,
            "token_state": token_state,
            "service_scope": service_scope,
            "client_secret_ready": client_secret_ready == "yes",
            "paths": {"client_secret": "oauth/client_secret.json"},
            "oauth_mock": "full_prot_pub"}


def _anchored(topics: str, intents: str) -> str:
    """Regex matching a question when a topic word and an intent word co-occur
    (either order, small gap) -- tolerant enough for a real model's improvised
    clarify wording, tight enough that the canonical question of a *different*
    intent does not collide."""
    return (rf"(?:{topics}).{{0,24}}(?:{intents})"
            rf"|(?:{intents}).{{0,24}}(?:{topics})")


# Clarify keys are regexes, matched by the clarify routing (substring OR
# re.search, IGNORECASE).  The extra alternatives exist so a real model
# improvising its own phrasing still lands on the scripted answer.  Protection
# is checked before secret ("管理员/允许列表" must beat the "oauth client"
# secret tell in a mixed question) and both before scope, whose broadened
# intent list additionally needs a topic word (gmail/邮箱/谷歌/...) so
# off-task questions fall through to the deterministic default.
_SECRET_INTENT = (r"client[_\s-]?secret|secret\s*json|json\s*文件|凭据|密钥文件"
                  r"|建项目|创建项目|oauth\s*client")
_PROTECTION_INTENT = (r"高级保护|advanced\s*protection|硬件安全密钥|安全密钥"
                      r"|管理员|允许列表|第三方应用")
_SCOPE_INTENT = (r"做什么|干啥|干嘛|用途|场景|哪种|范围|怎么连|怎么用|连接方式"
                 r"|什么功能|收发|imap|smtp|app\s*password"
                 r"|只需要|需要连接|需要哪|哪些服务|什么服务|还要什么|连接什么|还差什么")

# the ROUTED clarify keys, lifted to module constants so the process gate
# (`oauth_gate.py`) judges the disambiguate obligations against the EXACT same
# regexes the clarify routing matches on (a question that fulfils an
# obligation is precisely one that would land on the scripted answer).
_SCOPE_TOPICS = r"谷歌|google|gmail|邮箱|邮件|workspace"
SCOPE_CLARIFY_KEY = _anchored(_SCOPE_TOPICS, _SCOPE_INTENT)
PROTECTION_CLARIFY_KEY = _PROTECTION_INTENT
SECRET_CLARIFY_KEY = _SECRET_INTENT


def user_for(service_scope: str, advanced_protection: str, secret_answer: str) -> dict[str, Any]:
    scope_answers = {
        "email_only": "只收发邮件(用 App Password + himalaya，2 分钟搞定，不需要建 Google Cloud 项目)",
        "email_calendar": "邮件 + 日历/Drive/文档等完整 Google Workspace(需要 Google Cloud OAuth 授权,约5-10分钟)",
        "full_workspace": "邮件 + 日历/Drive/文档等完整 Google Workspace(需要 Google Cloud OAuth 授权,约5-10分钟)",
    }
    return {
        "prompt": USER_PROMPT,
        "clarify_answers": {
            PROTECTION_CLARIFY_KEY:
                ("没有 / 不确定(标准流程即可)" if advanced_protection == "no"
                 else "有(需要 Workspace 管理员先把 OAuth client 加入允许列表)"),
            SECRET_CLARIFY_KEY: secret_answer,
            SCOPE_CLARIFY_KEY: scope_answers[service_scope],
        },
        "browser": "scripted",
        "steer": {},
    }


def _secret_answer(combo: dict[str, str], injected: str | None = None) -> str:
    ready = combo["client_secret_ready"] == "yes"
    if injected == "abandon":
        return "未建项目"
    return "已创建并下载：oauth/client_secret.json" if ready else "未建项目"


def expected_outcome(combo: dict[str, str]) -> dict[str, str]:
    """Expected result of a combo, derived from the `server_for` behaviour
    table (which already encodes what a combo can reach -- this exporter adds
    no second source of truth).

    outcome=reach -- an agent acting well can finish OAuth: authorize success
        branch ∧ no lingering bad token ∧ client_secret ready; expected final
        state AUTHENTICATED.
    outcome=blocked -- a business block fires; expected final state is the
        correct branch handled cleanly, never a success claim:
        access_denied (test user) / abandon, code_expired / nosecret /
        token_expired, token_revoked.
    """
    server = server_for(combo.get("code_validity", "valid"),
                        combo.get("test_user_added", "yes"),
                        combo.get("token_state", "absent"))
    if server["auth_url_behavior"] != "success":
        reason = server["auth_url_behavior"]  # access_denied / abandon / code_expired
    elif combo.get("client_secret_ready", "yes") != "yes":
        reason = "nosecret"
    elif server["token_behavior"] != "success":
        reason = f"token_{combo.get('token_state', 'absent')}"
    else:
        reason = "reach"
    return {"outcome": "reach" if reason == "reach" else "blocked", "reason": reason}


def google_scenario_id_for(combo: dict[str, str], version: str = "v1") -> str:
    """Deterministic google-email id from a combo, in the preset's axis order."""
    return scenario_id_for(combo, version=version, dims=google_email_dimensions())
