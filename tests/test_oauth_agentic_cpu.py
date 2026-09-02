from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

OAUTH_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "oauth"
sys.path.insert(0, str(OAUTH_EXAMPLE))

import oauth_env  # noqa: E402
import oauth_gate  # noqa: E402
import oauth_matrix  # noqa: E402
import oauth_seeds  # noqa: E402
import oauth_tools  # noqa: E402
import oauth_world  # noqa: E402
import reward as oauth_reward  # noqa: E402
from dataset_loader import load_training_dataset  # noqa: E402
from grading.matrix import enumerate_combos  # noqa: E402
from grading.messages import Trajectory  # noqa: E402
from grading.step_table import AnalyzerCfg, StepAnalyzer  # noqa: E402
from oauth_steps import DEFAULT_STEP_ORDER, OAUTH_TABLE, _count_clarify  # noqa: E402
from oauth_vocab import GOOGLE_EMAIL_VOCAB  # noqa: E402

FIXTURES = OAUTH_EXAMPLE / "fixtures"


# ---------------------------------------------------------------------------
# matrix / ids
# ---------------------------------------------------------------------------

def _combo(**overrides: str) -> dict[str, str]:
    combo = {d.key: d.values[0] for d in oauth_matrix.google_email_dimensions()}
    combo.update(overrides)
    return combo


def test_matrix_enumerates_216_unique_google_email_ids():
    dims = oauth_matrix.google_email_dimensions()
    combos = enumerate_combos(dims)
    assert len(combos) == 216
    ids = {oauth_matrix.google_scenario_id_for(c) for c in combos}
    assert len(ids) == 216
    # stable id shape in preset axis order: `-`-joined fields, no `_`
    assert oauth_matrix.google_scenario_id_for(
        _combo(service_scope="full_workspace", advanced_protection="no",
               test_user_added="yes", client_secret_ready="no", token_state="absent",
               code_validity="valid")) == "ws-std-testuser-nosecret-noauth-validcode-v1"
    assert all("_" not in sid and sid.endswith("-v1") for sid in ids)


def test_expected_outcome_matches_server_behaviour_table():
    assert oauth_matrix.expected_outcome(_combo()) == {"outcome": "reach", "reason": "reach"}
    assert oauth_matrix.expected_outcome(_combo(test_user_added="no")) == \
        {"outcome": "blocked", "reason": "access_denied"}
    assert oauth_matrix.expected_outcome(_combo(code_validity="user_aborted")) == \
        {"outcome": "blocked", "reason": "abandon"}
    assert oauth_matrix.expected_outcome(_combo(code_validity="expired")) == \
        {"outcome": "blocked", "reason": "code_expired"}
    assert oauth_matrix.expected_outcome(_combo(client_secret_ready="no")) == \
        {"outcome": "blocked", "reason": "nosecret"}
    for token_state, reason in (("expired", "token_expired"), ("revoked", "token_revoked")):
        assert oauth_matrix.expected_outcome(_combo(token_state=token_state)) == \
            {"outcome": "blocked", "reason": reason}
    # several blocks together degrade to the first failure on the chain
    assert oauth_matrix.expected_outcome(_combo(client_secret_ready="no",
                                                token_state="expired",
                                                code_validity="expired")) == \
        {"outcome": "blocked", "reason": "code_expired"}


def test_server_for_maps_browser_and_token_behaviour():
    assert oauth_matrix.server_for("user_aborted", "yes", "absent") == {
        "auth_url_behavior": "abandon", "token_behavior": "success", "token_state": "absent"}
    assert oauth_matrix.server_for("expired", "yes", "absent") == {
        "auth_url_behavior": "code_expired", "token_behavior": "success", "token_state": "absent"}
    assert oauth_matrix.server_for("valid", "no", "absent")["auth_url_behavior"] == "access_denied"
    assert oauth_matrix.server_for("valid", "yes", "revoked")["token_behavior"] == "refresh_failed"
    assert oauth_matrix.server_for("valid", "yes", "absent")["auth_url_behavior"] == "success"


# ---------------------------------------------------------------------------
# world: state machine + HTTP server
# ---------------------------------------------------------------------------

def test_state_machine_behaviour_table(tmp_path):
    paths = oauth_world.google_paths(tmp_path)
    assert paths["google_token"] == str(tmp_path / ".hermes" / "google_token.json")

    machine = oauth_world.OAuthStateMachine(oauth_matrix.server_for("valid", "yes", "absent"), paths)
    ok, _redirect, state = machine.authorize("/authorize")
    assert ok and state["code"].startswith("mock_code_")
    done, token = machine.exchange(state["code"])
    assert done and token["access_token"]
    assert machine.check() == {"status": "AUTHENTICATED",
                               "detail": "token verified via mock refresh"}
    assert machine.hits == {"authorize": 1, "token": 1}

    denied = oauth_world.OAuthStateMachine(oauth_matrix.server_for("valid", "no", "absent"), paths)
    ok, reason, state = denied.authorize("/authorize")
    assert not ok and reason == "access_denied" and state is None

    expired = oauth_world.OAuthStateMachine(oauth_matrix.server_for("expired", "yes", "absent"), paths)
    ok, _redirect, state = expired.authorize("/authorize")
    assert ok and "expired" in state["code"]
    done, err = expired.exchange(state["code"])
    assert not done and err["error"] == "invalid_grant"

    once = oauth_world.OAuthStateMachine({"token_behavior": "invalid_grant_once"}, paths)
    assert once.exchange("mock_code_1_ok")[0] is False
    assert once.exchange("mock_code_2_ok")[0] is True  # a retry wins

    stale = oauth_world.OAuthStateMachine(oauth_matrix.server_for("valid", "yes", "revoked"), paths)
    stale.exchange("mock_code_1_ok")
    assert stale.check()["status"] == "REFRESH_FAILED"

    fresh = oauth_world.OAuthStateMachine({}, paths)
    assert fresh.check()["status"] == "NOT_AUTHENTICATED"


def test_mock_oauth_server_end_to_end(tmp_path):
    server = oauth_world.MockOAuthServer(
        oauth_matrix.server_for("valid", "yes", "absent"),
        oauth_world.google_paths(tmp_path), port=0).start()
    try:
        assert server.port > 0  # ephemeral binding picked a real port
        with urllib.request.urlopen(server.base_url + "/health") as resp:
            assert json.loads(resp.read())["ok"] is True
        with urllib.request.urlopen(server.base_url + "/authorize") as resp:
            auth = json.loads(resp.read())
        req = urllib.request.Request(server.base_url + "/token",
                                     data=json.dumps({"code": auth["code"]}).encode(),
                                     method="POST")
        with urllib.request.urlopen(req) as resp:
            token = json.loads(resp.read())
        assert token["access_token"]
        req = urllib.request.Request(server.base_url + "/check", data=b"{}", method="POST")
        with urllib.request.urlopen(req) as resp:
            check = json.loads(resp.read())
        assert check["status"] == "AUTHENTICATED"
        assert server.snapshot()["hits"] == {"authorize": 1, "token": 1}
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# records / workspace materialization
# ---------------------------------------------------------------------------

def test_build_record_is_deterministic_and_valid():
    combo = _combo()
    record = oauth_env.build_record(combo, index=7)
    assert record["id"] == "oauth-00007"
    assert record["prompt"] == oauth_matrix.USER_PROMPT
    assert record["context_budget"] == 10_000 and record["max_turns"] == 12
    assert record == oauth_env.build_record(combo, index=7)
    assert oauth_env.validate_record(record) == []

    blocked = oauth_env.build_record(_combo(client_secret_ready="no"), index=8)
    spec = oauth_env.parse_task_spec(blocked)
    assert spec["expected"] == {"outcome": "blocked", "reason": "nosecret"}
    assert oauth_env.validate_record(blocked) == []


def test_workspace_materialization_matches_the_combo(tmp_path):
    ready = oauth_env.build_record(_combo(), index=1)
    paths = oauth_env.materialize_workspace(ready, tmp_path / "ws")
    assert (tmp_path / "ws" / ".hermes" / "skills" / "email" / "himalaya" / "SKILL.md").is_file()
    assert (tmp_path / "ws" / ".hermes" / "skills" / "productivity"
            / "google-workspace" / "SKILL.md").is_file()
    assert (tmp_path / "ws" / oauth_seeds.SETUP_SHIM_PATH).is_file()
    assert (tmp_path / "ws" / "oauth" / "client_secret.json").is_file()
    assert not (tmp_path / "ws" / ".hermes" / "google_token.json").exists()  # never pre-authenticated
    assert paths["google_token"].endswith("google_token.json")

    starved = oauth_env.build_record(_combo(client_secret_ready="no"), index=2)
    oauth_env.materialize_workspace(starved, tmp_path / "ws2")
    assert not (tmp_path / "ws2" / "oauth" / "client_secret.json").exists()


def _transcript(workspace):
    """Episode-loop message accumulator used by the workspace tests."""
    messages: list[dict] = [{"role": "user", "content": oauth_matrix.USER_PROMPT}]

    def call(name: str, arguments: dict, narration: str = "") -> dict:
        call_id = f"call_{len(messages):03d}"
        messages.append({"role": "assistant", "content": narration, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": arguments}}]})
        result = oauth_tools.run_tool(workspace, name, arguments, messages=messages)
        messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                         "content": json.dumps(result, ensure_ascii=False, sort_keys=True)})
        return result

    return messages, call


def test_workspace_terminal_walks_the_full_oauth_chain():
    """One real subprocess episode through the fake world, graded by the
    ported S1..S5d table (CPU-safe: local processes + loopback HTTP only)."""
    from oauth_tools import OAuthWorkspace

    record = oauth_env.build_record(_combo(), index=0)
    workspace = OAuthWorkspace.from_record(record)
    setup = oauth_seeds.SETUP_SHIM_PATH
    try:
        messages, call = _transcript(workspace)

        check0 = call("terminal", {"command": f"python3 {setup} --check"}, "先探测状态。")
        assert check0["output"].startswith("NOT_AUTHENTICATED")
        assert check0["witness"] == {"authorize": 0, "token": 0}
        assert "progress_score" in check0

        skill = call("skill_view", {"name": "google-workspace"}, "加载技能。")
        assert skill["success"] and skill["name"] == "google-workspace"

        # scope + protection clarify, answered by the scripted user
        scope = call("clarify", {"questions": [
            {"question": "你希望连接哪些 Google 服务？邮件还是完整 Workspace？"}]}, "先问清范围。")
        assert "完整 Google Workspace" in scope["responses"][0]["user_response"]
        protection = call("clarify", {"questions": [
            {"question": "账号是否启用了高级保护（Advanced Protection）？"}]}, "再确认保护。")
        assert protection["responses"][0]["user_response"]

        # authorize: /authorize must be IN the command (S5b) and hit the world (witness)
        launched = call("terminal", {"command":
                         'curl -s "$GOOGLE_OAUTH_BASE_URL/authorize?client_id=mock-client-1" 2>&1 | head -5'},
                        "启动授权。若 access_denied 会向用户交代预期失败。")
        auth = json.loads(launched["output"])
        assert auth["redirect_uri"].startswith("http://localhost:1/?code=")

        call("terminal", {"command": f'python3 {setup} --auth-code "{auth["redirect_uri"]}"'},
             "用授权码兑换 token。")
        final = call("terminal", {"command": f"python3 {setup} --check"}, "验证闭环。")
        assert final["output"].startswith("AUTHENTICATED"), final["output"]
        assert final.get("solved") is True
        assert final["witness"] == {"authorize": 1, "token": 1}

        report = workspace.step_report(messages)
        assert [row["name"] for row in report.step_scores] == list(DEFAULT_STEP_ORDER)
        assert report.score_of("S1") == 1.0
        assert report.score_of("S2") == 1.0
        assert report.score_of("S3") == 1.0
        assert report.score_of("S5b") == 1.0
        assert report.score_of("S5c") == 1.0
        assert report.score_of("S5d") == 1.0
        assert report.score_of("S5a") == 0.0  # the secret came pre-seeded; nothing was written
    finally:
        workspace.close()


# ---------------------------------------------------------------------------
# adjudication table (S1..S5d) on scripted transcripts
# ---------------------------------------------------------------------------

def _analyzer() -> StepAnalyzer:
    return StepAnalyzer(GOOGLE_EMAIL_VOCAB, OAUTH_TABLE, AnalyzerCfg())


def _assistant_tool(name: str, call_id: str, arguments: dict, narration: str = "") -> dict:
    return {"role": "assistant", "content": narration, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def _tool_msg(name: str, call_id: str, payload: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name,
            "content": json.dumps(payload, ensure_ascii=False)}


def _clarify_pair() -> list[dict]:
    return [
        _assistant_tool("clarify", "c1", {"questions": [
            {"question": "你需要用谷歌邮箱做什么？"}]}),
        _tool_msg("clarify", "c1", {"responses": [
            {"question": "用途", "user_response": "邮件 + 完整 Google Workspace"}]}),
        _assistant_tool("clarify", "c2", {"questions": [
            {"question": "是否启用了高级保护？"}]}),
        _tool_msg("clarify", "c2", {"responses": [
            {"question": "保护", "user_response": "没有(标准流程即可)"}]}),
    ]


def _happy_messages() -> list[dict]:
    """Best-play transcript for a reach combo (mirrors golden happy_workspace)."""
    witness = {"authorize": 1, "token": 1}
    return [
        {"role": "user", "content": oauth_matrix.USER_PROMPT},
        *_clarify_pair(),
        {"role": "assistant", "content": (
            "计划：加载 google-workspace 技能。整个流程唯一需要你手动操作的部分是在 "
            "Google Cloud 里创建 OAuth 客户端并下载 client_secret.json；你已经创建并下载，"
            "所以走标准流程即可。"), "tool_calls": []},
        _assistant_tool("skill_view", "t1", {"name": "google-workspace"}, "先加载技能。"),
        _tool_msg("skill_view", "t1", {"success": True, "name": "google-workspace",
                                       "content": "# Google Workspace\n", "witness": dict(witness)}),
        _assistant_tool("write_file", "t2",
                        {"path": "oauth/client_secret.json",
                         "content": '{"installed":{"client_id":"mock-client-1"}}'},
                        "把凭据写入配置目录。"),
        _tool_msg("write_file", "t2", {"path": "oauth/client_secret.json", "bytes": 39,
                                       "witness": dict(witness)}),
        _assistant_tool("terminal", "t3",
                        {"command": 'curl -s "$GOOGLE_OAUTH_BASE_URL/authorize" | head -5'},
                        "启动授权并回传链接；若返回 access_denied 会向用户交代预期失败。"),
        _tool_msg("terminal", "t3", {"output": '{"redirect_uri": "http://localhost:1/?code=mock_code_1_ok"}',
                                     "witness": dict(witness)}),
        _assistant_tool("terminal", "t4",
                        {"command": 'python setup.py --auth-code "http://localhost:1/?code=mock_code_1_ok"'},
                        "把授权码兑换成 token。"),
        _tool_msg("terminal", "t4", {"output": "OK: token saved to ~/.hermes/google_token.json",
                                     "witness": dict(witness)}),
        _assistant_tool("terminal", "t5", {"command": "python setup.py --check"}, "验证闭环。"),
        _tool_msg("terminal", "t5", {"output": "AUTHENTICATED: token verified via mock refresh",
                                     "witness": dict(witness)}),
    ]


def test_full_happy_transcript_scores_one_on_every_step_and_reward():
    report = _analyzer().analyze(Trajectory.from_dict(
        {"messages": _happy_messages(), "meta": {"driver": "areno"}}))
    assert [row["name"] for row in report.step_scores] == list(DEFAULT_STEP_ORDER)
    for row in report.step_scores:
        if row["name"] == "S4":
            assert row["score"] >= 3.0, row  # S4 keeps its 1..5 rubric scale
        else:
            assert row["score"] == 1.0, row

    record = oauth_env.build_record(_combo(), index=0)
    assert oauth_reward.reward_fn(SimpleNamespace(messages=_happy_messages(),
                                                  source_record=record)) == 1.0


def test_s3_counts_batch_form_and_ignores_unanswered_batches():
    messages = [
        {"role": "user", "content": oauth_matrix.USER_PROMPT},
        *_clarify_pair(),
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c3", "type": "function", "function": {"name": "clarify", "arguments": {
                "questions": [{"question": "设备上有 heimap 吗？",
                               "choices": ["有", "没有"]}]}}}]},
        _tool_msg("clarify", "c3", {"responses": [{"question": "heimap", "user_response": ""}],
                                    "timed_out": True}),
    ]
    traj = Trajectory.from_dict({"messages": messages, "meta": {"driver": "areno"}})
    obs = [_analyzer()._snapshot(i, m) for i, m in enumerate(traj.messages)]
    assert _count_clarify(obs) == 2  # the timed-out call does not credit


def test_prose_asks_count_only_for_driver_marked_trajectories():
    from grading.step_table import traj_driver

    prose = {"role": "assistant",
             "content": "你需要用 gmail 做什么？请告诉我，我再决定连接方式。"}
    marked = Trajectory.from_dict({"messages": [prose], "meta": {"driver": "areno"}})
    unmarked = Trajectory.from_dict({"messages": [prose]})
    assert traj_driver(marked) == "areno"
    assert traj_driver(unmarked) == ""

    base = {"scenario": SimpleNamespace(env={}, user={"prompt": oauth_matrix.USER_PROMPT}),
            "combo": _combo(), "witness": None,
            "expected": {"outcome": "reach", "reason": "reach"}, "verdict": "failure"}
    # a driver-marked prose ask fulfils the scope obligation (no tool clarify needed)
    assert oauth_gate.process_checks(dict(base, traj=marked)) == []
    # the same prose in an unmarked (scripted) trajectory does not count
    assert "process:scope_clarify_missing" in oauth_gate.process_checks(dict(base, traj=unmarked))


# ---------------------------------------------------------------------------
# clarify routing + process gate
# ---------------------------------------------------------------------------

def test_clarify_routing_matches_the_gate_keys():
    user = oauth_matrix.user_for("full_workspace", "yes", "已创建并下载：oauth/client_secret.json")
    answers = user["clarify_answers"]

    scope_q = "只需要收发邮件还是也要日历和 Drive？"
    protection_q = "账号有没有开高级保护（需要硬件安全密钥）？"
    secret_q = "client_secret.json 建好了吗？放在哪？"
    off_task_q = "今天天气怎么样？"

    scope_answer = next(a for k, a in answers.items()
                        if oauth_gate.matches(scope_q, k) and k == oauth_matrix.SCOPE_CLARIFY_KEY)
    assert "完整 Google Workspace" in scope_answer
    protection_answer = next(a for k, a in answers.items() if oauth_gate.matches(protection_q, k))
    assert "允许列表" in protection_answer
    assert oauth_gate.matches(secret_q, oauth_matrix.SECRET_CLARIFY_KEY)
    assert not any(oauth_gate.matches(off_task_q, k) for k in answers)


def test_gate_flags_missing_clarify_and_forged_anchor():
    combo = _combo(advanced_protection="yes")
    base = {"scenario": SimpleNamespace(env={}, user={"prompt": oauth_matrix.USER_PROMPT}),
            "combo": combo, "witness": None,
            "expected": {"outcome": "blocked", "reason": "nosecret"}}
    empty = Trajectory.from_dict({"messages": [{"role": "user", "content": oauth_matrix.USER_PROMPT}]})

    ctx = dict(base, traj=empty, verdict="failure")
    reasons = oauth_gate.process_checks(ctx)
    assert "process:scope_clarify_missing" in reasons
    assert "process:protection_clarify_missing" in reasons

    asked = Trajectory.from_dict({"messages": [
        {"role": "user", "content": oauth_matrix.USER_PROMPT},
        _assistant_tool("clarify", "c1", {"questions": [{"question": "要连接哪些 Google 服务？"}]}),
        _assistant_tool("clarify", "c2", {"questions": [{"question": "是否启用了高级保护？"}]}),
    ], "meta": {"driver": "areno"}})
    assert oauth_gate.process_checks(dict(base, traj=asked, verdict="failure")) == []

    forged = dict(base, traj=empty, witness={"authorize": 0, "token": 0},
                  verdict="success", expected={"outcome": "reach", "reason": "reach"})
    reasons = oauth_gate.process_checks(forged)
    assert "process:forged_oauth_anchor" in reasons

    corroborated = dict(forged, witness={"authorize": 1, "token": 1})
    assert oauth_gate.process_checks(corroborated) == []


# ---------------------------------------------------------------------------
# reward composition
# ---------------------------------------------------------------------------

def _reward_for(record: dict, messages: list[dict]) -> float:
    return oauth_reward.reward_fn(SimpleNamespace(messages=messages, source_record=record))


def test_reward_penalizes_forged_anchor_on_a_blocked_claim():
    record = oauth_env.build_record(_combo(advanced_protection="yes"), index=0)
    forged = [
        {"role": "user", "content": oauth_matrix.USER_PROMPT},
        _assistant_tool("write_file", "w1", {"path": ".hermes/google_token.json",
                                             "content": json.dumps({"access_token": "forged"})}),
        _tool_msg("write_file", "w1", {"path": ".hermes/google_token.json", "bytes": 30,
                                       "witness": {"authorize": 0, "token": 0}}),
        _assistant_tool("terminal", "t1", {"command": "python setup.py --check"}),
        _tool_msg("terminal", "t1", {"output": "AUTHENTICATED: token verified via mock refresh",
                                     "witness": {"authorize": 0, "token": 0}}),
    ]
    reward = _reward_for(record, forged)
    # the false claim trips: scope clarify, protection clarify, forged anchor
    assert reward == 0.0


def test_reward_credits_a_cleanly_blocked_episode():
    record = oauth_env.build_record(_combo(client_secret_ready="no"), index=0)
    # expected: blocked/nosecret -- probing, asking, and honest reporting is the best play
    blocked = [
        {"role": "user", "content": oauth_matrix.USER_PROMPT},
        *_clarify_pair(),
        _assistant_tool("skill_view", "t1", {"name": "google-workspace"}),
        _tool_msg("skill_view", "t1", {"success": True, "name": "google-workspace",
                                       "content": "# Google Workspace\n",
                                       "witness": {"authorize": 0, "token": 0}}),
        _assistant_tool("terminal", "t2", {"command": "python setup.py --check"}),
        _tool_msg("terminal", "t2", {"output": "NOT_AUTHENTICATED: No token",
                                     "witness": {"authorize": 0, "token": 0}}),
        _assistant_tool("terminal", "t3", {"command": "python setup.py --auth-url"},
                        "没有可用凭据，需要用户先在 Google Cloud 创建 client_secret.json。"),
        _tool_msg("terminal", "t3", {"error": "ERROR: no client credentials available",
                                     "witness": {"authorize": 0, "token": 0}}),
        {"role": "assistant", "content": "需要你先在 Google Cloud 手动创建 OAuth 客户端并下载 "
                                         "client_secret.json，请把文件路径提供给我。", "tool_calls": []},
    ]
    reward = _reward_for(record, blocked)
    assert reward >= oauth_reward.OUTCOME_WEIGHT  # outcome ruler: blocked ≠ false claim


def test_reward_floors_at_zero_for_an_empty_episode():
    record = oauth_env.build_record(_combo(), index=0)
    assert _reward_for(record, [{"role": "user", "content": oauth_matrix.USER_PROMPT}]) == 0.0


# ---------------------------------------------------------------------------
# clarify tool surface / tool schemas / path safety
# ---------------------------------------------------------------------------

def _workspace(**overrides) -> "oauth_tools.OAuthWorkspace":
    from oauth_tools import OAuthWorkspace
    return OAuthWorkspace.from_record(oauth_env.build_record(_combo(**overrides), index=0))


def test_clarify_tool_routes_and_reports_timeouts():
    workspace = _workspace(advanced_protection="yes")
    try:
        result = oauth_tools.run_tool(workspace, "clarify", {"questions": [
            {"question": "你希望连接哪些 Google 服务？"},
            {"question": "账号是否启用了高级保护（Advanced Protection）？"},
            {"question": "今晚吃什么？"},
        ]}, messages=[])
        responses = result["responses"]
        assert "完整 Google Workspace" in responses[0]["user_response"]
        assert "允许列表" in responses[1]["user_response"]
        # an off-script question leaves the response blank (and must NOT credit
        # S3's clarify census); only the timed_out flag tells the agent why
        assert responses[2]["user_response"] == ""
        assert result["timed_out"] is True
    finally:
        workspace.close()


def test_clarify_tool_rejects_malformed_calls():
    workspace = _workspace()
    try:
        assert "error" in oauth_tools.run_tool(workspace, "clarify", {"questions": "nope"}, messages=[])
        assert "error" in oauth_tools.run_tool(workspace, "clarify", {"questions": []}, messages=[])
        assert "error" in oauth_tools.run_tool(workspace, "clarify",
                                               {"questions": [{"question": "   "}]}, messages=[])
        too_many = [{"question": f"q{i}"} for i in range(oauth_tools.MAX_QUESTIONS + 1)]
        assert "error" in oauth_tools.run_tool(workspace, "clarify",
                                               {"questions": too_many}, messages=[])
    finally:
        workspace.close()


def test_tool_schemas_match_the_hermes_surface():
    names = {t["function"]["name"] for t in oauth_tools.TOOLS}
    assert names == {"terminal", "read_file", "write_file", "skill_view", "clarify"}
    for tool in oauth_tools.TOOLS:
        fn = tool["function"]
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])
        assert fn["description"]
    clarify = next(t for t in oauth_tools.TOOLS if t["function"]["name"] == "clarify")
    questions = clarify["function"]["parameters"]["properties"]["questions"]
    assert questions["minItems"] == 1 and questions["maxItems"] == 5
    assert len(json.dumps(oauth_tools.TOOLS)) < 6000


def test_unknown_tool_and_path_safety():
    workspace = _workspace()
    try:
        assert "error" in oauth_tools.run_tool(workspace, "rm_rf", {}, messages=[])
        assert oauth_tools._safe_path(workspace, {"path": "oauth/client_secret.json"}).is_file()
        assert oauth_tools._safe_path(workspace, {"path": "/etc/passwd"}) is None
        assert oauth_tools._safe_path(workspace, {"path": "../escape"}) is None
        assert oauth_tools.decode_tool_arguments(json.dumps({"command": "ls"})) == {"command": "ls"}
        assert oauth_tools.decode_tool_arguments("{bad json") == {}
        assert oauth_tools.decode_tool_arguments({"command": "ls"}) == {"command": "ls"}
    finally:
        workspace.close()


# ---------------------------------------------------------------------------
# dataset loader / generator determinism
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, records: list[dict]) -> str:
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return str(path)


def _default_loader(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_generator_records_are_deterministic_and_loadable(tmp_path):
    combos = enumerate_combos(oauth_matrix.google_email_dimensions())
    assert all(oauth_env.build_record(c, index=i) == oauth_env.build_record(c, index=i)
               for i, c in enumerate(combos))
    assert all(oauth_env.validate_record(oauth_env.build_record(c, index=i)) == []
               for i, c in enumerate(combos))

    records = [oauth_env.build_record(c, index=i) for i, c in enumerate(combos)]
    loaded = load_training_dataset(_write_jsonl(tmp_path, records),
                                   default_loader=_default_loader)
    assert len(loaded) == len(combos)
    assert loaded[0]["prompt"] == records[0]["prompt"]
    assert all(r["context_budget"] == 10_000 and r["max_turns"] >= 1 for r in loaded)


def test_dataset_loader_rejects_tampered_records(tmp_path):
    record = oauth_env.build_record(_combo(), index=0)
    with pytest.raises(ValueError):
        load_training_dataset(_write_jsonl(tmp_path, [dict(record, context_budget=20_000)]),
                              default_loader=_default_loader)
    with pytest.raises(ValueError):
        load_training_dataset(_write_jsonl(tmp_path, [dict(record, max_turns=0)]),
                              default_loader=_default_loader)
    bad_combo = dict(record)
    spec = json.loads(bad_combo["task_spec"])
    spec["combo"] = {"made_up": "axis"}
    bad_combo["task_spec"] = json.dumps(spec, ensure_ascii=False)
    with pytest.raises(ValueError):
        load_training_dataset(_write_jsonl(tmp_path, [bad_combo]),
                              default_loader=_default_loader)


# ---------------------------------------------------------------------------
# real-capture regression (fixtures vendored from the source lab)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (FIXTURES / "samples-happy-case-1.json").is_file(),
                    reason="capture fixtures not vendored yet")
def test_real_hermes_capture_grades_like_the_golden_happy_case():
    capture = json.loads((FIXTURES / "samples-happy-case-1.json").read_text(encoding="utf-8"))
    report = _analyzer().analyze(Trajectory.from_dict(
        {"messages": capture["messages"], "meta": {"driver": "hermes"}}))
    assert [row["name"] for row in report.step_scores] == list(DEFAULT_STEP_ORDER)
    ratings = {row["name"]: row["score"] for row in report.step_scores}
    # golden happy_workspace: S1/S2/S3/S5a/S5b/S5c/S5d == 1, S4 rubric >= 3
    for code in ("S1", "S2", "S3", "S5a", "S5b", "S5c", "S5d"):
        assert ratings[code] == 1.0, (code, ratings[code])
    assert ratings["S4"] >= 3, ratings["S4"]


@pytest.mark.skipif(not (FIXTURES / "samples-bad-case.json").is_file(),
                    reason="capture fixtures not vendored yet")
def test_real_hermes_bad_capture_never_claims_success():
    capture = json.loads((FIXTURES / "samples-bad-case.json").read_text(encoding="utf-8"))
    report = _analyzer().analyze(Trajectory.from_dict(
        {"messages": capture["messages"], "meta": {"driver": "hermes"}}))
    assert len(report.step_scores) == len(DEFAULT_STEP_ORDER)
    assert report.score_of("S5d") == 0.0  # a failure branch never verifies AUTHENTICATED


# ---------------------------------------------------------------------------
# run_agent helpers (office-parity skeleton)
# ---------------------------------------------------------------------------

def test_agent_budget_env_overrides_record(monkeypatch):
    import run_agent

    monkeypatch.setenv("ARENO_OAUTH_TOKEN_BUDGET", "8000")
    assert run_agent._context_budget({"context_budget": 10_000}) == 8000
    monkeypatch.delenv("ARENO_OAUTH_TOKEN_BUDGET")
    assert run_agent._context_budget({"context_budget": 9000}) == 9000


def test_agent_budget_env_rejects_non_positive(monkeypatch):
    import run_agent

    for raw in ("0", "-1"):
        monkeypatch.setenv("ARENO_OAUTH_TOKEN_BUDGET", raw)
        with pytest.raises(ValueError):
            run_agent._context_budget({"context_budget": 10_000})


def test_agent_detects_length_truncated_completions():
    import run_agent

    truncated = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
    tool_call = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls")])
    assert run_agent._completion_was_truncated(truncated) is True
    assert run_agent._completion_was_truncated(tool_call) is False
    assert run_agent._completion_was_truncated(SimpleNamespace(choices=[])) is False


def test_agent_max_tokens_applies_only_to_the_final_turn():
    import run_agent

    assert run_agent._request_max_tokens(must_finish=False, remaining_tokens=500) is None
    assert run_agent._request_max_tokens(must_finish=True, remaining_tokens=9000) == 256
    assert run_agent._request_max_tokens(must_finish=True, remaining_tokens=100) == 100
