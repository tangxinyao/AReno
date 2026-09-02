"""Task environment + dataset records for the OAuth agentic example.

One record = one scenario combo from the 6-axis matrix (`oauth_matrix.py`),
plus the deterministic fixture set an episode's workspace is materialized
from (`materialize_workspace`).  The prompt is the canonical maximally
ambiguous seed ("帮我连接我的谷歌邮箱") for every record: the required
disambiguation is the clarify-user skill the reward grades, so prompt text
must not leak the answers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import oauth_matrix
import oauth_seeds
from oauth_world import google_paths

TASK = "connect_google_email"
DEFAULT_CONTEXT_BUDGET = 10_000
MAX_TURNS = 12
ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"

_VENDORED_SKILLS = (
    ("email", "himalaya"),
    ("productivity", "google-workspace"),
)


def parse_task_spec(record: dict) -> dict:
    """Parse a record's compact ``task_spec`` JSON string."""
    return json.loads(str(record["task_spec"]))


def build_record(combo: dict[str, str], *, index: int) -> dict:
    """Deterministic dataset row for one scenario combo."""
    scenario_id = oauth_matrix.google_scenario_id_for(combo)
    server = oauth_matrix.server_for(combo["code_validity"], combo["test_user_added"],
                                     combo["token_state"])
    env = oauth_matrix.env_for(combo["token_state"], combo["service_scope"],
                               combo["client_secret_ready"])
    secret_answer = oauth_matrix._secret_answer(combo)
    user = oauth_matrix.user_for(combo["service_scope"], combo["advanced_protection"],
                                 secret_answer)
    expected = oauth_matrix.expected_outcome(combo)
    spec = {
        "scenario_id": scenario_id,
        "combo": dict(combo),
        "server": server,
        "env": env,
        "user": user,
        "expected": expected,
    }
    return {
        "id": f"oauth-{index:05d}",
        "task": TASK,
        "scenario_id": scenario_id,
        "task_spec": json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
        "prompt": oauth_matrix.USER_PROMPT,
        "service_scope": combo["service_scope"],
        "advanced_protection": combo["advanced_protection"],
        "test_user_added": combo["test_user_added"],
        "client_secret_ready": combo["client_secret_ready"],
        "token_state": combo["token_state"],
        "code_validity": combo["code_validity"],
        "max_turns": MAX_TURNS,
        "context_budget": DEFAULT_CONTEXT_BUDGET,
    }


def materialize_workspace(record: dict, root: Path) -> dict[str, str]:
    """Seed an episode workspace with the skill docs, the setup shim and (when
    the combo promises it) the canonical client secret.  No token file ever
    pre-exists: authentication is the task."""
    root = Path(root)
    spec = parse_task_spec(record)

    hermes_home = root / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    for group, name in _VENDORED_SKILLS:
        skill_dir = hermes_home / "skills" / group / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SKILLS_DIR / group / name / "SKILL.md",
                        skill_dir / "SKILL.md")

    shim_path = root / oauth_seeds.SETUP_SHIM_PATH
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    shim_path.write_text(oauth_seeds.setup_shim_text(), encoding="utf-8")

    secret = oauth_seeds.client_secret_seed(bool(spec["env"].get("client_secret_ready")))
    if secret is not None:
        secret_path = root / oauth_seeds.CLIENT_SECRET_PATH
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        secret_path.write_text(secret, encoding="utf-8")

    return google_paths(root)


def validate_record(record: dict) -> list[str]:
    """Static self-consistency checks ``dataset_generator`` runs per record.
    Returns a list of issues (empty == valid)."""
    issues: list[str] = []
    spec = parse_task_spec(record)
    combo = spec["combo"]

    expected_combo_keys = {d.key for d in oauth_matrix.google_email_dimensions()}
    if set(combo) != expected_combo_keys:
        issues.append(f"combo keys {sorted(combo)} != {sorted(expected_combo_keys)}")
        return issues

    if spec["scenario_id"] != oauth_matrix.google_scenario_id_for(combo):
        issues.append("scenario_id does not match the combo")
    if spec["server"] != oauth_matrix.server_for(combo["code_validity"],
                                                 combo["test_user_added"],
                                                 combo["token_state"]):
        issues.append("server config does not match server_for(combo)")
    if spec["expected"] != oauth_matrix.expected_outcome(combo):
        issues.append("expected_outcome does not match the combo")
    if record.get("scenario_id") != spec["scenario_id"]:
        issues.append("record.scenario_id does not match task_spec")
    if record.get("prompt") != oauth_matrix.USER_PROMPT:
        issues.append("prompt must be the canonical seed prompt")

    # materialization smoke: fixtures exist exactly as the combo promises
    import tempfile

    with tempfile.TemporaryDirectory(prefix="areno-oauth-validate-") as tmp:
        tmp_root = Path(tmp) / "home"
        tmp_root.mkdir()
        materialize_workspace(record, tmp_root)
        shim = tmp_root / oauth_seeds.SETUP_SHIM_PATH
        if not shim.is_file() or "python3" not in shim.read_text(encoding="utf-8")[:40]:
            issues.append("setup shim missing or malformed")
        for group, name in _VENDORED_SKILLS:
            if not (tmp_root / ".hermes" / "skills" / group / name / "SKILL.md").is_file():
                issues.append(f"skill doc missing: {group}/{name}")
        secret_path = tmp_root / oauth_seeds.CLIENT_SECRET_PATH
        ready = bool(spec["env"].get("client_secret_ready"))
        if ready != secret_path.is_file():
            issues.append("client_secret presence does not match client_secret_ready")
        if (tmp_root / ".hermes" / "google_token.json").exists():
            issues.append("workspace must never be pre-authenticated")
    return issues
