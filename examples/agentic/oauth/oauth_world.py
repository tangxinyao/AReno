"""OAuth "world": mock Google OAuth server + state machine (ported from trajlab
`flows/google_email/world.py`).

Faithfully reproduces the real Google error shapes (HTTP status + JSON
`error` / `error_description`) of the mock behaviour table, and writes the
token file (``<home>/.hermes/google_token.json``) on successful /token so the
result-level verifier has a real anchor (`--check` + token file).

Porting deltas vs trajlab:
  * the ``WorldSpec`` registry binding (and its seed-file plumbing) was
    dropped -- the areno example drives the server directly;
  * the state machine is bound to the server *instance*, not a ``_Handler``
    class attribute, so episodes running concurrently (``asyncio.gather`` over
    a rollout batch) each grade against their own world;
  * ``port=0`` binds an ephemeral port; ``base_url`` reflects the real port
    after ``start()``.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Real error-shape contract --------------------------------------------------

AUTHORIZE_BEHAVIOURS = ("success", "access_denied", "abandon", "code_expired")
TOKEN_BEHAVIOURS = ("success", "invalid_grant", "invalid_client", "refresh_failed")

OAUTH_ERRORS = {
    "access_denied": (403, {"error": "access_denied",
                            "error_description": "The user account has not been granted the required permission in this project."}),
    "invalid_grant": (400, {"error": "invalid_grant",
                            "error_description": "The authorization code is invalid or has expired."}),
    "invalid_client": (401, {"error": "invalid_client",
                             "error_description": "The OAuth client credentials are invalid."}),
    "refresh_failed": (400, {"error": "invalid_grant",
                             "error_description": "The refresh token is invalid, expired, or revoked."}),
    "insufficient_permissions": (403, {"error": "insufficient_permissions",
                                       "error_description": "The token does not have sufficient scope for this request."}),
}


class OAuthStateMachine:
    """State machine a rollout drives and snapshots.

    `server` config decides `auth_url_behavior` and `token_behavior`.
    `token_state` ('absent'|'expired'|'revoked') decides the
    refresh-verification outcome used by `--check`.
    """

    def __init__(self, server_cfg: dict[str, Any], paths: dict[str, Any]) -> None:
        self.auth_url_behavior = server_cfg.get("auth_url_behavior", "success")
        self.token_behavior = server_cfg.get("token_behavior", "success")
        self.insufficient_permissions = server_cfg.get("insufficient_permissions", False)
        self.token_state = (server_cfg or {}).get("token_state")
        self.token_path = Path(paths.get("google_token") or "~/.hermes/google_token.json").expanduser()
        # per-scenario exchange counter (invalid_grant_once: fail the first
        # exchange only, succeed on retry)
        self._exchanges = 0
        # endpoint hit counts (witness): the world testifies which OAuth
        # endpoints the roll actually walked.  A reach-success with zero
        # authorize/token hits forged its anchor file (reward hacking) instead
        # of completing the chain.
        self.hits: dict[str, int] = {"authorize": 0, "token": 0}

    # -- authorize flow ------------------------------------------------------
    def authorize(self, auth_url: str) -> tuple[bool, str, Any]:
        """Returns (ok, redirect_or_reason, state)."""
        self.hits["authorize"] += 1
        if self.auth_url_behavior == "access_denied":
            return False, "access_denied", None
        if self.auth_url_behavior == "abandon":
            return False, "abandoned", None
        code = self._mint_code(expired=self.auth_url_behavior == "code_expired")
        redirect = "http://localhost:1/?code=%s&state=scaffold" % code
        return True, redirect, {"code": code}

    def _mint_code(self, expired: bool = False) -> str:
        self._codes_used = getattr(self, "_codes_used", 0) + 1
        return f"mock_code_{self._codes_used}_{'expired' if expired else 'ok'}"

    # -- token exchange ------------------------------------------------------
    def exchange(self, code: str | None) -> tuple[bool, Any]:
        """POST /token. Returns (ok, error-body). Writes token file on success."""
        self.hits["token"] += 1
        if self.token_behavior == "invalid_grant_once":
            # recovery-narration shape: the FIRST exchange errors, a retry wins
            self._exchanges += 1
            if self._exchanges <= 1:
                return False, OAUTH_ERRORS["invalid_grant"][1]
        if self.token_behavior == "invalid_grant":
            return False, OAUTH_ERRORS["invalid_grant"][1]
        if self.token_behavior == "invalid_client":
            return False, OAUTH_ERRORS["invalid_client"][1]
        if not code or "expired" in (code or ""):
            return False, OAUTH_ERRORS["invalid_grant"][1]

        refresh_token = f"rt_{code}"
        self._write_token_file(resp := {
            "access_token": f"at_{code}",
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 3599,
            "scope": "gmail.readonly calendar.events.readonly",
        })
        return True, resp

    # -- revocation / check --------------------------------------------------
    def revoke(self, token: str) -> bool:
        return True

    def check(self) -> dict[str, str]:
        """`setup.py --check` semantics: AUTHENTICATED / NOT_AUTHENTICATED / REFRESH_FAILED.

        Result-level anchor: success only when the token file exists AND the
        token is not expired/revoked (result-level rule).
        """
        if not self.token_path.exists():
            return {"status": "NOT_AUTHENTICATED",
                    "detail": "No token at %s" % self.token_path}
        if self.token_state in ("expired", "revoked"):
            return {"status": "REFRESH_FAILED", "detail": f"token_state={self.token_state}"}
        return {"status": "AUTHENTICATED", "detail": "token verified via mock refresh"}

    def _write_token_file(self, resp: dict[str, Any]) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(resp, indent=2), encoding="utf-8")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        machine = self.server.machine  # bound per-server (see module docstring)
        url = urlparse(self.path)
        if url.path.rstrip("/") == "/authorize":
            ok, result, st = machine.authorize(self.path)
            if ok:
                self._json(200, {"redirect_uri": result, "code": st["code"]})
            else:
                status, body = OAUTH_ERRORS.get(result, (400, {"error": result}))
                self._json(status, body)
        elif url.path.rstrip("/") == "/health":
            self._json(200, {"ok": True, "version": "v0.1.0"})
        else:
            self._json(404, {"error": "not_found", "error_description": f"Unknown endpoint {url.path}"})
        return

    def do_POST(self) -> None:  # noqa: N802
        machine = self.server.machine
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            params = json.loads(body.decode()) if body else {}
        except json.JSONDecodeError:
            params = parse_qs(body.decode())
        flat = {k: (v[0] if isinstance(v, list) else v) for k, v in params.items()}

        if url.path.rstrip("/") == "/token":
            ok, payload = machine.exchange(flat.get("code"))
            status = 200 if ok else self._error_status(payload)
            self._json(status, payload)
        elif url.path.rstrip("/") == "/revoke":
            machine.revoke(flat.get("token", ""))
            self._json(200, {})
        elif url.path.rstrip("/") == "/check":
            self._json(200, machine.check())
        else:
            self._json(404, {"error": "not_found"})
        return

    @staticmethod
    def _error_status(body: Any) -> int:
        err = body.get("error") if isinstance(body, dict) else None
        for _name, (status, payload) in OAUTH_ERRORS.items():
            if payload.get("error") == err:
                return status
        return 400

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:  # silence logs by default
        pass


class MockOAuthServer:
    """Owns one OAuthStateMachine and serves HTTP on a local port.

    ``port=0`` binds an ephemeral port; ``base_url`` / ``port`` reflect the
    real binding after ``start()`` -- each episode workspace gets its own
    state machine this way.
    """

    machine: OAuthStateMachine | None = None

    def __init__(self, server_cfg: dict[str, Any], paths: dict[str, Any],
                 host: str = "127.0.0.1", port: int = 9898) -> None:
        self.machine = OAuthStateMachine(server_cfg, paths)
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "MockOAuthServer":
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._server.machine = self.machine  # instance attr; handler reads it
        if self.port == 0:
            self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def snapshot(self) -> dict[str, Any]:
        """State snapshot used by golden-case regression."""
        return {"token_path": str(self.machine.token_path),
                "token_state": self.machine.token_state,
                "auth_url_behavior": self.machine.auth_url_behavior,
                "token_behavior": self.machine.token_behavior,
                "hits": dict(self.machine.hits)}


def run_server(host: str, port: int, server_cfg: dict[str, Any] | None = None,
               home: str | Path = "~", block: bool = True) -> None:
    """Standalone server for manual debugging / exploring the task."""
    paths = {"google_token": str(Path(home) / ".hermes" / "google_token.json")}
    server = MockOAuthServer(server_cfg or {}, paths, host=host, port=port).start()
    print(f"mock OAuth up at {server.base_url}  (auth_url_behavior={server.machine.auth_url_behavior}, "
          f"token_behavior={server.machine.token_behavior})")
    try:
        if block:
            while True:
                time.sleep(3600)
    finally:
        server.stop()


# Token paths for the example's workspace layout ----------------------------

def google_anchor(home: Any, cfg: Any = None) -> Path:
    """google-email anchor: the token file inside a roll's fresh home."""
    return Path(home) / ".hermes" / "google_token.json"


def google_paths(home: Any, cfg: Any = None) -> dict[str, Any]:
    """google-email world paths: token under the fresh home, secret beside it."""
    return {
        "google_token": str(google_anchor(home, cfg)),
        "client_secret": str(Path(home) / "oauth" / "client_secret.json"),
    }
