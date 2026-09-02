"""OAuth flow: files seeded into an episode workspace (ported from trajlab
`flows/google_email/real_seed.py`).

  * ``SETUP_SHIM_TMPL`` replaces the community `google-workspace` setup.py in
    the seeded skills dir.  Same CLI surface (--check / --client-secret /
    --auth-url / --auth-code / --revoke / --install-deps), stdlib only, and
    the OAuth endpoints point at the episode's mock world (``GOOGLE_OAUTH_BASE_URL``
    env, or the baked ``{base_url}`` fallback).  The scripted-browser leg is
    realized here: --auth-url answers with BOTH the auth_url and the
    already-visited redirect_url, so the model never needs a human at the
    login wall.
  * ``client_secret_seed`` returns the canonical ``oauth/client_secret.json``
    the user story promises (已创建并下载) -- but ONLY when the combo says the
    secret is ready.  A not-ready combo seeds nothing and the model has to
    face the real block.

Path-contract delta vs trajlab: the canonical secret resolves relative to
``HERMES_HOME`` (the workspace's ``.hermes``) instead of a hard-coded
container path ``/root/oauth/...`` -- the workspace root IS the roll home.
"""

from __future__ import annotations

import json

SETUP_SHIM_PATH = ".hermes/skills/productivity/google-workspace/scripts/setup.py"
CLIENT_SECRET_PATH = "oauth/client_secret.json"

# `{base_url}` is textually substituted (NOT str.format -- keep single braces).
SETUP_SHIM_TMPL = '''#!/usr/bin/env python3
"""Google Workspace OAuth setup (mock-world shim, stdlib only).

Same steps the skill describes; endpoints point at the training mock world.
  --check                     ask the mock whether we are authenticated
  --client-secret <path>      load the OAuth Desktop client credentials
                              (optional: auto-loads the canonical
                              oauth/client_secret.json when present)
  --auth-url [--services ...] mint an auth URL; needs a loaded client secret,
                              else reports that none was ever provided
  --auth-code <url-or-code>   exchange the code for a token
  --revoke                    drop the token
"""
import argparse, json, os, sys, urllib.request  # stdlib only
from urllib.error import HTTPError

BASE = os.environ.get("GOOGLE_OAUTH_BASE_URL", "{base_url}")
HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
CRED = os.path.join(HERMES_HOME, "google_client_secret.json")
TOKEN = os.path.join(HERMES_HOME, "google_token.json")
CANON_SECRET = os.path.join(os.path.dirname(HERMES_HOME), "oauth", "client_secret.json")


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


def _read(p):
    with open(p) as f:
        return json.load(f)


def _check():
    if os.path.exists(TOKEN):
        tok = _read(TOKEN)
        if tok.get("access_token"):
            print("AUTHENTICATED: token verified via mock refresh")
            return
    try:
        _st, body = _req("GET", "/check")
    except Exception as e:  # mock unreachable -> honest report
        print("UNKNOWN: mock world unreachable (%s)" % e)
        sys.exit(2)
    print("%s: %s" % (body.get("status", "UNKNOWN"), body.get("detail", "")))
    sys.exit(0 if body.get("status") == "AUTHENTICATED" else 1)


def _client_secret(path):
    c = _read(path)
    os.makedirs(HERMES_HOME, exist_ok=True)
    with open(CRED, "w") as f:
        json.dump(c, f)
    client = c.get("installed") or c.get("web") or {}
    print("client_id:", client.get("client_id"))
    print("client_secret saved. Next: run --auth-url")


def _maybe_load_canonical():
    """--client-secret is optional when the user already dropped the secret at
    the canonical path (the "已创建并下载" user story): load it once."""
    if os.path.exists(CRED) or not os.path.exists(CANON_SECRET):
        return
    try:
        _client_secret(CANON_SECRET)
    except Exception as e:  # a corrupt secret must not look like "no secret"
        print("WARN: canonical client secret unreadable (%s)" % e)


def _auth_url(_services):
    if not os.path.exists(CRED):
        # scripted world semantics: no client secret => no OAuth client was
        # ever created => the user story cannot authorize yet
        print("ERROR: no client credentials available -- the user has not "
              "created a Google Cloud OAuth client yet, so no auth URL can "
              "be minted. Ask the user for the client_secret.json path.")
        sys.exit(2)
    try:
        _st, body = _req("GET", "/authorize")
    except HTTPError as e:
        print(json.dumps(json.loads(e.read().decode())))
        sys.exit(1)
    except Exception as e:
        print("ERROR: mock world unreachable (%s)" % e)
        sys.exit(2)
    print(json.dumps({
        "auth_url": BASE + "/authorize",
        "redirect_url": body.get("redirect_uri"),
        "code": body.get("code"),
        "scripted_browser": True,
        "hint": "browser visit already scripted: run --auth-code with the redirect_url above",
    }))


def _extract_code(val):
    if val.startswith("http"):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(val).query)
        return q.get("code", [None])[0]
    return val


def _auth_code(val):
    code = _extract_code(val)
    if not code:
        print("ERROR: no code in input")
        sys.exit(1)
    try:
        _st, body = _req("POST", "/token", {"code": code})
    except HTTPError as e:
        body = json.loads(e.read().decode())
        print("REFRESH_FAILED:", json.dumps(body))
        print("code may be expired/used. Run --auth-url for a fresh one.")
        sys.exit(1)
    except Exception as e:
        print("ERROR: mock world unreachable (%s)" % e)
        sys.exit(2)
    os.makedirs(HERMES_HOME, exist_ok=True)
    with open(TOKEN, "w") as f:
        json.dump(body, f)
    print("OK: token saved to", TOKEN)


def _revoke():
    try:
        os.remove(TOKEN)
    except OSError:
        pass
    try:
        _req("POST", "/revoke", {"token": "local"})
    except Exception:
        pass
    print("token revoked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-live", action="store_true")
    ap.add_argument("--client-secret")
    ap.add_argument("--auth-url", action="store_true")
    ap.add_argument("--auth-code")
    ap.add_argument("--format", default=None)
    ap.add_argument("--services", default="all")
    ap.add_argument("--revoke", action="store_true")
    ap.add_argument("--install-deps", action="store_true")
    a = ap.parse_args()
    if a.install_deps:
        print("nothing to install: setup runs on the stdlib only")
        return
    if a.client_secret:
        _client_secret(a.client_secret)
    elif a.auth_url:
        _maybe_load_canonical()
        _auth_url(a.services)
    elif a.auth_code:
        _auth_code(a.auth_code)
    elif a.revoke:
        _revoke()
    elif a.check or a.check_live:
        _check()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
'''


def setup_shim_text(base_url: str | None = None) -> str:
    """Shim source with the mock-world default base URL baked in.

    At rollout time the workspace also exports ``GOOGLE_OAUTH_BASE_URL``, which
    takes precedence (shim reads the env first); the substitute only provides
    the fallback value inside the file.
    """
    return SETUP_SHIM_TMPL.replace("{base_url}", base_url or "http://127.0.0.1:9898")


def client_secret_seed(secret_ready: bool) -> str | None:
    """Fixture content for the canonical client_secret.json -- only when the
    combo's `client_secret_ready` says the user story already created it.

    The fixture carries a mock client id; the real Google URLs appear as
    inert strings (fixture only, never contacted: the endpoints the shim
    actually calls point at the mock world)."""
    if not secret_ready:
        return None
    return json.dumps({"installed": {
        "client_id": "mock-client-1", "project_id": "mock-proj",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "type": "authorized_user"}}, indent=2)
