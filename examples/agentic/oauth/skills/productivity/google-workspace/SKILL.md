---
name: google-workspace
description: "Gmail, Calendar, Drive, Docs, Sheets via OAuth + gws CLI."
version: 1.2.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials (downloaded from Google Cloud Console)
metadata:
  hermes:
    tags: [Google, Gmail, Calendar, Drive, Sheets, Docs, Contacts, Email, OAuth]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [himalaya]
  areno:
    trimmed: "Setup procedure kept verbatim in substance; the API usage
      reference (per-service command catalogue, output schemas) was cut to fit
      a training context window. Full doc: hermes-agent upstream."
---

# Google Workspace

Gmail, Calendar, Drive, Contacts, Sheets and Docs through Hermes-managed OAuth.
`scripts/setup.py` runs the one-time authorization; `scripts/google_api.py`
runs everything afterwards.

## First-Time Setup

Non-interactive — you drive it step by step. Define a shorthand first:

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
```

### Step 0: Check if already set up

```bash
$GSETUP --check
```

`AUTHENTICATED` → skip to Usage. `NOT_AUTHENTICATED` → continue.
`REFRESH_FAILED` → the stored token is expired or revoked; redo Steps 3-5.

### Step 1: Triage — ask the user TWO questions first

**Q1: "What Google services do you need? Just email, or also
Calendar/Drive/Sheets/Docs?"**

- **Email only** → They don't need this skill at all. Use the `himalaya`
  skill instead: a Gmail App Password, 2 minutes, no Google Cloud project.
  Load the himalaya skill and follow its setup.
- **Email + Calendar** → continue here with `--services email,calendar`.
- **Full Workspace** → continue here with the default `--services all`.

**Q2: "Does your Google account use Advanced Protection (hardware security
keys required to sign in)? If you're not sure, you probably don't."**

- **No / Not sure** → normal setup, continue below.
- **Yes** → their Workspace admin must add the OAuth client ID to the org's
  allowed apps list before Step 4 works. Tell them upfront.

### Step 2: Create OAuth credentials (one-time, ~5 minutes)

This is the only manual step. Tell the user:

> 1. Create or select a project at
>    https://console.cloud.google.com/projectselector2/home/dashboard
> 2. Enable the APIs you need (Gmail, Calendar, Drive, Sheets, Docs, People)
>    from https://console.cloud.google.com/apis/library
> 3. Create the OAuth client at
>    https://console.cloud.google.com/apis/credentials —
>    Create Credentials → OAuth 2.0 Client ID → type "Desktop app"
> 4. While the app is in Testing, add their own Google account as a test user
>    at https://console.cloud.google.com/auth/audience
> 5. Download the JSON and tell you the file path

Then load it:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

If they paste raw client id/secret values instead, write a valid Desktop OAuth
JSON yourself and run `--client-secret` against that file.

### Step 3: Get the authorization URL

```bash
$GSETUP --auth-url --services all --format json
```

Agent rules for this step:

- Send the `auth_url` field to the user as a single line.
- Tell them the browser will likely fail on `http://localhost:1` after
  approval — that failure is expected — and to copy the ENTIRE redirected URL.
- `Error 403: access_denied` → they are not on the test-user list; send them to
  https://console.cloud.google.com/auth/audience to add themselves.

### Step 4: Exchange the code

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED"
```

A URL or a bare code both work. If it fails because the code expired or was
already used, go back to Step 3 for a fresh one and use the newest redirect.

### Step 5: Verify

```bash
$GSETUP --check
```

Must print `AUTHENTICATED`. Never report success without this line.

### Notes

- Token lives at `~/.hermes/google_token.json` and auto-refreshes.
- Revoke with `$GSETUP --revoke`.

## Usage

```bash
GAPI="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"

$GAPI gmail search "is:unread" --max 10
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI calendar list
$GAPI drive search "quarterly report" --max 10
```

All commands return JSON. `$GAPI <service> --help` lists the rest.

## Rules

1. **Never send email, create/delete events, delete or share Drive files
   without confirming with the user first.**
2. **Check auth before first use** (`$GSETUP --check`); if it fails, guide the
   user through setup instead of guessing.
3. Calendar times need a timezone (ISO 8601 with offset, or `Z`).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup Steps 2-5 |
| `REFRESH_FAILED` | Token revoked or expired — redo Steps 3-5 |
| `403 Insufficient Permission` | Missing scope — `$GSETUP --revoke`, redo Steps 3-5 |
| `403 Access Not Configured` | The API is not enabled in the Cloud project |
| `ModuleNotFoundError` | `$GSETUP --install-deps` |
| Advanced Protection blocks auth | Workspace admin must allowlist the client ID |
