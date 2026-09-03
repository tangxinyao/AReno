---
name: himalaya
description: "Himalaya CLI: IMAP/SMTP email from terminal (Gmail: App Password)."
version: 1.1.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, IMAP, SMTP, CLI, Communication]
    homepage: https://github.com/pimalaya/himalaya
  areno:
    trimmed: "Setup + the Gmail App Password path kept; the full CLI reference
      (flags, folders, attachments, debugging) was cut to fit a training
      context window. Full doc: himalaya upstream."
prerequisites:
  commands: [himalaya]
---

# Himalaya Email CLI

Terminal email over IMAP/SMTP. For Gmail this is the *short* path to a
connected mailbox: an App Password, no Google Cloud project, no OAuth client.
Use it when the user only needs to read and send mail; use the
`google-workspace` skill when they also need Calendar/Drive/Docs.

## Prerequisites

1. The `himalaya` CLI (`himalaya --version` to verify — if it is missing and
   it cannot be installed here, say so instead of pretending it is set up)
2. `~/.config/himalaya/config.toml`
3. IMAP/SMTP credentials

```bash
# pre-built binary (Linux/macOS)
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh
brew install himalaya            # macOS
cargo install himalaya --locked  # any platform with Rust
```

## Gmail setup (App Password)

The user does one manual step: at https://myaccount.google.com/apppasswords
(2-Step Verification must be on) they create a 16-character App Password and
give it to you. Then write `~/.config/himalaya/config.toml`:

```toml
[accounts.gmail]
email = "you@gmail.com"
display-name = "Your Name"
default = true

backend.type = "imap"
backend.host = "imap.gmail.com"
backend.port = 993
backend.encryption.type = "tls"
backend.login = "you@gmail.com"
backend.auth.type = "password"
backend.auth.raw = "THE_APP_PASSWORD"

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.gmail.com"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "you@gmail.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.raw = "THE_APP_PASSWORD"

# Gmail's folder names are not himalaya's canonical ones
folder.aliases.inbox = "INBOX"
folder.aliases.sent = "[Gmail]/Sent Mail"
folder.aliases.drafts = "[Gmail]/Drafts"
folder.aliases.trash = "[Gmail]/Trash"
```

Use `folder.aliases.X` (plural, dotted, directly under `[accounts.NAME]`).
The pre-v1.2.0 `[accounts.NAME.folder.alias]` form parses but is ignored, and
save-to-Sent then fails *after* SMTP delivery — a retry sends duplicates.

Prefer a secret command (`backend.auth.cmd = "pass show email/imap"`) over a
raw password when the user has a password store.

Verify the account works:

```bash
himalaya folder list
himalaya envelope list --output json
```

## Common operations

```bash
himalaya envelope list                          # INBOX
himalaya envelope list --folder "Sent" --page 1 --page-size 20
himalaya envelope list from john@example.com subject meeting
himalaya message read 42
himalaya message move "Archive" 42
himalaya message delete 42
```

Compose non-interactively — pipe the message, never rely on `$EDITOR`:

```bash
cat << 'EOF' | himalaya template send
From: you@gmail.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya!
EOF
```

Reply keeps the thread: `himalaya template reply 42 | ... | himalaya template send`.

## Tips

- `--output json` for parseable output; `himalaya <command> --help` for the rest.
- Message IDs are relative to the current folder; re-list after moving.
- `himalaya account configure` is an interactive wizard — writing config.toml
  directly is more reliable from an agent.
