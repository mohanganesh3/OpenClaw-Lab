# Telegram Channel Contract

Telegram is the first remote-control channel for OpenResearchOS.

## Security Rules

- Private allowlist only.
- Pairing required before accepting commands.
- No public bot access.
- No experiment execution without approval.
- No outbound professor messages without human approval.

## Commands

```text
/start_research <topic>
/status <run_id>
/approve_micro_probe <run_id> <idea_id>
/approve_probe <run_id> <idea_id>
/reject <run_id> <idea_id>
/summarize <run_id>
/export_trace <run_id>
```

## Command Mapping

Local command-router smoke:

```bash
node channels/command_router.mjs \
  --allowlist channels/telegram_allowlist.example.json \
  --sender telegram:user-id-or-username \
  --message "/status <run_id>"
```

`/start_research <topic>`:

```bash
node src/openresearch.mjs start --topic "<topic>"
```

`/status <run_id>`:

```bash
node src/openresearch.mjs status --run <run_id>
```

`/approve_micro_probe <run_id> <idea_id>`:

```bash
node src/openresearch.mjs approve --run <run_id> --level micro_probe --idea <idea_id>
node src/openresearch.mjs run-micro-probe --run <run_id>
```

`/summarize <run_id>`:

```bash
node src/openresearch.mjs write --run <run_id>
node src/openresearch.mjs status --run <run_id>
```

## Sarvam Voice Add-On

Voice note flow:

1. Telegram receives audio.
2. Sarvam STT transcribes it.
3. OpenClaw maps the transcript to a command.
4. OpenResearchOS updates the run.
5. Sarvam TTS returns a short spoken status.

Voice is for command convenience. File artifacts remain the source of truth.
