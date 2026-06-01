# WhatsApp Channel Contract

WhatsApp is the second channel after Telegram works.

Use the same command router shape as Telegram, but keep experiment approval explicit.

Recommended commands:

```text
/status <run_id>
/approve_micro_probe <run_id> <idea_id>
/run_micro_probe <run_id>
/summarize <run_id>
/voice_summary <run_id> en-IN
```

Security rules:

- allowlist only,
- no public forwarding,
- no experiment execution without approval,
- no professor/outbound messages without human approval.

