# OpenClaw Integration

OpenResearchOS is designed as an OpenClaw workflow, not a detached script.

## Control Split

OpenClaw controls reasoning and tool use:

- `web_search` for live literature and competitor discovery,
- `web_fetch` for source snapshots,
- `pdf` for paper reading,
- `browser` for GitHub, OpenReview, Papers with Code, and dynamic pages,
- `sessions_spawn` for specialist subagents,
- `exec` for local deterministic runner commands,
- `memory/wiki` for durable lessons and failed ideas,
- Telegram for remote control,
- Sarvam for voice input/output,
- trajectory export for audit.

The local runner controls reproducibility:

- state machine,
- file artifacts,
- experiment workspace creation,
- approval files,
- metrics and logs,
- readiness decisions.

## Main Skill

Use the skill:

```text
openresearchos-coordinator
```

The coordinator should:

1. create or resume a run,
2. spawn specialist subagents when live research is needed,
3. call the local runner through OpenClaw `exec`,
4. enforce approval before execution,
5. export the trajectory at the end.

## Required Working Directory

Use:

```text
/Users/mohanganesh/OpenClaw-Lab/openresearchos
```

## Canonical Commands

```bash
node src/openresearch.mjs start --topic "<topic>"
node src/openresearch.mjs discover --run <run_id>
node src/openresearch.mjs read --run <run_id>
node src/openresearch.mjs map --run <run_id>
node src/openresearch.mjs ideas --run <run_id>
node src/openresearch.mjs approve --run <run_id> --level micro_probe --idea <idea_id>
node src/openresearch.mjs run-micro-probe --run <run_id>
node src/openresearch.mjs write --run <run_id>
```

Use `run-probe`, `run-ablation`, and `run-mvp` only after the previous level survives review.

Voice commands:

```bash
node src/openresearch.mjs voice-summary --run <run_id> --target-language en-IN
node src/openresearch.mjs voice-transcribe --run <run_id> --file <audio.wav>
```

Sarvam voice is used through OpenResearchOS command workflows:

- `voice-transcribe` converts a user voice note into text for a research command.
- `voice-summary` speaks a concise run status back to the user.
- OpenClaw `messages.tts` / `talk.speak` can use the local Sarvam CLI TTS provider.

Important boundary: the Control UI **Start Talk** button is browser realtime Talk. In OpenClaw 2026.5.28, that path calls browser realtime providers, and Sarvam local STT/TTS does not register as a browser realtime provider. If Start Talk shows a Google realtime-provider error, that means browser realtime Talk is trying to use Google Live Voice. The Sarvam command voice path can still be working correctly.

Channel-router command:

```bash
node channels/command_router.mjs --sender <allowed-sender> --message "/status <run_id>"
```

## Live Research Upgrade

The local runner can produce a complete fallback demo, but a professor-facing OpenClaw run should enrich it with live OpenClaw tools:

1. Use `web_search` to discover latest papers.
2. Use `web_fetch` to snapshot pages.
3. Use `pdf` to read top PDFs.
4. Use `browser` for OpenReview, GitHub, leaderboards, and Papers with Code.
5. Put discovered evidence into the run artifacts.
6. Use `sessions_spawn` for literature scout, novelty tribunal, and reviewer council.
7. Use `memory/wiki` to store failed ideas and stable lessons.
8. Export the OpenClaw trajectory.

## Demo Boundary

The local smoke demo proves the state machine and experiment loop.

The full OpenClaw demo proves:

- OpenClaw skills are loaded,
- live tools are used,
- local experiments run through `exec`,
- the trajectory export exists,
- the final output refuses unsupported claims.
