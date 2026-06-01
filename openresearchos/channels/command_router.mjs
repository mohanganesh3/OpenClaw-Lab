#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const RUNNER = resolve(ROOT, "src", "openresearch.mjs");

function parseArgs(argv) {
  const opts = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      opts._.push(token);
      continue;
    }
    const key = token.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    opts[key] = value;
    i += 1;
  }
  return opts;
}

async function loadAllowlist(path) {
  if (!path || !existsSync(path)) {
    return { allow_all: false, allowed_senders: [] };
  }
  return JSON.parse(await readFile(path, "utf8"));
}

function runRunner(args) {
  const result = spawnSync(process.execPath, [RUNNER, ...args], {
    cwd: ROOT,
    encoding: "utf8",
    timeout: 1000 * 600,
  });
  return {
    ok: result.status === 0,
    status: result.status,
    stdout: result.stdout,
    stderr: result.stderr,
    command: `node ${RUNNER} ${args.join(" ")}`,
  };
}

function splitCommand(message) {
  const trimmed = String(message || "").trim();
  const [command, ...rest] = trimmed.split(/\s+/);
  return { command, rest, text: rest.join(" ") };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const message = opts.message || opts._.join(" ");
  if (!message) throw new Error("Missing --message");
  const sender = opts.sender || "unknown";
  const allowlist = await loadAllowlist(opts.allowlist || resolve(__dirname, "telegram_allowlist.json"));
  const allowed = Boolean(allowlist.allow_all) || (allowlist.allowed_senders || []).includes(sender);
  if (!allowed) {
    console.log(JSON.stringify({ ok: false, error: "sender_not_allowed", sender }, null, 2));
    return;
  }
  const parsed = splitCommand(message);
  let result;
  switch (parsed.command) {
    case "/start_research":
      result = runRunner(["start", "--topic", parsed.text]);
      break;
    case "/status":
      result = runRunner(["status", "--run", parsed.rest[0], "--json"]);
      break;
    case "/approve_micro_probe":
      result = runRunner(["approve", "--run", parsed.rest[0], "--level", "micro_probe", "--idea", parsed.rest[1]]);
      break;
    case "/run_micro_probe":
      result = runRunner(["run-micro-probe", "--run", parsed.rest[0]]);
      break;
    case "/approve_probe":
      result = runRunner(["approve", "--run", parsed.rest[0], "--level", "probe", "--idea", parsed.rest[1]]);
      break;
    case "/summarize":
      result = runRunner(["write", "--run", parsed.rest[0]]);
      break;
    case "/voice_summary":
      result = runRunner(["voice-summary", "--run", parsed.rest[0], "--target-language", parsed.rest[1] || "en-IN"]);
      break;
    case "/verify":
      result = runRunner(["verify", "--run", parsed.rest[0]]);
      break;
    case "/export_trace":
      result = {
        ok: true,
        status: 0,
        stdout: `/export-trajectory ${parsed.rest[0]}`,
        stderr: "",
        command: "OpenClaw trajectory export instruction",
      };
      break;
    case "/runs": {
      const { readdirSync } = await import("node:fs");
      const runsDir = resolve(ROOT, "runs");
      let runIds = [];
      try {
        runIds = readdirSync(runsDir)
          .filter(d => d.startsWith("run_"))
          .sort()
          .reverse()
          .slice(0, 10);
      } catch {
        runIds = [];
      }
      result = {
        ok: true,
        status: 0,
        stdout: JSON.stringify({ recent_runs: runIds }, null, 2),
        stderr: "",
        command: "list runs directory",
      };
      break;
    }
    case "/demo":
      result = runRunner(["demo", "--topic", parsed.text || "autonomous research agents with reviewer simulation"]);
      break;
    case "/help":
      result = {
        ok: true,
        status: 0,
        stdout: [
          "OpenResearchOS — Commands:",
          "  /start_research <topic>     Start a new research run",
          "  /demo <topic>               Quick demo (full pipeline, offline-safe)",
          "  /status <run_id>            Show current state + metrics",
          "  /runs                       List 10 most recent run IDs",
          "  /approve_micro_probe <run_id> <idea_id>   Approve micro-probe",
          "  /run_micro_probe <run_id>   Execute approved micro-probe",
          "  /approve_probe <run_id> <idea_id>         Approve full probe",
          "  /summarize <run_id>         Write paper draft (if RRL≥5)",
          "  /voice_summary <run_id> [lang]  Generate voice summary",
          "  /verify <run_id>            Verify all artifacts",
          "  /export_trace <run_id>      Export full research trace",
          "  /help                       Show this message",
        ].join("\n"),
        stderr: "",
        command: "help",
      };
      break;
    default:
      result = {
        ok: false,
        status: 2,
        stdout: "",
        stderr: `Unknown command: ${parsed.command}. Try /help for the full command list.`,
        command: "none",
      };
  }
  console.log(JSON.stringify({ sender, message, ...result }, null, 2));
  if (!result.ok) process.exitCode = result.status || 1;
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
