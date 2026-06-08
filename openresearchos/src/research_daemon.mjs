#!/usr/bin/env node
/**
 * research_daemon.mjs — Telegram-Controlled Research Daemon
 *
 * Starts IDLE. Does NOTHING until you command it from Telegram.
 *
 * Telegram commands to control:
 *   /start <topic>  → start researching that topic NOW
 *   /queue          → show queue
 *   /add <topic>    → add topic to queue
 *   /run            → start next topic from queue
 *   /pause          → pause after current stage
 *   /resume         → resume
 *   /skip           → skip current topic
 *   /cancel         → cancel immediately
 *   /stop           → stop research, go back to idle
 *   /status         → what's happening right now
 *   /results        → last completed run summary
 *   /paper          → get the paper draft
 *   /gaps           → research gaps
 *   /ideas          → ideas with scores
 *   /runs           → all completed runs
 *   /logs           → daemon logs
 *   /help           → full command list
 */

import { spawn } from "node:child_process";
import {
  existsSync, readFileSync, writeFileSync,
  readdirSync, statSync
} from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import { startBot } from "./telegram_bot.mjs";
import { sendDocument as sendTelegramDocument } from "./telegram_notifier.mjs";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT   = join(__dir, "..");

// ─── Telegram push ────────────────────────────────────────────────────────────
const SECRETS   = join(homedir(), ".openclaw", "secrets");
const BOT_TOKEN = readSecret("telegram-bot-token.txt");
const CHAT_ID   = readSecret("telegram-chat-id.txt");

function readSecret(name) {
  const p = join(SECRETS, name);
  return existsSync(p) ? readFileSync(p, "utf8").trim() : null;
}

async function tg(text, silent = false) {
  if (!BOT_TOKEN || !CHAT_ID) { console.log("[TG]", text.slice(0, 80)); return; }
  const chunks = [];
  for (let i = 0; i < text.length; i += 4000) chunks.push(text.slice(i, i + 4000));
  for (const chunk of chunks) {
    try {
      await fetch(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: CHAT_ID, text: chunk, parse_mode: "HTML", disable_notification: silent }),
      });
    } catch (e) { console.error("[TG] error:", e.message); }
    if (chunks.length > 1) await sleep(300);
  }
}

function escHtml(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ─── State files ──────────────────────────────────────────────────────────────
const STATE_FILE = join(ROOT, "daemon_state.json");
const FLAGS_FILE = join(ROOT, "bot_flags.json");

function loadState() {
  try { return JSON.parse(readFileSync(STATE_FILE, "utf8")); } catch {}
  return { queue: [], completed: [], loop: 0, last_topic: null, last_run_id: null, current_stage: null, current_start: null, status: "idle" };
}

function saveState(s) { writeFileSync(STATE_FILE, JSON.stringify(s, null, 2)); }

function readFlags() {
  try { return JSON.parse(readFileSync(FLAGS_FILE, "utf8")); } catch { return {}; }
}

function setFlag(k, v) {
  const f = readFlags(); f[k] = v;
  writeFileSync(FLAGS_FILE, JSON.stringify(f, null, 2));
}

function clearFlag(k) {
  const f = readFlags(); delete f[k];
  writeFileSync(FLAGS_FILE, JSON.stringify(f, null, 2));
}

function clearAllFlags() { writeFileSync(FLAGS_FILE, "{}"); }

// ─── Flag checker between stages ─────────────────────────────────────────────
async function checkFlags(stageName, state) {
  const f = readFlags();

  if (f.stop) {
    clearAllFlags();
    state.status = "idle"; state.current_stage = null; saveState(state);
    await tg(`🛑 <b>Stopped at stage: ${escHtml(stageName)}</b>\nBack to idle. Send /start &lt;topic&gt; to begin new research.`);
    return "stop";
  }

  if (f.skip || f.cancel_stage) {
    clearFlag("skip"); clearFlag("cancel_stage");
    await tg(`⏭ <b>Skipped</b> — moving to next queued topic or going idle.`);
    return "skip";
  }

  if (f.pause) {
    console.log(`[Daemon] PAUSED at: ${stageName}`);
    await tg(`⏸ <b>Paused at: ${escHtml(stageName)}</b>\n\nSend:\n/resume — continue\n/skip — skip topic\n/stop — go idle`);
    while (true) {
      await sleep(2000);
      const f2 = readFlags();
      if (f2.stop) { clearAllFlags(); state.status = "idle"; saveState(state); await tg("🛑 Stopped. Send /start &lt;topic&gt; to begin."); return "stop"; }
      if (f2.skip) { clearFlag("skip"); clearFlag("pause"); return "skip"; }
      if (!f2.pause) break;
    }
    await tg(`▶️ <b>Resumed at stage: ${escHtml(stageName)}</b>`);
  }

  return "ok";
}

// ─── Run one pipeline stage ───────────────────────────────────────────────────
async function runStage(cmd, args, { timeoutMs = 3600000, allowTg = false } = {}) {
  return new Promise((resolve) => {
    console.log(`[Daemon] Running: ${cmd} ${args.join(" ")}`);
    const env = { ...process.env };
    if (!allowTg) env.OPENRESEARCHOS_SUPPRESS_TG = "1";
    const proc = spawn(cmd, args, {
      cwd: ROOT,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "", runId = null;
    proc.stdout.on("data", d => { const t = d.toString(); output += t; process.stdout.write(t); const m = t.match(/"run_id"\s*:\s*"([^"]+)"/); if (m) runId = m[1]; });
    proc.stderr.on("data", d => { const t = d.toString(); output += t; process.stderr.write(t); });
    const timer = setTimeout(() => { console.error("[Daemon] Timeout — killing"); proc.kill("SIGTERM"); }, timeoutMs);
    proc.on("close", code => { clearTimeout(timer); resolve({ ok: code === 0, runId, output }); });
    proc.on("error", err => { clearTimeout(timer); resolve({ ok: false, runId: null, output: err.message }); });
  });
}

function getLatestRun() {
  const d = join(ROOT, "runs");
  if (!existsSync(d)) return null;
  const dirs = readdirSync(d).filter(x => x.startsWith("run_")).map(x => ({ name: x, mtime: statSync(join(d, x)).mtime })).sort((a, b) => b.mtime - a.mtime);
  return dirs[0]?.name || null;
}

function getRunState(id) {
  try { return JSON.parse(readFileSync(join(ROOT, "runs", id, "run_state.json"), "utf8")); } catch { return null; }
}
function allExperimentResults(rs = {}) {
  const rows = [
    ...(rs.experiment_results || []),
    ...(rs.micro_probe_results || []),
    ...(rs.probe_results || []),
    ...(rs.ablation_results || []),
    ...(rs.mvp_results || []),
  ];
  const seen = new Set();
  return rows.filter((row) => {
    const key = row?.experiment_id || `${row?.idea_id || "idea"}:${row?.level || "level"}:${row?.completed_at || ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
function finalOutputInfo(rs = {}) {
  const output = rs.final_output_path || "";
  if (output.includes("paper_draft")) return { kind: "paper", label: "Paper draft ready", command: "/paper" };
  if (output.includes("research_failure_report")) return { kind: "failure", label: "Paper blocked — failure report ready", command: "/report" };
  if (output.includes("final_report")) return { kind: "report", label: "Final report ready", command: "/report" };
  return { kind: "pending", label: "Final output pending", command: "/status" };
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function fmt(ms) {
  const s = Math.floor(ms / 1000), m = Math.floor(s / 60), h = Math.floor(m / 60);
  if (s < 60) return `${s}s`; if (m < 60) return `${m}m ${s % 60}s`; return `${h}h ${m % 60}m`;
}

async function runReadUntilComplete(runId, expectedCount, t0) {
  let lastParsed = getRunState(runId)?.parsed_papers?.length || 0;
  const maxPasses = 4;
  const fullPdfCap = Math.max(0, Math.min(
    expectedCount,
    Number(process.env.OPENRESEARCHOS_FULL_PDF_READ_CAP || 20)
  ));
  for (let pass = 1; pass <= maxPasses; pass++) {
    await tg(
      `📖 <b>[3/7] Reading Pass ${pass}/${maxPasses}</b>\n` +
      `Parsed so far: <code>${lastParsed}/${expectedCount}</code>\n` +
      `Full-PDF cap: <code>${fullPdfCap}</code> priority papers\n` +
      `<i>All evidence gets a paper card. Top priority papers get full PDF extraction; the rest get abstract/metadata cards. The daemon will not advance while unread evidence remains.</i>`,
      true
    );
    // Use allowTg=true so per-paper Telegram updates are sent from the read subprocess
    const result = await runStage("node", ["src/openresearch.mjs", "read", "--run", runId, "--deep-read-cap", String(fullPdfCap)], { timeoutMs: 45 * 60 * 1000, allowTg: true });
    await sleep(1000);
    const state = getRunState(runId);
    const parsed = state?.parsed_papers?.length || 0;
    await tg(`📖 <b>Reading progress</b>: <code>${parsed}/${expectedCount}</code> papers parsed | ⏱ ${fmt(Date.now() - t0)}`, true);
    if (parsed >= expectedCount) return { ok: true, parsed, output: result.output };
    if (parsed <= lastParsed && !result.ok) {
      return { ok: false, parsed, output: result.output };
    }
    lastParsed = parsed;
  }
  return { ok: false, parsed: lastParsed, output: "read_pass_limit_reached" };
}

// ─── Full research pipeline for one topic ────────────────────────────────────
async function researchTopic(topic, state) {
  const t0 = Date.now();
  state.last_topic = topic; state.current_start = t0; state.status = "running"; saveState(state);

  await tg(
    `🔬 <b>Research Started!</b>\n\n` +
    `<b>Topic:</b> ${escHtml(topic)}\n` +
    `⏰ ${new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}\n\n` +
    `<b>Pipeline:</b> 7 stages\n` +
    `<i>Control anytime:\n/pause  /skip  /stop  /status</i>`
  );

  // ── Stage 1: Search planning ─────────────────────────────────────────────
  state.current_stage = "1/7 search-planning"; saveState(state);
  if ((await checkFlags("search-planning", state)) !== "ok") return;

  await tg(`🗂 <b>[1/7] Search Planning</b>\nGenerating queries for: <i>${escHtml(topic)}</i>`, true);
  const s1 = await runStage("node", ["src/openresearch.mjs", "start", "--topic", topic], { timeoutMs: 300000 });
  const runId = s1.runId || getLatestRun();
  if (!runId || !s1.ok) { await tg(`❌ Stage 1 failed — could not create search plan.\nSkipping topic.`); return; }
  state.last_run_id = runId; saveState(state);
  const nq = parseInt(s1.output.match(/"queries"\s*:\s*(\d+)/)?.[1] || "0");
  await tg(`✅ <b>[1/7] Done</b> — ${nq} search queries created\n<code>${runId.slice(0, 42)}</code>`, true);

  // ── Stage 2: Discover papers ─────────────────────────────────────────────
  state.current_stage = "2/7 discovering"; saveState(state);
  if ((await checkFlags("discovering", state)) !== "ok") return;

  await tg(`🔍 <b>[2/7] Discovering Papers</b>\nSearching Semantic Scholar/arXiv and locking evidence (${nq} queries)...`);
  const discoverResult = await runStage("node", ["src/openresearch.mjs", "discover", "--run", runId], { timeoutMs: 600000 });
  await sleep(500); // ensure state file is flushed
  const dState = getRunState(runId);
  const evCount = dState?.evidence_ids?.length || dState?.evidence_count || 0;
  if (!discoverResult.ok || evCount <= 0) {
    await tg(
      `🛑 <b>Discovery did not produce evidence — stopped safely</b>\n\n` +
      `Run: <code>${escHtml(runId)}</code>\n` +
      `Evidence locked: <code>${evCount}</code>\n\n` +
      `The daemon will not continue to reading, gaps, ideas, or paper writing without an evidence index.`
    );
    state.status = "idle";
    state.current_stage = null;
    state.current_start = null;
    saveState(state);
    return;
  }
  await tg(`✅ <b>[2/7] Done</b> — <code>${evCount}</code> papers found | ⏱ ${fmt(Date.now() - t0)}`);


  // ── Stage 3: Deep reading ─────────────────────────────────────────────────
  state.current_stage = "3/7 reading"; saveState(state);
  if ((await checkFlags("reading", state)) !== "ok") return;

  const fullPdfCap = Math.max(0, Math.min(evCount, Number(process.env.OPENRESEARCHOS_FULL_PDF_READ_CAP || 20)));
  await tg(`📖 <b>[3/7] Deep Reading</b>\nParsing ${evCount} papers.\nFull PDF extraction for top ${fullPdfCap}; abstract/metadata cards for the rest.\n<i>This will take ~${Math.ceil((fullPdfCap * 45 + evCount * 2) / 60)}–${Math.ceil((fullPdfCap * 90 + evCount * 4) / 60)} min. You'll get progress updates.</i>`, true);
  const readOutcome = await runReadUntilComplete(runId, evCount, t0);
  await sleep(1000);
  const rState = getRunState(runId);
  const papersRead = rState?.parsed_papers?.length || readOutcome.parsed || 0;
  const fullPdfs = (rState?.parsed_papers || []).filter(p => p.full_text_tool && p.full_text_tool !== 'abstract_only' && p.full_text_tool !== 'keyword_fallback').length;
  if (evCount > 0 && papersRead < evCount) {
    if (papersRead === 0) {
      // Truly nothing was read — cannot do gap analysis
      await tg(
        `🛑 <b>Reading failed — 0 papers parsed</b>\n\n` +
        `Parsed: <code>0/${evCount}</code>\n` +
        `Reason: ${escHtml(readOutcome.output?.slice(-500) || "read stage returned nothing")}\n\n` +
        `Cannot do gap analysis without paper evidence.\n` +
        `Restart with: <code>/start ${escHtml(topic)}</code>`
      );
      state.status = "idle";
      state.current_stage = null;
      state.current_start = null;
      saveState(state);
      return;
    }
    // Partial read — warn but continue with what we have
    await tg(
      `⚠️ <b>Reading partially complete — continuing with available evidence</b>\n` +
      `Parsed: <code>${papersRead}/${evCount}</code> papers\n` +
      `<i>Gap analysis and ideas will use ${papersRead} parsed papers. Results may be less complete than a full read.</i>`,
      true
    );
  }
  await tg(`✅ <b>[3/7] Done</b> — <code>${papersRead}</code> papers read\n📄 Full PDF: <code>${fullPdfs}</code> | Abstract: <code>${papersRead - fullPdfs}</code>\n⏱ ${fmt(Date.now() - t0)}`);
  if (papersRead === 0) {
    await tg(`⚠️ <b>Warning:</b> No papers parsed into state — read stage may have timed out. Continuing with abstract-based gap analysis.`);
  }


  // ── Stage 4: Gap analysis ─────────────────────────────────────────────────
  state.current_stage = "4/7 gap-analysis"; saveState(state);
  if ((await checkFlags("gap-analysis", state)) !== "ok") return;

  await tg(`🔬 <b>[4/7] Gap Analysis</b>\nFinding research gaps across ${papersRead} papers...`, true);
  await runStage("node", ["src/openresearch.mjs", "map", "--run", runId], { timeoutMs: 300000 });
  const gaps = getRunState(runId)?.research_gaps || [];
  if (gaps.length === 0) {
    // Gap analysis returned 0 gaps — this should be very rare since buildGaps has 3 fallback layers.
    // Log a warning but continue — the ideas stage has its own fallback templates.
    await tg(
      `⚠️ <b>[4/7] Gap Analysis — 0 gaps from LLM, continuing with topic templates</b>\n` +
      `Run: <code>${escHtml(runId)}</code>\n` +
      `<i>Ideas stage will use topic-derived templates as fallback. Research will continue.</i>`,
      true
    );
  }
  let gapMsg = `✅ <b>[4/7] Done</b> — ${gaps.length} gaps found\n\n`;
  gaps.slice(0, 4).forEach((g, i) => {
    gapMsg += `${i + 1}. <b>${escHtml((g.gap_title || g.text || "").slice(0, 72))}</b>\n`;
    if (g.testable_hypothesis) gapMsg += `   <i>${escHtml(g.testable_hypothesis.slice(0, 80))}</i>\n`;
  });
  gapMsg += `\n⏱ ${fmt(Date.now() - t0)} | /gaps for full list`;
  await tg(gapMsg);

  // ── Stage 5: Ideas + Adversarial Review ──────────────────────────────────
  state.current_stage = "5/7 ideas-review"; saveState(state);
  if ((await checkFlags("ideas-review", state)) !== "ok") return;

  const ideaCount = (getRunState(runId)?.research_gaps || []).length || 6;
  await tg(`💡 <b>[5/7] Ideas + Adversarial Review</b>\n${ideaCount} gaps → ideas → 4 reviewers × 2 adversarial phases\n<i>~${ideaCount * 4}–${ideaCount * 6} min. You'll get reviewer scores live.</i>`);
  await runStage("node", ["src/openresearch.mjs", "ideas", "--run", runId], { timeoutMs: 2700000 }); // 45 min for 6 ideas × 8 LLM calls
  const iState = getRunState(runId);
  const ideas = iState?.idea_tree || iState?.ideas || [];
  const cands = iState?.experiment_candidates || [];
  if (ideas.length === 0) {
    await tg(
      `⛔ <b>[5/7] Ideas stopped</b>\n` +
      `Ideas: <code>0</code> — No ideas could be generated. Cannot run experiments.\n\n` +
      `Run: <code>${escHtml(runId)}</code>`,
      true
    );
    state.status = "idle"; state.current_stage = null; state.current_start = null; saveState(state);
    return;
  }
  if (cands.length === 0) {
    // No formal candidates but ideas exist — use the top idea as a provisional candidate
    await tg(
      `⚠️ <b>[5/7] No formal experiment candidates after reviewer council</b>\n` +
      `Using top-scored idea as provisional experiment candidate.\n` +
      `Ideas: <code>${ideas.length}</code> | Auto-promoting top idea for experiments.`,
      true
    );
  }
  let ideaMsg = `✅ <b>[5/7] Done</b> — ${ideas.length} ideas, ${cands.length} approved for experiments\n\n`;
  ideas.slice(0, 5).forEach((idea, i) => {
    const score = idea.reviewer_average ?? idea.novelty_score ?? "?";
    ideaMsg += `${i + 1}. [<code>${score}</code>] <b>${escHtml((idea.title || "?").slice(0, 60))}</b>\n`;
  });
  ideaMsg += `\n⏱ ${fmt(Date.now() - t0)} | /ideas for full list`;
  await tg(ideaMsg);

  // ── Stage 6: Experiments ─────────────────────────────────────────────────
  state.current_stage = "6/7 experiments"; saveState(state);
  if ((await checkFlags("experiments", state)) !== "ok") return;

  const expCandidates = (getRunState(runId)?.experiment_candidates || []);
  if (expCandidates.length === 0) {
    // expCandidates will be auto-populated by ideasRun (top-idea fallback).
    // If still empty, run-micro-probe will use the top idea from idea_tree.
    await tg(
      `⚠️ <b>[6/7] 0 formal candidates — attempting auto-promotion from idea tree</b>\n` +
      `Run: <code>${escHtml(runId)}</code>\n` +
      `<i>The experiment loop will promote the top-scored idea automatically.</i>`,
      true
    );
  }
  await tg(`🧪 <b>[6/7] Local Experiment Loop</b>\n<code>${expCandidates.length}</code> ideas approved\nRunning micro-probes/probes on the Mac with saved code, logs, metrics, and next-action decisions.`);
  // Run micro-probe WITHOUT --offline so LLM self-healing can fix any code errors.
  // The deterministic pythonProbeSource is the fallback, but LLM repair gives us a better shot.
  await runStage("node", ["src/openresearch.mjs", "run-micro-probe", "--run", runId], { timeoutMs: 1800000 });
  const ladderState = getRunState(runId) || {};
  const chosenIdea = (ladderState.micro_probe_results || []).find((r) => r.success)?.idea_id
    || ladderState.experiment_candidates?.[0]?.idea_id
    || ladderState.idea_tree?.[0]?.idea_id;
  const microPassed = (ladderState.micro_probe_results || []).some((r) => r.success);
  if (chosenIdea && microPassed) {
    await tg(`🧪 <b>Micro-probe passed</b>\nPromoting <code>${escHtml(chosenIdea)}</code> to probe → ablation → MVP with deterministic local code and saved metrics.`, true);
    const levels = [
      { level: "probe", label: "Probe" },
      { level: "ablation", label: "Ablation" },
      { level: "mvp", label: "MVP" },
    ];
    for (const item of levels) {
      const before = getRunState(runId) || {};
      const previousOk =
        item.level === "probe" ||
        (item.level === "ablation" && (before.probe_results || []).some((r) => r.success)) ||
        (item.level === "mvp" && (before.ablation_results || []).some((r) => r.success));
      if (!previousOk) {
        await tg(`🧪 <b>${escHtml(item.label)} skipped</b>\nPrevious experiment level did not pass, so the ladder stopped honestly.`, true);
        break;
      }
      await tg(`🧪 <b>${escHtml(item.label)} starting</b>\nIdea: <code>${escHtml(chosenIdea)}</code>`, true);
      await runStage("node", ["src/openresearch.mjs", "approve", "--run", runId, "--level", item.level, "--idea", chosenIdea, "--skip-codegen"], { timeoutMs: 120000 });
      await runStage("node", ["src/openresearch.mjs", `run-${item.level}`, "--run", runId, "--idea", chosenIdea, "--skip-codegen", "--no-revision-loop", "--offline"], { timeoutMs: item.level === "mvp" ? 7200000 : 2700000 });
      const after = getRunState(runId) || {};
      const collection = item.level === "probe" ? after.probe_results : item.level === "ablation" ? after.ablation_results : after.mvp_results;
      const latest = (collection || []).find((r) => r.idea_id === chosenIdea);
      await tg(
        `${latest?.success ? "✅" : "⚠️"} <b>${escHtml(item.label)} result</b>\n` +
        `Success: <code>${latest?.success ? "yes" : "no"}</code>\n` +
        `Next: <code>${escHtml(latest?.next_action || latest?.decision || "review_required")}</code>`,
        true
      );
      if (!latest?.success) {
        await tg(`⚠️ <b>${escHtml(item.label)} did not pass</b> — stopping ladder, proceeding to paper gate.`, true);
        break;
      }
    }
  } else {
    await tg(`🧪 <b>Experiment ladder stopped</b>\nNo successful micro-probe, so probe/ablation/MVP were not run.`, true);
  }
  const exps = allExperimentResults(getRunState(runId) || {});
  const passed = exps.filter(e => e.success || e.status === "passed").length;
  let expMsg = `✅ <b>[6/7] Done</b> — ${exps.length} experiments (${passed} passed)\n\n`;
  exps.slice(0, 4).forEach(e => {
    const m = e.metrics || e.results || {};
    const acc = m.accuracy_mean ?? m.accuracy ?? m.f1_mean ?? m.score;
    expMsg += `• <b>${escHtml((e.idea_title || e.experiment_id || "exp").slice(0, 55))}</b>`;
    if (acc != null) expMsg += ` → <code>${typeof acc === "number" ? (acc > 1 ? acc.toFixed(2) : (acc * 100).toFixed(1) + "%") : acc}</code>`;
    expMsg += "\n";
  });
  expMsg += `\n⏱ ${fmt(Date.now() - t0)}`;
  await tg(expMsg);

  // ── Stage 7: Readiness gate + final package ───────────────────────────────
  state.current_stage = "7/7 writing"; saveState(state);
  if ((await checkFlags("writing", state)) !== "ok") return;

  await tg(`📝 <b>[7/7] Readiness Gate + Final Package</b>\nChecking evidence, novelty, experiments, and reviewer gates.\n<i>A paper draft is written only if gates pass; otherwise an honest failure report is produced.</i>`, true);
  await runStage("node", ["src/openresearch.mjs", "write", "--run", runId, "--offline"], { timeoutMs: 600000 });
  await runStage("node", ["src/openresearch.mjs", "verify", "--run", runId, "--offline"], { timeoutMs: 120000 });

  // ── Complete ──────────────────────────────────────────────────────────────
  const fs = getRunState(runId);
  const finalInfo = finalOutputInfo(fs || {});
  const finalExperiments = allExperimentResults(fs || {});
  const finalPassed = finalExperiments.filter(e => e.success || e.status === "passed").length;
  const elapsed = Date.now() - t0;

  state.completed.push({ topic, timestamp: Date.now(), run_id: runId });
  state.status = "idle"; state.current_stage = null; state.current_start = null;
  saveState(state);

  await tg(
    `${finalInfo.kind === "paper" ? "🎓" : "📋"} <b>Research Run Finished</b>\n\n` +
    `🔬 <b>${escHtml(topic)}</b>\n\n` +
    `📝 Output: <b>${escHtml(finalInfo.label)}</b>\n` +
    `📈 Readiness: <code>${escHtml(fs?.paper_readiness_level || "RRL-0")}</code>\n` +
    `📚 Evidence locked: <code>${fs?.evidence_ids?.length || 0}</code>\n` +
    `📄 Papers read: <code>${fs?.parsed_papers?.length || papersRead}</code>\n` +
    `🔎 Gaps found: <code>${fs?.research_gaps?.length || gaps.length}</code>\n` +
    `💡 Ideas: <code>${fs?.idea_tree?.length || ideas.length}</code>\n` +
    `🧪 Experiments: <code>${finalExperiments.length}</code> (<code>${finalPassed}</code> passed)\n` +
    `⏱ Total: <b>${fmt(elapsed)}</b>\n\n` +
    (finalInfo.kind === "failure" ? `<b>Honest gate:</b> no paper draft was sent because paper gates did not pass.\n\n` : "") +
    `<b>Get your results:</b>\n` +
    `${finalInfo.command} — final output\n` +
    `/paper — paper draft only if gates passed\n` +
    `/gaps — research gaps\n` +
    `/ideas — all ideas\n` +
    `/experiments — metrics and next actions\n` +
    `/results — summary\n\n` +
    `<i>Send /start &lt;topic&gt; to begin next research</i>`
  );

  const finalDocPath = fs?.pdf_output_path || fs?.final_output_path;
  if (finalDocPath && existsSync(finalDocPath)) {
    const caption = finalInfo.kind === "paper"
      ? `🎓 Paper draft ready\n<b>${escHtml(topic)}</b>\n\nReadiness: ${escHtml(fs?.paper_readiness_level || "RRL-0")}`
      : `📋 Final report ready\n<b>${escHtml(topic)}</b>\n\nReadiness: ${escHtml(fs?.paper_readiness_level || "RRL-0")}`;
    await sendTelegramDocument(finalDocPath, caption).catch((err) => {
      console.error("[Daemon] Final document send failed:", err.message);
    });
  }
}

// ─── Main: IDLE loop — only runs when Telegram commands it ───────────────────
async function main() {
  clearAllFlags();

  const state = loadState();
  state.status = "idle"; state.current_stage = null;
  saveState(state);

  // Start Telegram bot (bidirectional listener) in parallel
  startBot().catch(e => console.error("[Bot] Fatal:", e.message));

  // Announce readiness
  await tg(
    `🤖 <b>OpenResearchOS — Ready</b>\n\n` +
    `I'm idle and waiting for your command.\n\n` +
    `<b>To start research:</b>\n` +
    `/start &lt;topic&gt; — research a topic now\n` +
    `/add &lt;topic&gt; — add to queue\n` +
    `/run — run next from queue\n\n` +
    `<b>Queue:</b> <code>${state.queue.length}</code> topics waiting\n` +
    `<b>Completed:</b> <code>${state.completed.length}</code> topics\n\n` +
    `Send /help for all commands.`
  );

  console.log("[Daemon] IDLE — waiting for Telegram commands...");

  // ── IDLE LOOP — poll flags every 2s ──────────────────────────────────────
  while (true) {
    await sleep(2000);

    // Re-read state (bot may have modified queue)
    const s = loadState();
    Object.assign(state, s);

    const flags = readFlags();

    // Check for /start <topic> command
    if (flags.start_topic) {
      const topic = flags.start_topic;
      clearFlag("start_topic");
      console.log(`[Daemon] START command received: "${topic}"`);
      try {
        await researchTopic(topic, state);
      } catch (err) {
        console.error("[Daemon] Error:", err.message);
        await tg(`❌ <b>Error during research:</b>\n<code>${escHtml(err.message?.slice(0, 200))}</code>\n\nSend /start &lt;topic&gt; to try again.`);
        state.status = "idle"; state.current_stage = null; saveState(state);
      }
      console.log("[Daemon] Back to IDLE — waiting for next command...");
      continue;
    }

    // Check for /run command (run next from queue)
    if (flags.run_queue) {
      clearFlag("run_queue");
      const freshState = loadState();
      if (freshState.queue.length === 0) {
        await tg(`❌ <b>Queue is empty.</b>\nUse /add &lt;topic&gt; to add topics first.`);
        continue;
      }
      const topic = freshState.queue.shift();
      freshState.last_topic = topic;
      saveState(freshState);
      Object.assign(state, freshState);
      console.log(`[Daemon] RUN command — next from queue: "${topic}"`);
      try {
        await researchTopic(topic, state);
      } catch (err) {
        console.error("[Daemon] Error:", err.message);
        await tg(`❌ <b>Error:</b>\n<code>${escHtml(err.message?.slice(0, 200))}</code>\n\nSend /start &lt;topic&gt; to try again.`);
        state.status = "idle"; state.current_stage = null; saveState(state);
      }
      console.log("[Daemon] Back to IDLE...");
    }
  }
}

// ─── CLI entrypoint ──────────────────────────────────────────────────────────
if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  process.on("SIGINT",  () => { console.log("[Daemon] stopped SIGINT"); process.exit(0); });
  process.on("SIGTERM", () => { console.log("[Daemon] stopped SIGTERM"); process.exit(0); });
  process.on("uncaughtException", async err => { await tg(`🔥 <b>Crash (pm2 restarting)</b>\n<code>${escHtml(err.message?.slice(0,200))}</code>`).catch(() => {}); });

  main().catch(async err => {
    await tg(`💀 <b>Fatal:</b>\n<code>${escHtml(err.message)}</code>`).catch(() => {});
    process.exit(1);
  });
}
