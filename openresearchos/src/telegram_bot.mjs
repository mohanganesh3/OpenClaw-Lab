/**
 * telegram_bot.mjs — OpenResearchOS Telegram Control
 *
 * Every message has inline buttons. No need to remember commands.
 * Just tap buttons or type commands — works both ways.
 */

import {
  existsSync, readFileSync, writeFileSync, readdirSync, statSync
} from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT   = join(__dir, "..");

// ─── Secrets ──────────────────────────────────────────────────────────────────
const SECRETS   = join(homedir(), ".openclaw", "secrets");
const BOT_TOKEN = readSecret("telegram-bot-token.txt");
const MY_CHAT   = readSecret("telegram-chat-id.txt");
const API       = `https://api.telegram.org/bot${BOT_TOKEN}`;

function readSecret(name) {
  const p = join(SECRETS, name);
  return existsSync(p) ? readFileSync(p, "utf8").trim() : null;
}

// ─── State & flag files ────────────────────────────────────────────────────────
const DAEMON_STATE = join(ROOT, "daemon_state.json");
const BOT_FLAGS    = join(ROOT, "bot_flags.json");

function readState()    { try { return JSON.parse(readFileSync(DAEMON_STATE, "utf8")); } catch { return {}; } }
function writeState(s)  { writeFileSync(DAEMON_STATE, JSON.stringify(s, null, 2)); }
function readFlags()    { try { return JSON.parse(readFileSync(BOT_FLAGS, "utf8")); } catch { return {}; } }
function writeFlags(f)  { writeFileSync(BOT_FLAGS, JSON.stringify(f, null, 2)); }

// ─── Helpers ──────────────────────────────────────────────────────────────────
function esc(s)  { return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function ist()   { return new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }); }
function fmt(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}

function getLatestRun() {
  return listRunsNewestFirst().find((rid) => !isHiddenRun(getRunState(rid))) || null;
}
function listRunsNewestFirst() {
  const d = join(ROOT, "runs");
  if (!existsSync(d)) return [];
  return readdirSync(d).filter(x => x.startsWith("run_"))
    .map(x => ({ name: x, mtime: statSync(join(d, x)).mtime }))
    .sort((a, b) => b.mtime - a.mtime)
    .map((entry) => entry.name);
}
function getLatestRunWithFinalOutput({ preferPaper = false } = {}) {
  for (const rid of listRunsNewestFirst()) {
    const rs = getRunState(rid);
    if (!rs) continue;
    if (isHiddenRun(rs)) continue;
    const terminal = ["TRACE_EXPORTED", "PAPER_DRAFTED_OR_STOPPED", "VENUE_FIT_DONE"].includes(rs.current_state);
    const paperPdf = getRunFile(rid, "paper_draft.pdf");
    const paperMd = getRunFile(rid, "paper_draft.md");
    const failure = getRunFile(rid, "research_failure_report.md");
    const finalReport = getRunFile(rid, "final_report.md");
    if (preferPaper && (paperPdf || paperMd)) return rid;
    if (!preferPaper && (paperPdf || paperMd || (terminal && (failure || finalReport || rs.final_output_path)))) return rid;
  }
  return getLatestRun();
}
function getRunState(id) {
  if (!id) return null;
  try { return JSON.parse(readFileSync(join(ROOT, "runs", id, "run_state.json"), "utf8")); } catch { return null; }
}
function isHiddenRun(rs = {}) {
  return Boolean(rs.suppress_telegram) || String(rs.topic || "").toLowerCase().includes(" smoke ");
}
function getRunFile(id, name) {
  const f = join(ROOT, "runs", id, name);
  return existsSync(f) ? f : null;
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
function outputInfo(rid, rs = {}) {
  const finalPath = rs.final_output_path || "";
  const terminal = ["TRACE_EXPORTED", "PAPER_DRAFTED_OR_STOPPED", "VENUE_FIT_DONE"].includes(rs.current_state);
  const paperPdf = getRunFile(rid, "paper_draft.pdf");
  const paper = getRunFile(rid, "paper_draft.md");
  const failure = getRunFile(rid, "research_failure_report.md");
  const finalReport = getRunFile(rid, "final_report.md");
  if (paperPdf && finalPath.includes("paper_draft")) return { kind: "paper", label: "Paper PDF ready", path: paperPdf };
  if (paper && finalPath.includes("paper_draft")) return { kind: "paper", label: "Paper draft ready", path: paper };
  if (paper && !failure) return { kind: "paper", label: "Paper draft ready", path: paper };
  if (terminal && failure) return { kind: "failure", label: "Paper blocked — failure report", path: failure };
  if (terminal && finalReport) return { kind: "report", label: "Final report", path: finalReport };
  return { kind: "none", label: "No final output yet", path: null };
}

// ─── Telegram API ─────────────────────────────────────────────────────────────
async function tgReq(method, body) {
  try {
    const r = await fetch(`${API}/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await r.json();
  } catch (e) { console.error(`[Bot] API error:`, e.message); return { ok: false }; }
}

// Send plain message
async function send(chatId, text, keyboard = null) {
  const body = { chat_id: chatId, text, parse_mode: "HTML" };
  if (keyboard) body.reply_markup = { inline_keyboard: keyboard };
  const chunks = [];
  for (let i = 0; i < text.length; i += 4000) chunks.push(text.slice(i, i+4000));
  for (let i = 0; i < chunks.length; i++) {
    const b = { chat_id: chatId, text: chunks[i], parse_mode: "HTML" };
    if (keyboard && i === chunks.length - 1) b.reply_markup = { inline_keyboard: keyboard };
    await tgReq("sendMessage", b);
    if (chunks.length > 1) await sleep(300);
  }
}

// Edit an existing message
async function edit(chatId, msgId, text, keyboard = null) {
  const body = { chat_id: chatId, message_id: msgId, text, parse_mode: "HTML" };
  if (keyboard) body.reply_markup = { inline_keyboard: keyboard };
  return tgReq("editMessageText", body);
}

// Answer callback query (removes loading spinner)
async function answer(queryId, text = "") {
  return tgReq("answerCallbackQuery", { callback_query_id: queryId, text });
}

// Send file content
async function sendDoc(chatId, filePath, caption = "") {
  if (!existsSync(filePath)) {
    await send(chatId, `❌ File not found: <code>${esc(filePath)}</code>`);
    return;
  }
  const name = filePath.split("/").pop();

  try {
    const bytes = readFileSync(filePath);
    const form = new FormData();
    form.append("chat_id", String(chatId));
    if (caption) {
      form.append("caption", caption.slice(0, 1024));
      form.append("parse_mode", "HTML");
    }
    const mime = name.endsWith(".pdf") ? "application/pdf"
      : name.endsWith(".tex") ? "application/x-tex"
        : "text/markdown";
    form.append("document", new Blob([bytes], { type: mime }), name);
    const res = await fetch(`${API}/sendDocument`, { method: "POST", body: form });
    const body = await res.json().catch(() => ({ ok: false }));
    if (body.ok) return;
  } catch (err) {
    console.error("[Bot] sendDocument failed:", err.message);
  }

  const content = readFileSync(filePath, "utf8");
  if (content.length < 3600) {
    await send(chatId, `📄 <b>${esc(name)}</b>\n\n<pre>${esc(content.slice(0, 3300))}</pre>`);
  } else {
    await send(chatId, `📄 <b>${esc(name)}</b> (${Math.round(content.length/1024)}KB)\n\n<pre>${esc(content.slice(0, 3300))}</pre>`);
    await send(chatId, `<i>…file truncated. Full file at:\n<code>${esc(filePath)}</code></i>`);
  }
}

async function getUpdates(offset) {
  try {
    const r = await fetch(`${API}/getUpdates?offset=${offset}&timeout=25&allowed_updates=["message","callback_query"]`);
    const d = await r.json();
    return d.ok ? d.result : [];
  } catch { return []; }
}

// ─── Inline keyboard layouts ──────────────────────────────────────────────────

const KB_IDLE = [
  [{ text: "📊 Status", callback_data: "status" }, { text: "📋 Queue", callback_data: "queue" }],
  [{ text: "📁 Past Runs", callback_data: "runs" }, { text: "❓ Help", callback_data: "help" }],
];

const KB_RUNNING = [
  [{ text: "📊 Status", callback_data: "status" }, { text: "⏸ Pause", callback_data: "pause" }],
  [{ text: "⏭ Skip Topic", callback_data: "skip" }, { text: "🛑 Stop", callback_data: "stop" }],
];

const KB_PAUSED = [
  [{ text: "▶️ Resume", callback_data: "resume" }, { text: "⏭ Skip Topic", callback_data: "skip" }],
  [{ text: "🛑 Stop Daemon", callback_data: "stop" }, { text: "📊 Status", callback_data: "status" }],
];

const KB_RESULTS = [
  [{ text: "📎 Final Report", callback_data: "report" }, { text: "📄 Paper", callback_data: "paper" }],
  [{ text: "🔬 Gaps", callback_data: "gaps" }, { text: "💡 Ideas", callback_data: "ideas" }],
  [{ text: "🧪 Experiments", callback_data: "experiments" }, { text: "📊 Status", callback_data: "status" }],
];

const KB_BACK = [
  [{ text: "📊 Status", callback_data: "status" }, { text: "❓ Help", callback_data: "help" }],
];

// ─── Home screen ──────────────────────────────────────────────────────────────
function buildHomeScreen() {
  const state  = readState();
  const flags  = readFlags();
  const status = state.status || "idle";
  const isRunning = status === "running" && !flags.pause && !flags.stop;
  const isPaused  = flags.pause || false;

  const rid    = state.last_run_id || getLatestRun();
  const rs     = getRunState(rid);
  const queue  = state.queue || [];

  let icon, statusLine;
  if (flags.stop)        { icon = "🛑"; statusLine = "STOPPED — send /resume to restart"; }
  else if (isPaused)     { icon = "⏸"; statusLine = `PAUSED at: ${state.current_stage || "??"}`; }
  else if (isRunning)    { icon = "🟢"; statusLine = `WORKING — ${state.current_stage || "??"}`; }
  else                   { icon = "💤"; statusLine = "IDLE — waiting for your command"; }

  let text = `${icon} <b>OpenResearchOS</b>\n`;
  text += `━━━━━━━━━━━━━━━━━━━━━━━\n`;
  text += `<b>Status:</b> ${statusLine}\n`;
  text += `<b>Time:</b>   ${ist()}\n\n`;

  if (isRunning || isPaused) {
    const elapsed = state.current_start ? fmt(Date.now() - state.current_start) : "?";
    text += `<b>Topic:</b>\n<i>${esc((state.last_topic || "none").slice(0, 80))}</i>\n`;
    text += `<b>Stage:</b>   <code>${esc(state.current_stage || "?")}</code>\n`;
    text += `<b>Elapsed:</b> <code>${elapsed}</code>\n\n`;
    if (rs) {
      text += `📚 Papers: <code>${rs.evidence_count || 0}</code>  `;
      text += `🔎 Gaps: <code>${rs.research_gaps?.length || 0}</code>  `;
      text += `💡 Ideas: <code>${rs.idea_tree?.length || 0}</code>\n\n`;
    }
  } else {
    // Idle — show last completed and what to do next
    if (state.completed?.length > 0) {
      const last = state.completed[state.completed.length - 1];
      text += `<b>Last completed:</b>\n<i>${esc((last.topic || "").slice(0, 70))}</i>\n\n`;
    }
    text += `<b>To start research, type:</b>\n`;
    text += `<code>/start your research topic here</code>\n\n`;
  }

  text += `<b>Queue:</b> <code>${queue.length}</code> topics  |  <b>Done:</b> <code>${state.completed?.length || 0}</code> total`;

  const kb = isPaused ? KB_PAUSED : (isRunning ? KB_RUNNING : KB_IDLE);
  return { text, kb };
}

// ─── Command handlers ─────────────────────────────────────────────────────────

async function showHome(chatId) {
  const { text, kb } = buildHomeScreen();
  await send(chatId, text, kb);
}

async function cmdStart(chatId, topic) {
  if (!topic || topic.trim().length < 3) {
    await send(chatId,
      `🔬 <b>Start Research</b>\n\n` +
      `Type a research topic after /start:\n\n` +
      `<b>Examples:</b>\n` +
      `<code>/start mixture of experts routing strategies</code>\n` +
      `<code>/start retrieval augmented generation grounding</code>\n` +
      `<code>/start continual learning catastrophic forgetting</code>\n` +
      `<code>/start test-time compute scaling language models</code>\n\n` +
      `<i>Tip: Be specific — the more specific the topic, the better the paper.</i>`,
      KB_BACK
    );
    return;
  }
  const state = readState();
  if (state.status === "running") {
    await send(chatId,
      `⚠️ <b>Already running a research task</b>\n\n` +
      `<b>Current topic:</b>\n<i>${esc(state.last_topic)}</i>\n` +
      `<b>Stage:</b> <code>${esc(state.current_stage || "?")}</code>\n\n` +
      `<b>Options:</b>\n` +
      `• /stop — finish this and go idle\n` +
      `• /skip — skip this topic and start yours\n` +
      `• /add ${esc(topic.trim())} — add to queue`,
      [
        [{ text: "🛑 Stop Current", callback_data: "stop" }, { text: "⏭ Skip & Start", callback_data: "skip" }],
        [{ text: "📊 Status", callback_data: "status" }],
      ]
    );
    return;
  }
  const flags = readFlags();
  flags.start_topic = topic.trim();
  writeFlags(flags);
  await send(chatId,
    `🚀 <b>Research Starting!</b>\n\n` +
    `<b>Topic:</b>\n<i>${esc(topic.trim())}</i>\n\n` +
    `<b>Pipeline: 7 stages</b>\n` +
    `1️⃣ Search planning  →  2️⃣ Discover papers\n` +
    `3️⃣ Deep reading     →  4️⃣ Gap analysis\n` +
    `5️⃣ Ideas + review   →  6️⃣ Experiments\n` +
    `7️⃣ Write paper\n\n` +
    `<i>You'll get a message at every stage.\nControl anytime with the buttons below.</i>`,
    KB_RUNNING
  );
}

async function cmdRun(chatId) {
  const state = readState();
  if (state.status === "running") {
    await send(chatId,
      `⚠️ <b>Already running:</b>\n<i>${esc(state.last_topic)}</i>\n\nUse /stop or /skip first.`,
      KB_RUNNING
    );
    return;
  }
  const queue = state.queue || [];
  if (queue.length === 0) {
    await send(chatId,
      `📋 <b>Queue is empty</b>\n\n` +
      `Add a topic to run:\n<code>/add your research topic here</code>\n\n` +
      `Or start directly:\n<code>/start your research topic here</code>`,
      KB_IDLE
    );
    return;
  }
  const flags = readFlags();
  flags.run_queue = true;
  writeFlags(flags);
  await send(chatId,
    `▶️ <b>Starting next from queue</b>\n\n` +
    `<b>Topic:</b>\n<i>${esc(queue[0])}</i>\n\n` +
    `<code>${queue.length}</code> more topics in queue after this.`,
    KB_RUNNING
  );
}

async function cmdStatus(chatId) {
  await showHome(chatId);
}

async function cmdQueue(chatId) {
  const state = readState();
  const queue = state.queue || [];
  if (queue.length === 0) {
    await send(chatId,
      `📋 <b>Queue is empty</b>\n\n` +
      `<b>Add topics:</b>\n<code>/add efficient attention mechanisms</code>\n\n` +
      `<b>Or start directly:</b>\n<code>/start your topic here</code>\n\n` +
      `<b>Or reload defaults:</b>\n/refill`,
      [
        [{ text: "🔄 Refill Defaults", callback_data: "refill" }],
        [{ text: "📊 Status", callback_data: "status" }],
      ]
    );
    return;
  }
  let msg = `📋 <b>Research Queue — ${queue.length} topics</b>\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
  queue.forEach((t, i) => {
    msg += `${i + 1}. ${esc(t)}\n`;
  });
  msg += `\n<i>Tap ▶️ Run to start the first one.\nUse /add &lt;topic&gt; to add more.</i>`;
  await send(chatId, msg, [
    [{ text: "▶️ Run Next", callback_data: "run" }, { text: "🗑 Clear Queue", callback_data: "clear" }],
    [{ text: "📊 Status", callback_data: "status" }],
  ]);
}

async function cmdAdd(chatId, topic) {
  if (!topic || topic.trim().length < 3) {
    await send(chatId,
      `📝 <b>Add to Queue</b>\n\n` +
      `Type a topic after /add:\n<code>/add mixture of experts gating</code>`,
      KB_BACK
    );
    return;
  }
  const state = readState();
  state.queue = state.queue || [];
  state.queue.push(topic.trim());
  writeState(state);
  await send(chatId,
    `✅ <b>Added to queue</b>\n\n<i>${esc(topic.trim())}</i>\n\n` +
    `Queue now has <code>${state.queue.length}</code> topic${state.queue.length > 1 ? "s" : ""}.`,
    [
      [{ text: "▶️ Run Now", callback_data: "run" }, { text: "📋 View Queue", callback_data: "queue" }],
      [{ text: "📊 Status", callback_data: "status" }],
    ]
  );
}

async function cmdPause(chatId) {
  const flags = readFlags();
  flags.pause = true;
  writeFlags(flags);
  await send(chatId,
    `⏸ <b>Pause requested</b>\n\n` +
    `The daemon will pause after the current stage finishes.\n\n` +
    `<i>Stages are typically 2–10 minutes each.</i>`,
    KB_PAUSED
  );
}

async function cmdResume(chatId) {
  const flags = readFlags();
  delete flags.pause;
  delete flags.stop;
  writeFlags(flags);
  await send(chatId, `▶️ <b>Resumed!</b>\nDaemon is running again.`, KB_RUNNING);
}

async function cmdSkip(chatId) {
  const flags = readFlags();
  flags.skip = true;
  writeFlags(flags);
  const state = readState();
  await send(chatId,
    `⏭ <b>Skip requested</b>\n\n` +
    `Will skip: <i>${esc((state.last_topic || "current topic").slice(0, 70))}</i>\n\n` +
    `Moving to next topic in queue after current stage ends.`,
    KB_RUNNING
  );
}

async function cmdCancel(chatId) {
  const flags = readFlags();
  flags.skip = true;
  flags.cancel_stage = true;
  writeFlags(flags);
  const state = readState();
  await send(chatId,
    `🚫 <b>Cancelling topic</b>\n\n` +
    `<i>${esc((state.last_topic || "current topic").slice(0, 70))}</i>\n\n` +
    `Stopping as soon as possible and moving on.`,
    KB_RUNNING
  );
}

async function cmdStop(chatId) {
  const flags = readFlags();
  flags.stop = true;
  flags.pause = true;
  writeFlags(flags);
  await send(chatId,
    `🛑 <b>Stopping daemon</b>\n\n` +
    `Will halt after the current stage finishes.\n` +
    `Daemon stays alive but goes idle.\n\n` +
    `<b>To restart:</b> /resume\n` +
    `<b>To start new topic:</b> /start &lt;topic&gt;`,
    [
      [{ text: "▶️ Resume", callback_data: "resume" }],
      [{ text: "📊 Status", callback_data: "status" }],
    ]
  );
}

async function cmdClear(chatId) {
  const state = readState();
  const n = state.queue?.length || 0;
  state.queue = [];
  writeState(state);
  await send(chatId,
    `🗑 <b>Queue cleared</b>\n${n} topic${n !== 1 ? "s" : ""} removed.`,
    [
      [{ text: "🔄 Refill Defaults", callback_data: "refill" }],
      [{ text: "📊 Status", callback_data: "status" }],
    ]
  );
}

async function cmdRefill(chatId) {
  const DEFAULT_QUEUE = [
    "efficient long-context transformers",
    "test-time compute scaling for language models",
    "mixture of experts routing strategies",
    "multimodal foundation models visual grounding",
    "reinforcement learning from human feedback alignment",
    "neural architecture search automated machine learning",
    "continual learning catastrophic forgetting prevention",
    "retrieval augmented generation knowledge grounding",
  ];
  const state = readState();
  state.queue = [...DEFAULT_QUEUE];
  writeState(state);
  await send(chatId,
    `🔄 <b>Queue refilled — ${DEFAULT_QUEUE.length} topics</b>\n\n` +
    DEFAULT_QUEUE.map((t, i) => `${i + 1}. ${esc(t)}`).join("\n"),
    [
      [{ text: "▶️ Run Next", callback_data: "run" }, { text: "📋 View Queue", callback_data: "queue" }],
    ]
  );
}

async function cmdResults(chatId) {
  const state = readState();
  const completed = state.completed || [];
  const rid = getLatestRunWithFinalOutput() || completed.at(-1)?.run_id || state.last_run_id || getLatestRun();
  const rs = getRunState(rid);
  if (!rid || !rs) {
    await send(chatId,
      `📊 <b>No completed topics yet</b>\n\n` +
      `Start research with:\n<code>/start your topic here</code>`,
      KB_IDLE
    );
    return;
  }
  const last = completed.find((entry) => entry.run_id === rid) || completed.at(-1) || {};
  const exps = allExperimentResults(rs);
  const passed = exps.filter(e => e.success || e.status === "passed").length;
  const out = outputInfo(rid, rs);

  let msg = `📊 <b>Latest Research Run</b>\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
  msg += `<b>Topic:</b>\n<i>${esc(last.topic || rs.topic || "unknown topic")}</i>\n\n`;
  msg += `<b>Run:</b> <code>${esc(rid)}</code>\n`;
  if (last.timestamp) msg += `<b>Finished:</b> ${new Date(last.timestamp).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}\n`;
  msg += `<b>State:</b> <code>${esc(rs.current_state || "?")}</code>\n`;
  msg += `<b>Readiness:</b> <code>${esc(rs.paper_readiness_level || "RRL-0")}</code>\n`;
  msg += `<b>Output:</b> ${esc(out.label)}\n\n`;
  msg += `📚 Evidence locked: <code>${rs.evidence_ids?.length || 0}</code>\n`;
  msg += `📄 Papers read:     <code>${rs.parsed_papers?.length || 0}</code>\n`;
  msg += `🔎 Gaps found:      <code>${rs.research_gaps?.length || 0}</code>\n`;
  msg += `💡 Ideas:           <code>${rs.idea_tree?.length || 0}</code>\n`;
  msg += `🧪 Experiments:     <code>${exps.length}</code> (<code>${passed}</code> passed)\n\n`;
  if (out.kind === "failure") {
    msg += `<b>Honest gate:</b> no paper draft was produced because the evidence/experiment gates did not pass.\n\n`;
  }
  if (completed.length > 1) {
    msg += `<i>All-time completed: ${completed.length} topics</i>`;
  }
  await send(chatId, msg, KB_RESULTS);
}

async function cmdPaper(chatId, runId) {
  const rid = runId || getLatestRunWithFinalOutput({ preferPaper: true });
  if (!rid) { await send(chatId, "❌ No runs found yet.", KB_IDLE); return; }
  const rs = getRunState(rid);
  const paperPdf = getRunFile(rid, "paper_draft.pdf");
  const paper = getRunFile(rid, "paper_draft.md");
  if (paperPdf) {
    await send(chatId, `🎓 <b>Paper PDF ready</b>\nRun: <code>${esc(rid)}</code>`);
    await sendDoc(chatId, paperPdf, `🎓 Paper PDF — <b>${esc(rs?.topic || rid)}</b>`);
    if (paper) await send(chatId, `<i>Markdown/LaTeX are also in the run directory.</i>`, KB_RESULTS);
    return;
  }
  if (paper) {
    await send(chatId, `📄 <b>Paper draft ready</b>\nRun: <code>${esc(rid)}</code>`);
    await sendDoc(chatId, paper, `📄 Paper draft — <b>${esc(rs?.topic || rid)}</b>`);
    await send(chatId, `<i>Use /report ${esc(rid)} for the full final package.</i>`, KB_RESULTS);
    return;
  }
  const failure = getRunFile(rid, "research_failure_report.md");
  if (failure) {
    await send(
      chatId,
      `📋 <b>No paper draft for this run</b>\n\n` +
      `<b>Run:</b> <code>${esc(rid)}</code>\n` +
      `<b>Readiness:</b> <code>${esc(rs?.paper_readiness_level || "RRL-0")}</code>\n\n` +
      `The gate blocked paper writing. Sending the failure report instead.`,
      KB_RESULTS
    );
    await sendDoc(chatId, failure, `📋 Failure report — <b>${esc(rs?.topic || rid)}</b>`);
    return;
  }
  await send(chatId, `❌ No paper or failure report found yet.\n<code>${esc(rid)}</code>\n\n<i>Use /status to see the current stage.</i>`, KB_BACK);
}

async function cmdReport(chatId, runId) {
  const rid = runId || getLatestRunWithFinalOutput();
  if (!rid) { await send(chatId, "❌ No runs found yet.", KB_IDLE); return; }
  const rs = getRunState(rid);
  const out = outputInfo(rid, rs || {});
  if (!out.path) {
    await send(chatId, `❌ Final report is not ready yet.\n<code>${esc(rid)}</code>`, KB_BACK);
    return;
  }
  await send(
    chatId,
    `📎 <b>${esc(out.label)}</b>\n\n` +
    `<b>Topic:</b> ${esc(rs?.topic || "?")}\n` +
    `<b>Readiness:</b> <code>${esc(rs?.paper_readiness_level || "RRL-0")}</code>\n` +
    `<b>Run:</b> <code>${esc(rid)}</code>`,
    KB_RESULTS
  );
  await sendDoc(chatId, out.path, `📎 ${esc(out.label)} — <b>${esc(rs?.topic || rid)}</b>`);
}

async function cmdGaps(chatId, runId) {
  const rid = runId || getLatestRun();
  if (!rid) { await send(chatId, "❌ No runs found yet.", KB_IDLE); return; }
  const rs   = getRunState(rid);
  const gaps = rs?.research_gaps || [];
  if (gaps.length === 0) {
    await send(chatId, `❌ No gaps yet in this run.\n<i>Gaps are found in Stage 4.</i>`, KB_BACK);
    return;
  }
  let msg = `🔬 <b>Research Gaps — ${gaps.length} found</b>\n`;
  msg += `<i>${esc(rid.slice(0, 42))}</i>\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
  gaps.slice(0, 8).forEach((g, i) => {
    msg += `<b>${i + 1}. ${esc((g.gap_title || g.text || "?").slice(0, 75))}</b>\n`;
    if (g.testable_hypothesis) msg += `   <i>${esc(g.testable_hypothesis.slice(0, 90))}</i>\n`;
    msg += "\n";
  });
  await send(chatId, msg, KB_RESULTS);
}

async function cmdIdeas(chatId, runId) {
  const rid  = runId || getLatestRun();
  if (!rid) { await send(chatId, "❌ No runs found yet.", KB_IDLE); return; }
  const rs   = getRunState(rid);
  const ideas = rs?.idea_tree || rs?.ideas || [];
  if (ideas.length === 0) {
    await send(chatId, `❌ No ideas yet.\n<i>Ideas are generated in Stage 5.</i>`, KB_BACK);
    return;
  }
  let msg = `💡 <b>Research Ideas — ${ideas.length} generated</b>\n`;
  msg += `<i>${esc(rid.slice(0, 42))}</i>\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
  ideas.slice(0, 8).forEach((idea, i) => {
    const score = idea.reviewer_average ?? idea.novelty_score ?? "?";
    const status = idea.status === "shortlisted" ? "✅" : idea.status === "rejected" ? "❌" : "🔵";
    msg += `${status} <b>${i + 1}. ${esc((idea.title || "?").slice(0, 62))}</b>\n`;
    msg += `   Score: <code>${score}</code>\n\n`;
  });
  await send(chatId, msg, KB_RESULTS);
}

async function cmdExperiments(chatId, runId) {
  const rid = runId || getLatestRun();
  if (!rid) { await send(chatId, "❌ No runs found yet.", KB_IDLE); return; }
  const rs = getRunState(rid);
  const exps = allExperimentResults(rs || {});
  if (exps.length === 0) {
    await send(chatId, `🧪 <b>No experiments recorded yet</b>\n\nRun: <code>${esc(rid)}</code>\n<i>Experiments run after idea review and approval.</i>`, KB_BACK);
    return;
  }
  let msg = `🧪 <b>Experiments — ${exps.length}</b>\n`;
  msg += `<i>${esc(rid)}</i>\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
  exps.slice(0, 8).forEach((e, i) => {
    const m = e.metrics || {};
    const improvement = m.improvement_pct ?? (typeof m.improvement === "object" ? Object.values(m.improvement)[0] : m.improvement);
    const dataset = m.dataset || e.dataset || "unknown";
    const status = e.success || e.status === "passed" ? "✅" : "⚠️";
    msg += `${status} <b>${i + 1}. ${esc(e.experiment_id || e.idea_id || "experiment")}</b>\n`;
    msg += `   Level: <code>${esc(e.level || m.level || "?")}</code> | Dataset: <code>${esc(dataset)}</code>\n`;
    if (improvement != null) msg += `   Improvement: <code>${esc(typeof improvement === "number" ? improvement.toFixed(3) : improvement)}</code>\n`;
    if (e.next_action || e.decision) msg += `   Next: <code>${esc(e.next_action || e.decision)}</code>\n`;
    msg += "\n";
  });
  await send(chatId, msg, KB_RESULTS);
}

async function cmdRuns(chatId) {
  const runsDir = join(ROOT, "runs");
  if (!existsSync(runsDir)) { await send(chatId, "❌ No runs directory found.", KB_IDLE); return; }
  const runs = readdirSync(runsDir)
    .filter(d => d.startsWith("run_"))
    .map(d => ({ name: d, mtime: statSync(join(runsDir, d)).mtime }))
    .sort((a, b) => b.mtime - a.mtime)
    .filter((r) => !isHiddenRun(getRunState(r.name)))
    .slice(0, 8);
  if (runs.length === 0) { await send(chatId, "📁 No runs yet.", KB_IDLE); return; }
  let msg = `📁 <b>Recent Runs — ${runs.length}</b>\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━━━━\n\n`;
  runs.forEach((r, i) => {
    const topic = r.name.replace(/run_\d+_/, "").replace(/_[a-f0-9]+$/, "").replace(/-/g, " ").slice(0, 45);
    const d = r.mtime.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", dateStyle: "short", timeStyle: "short" });
    msg += `${i + 1}. <i>${esc(topic)}</i>\n   <code>${esc(r.name.slice(0, 38))}</code>\n   ${d}\n\n`;
  });
  msg += `<i>/report &lt;run_id&gt; for final output\n/paper &lt;run_id&gt; only if gates passed\n/gaps &lt;run_id&gt; for gaps from that run</i>`;
  await send(chatId, msg, KB_BACK);
}

async function cmdLogs(chatId) {
  const logFile = join(ROOT, "logs", "daemon.log");
  if (!existsSync(logFile)) {
    // Try pm2 logs path
    await send(chatId,
      `📋 <b>Daemon Logs</b>\n\n` +
      `<i>Log file not found at expected path.\nCheck pm2 logs with:\n<code>npx pm2 logs openresearchos --lines 30</code></i>`,
      KB_BACK
    );
    return;
  }
  const lines = readFileSync(logFile, "utf8").split("\n").filter(Boolean);
  const latestStart = Math.max(
    lines.findLastIndex((line) => line.includes("Telegram bot started")),
    lines.findLastIndex((line) => line.includes("IDLE — waiting for Telegram commands"))
  );
  const recent = latestStart >= 0 ? lines.slice(latestStart, latestStart + 30) : lines.slice(-20);
  const title = latestStart >= 0 ? "Current daemon session" : "Last 20 log lines";
  await send(chatId, `📋 <b>${title}</b>\n\n<pre>${esc(recent.join("\n").slice(0, 3500))}</pre>`, KB_BACK);
}

async function cmdHelp(chatId) {
  await send(chatId,
    `🤖 <b>OpenResearchOS — Command Guide</b>\n` +
    `━━━━━━━━━━━━━━━━━━━━━━━\n\n` +

    `<b>🚀 START RESEARCH</b>\n` +
    `<code>/start &lt;topic&gt;</code> — Start researching a topic NOW\n` +
    `<code>/run</code>           — Run next topic from queue\n\n` +

    `<b>📋 MANAGE QUEUE</b>\n` +
    `<code>/add &lt;topic&gt;</code>  — Add topic to queue\n` +
    `<code>/queue</code>         — See all queued topics\n` +
    `<code>/clear</code>         — Clear queue\n` +
    `<code>/refill</code>        — Load 8 default topics\n\n` +

    `<b>🎮 CONTROL</b>\n` +
    `<code>/pause</code>         — Pause after current stage\n` +
    `<code>/resume</code>        — Continue from pause\n` +
    `<code>/skip</code>          — Skip current topic → next\n` +
    `<code>/cancel</code>        — Cancel topic immediately\n` +
    `<code>/stop</code>          — Stop daemon (goes idle)\n\n` +

    `<b>📊 CHECK STATUS</b>\n` +
    `<code>/status</code>        — Full live status\n` +
    `<code>/logs</code>          — Recent daemon logs\n\n` +

    `<b>📄 GET RESULTS</b>\n` +
    `<code>/results</code>       — Summary of last run\n` +
    `<code>/report</code>        — Get final output: paper or failure report\n` +
    `<code>/paper</code>         — Get paper draft only if gates passed\n` +
    `<code>/gaps</code>          — Research gaps found\n` +
    `<code>/ideas</code>         — Ideas with scores\n` +
    `<code>/experiments</code>   — Experiment metrics and next actions\n` +
    `<code>/runs</code>          — Past 8 run IDs\n\n` +

    `<b>💡 QUICK START</b>\n` +
    `<code>/start mixture of experts routing</code>\n` +
    `<code>/start continual learning forgetting</code>\n` +
    `<code>/start retrieval augmented generation</code>`,
    [
      [{ text: "📊 Status", callback_data: "status" }, { text: "📋 Queue", callback_data: "queue" }],
      [{ text: "🔄 Refill & Run", callback_data: "refill" }],
    ]
  );
}

// ─── Dispatcher ───────────────────────────────────────────────────────────────
async function dispatch(chatId, text) {
  const parts = text.trim().split(/\s+/);
  const cmd   = (parts[0] || "").toLowerCase();
  const arg1  = parts[1] || "";
  const rest  = parts.slice(1).join(" ");
  console.log(`[Bot] Command: ${cmd} ${rest}`);

  switch (cmd) {
    case "/start":    return rest.length > 0 ? cmdStart(chatId, rest) : cmdHelp(chatId);
    case "/run":      return cmdRun(chatId);
    case "/status":   return cmdStatus(chatId);
    case "/queue":    return cmdQueue(chatId);
    case "/add":      return cmdAdd(chatId, rest);
    case "/pause":    return cmdPause(chatId);
    case "/resume":   return cmdResume(chatId);
    case "/skip":     return cmdSkip(chatId);
    case "/cancel":   return cmdCancel(chatId);
    case "/stop":     return cmdStop(chatId);
    case "/clear":    return cmdClear(chatId);
    case "/refill":   return cmdRefill(chatId);
    case "/results":  return cmdResults(chatId);
    case "/report":   return cmdReport(chatId, arg1);
    case "/paper":    return cmdPaper(chatId, arg1);
    case "/gaps":     return cmdGaps(chatId, arg1);
    case "/ideas":    return cmdIdeas(chatId, arg1);
    case "/experiments": return cmdExperiments(chatId, arg1);
    case "/runs":     return cmdRuns(chatId);
    case "/logs":     return cmdLogs(chatId);
    case "/help":     return cmdHelp(chatId);
    default:
      await send(chatId,
        `❓ Unknown: <code>${esc(cmd)}</code>\n\nSend /help for all commands.`,
        KB_BACK
      );
  }
}

// Button callback dispatcher
async function dispatchCallback(chatId, queryId, data) {
  await answer(queryId);
  switch (data) {
    case "status":  return showHome(chatId);
    case "run":     return cmdRun(chatId);
    case "pause":   return cmdPause(chatId);
    case "resume":  return cmdResume(chatId);
    case "skip":    return cmdSkip(chatId);
    case "cancel":  return cmdCancel(chatId);
    case "stop":    return cmdStop(chatId);
    case "queue":   return cmdQueue(chatId);
    case "clear":   return cmdClear(chatId);
    case "refill":  return cmdRefill(chatId);
    case "paper":   return cmdPaper(chatId, null);
    case "report":  return cmdReport(chatId, null);
    case "gaps":    return cmdGaps(chatId, null);
    case "ideas":   return cmdIdeas(chatId, null);
    case "experiments": return cmdExperiments(chatId, null);
    case "runs":    return cmdRuns(chatId);
    case "help":    return cmdHelp(chatId);
    case "results": return cmdResults(chatId);
  }
}

// ─── Security ─────────────────────────────────────────────────────────────────
function isAllowed(chatId) { return String(chatId) === String(MY_CHAT); }

// ─── Main bot loop ────────────────────────────────────────────────────────────
export async function startBot({ announce = false } = {}) {
  if (!BOT_TOKEN || !MY_CHAT) {
    console.log("[Bot] Telegram not configured — bot disabled");
    return;
  }

  console.log(`[Bot] 🤖 Telegram bot started — listening from chat ${MY_CHAT}`);

  if (announce) {
    const state = readState();
    const queue = state.queue || [];

    await send(MY_CHAT,
      `🤖 <b>OpenResearchOS — Ready</b>\n` +
      `━━━━━━━━━━━━━━━━━━━━━━━\n` +
      `⏰ ${ist()}\n\n` +
      `💤 <b>Status: IDLE</b>\n` +
      `I'm waiting for your command.\n` +
      `I won't do anything until you say so.\n\n` +
      `<b>To start research:</b>\n` +
      `<code>/start your research topic here</code>\n\n` +
      `<b>Example topics:</b>\n` +
      `• mixture of experts routing strategies\n` +
      `• retrieval augmented generation grounding\n` +
      `• continual learning catastrophic forgetting\n` +
      `• test-time compute scaling language models\n\n` +
      `<b>Queue:</b> <code>${queue.length}</code> topics waiting\n` +
      `<b>Completed:</b> <code>${state.completed?.length || 0}</code> all-time`,
      [
        [{ text: "📊 Status", callback_data: "status" }, { text: "❓ Help", callback_data: "help" }],
        [{ text: "📋 Queue", callback_data: "queue" }, { text: "🔄 Refill Queue", callback_data: "refill" }],
      ]
    );
  }

  let offset = 0;

  while (true) {
    try {
      const updates = await getUpdates(offset);
      for (const upd of updates) {
        offset = upd.update_id + 1;

        // Text message
        if (upd.message?.text) {
          const chatId = upd.message.chat.id;
          if (!isAllowed(chatId)) continue;
          try { await dispatch(chatId, upd.message.text); }
          catch (err) {
            console.error("[Bot] Error:", err.message);
            await send(chatId, `❌ Error: <code>${esc(err.message?.slice(0, 200))}</code>`);
          }
        }

        // Button press
        if (upd.callback_query) {
          const cq = upd.callback_query;
          const chatId = cq.message?.chat?.id;
          if (!isAllowed(chatId)) { await answer(cq.id); continue; }
          try { await dispatchCallback(chatId, cq.id, cq.data); }
          catch (err) {
            console.error("[Bot] Callback error:", err.message);
            await answer(cq.id, "Error — check /status");
          }
        }
      }
    } catch (e) {
      console.error("[Bot] Polling error:", e.message);
      await sleep(5000);
    }
    await sleep(800);
  }
}
