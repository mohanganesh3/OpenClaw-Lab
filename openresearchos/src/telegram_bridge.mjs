/**
 * telegram_bridge.mjs — OpenResearchOS Telegram Notifications
 *
 * Sends real-time research progress to your Telegram chat.
 * Uses your bot: @Researcher_open_claw_bot
 *
 * Reads from:
 *   ~/.openclaw/secrets/telegram-bot-token.txt
 *   ~/.openclaw/secrets/telegram-chat-id.txt
 *
 * Usage:
 *   import * as tg from './telegram_bridge.mjs';
 *   await tg.send("Research started!");
 *   await tg.sendProgress(runId, stage, details);
 *   await tg.sendResult(runId, paperTitle, score, gaps);
 */

import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";

// ─── Config ──────────────────────────────────────────────────────────────────

function readSecret(name) {
  const path = `${homedir()}/.openclaw/secrets/${name}`;
  if (!existsSync(path)) return null;
  return readFileSync(path, "utf8").trim();
}

const BOT_TOKEN = readSecret("telegram-bot-token.txt");
const CHAT_ID = readSecret("telegram-chat-id.txt");

const ENABLED = !!(BOT_TOKEN && CHAT_ID);

if (!ENABLED) {
  console.error("  [TG] WARNING: Telegram secrets not found. Notifications disabled.");
}

// ─── Core send ───────────────────────────────────────────────────────────────

/**
 * Send a plain or HTML-formatted message to Telegram.
 * Uses parse_mode=HTML for rich formatting (bold, italic, code, etc.)
 */
export async function send(text, { parseMode = "HTML", silent = false } = {}) {
  if (!ENABLED) return false;

  // Telegram message limit is 4096 chars — split if needed
  const chunks = splitMessage(text, 4000);
  let success = true;

  for (const chunk of chunks) {
    try {
      const body = new URLSearchParams({
        chat_id: CHAT_ID,
        text: chunk,
        parse_mode: parseMode,
        disable_notification: silent ? "true" : "false",
      });

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      const res = await fetch(
        `https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`,
        { method: "POST", body, signal: controller.signal }
      );
      clearTimeout(timeout);

      if (!res.ok) {
        const err = await res.text();
        console.error(`  [TG] Send failed: ${res.status} — ${err.slice(0, 100)}`);
        success = false;
      }
    } catch (err) {
      console.error(`  [TG] Network error: ${err.message}`);
      success = false;
    }

    // Small delay between chunks
    if (chunks.length > 1) await sleep(500);
  }

  return success;
}

function splitMessage(text, maxLen) {
  if (text.length <= maxLen) return [text];
  const chunks = [];
  let remaining = text;
  while (remaining.length > 0) {
    let split = maxLen;
    // Try to split at newline
    const newline = remaining.lastIndexOf("\n", maxLen);
    if (newline > maxLen * 0.7) split = newline;
    chunks.push(remaining.slice(0, split));
    remaining = remaining.slice(split).trim();
  }
  return chunks;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Formatted updates ───────────────────────────────────────────────────────

const STAGE_EMOJI = {
  SEARCH_PLAN_CREATED:       "📋",
  LITERATURE_DISCOVERY:      "🔍",
  DEEP_READING:              "📖",
  GAP_ANALYSIS:              "🔬",
  IDEAS_GENERATED:           "💡",
  NOVELTY_CHECK:             "🆕",
  REVIEWER_COUNCIL:          "👩‍⚖️",
  EXPERIMENT_RUNNING:        "🧪",
  RESULTS_READY:             "📊",
  PAPER_WRITTEN:             "📄",
  NEURIPS_READY:             "🏆",
  COMPLETED:                 "✅",
  ERROR:                     "❌",
  LOOP_START:                "🔄",
  IDLE:                      "😴",
};

/**
 * Send a stage progress update.
 */
export async function sendStage(runId, stage, details = "", { silent = false } = {}) {
  const emoji = STAGE_EMOJI[stage] || "⚙️";
  const shortRun = runId?.split("_").slice(0, 3).join("_") || "run";
  const detailLine = details ? `\n<i>${escapeHtml(details.slice(0, 200))}</i>` : "";

  return send(
    `${emoji} <b>OpenResearchOS</b>\n` +
    `Stage: <code>${stage}</code>${detailLine}\n` +
    `Run: <code>${shortRun}</code>`,
    { parseMode: "HTML", silent }
  );
}

/**
 * Send a rich research summary (gap list, ideas shortlisted, etc.)
 */
export async function sendSummary(runId, topic, data = {}) {
  const {
    papersRead = 0,
    gaps = [],
    ideasGenerated = 0,
    ideasPassed = 0,
    bestIdea = null,
    bestScore = null,
    stage = "RUNNING",
    elapsed = null,
  } = data;

  const shortRun = runId?.split("_").slice(0, 3).join("_") || "run";
  const elapsedStr = elapsed ? `⏱ ${formatElapsed(elapsed)}` : "";

  let msg = `📊 <b>Research Update</b> ${elapsedStr}\n`;
  msg += `🔬 Topic: <b>${escapeHtml(topic?.slice(0, 80) || "?")}</b>\n`;
  msg += `📖 Papers read: <code>${papersRead}</code>\n`;

  if (gaps.length > 0) {
    msg += `\n🔎 <b>Top Research Gaps:</b>\n`;
    gaps.slice(0, 3).forEach((g, i) => {
      msg += `  ${i + 1}. ${escapeHtml((g.gap_title || g.text || "").slice(0, 80))}\n`;
    });
  }

  if (ideasGenerated > 0) {
    msg += `\n💡 Ideas generated: <code>${ideasGenerated}</code> → passed review: <code>${ideasPassed}</code>\n`;
  }

  if (bestIdea) {
    msg += `\n🏅 <b>Best Idea:</b> ${escapeHtml(bestIdea.slice(0, 100))}\n`;
    if (bestScore) msg += `   Reviewer score: <code>${bestScore}/10</code>\n`;
  }

  msg += `\n📍 Stage: <code>${stage}</code>\n`;
  msg += `🔑 Run: <code>${shortRun}</code>`;

  return send(msg, { parseMode: "HTML" });
}

/**
 * Send experiment result notification.
 */
export async function sendExperimentResult(runId, ideaTitle, metrics = {}, verdict = "PROMOTE") {
  const verdictEmoji = { PROMOTE: "🚀", REVISE: "🔧", KILL: "💀" }[verdict] || "📊";
  const accuracy = metrics.accuracy_mean ?? metrics.accuracy ?? metrics.f1_mean ?? null;
  const pval = metrics.p_value ?? null;

  let msg = `${verdictEmoji} <b>Experiment Result</b>\n`;
  msg += `💡 Idea: <b>${escapeHtml(ideaTitle?.slice(0, 80))}</b>\n`;
  msg += `📊 Verdict: <code>${verdict}</code>\n`;
  if (accuracy !== null) msg += `📈 Accuracy: <code>${(accuracy * 100).toFixed(1)}%</code>\n`;
  if (pval !== null) msg += `📉 p-value: <code>${pval.toFixed(4)}</code>\n`;
  msg += `🔑 Run: <code>${runId?.split("_").slice(0, 3).join("_")}</code>`;

  return send(msg, { parseMode: "HTML" });
}

/**
 * Send paper completion notification.
 */
export async function sendPaperReady(runId, topic, paperPath) {
  return send(
    `📄 <b>Paper Draft Ready!</b>\n` +
    `🔬 Topic: <b>${escapeHtml(topic?.slice(0, 80))}</b>\n` +
    `📁 File: <code>${paperPath}</code>\n` +
    `🔑 Run: <code>${runId?.split("_").slice(0, 3).join("_")}</code>`,
    { parseMode: "HTML" }
  );
}

/**
 * Send error notification.
 */
export async function sendError(runId, stage, errorMsg) {
  return send(
    `❌ <b>Error in OpenResearchOS</b>\n` +
    `Stage: <code>${stage}</code>\n` +
    `Error: <code>${escapeHtml(errorMsg?.slice(0, 300))}</code>\n` +
    `Run: <code>${runId?.split("_").slice(0, 3).join("_") || "unknown"}</code>`,
    { parseMode: "HTML", silent: false }
  );
}

/**
 * Send a heartbeat / "still working" ping for long-running tasks.
 */
export async function sendHeartbeat(runId, stage, papersDone, papersTotal, elapsed) {
  return send(
    `💓 <b>Still working...</b>\n` +
    `📖 ${papersDone}/${papersTotal} papers | Stage: <code>${stage}</code>\n` +
    `⏱ Elapsed: ${formatElapsed(elapsed)}\n` +
    `Run: <code>${runId?.split("_").slice(0, 3).join("_")}</code>`,
    { parseMode: "HTML", silent: true }
  );
}

/**
 * Send loop status when starting a new topic in daemon mode.
 */
export async function sendLoopStatus(loopNum, topic, queueRemaining) {
  return send(
    `🔄 <b>Starting Research Loop #${loopNum}</b>\n` +
    `🔬 Topic: <b>${escapeHtml(topic?.slice(0, 80))}</b>\n` +
    `📋 Queue remaining: <code>${queueRemaining}</code> topics`,
    { parseMode: "HTML" }
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatElapsed(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export { ENABLED };
