/**
 * Telegram Notifier for OpenResearchOS.
 *
 * Sends rich pipeline milestone updates, live status, and final research
 * reports directly to your Telegram chat via the Bot API.
 *
 * Config (stored in ~/.openclaw/secrets/):
 *   telegram-bot-token.txt  — Bot token from @BotFather
 *   telegram-chat-id.txt    — Your personal chat ID (auto-detected)
 */

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { readFile } from "node:fs/promises";

const SECRETS_DIR = join(homedir(), ".openclaw", "secrets");
const API_BASE = "https://api.telegram.org";

function getSecret(name) {
  const path = join(SECRETS_DIR, name);
  if (!existsSync(path)) return null;
  return readFileSync(path, "utf8").trim();
}

function getBotToken() { return getSecret("telegram-bot-token.txt"); }
function getChatId()   { return getSecret("telegram-chat-id.txt"); }

/** Check if Telegram is configured and ready. */
export function isConfigured() {
  return !!(getBotToken() && getChatId());
}

/**
 * Send a text message (supports Telegram MarkdownV2).
 */
export async function sendMessage(text, options = {}) {
  const token = getBotToken();
  const chatId = getChatId();
  if (!token || !chatId) return false;

  try {
    const res = await fetch(`${API_BASE}/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text: text.slice(0, 4096), // Telegram limit
        parse_mode: "HTML",
        ...options,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Send a file (research report) as a Telegram document.
 */
export async function sendDocument(filePath, caption = "") {
  const token = getBotToken();
  const chatId = getChatId();
  if (!token || !chatId) return false;

  try {
    const fileContent = await readFile(filePath);
    const filename = filePath.split("/").pop();

    const form = new FormData();
    form.append("chat_id", chatId);
    form.append("caption", caption.slice(0, 1024));
    form.append("parse_mode", "HTML");
    const mime = filename.endsWith(".pdf") ? "application/pdf"
      : filename.endsWith(".tex") ? "application/x-tex"
        : "text/markdown";
    form.append("document", new Blob([fileContent], { type: mime }), filename);

    const res = await fetch(`${API_BASE}/bot${token}/sendDocument`, {
      method: "POST",
      body: form,
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ─── Milestone messages ───────────────────────────────────────────────────────

/** Milestone states that trigger a Telegram notification (not every save). */
const MILESTONE_STATES = new Set([
  "SEARCH_PLAN_CREATED",
  "EVIDENCE_LOCKED",
  "PAPERS_PARSED",
  "GAPS_IDENTIFIED",
  "IDEAS_SHORTLISTED",
  "MICRO_PROBE_REVIEWED",
  "PAPER_READINESS_REVIEW",
  "PAPER_DRAFTED_OR_STOPPED",
  "TRACE_EXPORTED",
]);

const STATE_EMOJI = {
  SEARCH_PLAN_CREATED:      "📋",
  EVIDENCE_LOCKED:          "🔒",
  PAPERS_PARSED:            "📄",
  GAPS_IDENTIFIED:          "🔬",
  IDEAS_SHORTLISTED:        "💡",
  AWAITING_HUMAN_APPROVAL:  "⏸️",
  MICRO_PROBE_REVIEWED:     "🧪",
  PAPER_READINESS_REVIEW:   "📝",
  PAPER_DRAFTED_OR_STOPPED: "📃",
  TRACE_EXPORTED:           "✅",
};

/**
 * Called on every pipeline saveState().
 * Only sends a message at key milestones to avoid spam.
 */
export async function notifyMilestone(state) {
  if (!isConfigured()) return;
  if (state.suppress_telegram || String(state.topic || "").toLowerCase().includes(" smoke ")) return;
  if (!MILESTONE_STATES.has(state.current_state)) return;

  const icon  = STATE_EMOJI[state.current_state] || "⚙️";
  const rrl   = state.paper_readiness_level || "RRL-0";
  const topic = state.topic || "Unknown topic";

  const lines = [
    `${icon} <b>OpenResearchOS Update</b>`,
    ``,
    `<b>Topic:</b> ${escHtml(topic)}`,
    `<b>State:</b> <code>${state.current_state}</code>`,
    `<b>RRL:</b> ${rrl}`,
    ``,
  ];

  // State-specific detail lines
  switch (state.current_state) {
    case "EVIDENCE_LOCKED":
      lines.push(`📚 <b>${state.evidence_ids?.length || 0} papers</b> discovered and locked`);
      break;
    case "PAPERS_PARSED":
      lines.push(`📄 <b>${state.parsed_papers?.length || 0} papers</b> fully parsed and claim-extracted`);
      break;
    case "GAPS_IDENTIFIED":
      lines.push(`🔬 <b>${state.research_gaps?.length || 0} research gaps</b> identified`);
      if (state.research_gaps?.length > 0) {
        lines.push(``, `<b>Key gaps:</b>`);
        state.research_gaps.slice(0, 3).forEach((g, i) => {
          const gapLabel = String(g.gap_title || g.gap || g.text || `Gap ${i+1}`).slice(0, 80);
          lines.push(`  ${i + 1}. ${escHtml(gapLabel)}`);
          const hypothesis = g.testable_hypothesis || g.gap_description;
          if (hypothesis) lines.push(`     <i>${escHtml(String(hypothesis).slice(0, 80))}</i>`);
        });
      }
      break;
    case "IDEAS_SHORTLISTED": {
      const total = state.idea_tree?.length || 0;
      const shortlisted = state.experiment_candidates?.length || 0;
      const avgScore = state.reviewer_scores?.length > 0
        ? (state.reviewer_scores.reduce((s, r) => s + (r.score || 0), 0) / state.reviewer_scores.length).toFixed(1)
        : "—";
      lines.push(`💡 <b>${total} ideas</b> generated → <b>${shortlisted} shortlisted</b>`);
      lines.push(`⚖️ Reviewer avg: <b>${avgScore}/10</b>`);
      if (state.experiment_candidates?.length > 0) {
        lines.push(``, `<b>Top ideas:</b>`);
        state.experiment_candidates.slice(0, 3).forEach((c, i) => {
          lines.push(`  ${i + 1}. ${escHtml(c.title || c.idea_id || String(c))}`);
        });
      }
      break;
    }
    case "AWAITING_HUMAN_APPROVAL":
      lines.push(`⏸️ Pipeline paused — <b>awaiting your approval</b> to run experiments`);
      lines.push(`Reply <code>/approve</code> or <code>/reject</code> in the OpenClaw app`);
      break;
    case "PAPER_DRAFTED_OR_STOPPED":
    case "TRACE_EXPORTED": {
      const output = state.pdf_output_path || state.final_output_path;
      if (output?.includes("research_failure_report")) {
        lines.push(`📋 <b>Paper blocked — failure report generated</b>`);
        lines.push(`The system stopped honestly because evidence/experiment gates did not justify a paper draft.`);
      } else if (output?.includes("paper_draft")) {
        lines.push(`🎓 <b>Paper Draft</b> generated — ${rrl} readiness`);
      } else if (output) {
        lines.push(`📎 <b>Final report</b> generated`);
      }
      lines.push(``, `<b>Output:</b> <code>${escHtml(output || "—")}</code>`);
      break;
    }
    default:
      break;
  }

  lines.push(``, `<i>Run: <code>${escHtml(state.run_id)}</code></i>`);
  await sendMessage(lines.join("\n"));

  // Send final report as a document
  if (
    state.current_state === "TRACE_EXPORTED"
    && (state.pdf_output_path || state.final_output_path)
    && existsSync(state.pdf_output_path || state.final_output_path)
  ) {
    const docPath = state.pdf_output_path || state.final_output_path;
    const isFailure = docPath.includes("failure");
    const caption = isFailure
      ? `📋 Paper blocked — failure report\n<b>${escHtml(topic)}</b>\n\nReadiness: ${rrl}`
      : `🎓 Paper draft\n<b>${escHtml(topic)}</b>\n\nReadiness: ${rrl}`;
    await sendDocument(docPath, caption);
  }
}

/**
 * Send a startup banner when a new run begins.
 */
export async function notifyRunStart(runId, topic) {
  if (!isConfigured()) return;
  await sendMessage([
    `🚀 <b>OpenResearchOS — New Run Started</b>`,
    ``,
    `<b>Topic:</b> ${escHtml(topic)}`,
    `<b>Run ID:</b> <code>${escHtml(runId)}</code>`,
    ``,
    `<i>Pipeline: discover → read → map → ideas → reviewer council → write</i>`,
    `You'll get updates at each major milestone. 🔔`,
  ].join("\n"));
}

/**
 * Send the full research map as a rich formatted message.
 */
export async function sendResearchMap(state) {
  if (!isConfigured()) return;
  if (!state.research_gaps?.length) return;

  const lines = [
    `🗺️ <b>Research Map — ${escHtml(state.topic)}</b>`,
    ``,
    `<b>${state.evidence_ids?.length || 0} papers</b> analyzed → <b>${state.research_gaps.length} gaps</b> found`,
    ``,
    `<b>Research Gaps:</b>`,
  ];
  state.research_gaps.forEach((g, i) => {
    const gapLabel = String(g.gap_title || g.gap || g.text || `Gap ${i+1}`).slice(0, 90);
    lines.push(`${i + 1}. <b>${escHtml(gapLabel)}</b>`);
    const hyp = g.testable_hypothesis || g.gap_description;
    if (hyp) lines.push(`   <i>H: ${escHtml(String(hyp).slice(0, 80))}</i>`);
  });
  await sendMessage(lines.join("\n"));
}

/** Escape HTML special chars for Telegram HTML parse mode. */
function escHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
