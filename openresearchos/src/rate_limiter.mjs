/**
 * rate_limiter.mjs — Global cross-process token bucket for Sarvam.
 *
 * Sarvam-105b (Starter) = 40 requests/min, enforced PER ACCOUNT (token bucket,
 * shared across all API keys AND all sub-agents/processes). A process-local
 * timer is therefore unsafe the moment anything runs in parallel.
 *
 * This limiter coordinates EVERY Sarvam call across all OpenResearchOS processes
 * via a shared state file + lock file under ~/.openclaw. It targets a safe
 * sub-cap (default 36/min) and counts retries too, so we never exceed 40/min.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync, openSync, closeSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const DIR = join(homedir(), ".openclaw");
const STATE_PATH = join(DIR, "ors_rate_limit.json");
const LOCK_PATH = join(DIR, "ors_rate_limit.lock");

const WINDOW_MS = 60_000;
const MAX_PER_WINDOW = Number(process.env.ORS_RATE_MAX || 36); // safe sub-cap under 40/min
const LOCK_STALE_MS = 15_000;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function ensureDir() {
  try { if (!existsSync(DIR)) mkdirSync(DIR, { recursive: true }); } catch {}
}

/** Acquire the cross-process lock (spin with small backoff). Best-effort, self-healing. */
async function acquireLock() {
  ensureDir();
  for (let i = 0; i < 200; i++) {
    try {
      const fd = openSync(LOCK_PATH, "wx"); // fails if exists
      closeSync(fd);
      writeFileSync(LOCK_PATH, String(Date.now()), "utf8");
      return true;
    } catch {
      // Lock held — check staleness
      try {
        const ts = Number(readFileSync(LOCK_PATH, "utf8")) || 0;
        if (Date.now() - ts > LOCK_STALE_MS) { try { unlinkSync(LOCK_PATH); } catch {} continue; }
      } catch {}
      await sleep(25 + Math.floor(Math.random() * 50));
    }
  }
  return false; // give up on lock; proceed unsynchronized rather than hang
}

function releaseLock() { try { unlinkSync(LOCK_PATH); } catch {} }

function readTimestamps() {
  try {
    const arr = JSON.parse(readFileSync(STATE_PATH, "utf8"));
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

function writeTimestamps(arr) {
  try { writeFileSync(STATE_PATH, JSON.stringify(arr), "utf8"); } catch {}
}

/**
 * Block until a request slot is available, then reserve it.
 * Returns the number of ms waited (for logging).
 */
export async function acquireSlot() {
  const start = Date.now();
  for (let attempt = 0; attempt < 600; attempt++) {
    const locked = await acquireLock();
    try {
      const now = Date.now();
      let ts = readTimestamps().filter((t) => now - t < WINDOW_MS);
      if (ts.length < MAX_PER_WINDOW) {
        ts.push(now);
        writeTimestamps(ts);
        return now - start;
      }
      // Need to wait until the oldest timestamp leaves the window.
      const waitMs = Math.max(50, WINDOW_MS - (now - Math.min(...ts)) + 20);
      if (locked) releaseLock();
      await sleep(Math.min(waitMs, 2000));
      continue;
    } finally {
      if (locked) releaseLock();
    }
  }
  return Date.now() - start; // fail-open after ~ minutes
}

export function rateStatus() {
  const now = Date.now();
  const ts = readTimestamps().filter((t) => now - t < WINDOW_MS);
  return { used: ts.length, max: MAX_PER_WINDOW, windowMs: WINDOW_MS };
}
