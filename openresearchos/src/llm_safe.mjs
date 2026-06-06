/**
 * llm_safe.mjs — Hardening layer for a weak, rate-limited single model (Sarvam).
 *
 * Why this exists:
 *   The v1 idea stage made 160+ sequential Sarvam calls and only saved at the
 *   very end. When the daemon timed out mid-stage, NOTHING was saved → every
 *   autonomous run showed 0 ideas. This module enforces the v2 principles:
 *
 *   - Tiny calls, hard budgets    → a stage cannot run forever.
 *   - Never throw                 → a failed call returns a typed fallback.
 *   - Full audit                  → every call is logged to llm_calls.jsonl.
 *
 * It does NOT replace llm_client.mjs; it wraps individual calls so the
 * orchestrator can stay alive and resumable.
 */

import { appendFile, mkdir } from "node:fs/promises";
import { join, dirname } from "node:path";

/**
 * Tracks how much a single stage is allowed to spend.
 * When the budget is exhausted, callers degrade gracefully instead of looping.
 */
export class RunBudget {
  constructor({ maxCalls = 60, maxMs = 8 * 60 * 1000, label = "stage" } = {}) {
    this.maxCalls = maxCalls;
    this.maxMs = maxMs;
    this.label = label;
    this.calls = 0;
    this.start = Date.now();
  }

  get elapsedMs() {
    return Date.now() - this.start;
  }

  /** True if there is still budget for another LLM call. */
  canSpend() {
    return this.calls < this.maxCalls && this.elapsedMs < this.maxMs;
  }

  spend() {
    this.calls += 1;
  }

  reason() {
    if (this.calls >= this.maxCalls) return `call budget reached (${this.maxCalls})`;
    if (this.elapsedMs >= this.maxMs) return `time budget reached (${Math.round(this.maxMs / 1000)}s)`;
    return "ok";
  }

  summary() {
    return { label: this.label, calls: this.calls, elapsed_ms: this.elapsedMs };
  }
}

/**
 * Append one LLM call record to runs/<run>/llm_calls.jsonl for the audit trail.
 * Never throws.
 */
export async function logCall(runDirPath, record) {
  try {
    const path = join(runDirPath, "llm_calls.jsonl");
    await mkdir(dirname(path), { recursive: true });
    await appendFile(path, JSON.stringify({ ts: new Date().toISOString(), ...record }) + "\n", "utf8");
  } catch {
    /* logging must never break the run */
  }
}

/**
 * Run an async LLM operation with a budget, a timeout, audit logging, and a
 * guaranteed fallback. NEVER throws.
 *
 * @param {object}   o
 * @param {string}   o.label      - what this call is for (logged)
 * @param {RunBudget} o.budget    - shared stage budget (optional)
 * @param {string}   o.runDir     - run directory for the audit log (optional)
 * @param {Function} o.fn         - async () => result
 * @param {*}        o.fallback   - returned if budget is gone or fn fails/empties
 * @param {Function} [o.valid]    - result => boolean; invalid results use fallback
 * @param {number}   [o.timeoutMs]
 */
export async function safeCall({ label, budget, runDir, fn, fallback, valid, timeoutMs = 120000 }) {
  if (budget && !budget.canSpend()) {
    await logCall(runDir, { label, skipped: true, reason: budget.reason() });
    return { ok: false, value: fallback, skipped: true };
  }
  if (budget) budget.spend();

  const started = Date.now();
  try {
    const result = await withTimeout(fn(), timeoutMs);
    const isValid = result != null && (typeof valid !== "function" || valid(result));
    await logCall(runDir, {
      label,
      ok: isValid,
      ms: Date.now() - started,
      calls: budget?.calls,
    });
    if (isValid) return { ok: true, value: result };
    return { ok: false, value: fallback };
  } catch (err) {
    await logCall(runDir, { label, ok: false, error: String(err?.message || err), ms: Date.now() - started });
    return { ok: false, value: fallback, error: err };
  }
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms)),
  ]);
}
