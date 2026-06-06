/**
 * openclaw_subagents.mjs — OpenClaw-native specialist sub-agents (Phase D).
 *
 * Spawns REAL OpenClaw agent turns (`openclaw agent --session-key ... --json`),
 * each in its own session under the run's namespace, so the work shows up in the
 * OpenClaw trajectory as a genuine multi-agent research team. Every spawn first
 * acquires a slot from the GLOBAL Sarvam rate bucket (the 40/min cap is shared).
 *
 * These are bounded, single-purpose specialists (not a free-running loop):
 *   - literature-scout : find candidate papers/repos for a topic
 *   - skeptic          : adversarially critique the run's ideas
 *   - synthesizer      : write a short cross-paper synthesis
 * The deterministic engine remains the workhorse; this is the OpenClaw showcase.
 */

import { spawnSync } from "node:child_process";
import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { acquireSlot } from "./rate_limiter.mjs";

function parseAgentText(stdout) {
  try {
    const j = JSON.parse(stdout);
    const payloads = j?.result?.payloads || [];
    const text = payloads.map((p) => p?.text).filter(Boolean).join("\n").trim();
    return { text: text || null, usage: j?.result?.meta?.agentMeta?.usage || null, sessionId: j?.result?.meta?.agentMeta?.sessionId || null };
  } catch {
    return { text: (stdout || "").trim() || null, usage: null, sessionId: null };
  }
}

/**
 * Spawn one OpenClaw sub-agent turn. Never throws.
 * @returns {{ ok, text, sessionKey, usage }}
 */
export async function spawnSpecialist({ runId, label, task, timeoutMs = 180000 }) {
  const sessionKey = `openresearchos-${runId}-${label}`;
  await acquireSlot(); // respect the shared 40/min account cap
  try {
    const r = spawnSync("openclaw", ["agent", "--session-key", sessionKey, "--message", task, "--json"], {
      encoding: "utf8", timeout: timeoutMs, maxBuffer: 32 * 1024 * 1024, env: { ...process.env },
    });
    if (r.status !== 0) {
      return { ok: false, text: null, sessionKey, error: (r.stderr || "agent failed").slice(0, 200) };
    }
    const { text, usage, sessionId } = parseAgentText(r.stdout || "");
    return { ok: !!text, text, sessionKey, sessionId, usage };
  } catch (err) {
    return { ok: false, text: null, sessionKey, error: err.message };
  }
}

/**
 * Run a small specialist team for a run and save each output to subagents/.
 * Bounded (3 specialists). Returns a summary; writes artifacts for the trajectory.
 */
export async function coordinateTeam({ runId, runDirPath, topic, ideas = [], papers = [] }) {
  const dir = join(runDirPath, "subagents");
  await mkdir(dir, { recursive: true }).catch(() => {});

  const ideaList = ideas.slice(0, 5).map((i, n) => `${n + 1}. ${i.title}`).join("\n") || "(none yet)";
  const paperList = papers.slice(0, 8).map((p, n) => `${n + 1}. ${p.title}`).join("\n") || "(none)";

  const specialists = [
    {
      label: "literature-scout",
      task: `You are a literature scout. For the research topic "${topic}", list 6 concrete, real, recent papers or GitHub repos worth reading (title + why). Be specific and factual; if unsure, say so. Keep under 250 words.`,
    },
    {
      label: "skeptic",
      task: `You are a harsh research skeptic. Given the topic "${topic}" and these candidate ideas:\n${ideaList}\nName the single biggest risk or likely-prior-work for each, and which one idea is most worth testing first. Under 250 words.`,
    },
    {
      label: "synthesizer",
      task: `You are a synthesis writer. Given the topic "${topic}" and these papers already read:\n${paperList}\nWrite a 150-word synthesis of the current state and the clearest open gap. Be concrete.`,
    },
  ];

  const results = [];
  for (const s of specialists) {
    console.error(`  [Subagent] spawning ${s.label}...`);
    const res = await spawnSpecialist({ runId, label: s.label, task: s.task });
    const md = `# Sub-agent: ${s.label}\n\nSession: \`${res.sessionKey}\`\nOK: ${res.ok}\n\n## Task\n${s.task}\n\n## Output\n${res.text || "(no output: " + (res.error || "empty") + ")"}\n`;
    await writeFile(join(dir, `${s.label}.md`), md, "utf8").catch(() => {});
    console.error(`  [Subagent] ${s.label}: ${res.ok ? "ok (" + (res.text?.length || 0) + " chars)" : "failed (" + (res.error || "empty") + ")"}`);
    results.push({ label: s.label, ok: res.ok, sessionKey: res.sessionKey, chars: res.text?.length || 0 });
  }
  return results;
}
