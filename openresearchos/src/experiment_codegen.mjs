/**
 * Experiment Codegen for OpenResearchOS.
 *
 * Generates custom Python experiment scripts tailored to the research topic
 * and idea, with deterministic templates as safe fallbacks.
 */

import * as llm from "./llm_client.mjs";

/**
 * Generates the Python experiment code.
 * Falls back to standard probe if LLM fails or is offline.
 */
export async function generateExperimentCode(idea, topic, level, fallbackSource) {
  try {
    console.error(`  [LLM] Designing custom experiment code for topic: "${topic}"...`);
    const rawCode = await llm.designExperiment(idea, topic, level);
    if (rawCode) {
      // Clean up markdown fences
      const cleaned = rawCode
        .replace(/^```(?:python)?\s*/i, "")
        .replace(/\s*```$/i, "")
        .trim();

      if (cleaned.length > 50 && cleaned.includes("import ") && cleaned.includes("metrics.json")) {
        console.error(`  [LLM] Custom experiment code generated successfully (${cleaned.length} chars).`);
        return cleaned;
      }
    }
  } catch (err) {
    console.error(`  [LLM] Codegen failed, falling back:`, err);
  }

  console.error("  [FALLBACK] Using standard fallback experiment script.");
  return fallbackSource;
}
