#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenResearchOS Orchestrator
#
# Runs the full research pipeline through OpenClaw agent turns.
# This is the CORRECT way to run OpenResearchOS — not `node src/openresearch.mjs demo`.
#
# Usage:
#   ./scripts/orchestrate.sh "calibration-aware active learning for medical imaging"
#   ./scripts/orchestrate.sh "traceable autonomous research agents" --offline
#
# The OpenClaw agent (guided by SKILL.md files) will:
#   1. Search for papers using openclaw infer web search
#   2. Fetch evidence snapshots using openclaw infer web fetch
#   3. Check OpenClaw memory for past failures
#   4. Generate ideas, run reviewer council, novelty tribunal
#   5. Run experiment ladder with human approval gates
#   6. Write final report
#   7. Export full trajectory via openclaw sessions export-trajectory
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

TOPIC="${1:-}"
OFFLINE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$TOPIC" ]]; then
  echo "Usage: $0 \"<research topic>\" [--offline]"
  echo "Example: $0 \"calibration-aware active learning for medical imaging\""
  exit 1
fi

# Build the session key for this run (deterministic from topic)
TOPIC_SLUG=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g' | cut -c1-40)
SESSION_KEY="openresearchos-${TOPIC_SLUG}-$(date +%s)"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         OpenResearchOS via OpenClaw                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Topic: $TOPIC"
echo "║  Session: $SESSION_KEY"
echo "║  Mode: $([ -n "$OFFLINE" ] && echo "OFFLINE (no web search)" || echo "ONLINE (full OpenClaw web search)")"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check gateway is running ──────────────────────────────────────────
echo "[1/8] Checking OpenClaw gateway..."
if ! openclaw health &>/dev/null; then
  echo "  ERROR: OpenClaw gateway is not running."
  echo "  Start it with: openclaw gateway run"
  exit 1
fi
echo "  ✓ Gateway reachable"

# ── Step 2: Check web search capability ───────────────────────────────────────
if [[ -z "$OFFLINE" ]]; then
  echo "[2/8] Checking OpenClaw web search capability..."
  WEB_CHECK=$(openclaw infer web search --query "test" --limit 1 --json 2>/dev/null || echo "error")
  if [[ "$WEB_CHECK" == "error" ]] || [[ -z "$WEB_CHECK" ]]; then
    echo "  WARNING: web search not available — falling back to OpenAlex + Semantic Scholar only"
  else
    echo "  ✓ Web search available"
  fi
fi

# ── Step 3: Memory check for prior failures ────────────────────────────────────
echo "[3/8] Checking OpenClaw memory for prior failures on this topic..."
MEMORY_RESULTS=$(openclaw memory search --query "$TOPIC failed" --max-results 5 --json 2>/dev/null || echo "[]")
MEMORY_COUNT=$(echo "$MEMORY_RESULTS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else len(d.get('results',[])))" 2>/dev/null || echo "0")
if [[ "$MEMORY_COUNT" -gt 0 ]]; then
  echo "  ⚠ Found $MEMORY_COUNT prior failure(s) related to this topic."
  echo "  Check ~/.openclaw/workspace/memory/failed_ideas/ before proceeding."
else
  echo "  ✓ No prior failures found — topic is fresh"
fi

# ── Step 4: Run full pipeline through OpenClaw agent ─────────────────────────
echo "[4/8] Running full OpenResearchOS pipeline through OpenClaw agent..."
echo "  Session key: $SESSION_KEY"
echo "  Working dir: $PROJECT_DIR"
echo ""

OFFLINE_FLAG=""
if [[ -n "$OFFLINE" ]]; then
  OFFLINE_FLAG="Use --offline mode (no web search, use fallback sources only)."
fi

MESSAGE="You are running OpenResearchOS. Working directory: $PROJECT_DIR.

Research topic: $TOPIC

$OFFLINE_FLAG

Follow the openresearchos-coordinator skill protocol exactly:
1. Start the run with: node src/openresearch.mjs start --topic \"$TOPIC\"
2. Check OpenClaw memory for prior failures
3. Run discovery: node src/openresearch.mjs discover --run <run_id>
4. Read and map: node src/openresearch.mjs read --run <run_id> && node src/openresearch.mjs map --run <run_id>
5. Generate ideas: node src/openresearch.mjs ideas --run <run_id>
6. Present the top shortlisted idea and its experiment plan. WAIT for user approval.
7. After approval: node src/openresearch.mjs run-experiment --run <run_id> --idea <idea_id> --level micro_probe
8. Write final output: node src/openresearch.mjs write --run <run_id>
9. Export trajectory: openclaw sessions export-trajectory --session-key $SESSION_KEY
10. Status: node src/openresearch.mjs status --run <run_id>

Use openclaw infer web search for paper discovery.
Use openclaw infer web fetch for evidence snapshots.
Use openclaw memory search to check for past failures.
Session key for this run: $SESSION_KEY"

openclaw agent \
  --message "$MESSAGE" \
  --session-key "$SESSION_KEY" \
  --thinking high \
  --json

AGENT_STATUS=$?

# ── Step 5: Get run status ─────────────────────────────────────────────────────
echo ""
echo "[5/8] Retrieving run status..."
LATEST_RUN=$(ls -t "$PROJECT_DIR/runs/" 2>/dev/null | head -1 || echo "")
if [[ -n "$LATEST_RUN" ]]; then
  echo "  Latest run: $LATEST_RUN"
  node "$PROJECT_DIR/src/openresearch.mjs" status --run "$LATEST_RUN" 2>/dev/null || true
fi

# ── Step 6: Export trajectory ──────────────────────────────────────────────────
echo ""
echo "[6/8] Exporting OpenClaw trajectory..."
if [[ -n "$LATEST_RUN" ]]; then
  EXPORT_DIR="$PROJECT_DIR/runs/$LATEST_RUN/trajectory_export"
  mkdir -p "$EXPORT_DIR"
  openclaw sessions export-trajectory \
    --session-key "$SESSION_KEY" \
    --output "$EXPORT_DIR" \
    --json 2>/dev/null || echo "  (Trajectory export: session may not have agent turns yet)"
  echo "  Trajectory export dir: $EXPORT_DIR"
fi

# ── Step 7: Memory index refresh ──────────────────────────────────────────────
echo ""
echo "[7/8] Refreshing OpenClaw memory index..."
openclaw memory index --force 2>/dev/null || true
echo "  ✓ Memory indexed"

# ── Step 8: Summary ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         OpenResearchOS Run Complete                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Session:   $SESSION_KEY"
echo "║  Run dir:   $PROJECT_DIR/runs/$LATEST_RUN"
echo "║  Artifacts: $(ls "$PROJECT_DIR/runs/$LATEST_RUN/" 2>/dev/null | wc -l | tr -d ' ') files"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  To resume this session:"
echo "║    openclaw agent --session-key \"$SESSION_KEY\" \\"
echo "║      --message \"continue the OpenResearchOS run\""
echo "║  To view session:"
echo "║    openclaw sessions --json | grep $SESSION_KEY"
echo "╚══════════════════════════════════════════════════════════════╝"

exit $AGENT_STATUS
