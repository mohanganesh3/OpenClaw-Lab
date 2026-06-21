---
name: openresearchos-literature-scout
description: Discover, fetch, and lock research evidence using OpenClaw web search and web fetch for OpenResearchOS runs.
---

# OpenResearchOS Literature Scout

Use this skill to find papers, read their content, build the evidence index, and snowball citations for any OpenResearchOS run.

## Working Directory
```
/Users/mohanganesh/OpenClaw-Lab/openresearchos
```

## Step-by-Step Discovery Protocol

### 1. Multi-Query Web Search (OpenClaw Native)

Run at least 4 search queries using `openclaw infer web search`:

```bash
# Core topic papers
openclaw infer web search \
  --query "<topic> research paper arxiv 2023 2024" \
  --limit 10 --json

# Survey and benchmark papers  
openclaw infer web search \
  --query "<topic> survey state of the art benchmark" \
  --limit 10 --json

# Limitation and gap papers
openclaw infer web search \
  --query "<topic> limitations failure modes open problems" \
  --limit 8 --json

# Competitor and workshop papers
openclaw infer web search \
  --query "<topic> workshop NeurIPS ICML ICLR CVPR 2024" \
  --limit 8 --json
```

Parse the JSON output for `title`, `url`, and `snippet` fields.

### 2. Trigger Structured Discovery

Run the full pipeline discovery (OpenClaw web search + OpenAlex + Semantic Scholar):
```bash
node src/openresearch.mjs discover --run <run_id>
```

This uses OpenClaw infer web fetch internally to snapshot evidence.

### 3. Deep Evidence Fetching

For the top 8 papers by relevance, fetch full abstract pages:
```bash
openclaw infer web fetch \
  --url "https://arxiv.org/abs/<arxiv_id>" \
  --format text --json
```

For papers with PDF URLs, fetch the PDF text:
```bash
openclaw infer web fetch \
  --url "https://arxiv.org/pdf/<arxiv_id>.pdf" \
  --format text --json
```

### 4. Snowball Citations

For the most-cited paper found, search for papers citing it:
```bash
openclaw infer web search \
  --query "\"<exact paper title>\" cited by related work" \
  --limit 8 --json
```

### 5. Competitor Audit (Novelty Protection)

Before generating ideas, search specifically for existing implementations:
```bash
openclaw infer web search \
  --query "<proposed mechanism> github implementation code" \
  --limit 5 --json

openclaw infer web search \
  --query "<proposed idea keywords> arxiv preprint 2024 2025" \
  --limit 5 --json
```

If matches are found, log them as novelty risks in the evidence index.

## Target Deliverables

- Minimum **20 high-quality papers** in `evidence_index.json`
- Each evidence item has: `evidence_id`, `title`, `url`, `source`, `abstract`, `snapshot_path`
- At least 5 papers have full text snapshots via `openclaw infer web fetch`
- Competitor audit complete — novelty risks documented

## Tool Selection Matrix

| Objective | OpenClaw Tool |
|---|---|
| Broad topic search | `openclaw infer web search --query "..." --json` |
| Full paper text | `openclaw infer web fetch --url "..." --format text` |
| PDF reading | `openclaw infer web fetch --url "...pdf" --format text` |
| Citation snowball | `openclaw infer web search --query "\"<title>\" cited"` |
| Competitor check | `openclaw infer web search --query "<mechanism> github"` |
