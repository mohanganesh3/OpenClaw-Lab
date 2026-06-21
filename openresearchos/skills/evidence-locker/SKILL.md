---
name: openresearchos-evidence-locker
description: Lock research evidence into stable OpenResearchOS artifacts using OpenClaw web fetch and memory, with source snapshots, evidence IDs, and claim links.
---

# OpenResearchOS Evidence Locker

Use this skill after papers are discovered to create stable, immutable evidence snapshots using OpenClaw's web fetch capability.

## Protocol

### 1. Fetch Full Evidence Snapshots (OpenClaw Native)

For each paper in the evidence index, use `openclaw infer web fetch`:

```bash
# Fetch abstract page (HTML → clean text)
openclaw infer web fetch \
  --url "https://arxiv.org/abs/<arxiv_id>" \
  --format text --json

# Fetch PDF when available (PDF → extracted text)
openclaw infer web fetch \
  --url "https://arxiv.org/pdf/<arxiv_id>.pdf" \
  --format text --json

# Fetch from other sources (Semantic Scholar, conference pages)
openclaw infer web fetch \
  --url "<paper_url>" \
  --format text --json
```

Save output to `runs/<run_id>/evidence/<EVXXX>/source.md`.

### 2. Structured Claim Extraction

After snapshots are locked, run the claim extractor:
```bash
node src/openresearch.mjs read --run <run_id>
```

This uses the Sarvam LLM to extract: core claim, methods, limitations, datasets, metrics, and research gaps.

### 3. Build Research Map

```bash
node src/openresearch.mjs map --run <run_id>
```

Generates `research_map.md` with actionable gaps linked to specific evidence IDs.

### 4. Lock Evidence in Memory

After evidence is indexed, write a memory note:
```bash
openclaw memory index --force
```

Then search to confirm it's indexed:
```bash
openclaw memory search --query "<topic> evidence locked" --max-results 5 --json
```

## File Structure Contract

```text
runs/<run_id>/
├── evidence_index.json            # Master list (all 20 papers with evidence_ids)
├── research_map.md                # Gaps → Evidence ID links
├── paper_summaries/
│   ├── EV001.md                   # core_claim, methods, limitations, possible_gap
│   └── ...
└── evidence/
    ├── EV001/
    │   └── source.md              # URL + abstract + OpenClaw web fetch snapshot
    └── ...
```

## Evidence Quality Checklist

- [ ] Every evidence item has a stable `EV001`-style ID
- [ ] At least 10 items have abstract text (not just titles)
- [ ] At least 5 items have full-text snapshots via `openclaw infer web fetch`
- [ ] `evidence_index.json` is committed as the immutable evidence anchor
- [ ] `research_map.md` has specific gaps with `EV` ID citations

## Tool Selection Matrix

| Task | OpenClaw Tool |
|---|---|
| Fetch paper page | `openclaw infer web fetch --url "<url>" --format text` |
| Fetch PDF text | `openclaw infer web fetch --url "<pdf_url>" --format text` |
| Index memory | `openclaw memory index --force` |
| Search evidence memory | `openclaw memory search --query "..."` |
| Run extraction | `exec: node src/openresearch.mjs read --run <run_id>` |
