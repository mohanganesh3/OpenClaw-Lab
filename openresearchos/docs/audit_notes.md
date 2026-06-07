# OpenResearchOS Audit Notes — Harsh

## CRITICAL PROBLEMS FOUND

### Problem 1: experiment_spec.json is MISSING `idea_title`, `selected_idea`, `topic`
The spec written by `createExperimentWorkspace()` in openresearch.mjs does NOT include:
- `idea_title` (missing field entirely)  
- `selected_idea` (missing object)
- `topic` (missing)
So `research_probe.py` reads `spec.get("selected_idea", {}).get("title", "Unknown idea")`
→ always gets "Unknown idea" → metrics.json shows `idea_title: "Unknown idea"`.
Also metrics.json always shows `proposed_name: "Proposed (topic-tailored)"`.

**Fix:** Add `idea_title`, `selected_idea`, `topic` to the spec object in `createExperimentWorkspace`.

### Problem 2: Fallback experiment is NOT idea-aware
Even though the topic is "calibration-aware active learning", the fallback template
uses `is_topic()` keywords but "active learning" and "calibration" don't match 
the defined keyword sets. Falls through to `digits + LogisticRegression vs PCA+GradientBoosting`
which has NOTHING to do with calibration-aware active learning.
`proposed_name` is hardcoded as "Proposed (topic-tailored)" regardless.

**Fix:** 
1. Add calibration+active learning specific branch with ECE metric, temperature scaling, acquisition functions
2. Make proposed_name reflect the actual method used

### Problem 3: LLM-generated code is not being used — need to check experiment_codegen.mjs

### Problem 4: ideas don't reference EV IDs — generateResearchIdeas feeds minimal context
The `analyzeGaps` function only feeds 80 chars/paper and `generateResearchIdeas` feeds
only gap titles without citing EV IDs. The idea_tree.md has no EV references.

**Fix:** Feed actual gaps_this_enables + evidence_ids from Paper Cards into both functions.

## STATUS
- [ ] Fix experiment_spec.json (idea_title, selected_idea, topic)
- [ ] Fix pythonProbeSource (idea-aware, correct proposed_name)  
- [ ] Add calibration-AL branch with ECE + acquisition functions
- [ ] Feed Paper Card gaps into gap analysis + idea generation
- [ ] Run fresh demo
- [ ] Verify passes ok:true
