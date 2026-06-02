# EV004: Verifier-Guided Code Translation via Meta-Step Decoding

URL: https://www.semanticscholar.org/paper/0033708ee454d5ea4c7b6ed0f9424e63baf9b395
Year: 2026
Source: semantic_scholar
Arxiv: 2605.17626

## Abstract

Test-time scaling is an important mechanism for improving large language models, especially on tasks with deterministic verifiers. Code translation is a canonical example: the source program constrains valid outputs, while compilers, type check- ers, and behavioral checks provide exact pass/fail feedback. Existing approaches typically apply these verifiers only after generation, which is inefficient because early errors corrupt the autoregressive context and are rarely corrected later. We introduce Decoding Time Verification (DTV), a framework that treats structural boundaries as meta steps for verifier-guided decoding. DTV interleaves generation with verifier calls under a state-machine controller that enforces valid prefixes, using structural-boundary checks and structure-aware rollback to prevent error propagation while reducing wasted tokens. We evaluate DTV on C-to-Rust and JavaScript-to-TypeScript translation. Using Qwen3-4B as the primary generator under matched token budgets, DTV improves pass rates from 72.3% to 82.0% on C-to-Rust and from 33.3% to 46.0% on JavaScript-to-TypeScript relative to matched self-refinement baselines, while using fewer tokens per case; the same trend largely transfers to Gemma-4-E4B. In the evaluated cost-matched grid, DTV achieves a more favorable pass-rate-cost tradeoff than post-hoc verification or sampling-based scaling. These results show that verifier-guided decoding is an effective use of inference-time compute for code translation.
