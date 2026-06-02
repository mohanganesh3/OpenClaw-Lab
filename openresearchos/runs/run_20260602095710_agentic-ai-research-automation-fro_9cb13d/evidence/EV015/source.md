# EV015: Right-to-Act: A Pre-Execution Non-Compensatory Decision Protocol for AI Systems

URL: https://arxiv.org/abs/2604.24153
Year: 2026
Source: arxiv_api
Arxiv: 2604.24153

## Abstract

Current AI systems increasingly operate in contexts where their outputs directly trigger real-world actions. Most existing approaches to AI safety, risk management, and governance focus on post-hoc validation, probabilistic risk estimation, or certification of model behavior. However, these approaches implicitly assume that once a decision is produced, it is eligible for execution. In this work, we introduce the Right-to-Act protocol, a deterministic, non-compensatory pre-execution decision layer that evaluates whether an AI-generated decision is permitted to be realized at all. Unlike compensatory systems, where high-confidence signals can override failed conditions, the proposed framework enforces strict structural constraints: if any required condition is unmet, execution is halted or deferred. We formalize the distinction between compensatory and non-compensatory decision regimes and define a pre-execution legitimacy boundary. Through a scenario-based case study, we demonstrate how identical AI outputs can lead to divergent outcomes when evaluated under a Right-to-Act protocol, preserving reversibility and preventing premature or irreversible actions. The proposed approach reframes AI control from optimizing decisions to governing their admissibility, introducing a protocol-level abstraction that operates independently of model architecture or training methodology.
