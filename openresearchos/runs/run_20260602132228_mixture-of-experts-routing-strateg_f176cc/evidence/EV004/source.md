# EV004: Mixture-of-Transformers Learn Faster: A Theoretical Study on Classification Problems

URL: https://www.semanticscholar.org/paper/16a5698fa910ebe5d55438977c0a1dcaa8bcf8d4
Year: 2025
Source: semantic_scholar
Arxiv: 2510.27004

## Abstract

Mixture-of-Experts (MoE) models improve transformer efficiency but lack a unified theoretical explanation, especially when both feed-forward and attention layers are allowed to specialize. To this end, we study the Mixture-of-Transformers (MoT), a tractable theoretical framework in which each transformer block acts as an expert governed by a continuously trained gating network. This design allows us to isolate and study the core learning dynamics of expert specialization and attention alignment. In particular, we develop a three-stage training algorithm with continuous training of the gating network, and show that each transformer expert specializes in a distinct class of tasks and that the gating network accurately routes data samples to the correct expert. Our analysis shows how expert specialization reduces gradient conflicts and makes each subtask strongly convex. We prove that the training drives the expected prediction loss to near zero in $O(\log(\epsilon^{-1}))$ iteration steps, significantly improving over the $O(\epsilon^{-1})$ rate for a single transformer. We further validate our theoretical findings through extensive real-data experiments, demonstrating the practical effectiveness of MoT. Together, these results offer the first unified theoretical account of transformer-level specialization and learning dynamics, providing practical guidance for designing efficient large-scale models.
