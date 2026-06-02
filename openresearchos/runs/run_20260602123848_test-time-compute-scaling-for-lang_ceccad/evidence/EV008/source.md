# EV008: Stable Reasoning, Unstable Responses: Mitigating LLM Deception via Stability Asymmetry

URL: https://www.semanticscholar.org/paper/003a43ef5c155cfc95d1da28787a2bb927c4914e
Year: 2026
Source: semantic_scholar
Arxiv: 2603.26846

## Abstract

As Large Language Models (LLMs) expand in capability and application scope, their trustworthiness becomes critical. A vital risk is intrinsic deception, wherein models strategically mislead users to achieve their own objectives. Existing alignment approaches based on chain-of-thought (CoT) monitoring supervise explicit reasoning traces. However, under optimization pressure, models are incentivized to conceal deceptive reasoning, rendering semantic supervision fundamentally unreliable. Grounded in cognitive psychology, we hypothesize that a deceptive LLM maintains a stable internal belief in its CoT while its external response remains fragile under perturbation. We term this phenomenon stability asymmetry and quantify it by measuring the contrast between internal CoT stability and external response stability under perturbation. Building on this structural signature, we propose the Stability Asymmetry Regularization (SAR), a novel alignment objective that penalizes this distributional asymmetry during reinforcement learning. Unlike CoT monitoring, SAR targets the statistical structure of model outputs, rendering it robust to semantic concealment. Extensive experiments confirm that stability asymmetry reliably identifies deceptive behavior, and that SAR effectively suppresses intrinsic deception without degrading general model capability.
