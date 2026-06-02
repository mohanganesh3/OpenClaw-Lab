# EV018: Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving

URL: https://www.semanticscholar.org/paper/043aef64488f36fad56b71fb852c5cb027e87245
Year: 2025
Source: semantic_scholar
Arxiv: 2503.09730

## Abstract

The most promising recent methods for AI reasoning require applying variants of reinforcement learning (RL) either on rolled out trajectories from the LLMs, even for the step-wise rewards, or large quantities of human-annotated trajectory data. The reliance on the rolled-out trajectory renders the compute cost and time prohibitively high. In particular, the correctness of a reasoning trajectory can typically only be judged at its completion, leading to sparse rewards in RL or requiring expensive synthetic data generation in expert iteration-like methods. In this work, we focus on the Automatic Theorem Proving (ATP) task and propose a novel verifier-in-the-loop design, which, unlike existing approaches that leverage feedback on the entire reasoning trajectory, employs an automated verifier to give intermediate feedback at each step of the reasoning process. Using Lean as the verifier, we empirically show that the step-by-step local verification produces a global improvement in the model's reasoning accuracy and efficiency.
