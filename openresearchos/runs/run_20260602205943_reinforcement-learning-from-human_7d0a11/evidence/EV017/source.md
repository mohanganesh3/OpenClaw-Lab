# EV017: Aligning as Debiasing: Causality-Aware Alignment via Reinforcement Learning with Interventional Feedback

URL: https://www.semanticscholar.org/paper/0425c47e19b5f1fcc680967ebd6c6e7cebc0b768
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

Large language models (LLMs) often generate biased outputs containing offensive, toxic, or stereotypical text. Existing LLM alignment methods such as reinforcement learning from human feedback (RLHF) alleviate biases primarily based on reward signals from current model outputs without considering the source of biases. In this work, to explore how biases are formed, we revisit LLMs’ text generation from a causal perspective. We identify pretraining data and input prompts, which contain semantic correlations of textual phrases, as two confounders between LLMs and model outputs causing biases. Inspired by our causal view, we leverage the reward model in RL alignment as an instrumental variable to perform causal intervention on LLMs. Utilizing the reward difference between an initial LLM and intervened LLM as interventional feedback to guide RL finetuning, we propose Causality-Aware Alignment (CAA) for LLM debiasing. Experiments on two text generation tasks with three different alignment objectives demonstrate the advantages of our method in aligning LLMs to generate less biased and safer outputs.
