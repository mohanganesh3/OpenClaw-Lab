# EV003: Boundary Conditions of Cost-Aware Active Learning: A Multi-Dataset Taxonomy of Calibration and Length-Variance Failure Modes

URL: https://www.semanticscholar.org/paper/7e3b6940940bd618c336ea29b5e36c579cc13ae5
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Active Learning (AL) strategies that incorporate annotation cost, such as the recently proposed CAL-Log heuristic, have demonstrated significant efficiency gains in verbose, document-heavy Data Mining and Natural Language Processing (NLP) tasks. However, the stability of cost-aware selection remains poorly understood across the diverse spectrum of linguistic regimes. This paper presents a systematic diagnostic study of CAL-Log's performance boundaries across ten diverse text classification benchmarks, ranging from short-form social media tweets to long-form cinematic reviews. Our analysis reveals that the utility of cost-normalization is governed by a “Variance Law”: substantial gains are realized only when document length variance ($\sigma_{L}$) exceeds a critical threshold ($\sigma_{L} \approx 15$), below which the acquisition engine undergoes “Selection Collapse” and introduces stochastic noise. Furthermore, we identify a “Calibration Trap”—a phenomenon where high Expected Calibration Error (ECE) in high-cardinality tasks causes the learner to prioritize “cheaply wrong” samples, actively undermining the cost-utility tradeoff. We provide a comprehensive taxonomy of these failure modes and introduce a metadata-driven practitioner's decision framework for selecting optimal acquisition strategies. Our findings establish the first empirical boundary conditions for deploying cost-aware active learning in production NLP pipelines.
