# EV013: SycoPhantasy: Quantifying Sycophancy and Hallucination in Small Open Weight VLMs for Vision-Language Scoring of Fantasy Characters

URL: https://www.semanticscholar.org/paper/279b2280cc5bdc943fb1ef7ba3b0b9a084303eee
Year: 2026
Source: semantic_scholar
Arxiv: 2604.24346

## Abstract

Vision-language models (VLMs) are increasingly deployed as evaluators in tasks requiring nuanced image understanding, yet their reliability in scoring alignment between images and text descriptions remains underexplored. We investigate whether small, open-weight VLMs exhibit \emph{sycophantic} behavior when evaluating image-text alignment: assigning high scores without grounding their judgments in visual evidence. To quantify this phenomenon, we introduce the \emph{Bluffing Coefficient} (\bc), a metric that measures the mismatch between a model's score and its evidence recall. We evaluate six open-weight VLMs ranging from 450M to 8B parameters on a benchmark of 173,810 AI-generated character portraits paired with detailed textual descriptions. Our analysis reveals a significant inverse correlation between model size and sycophancy rate ($r = -0.96$, $p = 0.002$), with smaller models exhibiting substantially higher rates of unjustified high scores. The smallest model tested (LFM2-VL, 450M) produced sycophantic evaluations in 22.3\% of cases, compared to 6.0\% for the largest (LLaVA-1.6, 7B). These findings have direct implications for the deployment of small, open-weight VLMs as automated evaluators within attribute-rich, synthetic image evaluation tasks, where the gap between assigned scores and cited visual evidence is both measurable and consequential.
