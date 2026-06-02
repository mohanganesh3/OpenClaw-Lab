[2601.14525] Towards Execution-Grounded Automated AI Research

Towards Execution-Grounded Automated AI Research

Chenglei Si

Zitong Yang

Yejin Choi

Emmanuel Candès

Diyi Yang

Tatsunori Hashimoto

Abstract

Automated AI research
holds great potential to
accelerate scientific discovery.
However, current LLMs often generate plausible-looking but ineffective ideas.
Execution grounding may help, but it is unclear whether automated execution is feasible and whether LLMs can learn from the execution feedback.
To investigate these, we first build an automated executor to implement ideas and launch large-scale parallel GPU experiments to verify their effectiveness.
We then convert two realistic research problems – LLM pre-training and post-training – into execution environments and demonstrate that our automated executor can implement a large fraction of the ideas sampled from frontier LLMs.
We analyze two methods to learn from the execution feedback: evolutionary search and reinforcement learning.
Execution-guided evolutionary search is sample-efficient: it finds a method that significantly outperforms the GRPO baseline
(69.4% vs 48.0%)
on post-training, and finds a pre-training recipe that outperforms the nanoGPT baseline
(19.7 minutes vs 35.9 minutes)
on pre-training, all within just ten search epochs.
Frontier LLMs often generate meaningful algorithmic ideas during search, but they tend to saturate early and only occasionally exhibit scaling trends.
Reinforcement learning from execution reward, on the other hand, suffers from mode collapse.
It successfully improves the average reward of the ideator model but not the upper-bound, due to models converging on simple ideas.
We thoroughly analyze the executed ideas and training dynamics to facilitate future efforts towards execution-grounded automated AI research.

Machine Learning, ICML

1  Introduction

Figure 1:  We build an automated idea executor involving Implementer, Scheduler, and Worker. We then use this automated executor as a reward function to teach LLMs to generate more effective ideas through evolutionary search and RL. We only update the ideator in the learning process.

We envision automated AI research: LLMs generate research ideas to tackle important research problems, implement the ideas as code, run experiments to verify the effectiveness, and continuously learn from the execution results.
If successful, these automated AI researchers could automatically develop and identify effective research ideas in a massive search space, thereby scalably converting compute into scientific discovery; the discovered ideas could, in turn, improve frontier AI models themselves, enabling recursive self-improvement.
Despite the promise, automated AI research is bottlenecked by the ability of LLMs to generate effective ideas.
 Si et al. ( 2025b )  and  Si et al. ( 2025a )  evaluated the quality of LLM-generated research ideas through large-scale expert review and found that LLM ideas often look convincing but are ineffective after being executed by human researchers.

This highlights the need to ground idea generation in execution. However, obtaining execution results of ideas in an automated and scalable manner is challenging, especially since we are targeting
open-ended AI research where any ideas expressible in natural language are within our action space.
To tackle this, we design and build a high-throughput automated idea executor that can implement hundreds of model-generated ideas and execute them in parallel to obtain the experiment results as execution feedback.

To study the extent to which we can automate realistic LLM research, we chose two GPU-intensive research problems (LLM pre-training and post-training) that are critical for improving the capabilities of LLMs as the research environments for our automated AI researchers.
For the first time, we demonstrate that our automated executor can implement a large fraction of LLM-generated ideas on these challenging open-ended research problems, with over 90% execution rates on the pre-training environment with Claude-4.5-Sonnet and Claude-4.5-Opus.

To analyze whether grounding on execution-feedback can improve LLM idea generation, we define objective performance metrics for both environments and analyze the strengths and weaknesses of two popular learning algorithms: evolutionary search and reinforcement learning.

We use our automated executor to guide evolutionary search.
Within ten search epochs, this execution-guided search finds a post-training recipe that outperforms the GRPO baseline (69.4% vs 48.0%) on the task of post-training a 1.5B model for math reasoning, and a pre-training recipe that outperforms the nanoGPT baseline (19.7 minutes vs 35.9 minutes) on the task of minimizing the training wall-clock time to reach the target validation loss (Table

1  ).
Our analysis shows that models are often generating algorithmic ideas apart from tuning hyper-parameters, and evolutionary search significantly outperforms best-of-N under the same sampling budget.
However, when analyzing the scaling trend, only Claude-4.5-Opus shows a clear scaling curve, while both Claude-4.5-Sonnet and GPT-5 tend to saturate early.

We then use the automated executor as the reward function in an RL loop to finetune Qwen3-30B.
We show that RL with execution reward can successfully
improve the average reward of the ideator model, similar to typical RL from verifiable rewards.
However, RL does not improve the max reward, which is the more important metric for scientific discovery.
In fact, we reveal that RL causes the ideator model to converge on a few easy-to-implement ideas, resulting in a collapse in thinking length and idea diversity.

In summary, we develop a large-scale automated idea executor system that can implement research ideas for open-ended and realistic research problems.
Using this automated executor, we conduct an in-depth analysis of how well LLM ideators can learn from execution feedback to improve effectiveness through evolutionary search and RL.
Execution-guided evolutionary search is sample-efficient and effective, but shows limited scaling.
RL from execution reward suffers from diversity collapse and does not improve the upper-bound.
We additionally provide extensive analysis on the executed ideas and suggest promising directions to improve the existing learning algorithms.
Altogether, we demonstrate the feasibility and potential of grounding LLM ideation in automated execution and uncover important limitations for future improvement.

Table 1:  Performance of our execution-guided search in comparison with the provided baselines and best human experts. The post-training task is to finetune a 1.5B model for math reasoning, and the metric is validation accuracy. The pre-training task is to train a 124M Transformer on FineWeb, and the metric is the training time to reach 3.28 validation loss.

Post-Training

↑

Pre-Training

↓

Baseline

48.0%

35.9 min

Execution-Guided Search

69.4%

19.7 min

Best Human Expert

68.8%

2.1 min

2  Automated Idea Executor

To measure the effectiveness of model-generated ideas, we build an automated executor that takes natural language research ideas as input, generates code implementations, runs the experiments on the backend, and returns the idea’s benchmark performance as the final output.

2.1  Research Environments for Ideation

Our automated idea executor is grounded in specific research environments, where each environment consists of a research problem, a baseline codebase, a benchmark to measure performance on, fixed training and evaluation data, and evaluation metrics. When constructing the research environments, we aim to select research problems that are open-ended, so that there is ample room for new algorithmic innovations, and at the same time have well-established baselines and benchmarking metrics so that measuring effectiveness is straightforward. In this work, we construct both a pre-training e