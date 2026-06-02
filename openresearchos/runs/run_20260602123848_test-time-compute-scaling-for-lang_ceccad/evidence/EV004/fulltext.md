[2505.14604] Let LLMs Break Free from Overthinking via Self-Braking Tuning

Let LLMs Break Free from Overthinking

via Self-Braking Tuning

Haoran Zhao  1,2,*

&amp; Yuchen Yan  1,*

&amp; Yongliang Shen  1,

†  \dagger

&amp; Haolei Xu  1

&amp; Wenqi Zhang  1  Kaitao Song  3

&amp; Jian Shao  1

&amp; Weiming Lu  1

&amp; Jun Xiao  1

&amp; Yueting Zhuang  1

1

Zhejiang University,
 2  Tianjin University,
 3  Microsoft Research Asia

ran159753@tju.edu.cn, {yanyuchen, syl}@zju.edu.cn

GitHub:  https://github.com/ZJU-REAL/Self-Braking-Tuning

Project:  https://ZJU-REAL.github.io/SBT

Abstract

Large reasoning models (LRMs), such as OpenAI o1 and DeepSeek-R1, have significantly enhanced their reasoning capabilities by generating longer chains of thought, demonstrating outstanding performance across a variety of tasks. However, this performance gain comes at the cost of a substantial increase in redundant reasoning during the generation process, leading to high computational overhead and exacerbating the issue of overthinking.
Although numerous existing approaches aim to address the problem of overthinking, they often rely on external interventions.
In this paper, we propose a novel framework,  Self-Braking Tuning (SBT), which tackles overthinking from the perspective of allowing the model to regulate its own reasoning process, thus eliminating the reliance on external control mechanisms. We construct a set of overthinking identification metrics based on standard answers and design a systematic method to detect redundant reasoning. This method accurately identifies unnecessary steps within the reasoning trajectory and generates training signals for learning self-regulation behaviors. Building on this foundation, we develop a complete strategy for constructing data with adaptive reasoning lengths and introduce an innovative braking prompt mechanism that enables the model to naturally learn when to terminate reasoning at an appropriate point.
Experiments across mathematical benchmarks (AIME, AMC, MATH500, GSM8K) demonstrate that our method reduces token consumption by up to 60% while maintaining comparable accuracy to unconstrained models.

1

1  footnotetext:

The first two authors have equal contributions. This work was done when the first author was an intern at Zhejiang University.

2

2  footnotetext:

Corresponding author.

Figure 1 :

Demonstration of Self-Braking Tuning Effectiveness. In the single-example case (a), the self-braking tuned model exhibits spontaneous termination of overthinking and significantly reduces token usage. On major mathematical benchmarks (b), compared to using OpenR1-Math

openr1math220k

as the SFT dataset, the self-braking tuned Qwen2.5-Math-1.5B-Instruct

yang2024qwen2

achieves a substantial reduction in tokens consumed during inference while maintaining comparable accuracy.

1  Introduction

Large reasoning models (LRMs) such as OpenAI’s o1

jaech2024openai

, Deepseek-R1

guo2025deepseek

, QwQ

qwen2024qwq

, Gemini 2.0 Flash Thinking

gemini_flash_thinking

and Kimi-1.5

team2025kimi

, excel at mathematical and logical tasks by generating detailed multi-step reasoning, boosting accuracy on complex benchmarks

latif2024can

. However, this often results in excessively long inference trajectories, frequently consuming thousands of tokens per problem

learning_to_reason_with_llms

;

wang2025thoughts

, leading to increased computational cost, latency, and redundant reasoning that can obscure core solutions

wu2025more

. This “overthinking”

chen2024not

poses a significant challenge for practical deployment.

Many recent studies have focused on addressing the problem of overthinking

sui2025stop

;

wang2025harnessing

, which can be broadly categorized into three approaches: (1)  Model optimization:  Apply reinforcement learning (RL) or supervised fine-tuning (SFT) to equip models with the ability to control reasoning length  (  aggarwal2025l1controllinglongreasoning ,  ;

luo2025o1 ,  ;

yeo2025demystifying ,  ) ; (2)  Reasoning output optimization : Dynamically reducing the number of reasoning steps and output length during inference

ma2025reasoning

;

yang2025dynamic

;

zhang2025lightthinker

; (3)  Adding external restrictions : Imposing external constraints, such as token budgets, to reduce overthinking

han2024token

;

xu2025chain

. Most existing methods follow the paradigm of external intervention, relying on complex optimization strategies or introducing additional constraint mechanisms, and have yet to fully explore the intrinsic ability of the model to mitigate overthinking on its own.

This reliance on external control prompts a fundamental question:  Can we enable large reasoning models to autonomously recognize excessive reasoning and terminate their thinking process appropriately?  Ideally, a model should intrinsically understand when additional reasoning becomes redundant and halt its thought process without external triggers, similar to how humans naturally conclude their reasoning when reaching sufficient certainty.

To address this challenge, we propose  Self-Braking Tuning(SBT) , a novel framework that teaches LRMs to autonomously identify and terminate redundant reasoning. Unlike previous approaches that impose external constraints, SBT fundamentally reshapes how models perceive and regulate their own reasoning processes. Our key insight is that LRMs can be trained to develop an internal braking mechanism that recognizes when further reasoning becomes unproductive, enabling them to naturally conclude the thought process and transition to formulating the final solution.

Our approach begins with a systematic methodology for identifying overthinking patterns in reasoning trajectories. By combining metrics such as reasoning efficiency ratio and overthinking label ratio, we precisely pinpoint the transition point at which the model shifts from effective reasoning to redundant computation. Based on this analysis, we develop two complementary data construction strategies: (1) Self-Braking Tuning Exact (SBT-E), which strictly removes redundant reasoning segments based on predefined braking points, allowing the model to enter the conclusion phase earlier; (2) Self-Braking Tuning Dynamic (SBT-D), which implements step-level monitoring and dynamically halts reasoning when overthinking patterns emerge.

Building upon the high-quality OpenR1-Math

openr1math220k

dataset of reasoning trajectories, we constructed two specialized training datasets using these strategies: OpenR1-Math-SBT-E and OpenR1-Math-SBT-D. To further enhance the model’s self-awareness of its reasoning state, we introduce braking prompts at the identified braking points, explicitly simulating the recognition of having sufficiently completed the reasoning process. These prompts enable the model to naturally express an awareness of having reached adequate reasoning depth, thus promoting autonomous termination without the need for external signals. Our experimental results demonstrate consistent improvements in reasoning efficiency across mathematical benchmarks of varying difficulty. While maintaining high accuracy, token consumption was reduced by 30% to 60%. Our contributions can be summarized as follows:

•

We introduce a novel tuning framework that enables LRMs to self-regulate reasoning length without external constraints. Self-Braking Tuning cultivates models’ intrinsic ability to recognize and inhibit excessive reasoning, fundamentally improving inference efficiency and response quality.

•

We propose a systematic methodology for identifying overthinking patterns and develop two complementary data construction strategies: SBT-E and SBT-D, resulting in specialized training datasets for addressing the overthinking problem. These datasets systematically prune redundant reasoning while preserving essential thinking steps.

•

We demonstrate that models trained with our SBT framework maintain original accuracy levels while