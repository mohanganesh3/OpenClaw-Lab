[2505.20622] SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation

SeqPO-SiMT: Sequential Policy Optimization for

Simultaneous Machine Translation

Ting Xu

♠  \spadesuit

1

1

1

Work done when Ting Xu was interned at Bytedance.

,
 Zhichao Huang

♣  \clubsuit

2

2

2

Corresponding authors.

,
 Jiankai Sun

♢  \diamondsuit

,
 Shanbo Cheng

♣  \clubsuit

2

2

2

Corresponding authors.

,
 Wai Lam

♠  \spadesuit

,

♠  \spadesuit

The Chinese University of Hong Kong,

♣  \clubsuit

Bytedance,

♢  \diamondsuit

Stanford University,

xut0092@link.cuhk.edu.hk ,  jksun@stanford.edu ,

{zhichao.huang, chengshanbo}@bytedance.com, wlam@se.cuhk.edu.hk

Abstract

We present Sequential Policy Optimization for Simultaneous Machine Translation (SeqPO-SiMT), a new policy optimization framework that defines the simultaneous machine translation (SiMT) task as a sequential decision making problem, incorporating a tailored reward to enhance translation quality while reducing latency.
In contrast to popular Reinforcement Learning from Human Feedback (RLHF) methods, such as PPO and DPO, which are typically applied in single-step tasks, SeqPO-SiMT effectively tackles the multi-step SiMT task.
This intuitive framework allows the SiMT LLMs to simulate and refine the SiMT process using a tailored reward. We conduct experiments on six datasets from diverse domains for En

→  \to

Zh and Zh

→  \to

En SiMT tasks, demonstrating that SeqPO-SiMT consistently achieves significantly higher translation quality with lower latency. In particular, SeqPO-SiMT outperforms the supervised fine-tuning (SFT) model by 1.13

3

3  3  Kocmi et al. ( 2024b )  has indicated that an increase of 1 point in COMET represents a significant improvement.

points in COMET, while reducing the Average Lagging by 6.17 in the NEWSTEST2021 En

→  \to

Zh dataset.
While SiMT operates with far less context than offline translation, the SiMT results of SeqPO-SiMT on 7B LLM surprisingly rival the offline translation of high-performing LLMs, including Qwen-2.5-7B-Instruct and LLaMA-3-8B-Instruct.

SeqPO-SiMT: Sequential Policy Optimization for

Simultaneous Machine Translation

Ting Xu

♠  \spadesuit

1

1

1  Work done when Ting Xu was interned at Bytedance.

,
Zhichao Huang

♣  \clubsuit

2

2

2  Corresponding authors.

,
Jiankai Sun

♢  \diamondsuit

,
Shanbo Cheng

♣  \clubsuit

2

2

2  Corresponding authors.

,
Wai Lam

♠  \spadesuit

,

♠  \spadesuit

The Chinese University of Hong Kong,

♣  \clubsuit

Bytedance,

♢  \diamondsuit

Stanford University,

xut0092@link.cuhk.edu.hk ,  jksun@stanford.edu ,

{zhichao.huang, chengshanbo}@bytedance.com, wlam@se.cuhk.edu.hk

1  Introduction

Figure 1:  Two examples of SiMT, which translates streaming source texts into target texts. The source texts of SiMT are fed in a sequential manner. At each step, SiMT receives a new source text chunk and generates corresponding translations. Translations at each step can influence overall translations.
A forward slash (/) indicates an empty translation, which means the model chooses not to translate and waits for subsequent texts.

Simultaneous Machine Translation (SiMT) has made huge progress by leveraging large language models (LLMs)  Cheng et al. ( 2024 ); Koshkin et al. ( 2024 ) . These approaches generally use partial translation data to finetune LLMs, enabling LLMs to translate partial source texts to target texts. However, the partial translation data are often generated using simple alignment tools, like heuristic methods  (Ma et al.,  2019 )  or attention mechanisms  (Arivazhagan et al.,  2019 ) , which may introduce noise and degrade performance.

In parallel, Reinforcement Learning from Human Feedback (RLHF;  Ouyang et al.,  2022  ) has gained huge success in improving the performance of fine-tuned LLMs  (DeepSeek-AI,  2025 ; Yang et al.,  2024 ; Sun et al.,  2025 ) .
RLHF is reward-driven and does not rely on partial translation data.
Applying these techniques to SiMT appears to be a promising approach for improving its performance. However, we find that traditional RLHF methods like PPO  (Schulman et al.,  2017 )  and DPO  (Rafailov et al.,  2024 )  commonly work for a  single-step  process, while SiMT translates streaming inputs in a  multi-step  manner.
The multi-step dependency between the source and target in SiMT is complex.
First, the source texts of SiMT are provided step by step, and each step’s source may have ambiguous meanings that require subsequent context for clarification. For example, the first step’s source ( bark ) in Figure

1

is ambiguous, which requires subsequent source texts ( tree ) to clarify. Second, previous translation results can influence the overall translation quality. For example, in the second example of Figure

1  , misinterpreting  "the bark of"  leads to errors in the overall translations. We claim that traditional RLHF methods commonly used in single-step reveal deficiencies in modeling the complex dependence relations in the multi-step SiMT setting.

To this end, we propose a new policy optimization method,  Seq uential  P olicy  O ptimization (SeqPO), and apply it to the SiMT task.
As shown in Figure

2  , SeqPO-SiMT defines SiMT as a sequential decision-making process.
To simulate the SiMT process, we segment a full sentence into multiple chunks and feed these chunks to the LLM sequentially. At each step, the model evaluates the new chunk and translation history to decide whether to translate or wait for subsequent context. After completing all the steps, we construct a tailed reward to assess the entire SiMT process according to quality and latency. Finally, we optimize the SiMT process using policy gradient  (Sutton and Barto,  2018 ) . SeqPO-SiMT enables us to simulate the complex SiMT process as a multi-step decision problem and refine it based on quality and latency.

To validate the effectiveness of SeqPO-SiMT, we conduct experiments on extensive datasets for En

→  \to

Zh and Zh

→  \to

En SiMT tasks, including formal spoken language, informal spoken language, specialized knowledge domains, and news articles. Performance is measured using a range of comprehensive metrics.
Extensive experimental results demonstrate that SeqPO-SiMT not only attains superior translation quality but also reduces latency.
In low latency and high latency scenarios, the average COMET scores of SeqPO-SiMT are 1.3 and 1.25 points higher than those of the supervised fine-tuning (SFT) method, respectively.
In particular, SeqPO-SiMT outperforms the SFT model by 1.13 points in COMET, while reducing the Average Lagging (AL) by 6.17 in NEWSTEST2021 En

→  \to

Zh dataset.
Because the SiMT task has only a limited amount of contexts while offline translation utilizes the full context, SiMT is more challenging than offline translation. Remarkably, the SiMT performance of SeqPO-SiMT is comparable to the offline translation performance of high-performing open-source models, such as Qwen-2.5-7B-Instruct  (Yang et al.,  2024 )  and LLaMA-3-8B-Instruct.
These findings underscore the effectiveness of SeqPO-SiMT.
We summarize our contribution as follows:

1. We define the RLHF process of SiMT LLMs as a sequential decision-making process
to model the complex dependencies among steps in SiMT.

2. SeqPO-SiMT fuses both translation quality and latency into a reward. With a carefully designed fusion function, SeqPO-SiMT successfully improves the two metrics.

3. Extensive experiments demonstrate the superiority of SeqPO-SiMT, which not only enhances translation quality but also reduces the latency of SiMT. Furthermore, SeqPO-SiMT achieves SiMT translation quality comparable to the offline performance of strong LLMs like Qwen-2.5-7B-Instruct.

2  Sequential Policy Optimization

Figure 2:  Model structure of SeqPO-SiMT. We first segment a full sentence in multiple chunks. At each step, we feed a new chunk to the LLM. Concatenating new source chunk and translation history, we