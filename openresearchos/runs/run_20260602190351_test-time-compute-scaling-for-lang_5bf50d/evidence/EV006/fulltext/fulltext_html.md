[2510.14232] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models

Scaling Test-Time Compute to Achieve
 IOI Gold Medal with Open-Weight Models

Mehrzad Samadi, Aleksander Ficek, Sean Narenthiran, Siddhartha Jain

Wasi Uddin Ahmad, Somshubra Majumdar, Vahid Noroozi, Boris Ginsburg

NVIDIA
 Santa Clara, CA 95051, USA

{msamadi,aficek,snarenthiran,siddjain
 wasiuddina,smajumdar,vnoroozi,bginsburg}@nvidia.com

Abstract

Competitive programming has become a rigorous benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). The International Olympiad in Informatics (IOI) stands out as one of the most prestigious annual competitions in competitive programming and has become a key benchmark for comparing human and AI-level programming ability. While several proprietary models have been claimed to achieve gold medal-level performance at the IOI, often with undisclosed methods, achieving comparable results with open-weight models remains a significant challenge. In this paper, we present  GenCluster , a scalable and reproducible test-time compute framework that attains IOI gold-level performance using open-weight models. It combines large-scale generation, behavioral clustering, ranking, and a round-robin submission strategy to efficiently explore diverse solution spaces under limited validation budgets. Our experiments show that the performance of our proposed approach scales consistently with available compute, narrowing the gap between open and closed systems. Notably, we will show that  GenCluster  can achieve a gold medal at IOI 2025 for the first time with an open-weight model  gpt-oss-120b , setting a new benchmark for transparent and reproducible evaluation of reasoning in LLMs.

1  Introduction

As large language models (LLMs) have advanced in solving coding problems, traditional benchmarks such as HumanEval  (Chen et al.,  2021a )  and MBPP  (Austin et al.,  2021 )  have reached saturation. This has motivated a shift toward more challenging benchmarks like LiveCodeBench  (Jain et al.,  2025 )  and Codeforces  (Penedo et al.,  2025 ) , which feature significantly more complex competitive programming problems. Furthermore, considering the challenging aspects of understanding and solving the competitive programming questions, they have been shown to be effective for both training and evaluating the reasoning capabilities of LLMs  (DeepSeek-AI et al.,  2025 ) . Recent progress on these benchmarks has been driven by both improved training strategies and test-time compute methods. For example, AlphaCode2  (Leblond et al.,  2023 )  achieved performance exceeding that of 85% of Codeforces participants by generating over one million candidate programs per problem and then applying a pipeline of filtering, clustering, and ranking to select 10 solutions for final submissions.

Analogous to the International Mathematical Olympiad (IMO)  1

1  1 https://www.imo-official.org/

, the International Olympiad in Informatics (IOI)  2

2  2 https://ioinformatics.org/

and International Collegiate Programming Contest (ICPC)  3

3  3 https://icpc.global/

are widely regarded as the pinnacle of algorithmic programming competitions. These competitions serve as an important milestone in assessing AI competency on competitive programming and general reasoning capabilities as a whole. OpenAI reported achieving a gold medal at IOI 2024 with their  o1-ioi  and  o3  models  (OpenAI et al.,  2025 ) , using specialized test-time compute approaches.  o1-ioi  is a dedicated version of their  o1  model fine-tuned for competitive programming and leveraging external tools. Additionally, OpenAI claimed a gold medal and a 6th-place human-equivalent ranking at IOI 2025 with their latest models  (Decoder,  2025 ) , though the details of these systems have not yet been fully disclosed. Recently, both OpenAI and Google DeepMind  (Lin &amp; Cheng,  2025 )  have reported achieving gold-level performance at ICPC 2025 using proprietary methods.

Despite their impressive results, the techniques and models used to achieve gold-level performance at such competitions are rarely disclosed. In contrast, open-weight models such as  DeepSeek-R1-0528

(DeepSeek-AI et al.,  2025 )  and  Qwen3-235B-A22B

(Team,  2025 )  have achieved increasingly competitive results on LiveCodeBench and Codeforces, but still lag behind proprietary models. In this paper, we propose  GenCluster , a scalable test-time compute approach that substantially improves the performance of open-weight LLMs on IOI problems. Our approach first generates a large pool of candidate solutions for each problem, which are then refined through a pipeline of filtering, behavioral clustering, and ranking with tournament. Finally, a round-robin submission strategy selects the best candidates while adhering to the same submission constraints imposed on human contestants at IOI (Figure

1  ). In our experiments, we demonstrate that  GenCluster  enables open-weight model  gpt-oss-120b

(OpenAI,  2025 )  to achieve gold-level performance at IOI 2025.

In our experiments, we evaluated and analyzed some of the best open-weight models publicly available including  DeepSeek-R1-0528  and  Qwen3-235B-A22B , and we show that  gpt-oss-120b  is significantly superior on IOI problems, providing better scalability in terms of the number of generations. Furthermore, we demonstrate that  GenCluster  improves scores with increased compute and larger generation budgets which is an indicator of scalability of our proposed test-time compute approach.

In summary, our contributions are as follows:

1.

We propose  GenCluster , a scalable test-time compute approach to improve the competitive programming performance of LLMs by generating many solutions in parallel and then using behavioral clustering and tournaments to select the best candidates.

2.

We demonstrate, for the first time, that  gold-level performance  at IOI 2025 can be achieved using only  open-weight models  combined with a transparent and reproducible  test-time compute  strategy.

3.

We show that scaling  GenCluster  consistently improves scores, providing a promising path to surpass gold-level performance.

Figure 1:  The overall pipeline of  GenCluster  for a single subtask, a process to be repeated for every subtask in the IOI benchmark.

2  Background

2.1  Related Work

This paper investigates scalable test-time compute (TTC) for large language models (LLMs) applied to coding problems, particularly those with a limited verification budget. Test-time compute refers to allocating additional computational resources during inference to improve output quality. This can be achieved through extended chain-of-thought reasoning  (DeepSeek-AI et al.,  2025 ; Li et al.,  2025 ; Wei et al.,  2022 ; Zhang et al.,  2025b ) , generative verifiers for scoring and selection  (Fu et al.,  2025 ; Mahan et al.,  2024 ; Zhang et al.,  2025a ) , or best-of-N strategies  (Chen et al.,  2025 ; Toshniwal et al.,  2025 ; Brown et al.,  2024 ; Li et al.,  2022 ; Leblond et al.,  2023 ; OpenAI et al.,  2025 ) . While there exist some limited works that have leveraged large scale test-time compute pipelines to achieve state-of-the-art results on coding problems  (Li et al.,  2022 ; Leblond et al.,  2023 ; OpenAI et al.,  2025 ) , most works have explored smaller compute scales  (Chen et al.,  2023 ; Le et al.,  2022 ; Zhang et al.,  2023 )  or less challenging benchmarks  (Chen et al.,  2021b ; Inala et al.,  2022 ; Zhao et al.,  2025 ; To et al.,  2024 ) . These studies have not yet explored their methods at the scale required for competitive programming, where problems are highly complex and only a small fraction of the generated candidates are correct. In the following, we highlight works that specifically explore scalable approaches to competitive programming and coding competitions.

AlphaCode  (Li et al.,  2022 )  followed by its successor Alpha