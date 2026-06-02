[2504.15466] Learning Adaptive Parallel Reasoning with Language Models

Learning Adaptive Parallel Reasoning with Language Models

Jiayi Pan*  1  , Xiuyu Li*  1  , Long Lian*  1  , Charlie Snell  1  , Yifei Zhou  1  ,

Adam Yala  1,2  , Trevor Darrell  1  , Kurt Keutzer  1  , Alane Suhr  1

1  UC Berkeley

2  UCSF

{jiayipan,xiuyu,longlian,suhr}@berkeley.edu

Abstract

Scaling inference-time computation has substantially improved the reasoning capabilities of language models. However, existing methods have significant limitations: serialized chain-of-thought approaches generate overly long outputs, leading to increased latency and exhausted context windows, while parallel methods such as self-consistency suffer from insufficient coordination, resulting in redundant computations and limited performance gains.
To address these shortcomings, we propose Adaptive Parallel Reasoning (APR), a novel reasoning framework that enables language models to orchestrate both serialized and parallel computations end-to-end.
APR generalizes existing reasoning methods by enabling adaptive multi-threaded inference using  spawn()  and  join()  operations. A key innovation is our end-to-end reinforcement learning strategy, optimizing both parent and child inference threads to enhance task success rate without requiring predefined reasoning structures. Experiments on the Countdown reasoning task demonstrate significant benefits of APR:
(1) higher performance within the same context window (83.4% vs. 60.0% at 4k context);
(2) superior scalability with increased computation (80.1% vs. 66.6% at 20k total tokens);
(3) improved accuracy at equivalent latency (75.2% vs. 57.3% at approximately 5,000ms).
APR represents a step towards enabling language models to autonomously optimize their reasoning processes through adaptive allocation of computation.

†

†  footnotetext:  *Equal contribution. Code, model, and data are available at
 github.com/Parallel-Reasoning/APR .

Figure 1:

Serialized search  (Gandhi et al.,  2024 )  (top) vs Adaptive Parallel Reasoning (bottom)  illustrated on an example of the Countdown task, with a target number of 27 and input numbers of {22, 26, 31, 53}. Each box represents a node in the search representing the value of the intermediate expression (

x  x

). Edges are annotated with explored arithmetic operations relative to the parent-node

x  x

using remaining input numbers. In serialized search, the context window of the single inference thread is exhausted before a solution is found. In Adaptive Parallel Reasoning, the parent thread (blue) spawns two child threads (orange), which are executed in parallel. Child threads have access only to a limited context passed to them by the parent thread and return a summary of their execution to the parent thread. The parent thread can then continue to decode with access to these summaries. This parallel distribution of computation prevents context window exhaustion while reducing latency.

1  Introduction

Recent progress in language model reasoning, like OpenAI o1  (OpenAI,  2024 )  and DeepSeek-R1  (DeepSeek-AI,  2025 ) , has demonstrated the promise of exploiting test-time compute to perform search and of using reinforcement learning to optimize the search.
However, these current approaches face fundamental limitations: serialized chain-of-thought methods  (DeepSeek-AI,  2025 )  produce lengthy output sequences that increase latency and strain context window limits, while parallel methods like best-of-N or self-consistency  (Wang et al.,  2023 )  often
lack coordination between inference paths and are not end-to-end optimized, leading to redundant computation and limiting improvement. Structured inference-time search methods like tree-of-thought  (Yao et al.,  2023 )  require hand-designed search structures, limiting their flexibility and scalability.

We propose Adaptive Parallel Reasoning (APR), a simple yet effective approach that enables language models to reason by adaptively distributing inference-time computation in a manner that exploits both serial and parallel operations.
Our method generalizes existing approaches to reasoning with language models, including serialized chain-of-thought reasoning, parallelized inference with self-consistency, and structured search: rather than imposing fixed search structures through prompting or external orchestration, we train language models to  learn  when and how to parallelize their inference operations.

Adaptive Parallel Reasoning employs two key innovations:
First, we supply language models with a parent-child threading mechanism. Parent inference threads can, at any point during decoding, delegate subtasks to multiple child inference threads using a  spawn()  operation. Child threads independently explore distinct reasoning paths in parallel and return outcomes to the parent thread through a  join()  operation. The parent thread then continues decoding conditioned on the information returned by the children.
We build on the model serving framework SGLang  (Zheng et al.,  2024 )  to perform inference in child threads simultaneously through batching, which significantly reduces real-time latency.
Our second key innovation is to fine-tune the language model that performs inference in both parent and child threads via end-to-end reinforcement learning, which optimizes overall task success and eliminates the need to predefine explicit reasoning structure.

Figure

1

illustrates how APR facilitates more efficient and effective reasoning in the Countdown reasoning task  (Yao et al.,  2023 ; Gandhi et al.,  2024 ; Pan et al.,  2025 )  when compared to serialized methods. Our experiments demonstrate that we achieve three key benefits over prior approaches:

1.  Higher performance within same context window : Our approach performs more effective search and reasoning within fixed context size constraints (83.4 vs 60.0% at 4k context) compared to sequential methods that quickly exhaust available context.

2.  Superior scaling behavior : When scaling the total compute budget, APR exhibits better performance improvements (80.1 vs 66.6% with a budget of 20k total tokens) through wider parallelization in addition to increasing the length of individual reasoning chains.

3.  Improved performance at the same latency : Adaptive Parallel Reasoning achieves significantly higher success rates compared to serialized search methods with same latency (75.2 vs 57.3% at around 5,000ms).

These results highlight the potential of training language models to adaptively allocate their own inference-time compute resources. By learning when to reason serially and when to branch out into parallel computation, models can more efficiently explore the solution space of complex reasoning problems.

2  Related Work

Inference-time scaling

Prior work has shown that increasing test-time compute can improve language model performance on downstream tasks  (Wei et al.,  2022 ; Zelikman et al.,  2022 ; Zhu et al.,  2024 ; DeepSeek-AI,  2025 ; OpenAI,  2024 ; Team,  2025 ; Gandhi et al.,  2024 ; Snell et al.,  2025 ; Li et al.,  2025 ) . However, these methods typically result in significantly longer output sequences, introducing key limitations given the inherently sequential nature of autoregressive decoding: longer output sequences mean higher latency, and fitting an entire sequence into a single context window makes it hard for models to attend to relevant information when scaled with more output tokens. Compared to serialized inference, parallelizing reasoning traces reduces both latency and the pressure of context window limitations. Our experiments demonstrate that models trained end-to-end to adaptively distribute inference-time compute can outperform serialized approaches under the same computational budgets.

Parallelization in language model inference

Parallelizing inference with multiple independent language model calls has emerged as another avenue for inference-time scaling tow