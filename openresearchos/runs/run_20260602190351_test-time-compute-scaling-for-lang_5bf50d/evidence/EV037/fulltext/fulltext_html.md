GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models

1  Introduction

2  Related Works

3  The limitation of graph-to-sequence based inference

4  Methods

4.1  Preliminaries

4.2  Graph context memorization

4.3  Enabling in-parameter graph reasoning through question answering

4.4  Fine-tuning and inference of GRIP

5  Experiments

5.1  Setup

5.2  Results on performance comparison

5.3  Results on the efficiency

5.4  Ablation studies on GRIP design

6  Further discussion on limitation and application of GRIP

7  Conclusion and Limitation

A  Implementation Details

A.1  Prompts Design

A.1.1  Summarization task

A.1.2  Context QA task

A.1.3  Reasoning QA task

A.2  Implementation setting

A.3  Fine-tuning setting

B  Experiment details

B.1  Dataset details

B.2  Training process and hyperparameters

GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models

Jiarui Feng

feng.jiarui@wustl.edu

Washington University in Saint Louis  USA

,

Donghong Cai

cai.d@wustl.edu

Washington University in Saint Louis  USA

,

Yixin Chen

ychen25@wustl.edu

Washington University in Saint Louis  USA

and

Muhan Zhang

muhan@pku.edu.cn

Institute for Artificial Intelligence, Peking University  State Key Laboratory of General Artificial Intelligence, BIGAI  China

Abstract.

Large Language Models (LLMs) have demonstrated remarkable capabilities in modeling sequential textual data and generalizing across diverse tasks. However, adapting LLMs to effectively handle structural data, such as knowledge graphs or web data, remains a challenging problem. Some approaches adopt complex strategies to convert graphs into text sequences, resulting in significant token overhead and rendering them impractical for large-scale graphs. Others introduce additional modules to encode graphs into fixed-size token representations for LLMs. However, these methods typically require large-scale post-training on graph-text corpus and complex alignment procedures, yet often yield sub-optimal results due to poor modality alignment. Inspired by in-parameter knowledge injection for test-time adaptation of LLMs, we propose GRIP, a novel framework that equips LLMs with the ability to internalize complex relational information from graphs through carefully designed fine-tuning tasks. This knowledge is efficiently stored within lightweight LoRA parameters, enabling the fine-tuned LLM to perform a wide range of graph-related tasks  without requiring access to the original graph  at inference time. Extensive experiments across multiple benchmarks validate the effectiveness and efficiency of our approach.

Graph Learning, Graph Algorithm, Test Time Adaptation, Large Language Models

†

†  ccs:  Computing methodologies Machine learning

1.  Introduction

In recent years, Large Language Models (LLMs) such as ChatGPT  ( chatgpt )  and DeepSeek  ( deepseek )  have revolutionized the field of artificial intelligence. Pre-trained on large-scale corpora of human knowledge using next-token prediction, these models have demonstrated remarkable generalization capabilities across a variety of downstream tasks, including math problem solving  ( GSM8K ) , coding  ( deepseekcoder ) , tool use  ( hugginggpt ) , and knowledge-intensive applications  ( rag ;  paperqa ) . However, to fully leverage the strong reasoning abilities of LLMs, tasks must first be formulated and expressed in human-readable textual formats. This conversion process is often complex and, in some cases, infeasible—particularly for tasks involving intricate data structures such as graphs.

There has been extensive work on adapting LLMs for graph-related tasks, which can be broadly classified into two categories. The first class of approaches directly applies LLMs to graph data by converting graphs into text sequences  ( NLGraph ;  instructGLM ;  langGFM ) . However, due to the complex structure of graphs, representing them as sequences is nontrivial and often results in  excessive token overhead or degraded inference performance

( instructGLM ) , making this approach impractical for large graphs. In particular, for scenarios requiring  reasoning over the same graph with multiple queries —common in web-based retrieval or knowledge graph reasoning—the overhead scales linearly with the number of queries, since the same graph context must be combined with each query repeatedly. Moreover, for large-scale graphs, the size of the graph can easily exceed the context window of LLMs. A recent study has also highlighted the inherent limitation of sequential LLMs in solving graph tasks under node permutation  ( gnntaskplanner ) . We provide a more detailed discussion in the next section.

The second class of approaches integrates specialized modules, such as Graph Neural Networks (GNNs), to process graph data prior to interaction with the LLM. The resulting graph embeddings are then aligned with the LLM’s representation space via an additional adapter and alignment  ( gofa ;  graphgpt ;  llaga ) . While effective in some cases, these methods typically require  careful model design and fine-tuning tasks to achieve alignment between language and graph modalities . Furthermore, such frameworks typically require modifications to the model itself, which prevents the use of common acceleration techniques designed for standard LLMs and restricts their practical applicability.

Inspired by recent advances in parameterized knowledge injection for LLM test-time adaptation  ( Temp-Lora ;  lift ;  parametricRAG ) , we investigate the potential of embedding graph information directly into LLM parameters using Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA  ( lora ) . To this end, we introduce Graph Reasoning In-Parameterization (GRIP), a unified framework illustrated in Figure

1  . In GRIP, we generate diverse fine-tuning tasks to guide LLMs in both memorizing graph contexts and leveraging this knowledge for downstream tasks. Given an input graph, we design context memorization and summarization tasks to inject graph knowledge into the LoRA parameters. We then construct various QA tasks on the graph to enable the LLM to effectively retrieve and apply the memorized knowledge to answer queries. Through fine-tuning, the LLM internalizes the structural information of the graph and encodes it into the LoRA parameters. At inference time, since the graph context is already embedded in the parameters, the model can perform a wide range of tasks related to the injected graph without requiring explicit graph context, simply by applying the trained LoRA. GRIP offers several key advantages:

•

Simplicity.  It eliminates the need for specialized graph modules or graph-to-sequence conversions during inference.

•

Generalization ability.  The framework of GRIP is largely agnostic to evaluation tasks, giving it strong capability in out-of-domain task solving.

•

Efficiency.  After training, it can perform various tasks on the trained graph without requiring access to the original graph data as context, which significantly reduces token cost and improves inference efficiency.

•

Reusability.  Once the LoRA is trained, it can be stored and reused for inference on the same graph at any time.

We conduct extensive experiments to validate the effectiveness of GRIP. For graph tasks where the graph size exceeds the context window of the LLM, GRIP consistently outperforms baseline LLMs with graph-to-sequence conversion. This improvement stems from its ability to internalize the entire graph into the LoRA parameters and leverage this knowledge during inference. For tasks where the graph fits within the LLM’s context window, GRIP achieves comparable performance to baselines without requiring explicit graph context, demonstrating its efficiency. Moreover, when provided with the graph context, GRIP further improves performance and surpasses the baselines, highlighting its potential as a test-time adaptation method for graph tas