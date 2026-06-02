[2502.06327] Prompt-Driven Continual Graph Learning

Prompt-Driven Continual Graph Learning

Qi Wang, Tianfei Zhou, Ye Yuan, and Rui Mao

Qi Wang is with Beijing Institute of Technology, Zhuhai, China.
(e-mail: qiwang@bit.edu.cn)
Tianfei Zhou and Ye Yuan are with the School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China.
(e-mail: tfzhou@bit.edu.cn, yuan-ye@bit.edu.cn)
Rui Mao is with the College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China. (e-mail: mao@szu.edu.cn)

Abstract

Continual Graph Learning (CGL), which aims to accommodate new tasks over evolving graph data without forgetting prior knowledge, is garnering significant research interest. Mainstream solutions adopt the memory replay-based idea,  i.e. , caching representative data from earlier tasks for retraining the graph model. However, this strategy struggles with scalability issues for constantly evolving graphs and raises concerns regarding data privacy. Inspired by recent advancements in the prompt-based learning paradigm, this paper introduces a novel prompt-driven continual graph learning ( PromptCGL ) framework, which learns a separate prompt for each incoming task and maintains the underlying graph neural network model fixed. In this way,  PromptCGL  naturally avoids catastrophic forgetting of knowledge from previous tasks. More specifically, we propose  hierarchical prompting  to instruct the model from both feature- and topology-level to fully address the variability of task graphs in dynamic continual learning. Additionally, we develop a  personalized prompt generator  to generate tailored prompts for each graph node while minimizing the number of prompts needed, leading to constant memory consumption regardless of the graph scale. Extensive experiments on four benchmarks show that  PromptCGL  achieves superior performance against existing CGL approaches while significantly reducing memory consumption. Our code is available at  https://github.com/QiWang98/PromptCGL .

Index Terms:

Graph Neural Networks, Continue Graph Learning, Prompt Learning, Graph Prompt Learning.

I

Introduction

Graphs are prevalent in numerous real-world applications, including social networks, biochemistry, and recommendation systems  [ 2 ,  1 ,  3 ,  4 ] . Consequently, Graph Neural Networks (GNNs) have emerged as powerful tools for processing graph-structured data  [ 1 ,  5 ,  6 ,  7 ] . However, traditional GNN methodologies typically assume static graph structures, which fail to capture the dynamic nature of the real world where graphs evolve continuously  [ 8 ,  9 ,  10 ] . For instance, citation networks continually expand with the publication of new research papers, and co-purchasing networks grow as new categories of products are introduced. This necessitates models that can efficiently incorporate the features and topological information of new graphs in a continuous manner.

(a)  Conceptual illustration of replay-based method (left) and ours (right).

(b)  Reddit Dataset

(c)  Products Dataset

Figure 1:

Main Idea . Replay-based methods,  e.g. , CaT  [ 20 ] , SSM  [ 18 ] , ER-GNN  [ 17 ] , require a memory buffer to store a number of graph nodes per task, which is merged with the incoming graph for model retraining (see (a)). However, they face a severe degradation when the buffer size decreases (see (b) and (c)). In contrast,  PromptCGL  represents a novel prompt-based learning paradigm, which learns a fixed number of prompts for each unique task, and leaves GNNs parameters unchanged during the continual learning process. From (b) and (c),  PromptCGL  shows leading performance, regardless of the size of memory buffer.

Due to the limitations in time overhead and computational resources, retraining the GNN models on entire datasets is impractical. Continual Graph Learning (CGL) thus emerges as a crucial paradigm to address the challenges posed by evolving graphs in the real world  [ 11 ,  12 ,  13 ,  14 ] . Recent advancements in CGL can be broadly categorized into regularization  [ 8 ] , architectural design  [ 15 ,  16 ] , and memory replay-based methods  [ 17 ,  18 ,  19 ] . Among these, replay-based methods have shown state-of-the-art performance by storing sampled graphs in a memory buffer and replaying previous data while learning new tasks, as illustrated in Fig.

1

(a). Despite their effectiveness, replay-based CGL methods encounter two significant limitations. First, these methods demand substantial memory resources to store historical data, leading to performance degradation as buffer sizes decrease  [ 20 ,  21 ] , as demonstrated in Fig.

1

(b) and (c). Second, the storage of historical node information raises privacy concerns, especially in contexts involving sensitive data, such as purchase records in co-purchasing networks  [ 22 ] . These limitations indicate that simply buffering past data and retraining the model is not the optimal approach for CGL.

With the success of foundation models, prompt learning has emerged as a key approach for transfer learning in large models  [ 24 ,  23 ] . It shifts the focus from directly tuning model weights to designing prompts that effectively instruct the model to perform specific tasks while keeping the number of parameters fixed  [ 26 ,  25 ] . Recently, significant advancements in prompt learning for natural language processing and computer vision have inspired its application in graph learning  [ 28 ,  29 ,  27 ,  30 ,  31 ] . These studies use graph prompts to bridge the gap between pre-trained pretext tasks and various downstream graph tasks on the same graph  [ 30 ] . By leveraging easily obtainable graph information as pretext tasks, these methods pre-train GNNs, and then learn the prompts to reformulate other types of graph tasks as the pretext task  [ 31 ] . This highlights the potential of prompt learning to transfer knowledge across different tasks on the graph.

However, the challenges posed by CGL are distinct, making existing graph prompt learning methods unsuitable for CGL scenarios. CGL requires the ability to continuously learn across  dynamic, incremental  settings, where new task graphs introduce  unseen classes and varying topologies . This fundamentally differs from the  static, non-incremental  setting of previous graph prompt learning methods, which focus on addressing task differences ( e.g. , node classification and edge prediction) within the same graph. Existing graph prompting methods are primarily designed to bridge the gap between different task types, without considering the variations in node features and topological structures that occur between different graphs. Additionally, these methods typically focus on transferring pre-trained models to downstream tasks, without addressing the critical issue of catastrophic forgetting, a central challenge in CGL setting.

Motivated by the above analysis, we propose a novel approach  PromptCGL  to explore prompt techniques in CGL. The basic idea of  PromptCGL  is to learn a set of unique prompts for each task to encode task-specific knowledge, and maintain graph neural networks mostly unchanged. This alleviates catastrophic forgetting and avoids privacy issues associated with retaining historical information. More concretely, we propose  hierarchical prompting  to instruct the model from both feature- and topology-level to address the variability of task graphs in CGL fully. Additionally, to minimize memory consumption,  PromptCGL  develops a personalized prompt generator that produces personalized prompts based on different queries while maintaining a small prompt set, ensuring consistent memory usage regardless of graph scale. Furthermore, in the CGL setting, we implement a parameter-efficient prompt-tuning strategy: we freeze the pre-trained GNN weights of all layers except the prediction layer, updating only the prediction layer’s parameters and prompt-related parameters.

To summarize,