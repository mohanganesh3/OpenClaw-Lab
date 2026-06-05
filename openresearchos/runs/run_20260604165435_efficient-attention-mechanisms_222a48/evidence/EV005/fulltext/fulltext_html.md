[2011.02705] Improving Commonsense Question Answering by Graph-based Iterative Retrieval over Multiple Knowledge Sources

\floatsetup

[table]capposition=top
 \newfloatcommand capbtabboxtable[][ \FBwidth ]

Improving Commonsense Question Answering by Graph-based Iterative Retrieval over Multiple Knowledge Sources

Qianglong Chen 1 , Feng Ji 2 , Haiqing Chen 2 , Yin Zhang 1

1 College of Computer Science and Technology, Zhejiang University, China

2 DAMO Academy, Alibaba Group, China

{chenqianglong,zhangyin98}@zju.edu.cn

{zhongxiu.jf, haiqing.chenhq}@alibaba-inc.com

* Corresponding Author: Yin Zhang

Abstract

In order to facilitate natural language understanding, the key is to engage commonsense or background knowledge. However, how to engage commonsense effectively in question answering systems is still under exploration in both research academia and industry. In this paper, we propose a novel question-answering method by integrating multiple knowledge sources, i.e. ConceptNet, Wikipedia, and the Cambridge Dictionary, to boost the performance. More concretely, we first introduce a novel graph-based iterative knowledge retrieval module, which iteratively retrieves concepts and entities related to the given question and its choices from multiple knowledge sources. Afterward, we use a pre-trained language model to encode the question, retrieved knowledge and choices, and propose an answer choice-aware attention mechanism to fuse all hidden representations of the previous modules. Finally, the linear classifier for specific tasks is used to predict the answer. Experimental results on the CommonsenseQA dataset show that our method significantly outperforms other competitive methods and achieves the new state-of-the-art. In addition, further ablation studies demonstrate the effectiveness of our graph-based iterative knowledge retrieval module and the answer choice-aware attention module in retrieving and synthesizing background knowledge from multiple knowledge sources.

1  Introduction

In the past decade, most previous work on question answering systems can be divided into two categories, namely, answering questions within a given context  [ Rajpurkar et al., 2016 ,  Mihaylov et al., 2018 ]  or without any context  [ Talmor et al., 2018 ,  Yang et al., 2018 ,  Sap et al., 2019b ] . Although some methods  [ Lan et al., 2020 ,  Zhang et al., 2020 ]  have been reported to exceed human performance on a few metrics, those methods are seldom involved with external knowledge, such as commonsense or background knowledge, which is necessary for better understanding natural language questions, especially in some settings that do not mention the background knowledge.

To effectively leverage external human-made knowledge graphs, previous methods  [ Chen et al., 2017 ,  Min et al., 2019 ,  Lin et al., 2019a ,  Lv et al., 2019 ,  Ye et al., 2019 ,  Shen et al., 2019 ]  adopt a two-stage framework, in which the first stage is to find knowledge facts related to a given question from a wide range of knowledge sources, and then the second stage is to fuse them with the question to predict the answer.  [ Chen et al., 2017 ]  use a TF-IDF-based Document Retriever to locate relevant paragraphs from Wikipedia documents, and then use a Document Reader to predict the start and end positions of the answer span.  [ Lin et al., 2018 ]  introduce an additional module of Paragraph Selector to remove noised paragraphs that explicitly do not contain an answer.  [ Das et al., 2019 ]  employ a multi-step reasoner to build a new query, then rerank the paragraphs and spot the answer after reading the top paragraph.

Although previous experimental results have proved the effectiveness of incorporating additional knowledge into question answering systems  [ Speer et al., 2017 ,  Sap et al., 2019a ] , one critical issue still remains to be addressed. As shown in Table

1  , in ConceptNet, three tail entities,  midwest ,  countryside  and  illinois , which are also three choices of the example question in CommonsenseQA, are all directly related to the same entity-relation pair

⟨  farmland  ,  AtLocation  ⟩

farmland

AtLocation

\langle\textbf{farmland},\textit{AtLocation}\rangle

. If only ConceptNet is given, it is difficult for machine to make the correct choice, since every choice seems correct. We call this issue as multi-value property of knowledge. It is a common phenomenon for question answering system, and the multi-value property of the entity-relation pair will hurt the performance of existing models.

In order to address the above issue, in this paper, we propose a novel question-answering method over multiple knowledge sources. We argue that it is critically important to engage multiple knowledge sources and establish the precise connection between the required background knowledge and the original question as well as choices, to solve the challenge induced by the multi-value property.

We develop our method from three perspectives: 1) We propose a graph-based iterative retrieval module to narrow and refine the potential useful knowledge facts by hidden relations among entities in the question, inspired by  [ Banerjee et al., 2019 ,  Asai et al., 2019 ,  Qi et al., 2019 ] .
2) Different from previous methods only using Wikipedia or ConceptNet, we adopt another extra knowledge source, namely a dictionary, to provide explanations for entities or concepts. Synthesizing entity or concept explanations and iteratively retrieved knowledge facts can help machine precisely distinguish the deceptive answer choices. 3)
Before feeding hidden representations into a linear classifier for final prediction, we introduce an answer choice-aware attention mechanism to compute attention scores between hidden representations of the given question, retrieved knowledge, and candidate choice, which are encoded through a pre-trained language model.

We evaluated our proposed method on the CommonsenseQA dataset, and compared it to other baseline methods. The experimental results show that our method achieves the new state-of-the-art with an improvement of 1.2% over UnifiedQA  [ Khashabi et al., 2020 ]  in test accuracy. Furthermore, we tested our method using different combinations of external knowledge sources to evaluate the impact of each knowledge source. Furthermore, we carried out ablation studies to evaluate the performance impact of the graph-based iterative knowledge retrieval module and the answer choice-aware attention mechanism on the commonsense question answering task.

The contributions of this paper are summarized as follows:

•

To improve commonsense question answering, we propose a graph-based iterative knowledge retrieval module, which uses the combination of entities and their potential relations pre-defined in ConceptNet, to find the background knowledge related to the question or its choices from multiple knowledge sources. In addition, we introduce an answer choice-aware attention mechanism to fuse the hidden representation of question, extended knowledge and choices.

•

To the best of our knowledge, it is the first time to integrate the explanations of entities or concepts from a dictionary with commonsense or background knowledge to answer questions. Through the extra explanations, our method is able to better distinguish the deceptive answer choices and achieve the new state-of-the-art on the CommonsenseQA dataset.

•

We also qualitatively and quantitatively evaluated the effectiveness of the proposed graph-based iterative knowledge retrieval module and answer choice-aware attention mechanism in commonsense reasoning setting, and conclude that both of them are useful to improve the performance.

Question

James was looking for a good place to buy  farmland .

Where might he look?

Choices

A: midwest B: countryside C: estate D: farming areas E: illinois

Evidences from different knowledge sources

ConceptNet

farmland

AtLocation

midwest

farmland

AtLocation

count