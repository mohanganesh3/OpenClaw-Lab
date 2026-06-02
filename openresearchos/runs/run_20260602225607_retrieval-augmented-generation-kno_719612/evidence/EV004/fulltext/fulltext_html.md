[2105.06232] Retrieval-Free Knowledge-Grounded Dialogue Response Generation with Adapters

Retrieval-Free Knowledge-Grounded Dialogue Response Generation with Adapters

Yan Xu, Etsuko Ishii ∗ , Samuel Cahyawijaya, Zihan Liu, Genta Indra Winata,

Andrea Madotto, Dan Su, Pascale Fung

Center for Artificial Intelligence Research (CAiRE)
 The Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong

{yxucb,eishii}@connect.ust.hk, pascale@ece.ust.hk

∗  These two authors contributed equally.

Abstract

To diversify and enrich generated dialogue responses, knowledge-grounded dialogue has been investigated in recent years.
The existing methods tackle the knowledge grounding challenge by retrieving the relevant sentences over a large corpus and augmenting the dialogues with explicit extra information. Despite their success, however, the existing works have drawbacks in inference efficiency. This paper proposes  KnowExpert , a framework to bypass the explicit retrieval process and inject knowledge into the pre-trained language models with lightweight adapters and adapt to the knowledge-grounded dialogue task. To the best of our knowledge, this is the first attempt to tackle this challenge without retrieval in this task under an open-domain chit-chat scenario. The experimental results show that  KnowExpert  performs comparably with some retrieval-based baselines while being time-efficient in inference, demonstrating the effectiveness of our proposed method.  1

1  1 Our code and models are available at  https://github.com/HLTCHKUST/KnowExpert .

1  Introduction

Numerous studies in recent years have established sophisticated techniques to build open-domain dialogue systems. Although such systems can generate fluent and grammatically correct responses based on the dialogue history, they are unsatisfactory compared to human-to-human conversations.
One primary reason is that existing dialogue systems are incapable of understanding and leveraging relevant knowledge, resulting in superficial and un-intelligent responses when they dive into a specific topic  Li et al. ( 2020 ) . To overcome this limitation, many research works have focused on developing knowledge-grounded dialogue (KGD) systems  (Dinan et al.,  2019 ; Chen et al.,  2020 ; Zhao et al.,  2020 ) .

Figure 1:  High-level architecture of the model. Taking a dialogue history as an input, the adapters are inserted upon the GPT-2 layers, acting as the knowledge experts, to enhance the response generation with the help from a topic model which assigns weights over the knowledge experts.

Figure 2:  Illustration of the training procedure, where the thick lined modules are trained while the rest (dash lined) kept frozen in each training step.

To equip the ability to incorporate knowledge, many recently proposed KGD systems  (Lian et al.,  2019 ; Kim et al.,  2019 ; Roller et al.,  2020 ; Chen et al.,  2020 ; Zhao et al.,  2020 )  comprise the following modules: (1) Knowledge Retrieval, for retrieving the related knowledge sentences from a large corpus (e.g., Wikipedia); (2) Knowledge Selection, for selecting the most relevant knowledge sentences for generation; and (3) Knowledge-augmented Generation, for augmenting the retrieved knowledge and conversation history to generate more knowledgeable responses.
The key to this approach is the explicit retrieval phase to enhance the quality of generated responses.

Despite demonstrating remarkable progress and promising performance on the KGD task, the retrieval-based approaches have drawbacks in their efficiency. First, knowledge retrieval in corpora requires a model to search over a large amount of data, consuming considerable memory resources to store the whole knowledge corpus. It also takes additional processing time to retrieve knowledge and conduct further knowledge selection. Second, adding knowledge as additional context to the language generation model also causes significant computation overhead, which slows the language generation process. Efficiency plays an essential role in the practical use of dialogue systems, and it is necessary to limit resource requirements so as to generate responses faster and support more active users.

Recently, large pre-trained language models (LMs) have been shown to have the capability to carry implicit knowledge  Wang et al. ( 2020 ); Lauscher et al. ( 2020 ) , which can be further applied to downstream classification tasks  Shwartz et al. ( 2020 ) . Many existing works have proved that the “knowledge” can be embedded in the pre-training process  (Brown et al.,  2020 ) . The explorations on the closed-book question answering (QA) task  Petroni et al. ( 2019 ); Roberts et al. ( 2020 ); Wang et al. ( 2021 )  with large pre-trained LMs also indicates the potential of leveraging the knowledge embedded inside LMs. For task-oriented dialogue systems,  Madotto et al. ( 2020 )  store knowledge bases (KBs) of different sizes directly into the model parameters by aligning auto-extracted dialogue templates with the corresponding KBs for each data sample. Based on their success in other tasks, LMs have potential to apply their implicit knowledge for open-domain KGD tasks. However, our scenario is different from both the closed-book QA and task-oriented dialogue tasks, where given a question or user query, relevant knowledge choices are highly constrained by the inputs. In contrast, open-domain chit-chat suffers much from the one-to-many issue between the inputs and possible outputs. In other words, given the inputs on a specific topic, the choice of knowledge candidates is varying, which brings new challenges to embedding knowledge in this task.

Inspired by the previous explorations on other tasks, we propose to tackle the KGD challenge by using the implicit knowledge in LMs under the open-domain chit-chat scenario.
In contrast to existing KGD systems, we bypass the retrieval step and propose a framework,  KnowExpert , to learn the knowledge corpus with the parameters of pre-trained LMs and incorporate the acquired knowledge for KGD generation. In the model, lightweight adapters  Bapna and Firat ( 2019 )  are inserted into the pre-trained GPT-2  Radford et al. ( 2019 ) , acting as knowledge experts. Taking advantage of latent topics, the knowledge sentences are embedded into different knowledge experts by pseudo-conversation style training, while the latent topics measure the relevance between the dialogue samples and the clusters.
We thus fine-tune LM layers where frozen pre-trained adapters are inserted for task adaptation.
Experimental results show that  KnowExpert performs comparably with some strong retrieval-based baselines, while its inference process is much more efficient since extra knowledge sentences are not required as a component of the inputs.

Our contributions are three-fold: (1) to the best of our knowledge, we are the first to explore learning prior knowledge with generative models for KGD tasks under open-domain chit-chat scenario; (2) our model bypass an explicit knowledge retrieval process, and has constant inference time regardless of the size of the knowledge corpus; and (3) our model performs comparably with some strong baselines and shows that a purely generation-based method for the KGD task is promising.

Figure 3:  The demonstration of the procedure of converting a document in the knowledge corpus (e.g., a Wikipedia article) into the pseudo-conversation style. First, the article is split into sentences so that each to represent one utterance. Then, random two utterances are permuted to avoid over-fitting (presented with dashed lines).

2  Related Work

2.1  Knowledge-Grounded Dialogue

The KGD task requires models to identify relevant knowledge sentences from a large corpus for grounding the response generation.
Information retrieval (IR) systems, such as TF-IDF, can quickly retrieve related knowledge over a large corpus. However, the effectiveness is limited as they can only be lev