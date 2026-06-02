[2508.19272] RAGaphene A RAG Annotation Platform with Human ENhancements and Edits

RAGaphene

A RAG Annotation Platform with Human ENhancements and Edits

Kshitij Fadnis,
Sara Rosenthal  1

1  footnotemark:

1

,
Maeda Hanafi,
Yannis Katsis,
Marina Danilevsky

{kpfadnis,sjrosenthal}@us.ibm.com

IBM Research - AI

Authors Contributed Equally

Abstract

Retrieval Augmented Generation (RAG) is an important aspect of conversing with Large Language Models (LLMs) when factually correct information is important. LLMs may provide answers that appear correct, but could contain hallucinated information. Thus, building benchmarks that can evaluate LLMs on multi-turn RAG conversations has become an increasingly important task. Simulating real-world conversations is vital for producing high quality evaluation benchmarks. We present  RAGaphene , a chat-based annotation platform that enables annotators to simulate real-world conversations for benchmarking and evaluating LLMs.  RAGaphene  has been successfully used by approximately 40 annotators to build thousands of real-world conversations.

RAGaphene

A RAG Annotation Platform with Human ENhancements and Edits

Kshitij Fadnis  †

†  thanks:  Authors Contributed Equally

,
Sara Rosenthal  1

1  footnotemark:

1

,
Maeda Hanafi,
Yannis Katsis,
Marina Danilevsky

{kpfadnis,sjrosenthal}@us.ibm.com

IBM Research - AI

1  Introduction

Figure 1:  The pipeline of  RAGaphene : A collection of documents and an LLM are chosen as the desired retriever and generator. The user uses  RAGaphene  to create a conversation which can be exported in a structured json format for further review and optional experimentation and analysis.

Figure 2:  Screenshot of  RAGaphene ’s create mode annotated with its main components.

Chat-Based web tools for conversing with Large Language Models (LLMs) such as ChatGPT  OpenAI ( 2024 )  and Claude  Anthropic ( 2024 )  have become extremely popular. One common use is the seeking of factual information where Retrieval Augmented Generation (RAG) is extremely important to ensure the model is answering faithfully to the relevant passages and not hallucinating. Thus, the ability to evaluate the performance of LLMs on multi-turn RAG-based conversations has become increasingly important and largely overlooked until recently  Katsis et al. ( 2025 ); Dziri et al. ( 2022 ); Feng et al. ( 2021 ); Kuo et al. ( 2024 ); Es et al. ( 2024 ) . Multi-Turn RAG is particularly important for enterprise use cases where domains are specific and there may be unique requirements such as specialized retrievers, generators and custom prompts

Sharma et al. ( 2024 ) . Building a challenging high-quality Multi-Turn RAG benchmark requires in-depth fine-grained human annotation that employs these requirements. Existing annotation platforms  BasicAI ( 2019 ); HumanSignal ( 2020 ); LabelBox ( 2024 ); Joko et al. ( 2021 )  include some basic features for annotating conversational data, such as thumbs up/down of questions/responses, adding metadata and tagging entities in the conversation. However, they do not support the creation of conversational multi-turn RAG datasets with real-time agent response generation. The First-Aid platform  Menini et al. ( 2025 )  is a conversational annotation interface which does include real-time agent response generation and editing. However, it doesn’t have any other feedback mechanisms (e.g. thumbs up/down) and does not incorporate a retrieval component - rather the conversations are generated based on a small set of pre-loaded documents as opposed to passages dynamically retrieved from a potentially very large underlying corpus. In contrast to prior work, our annotation tool enables editing/correcting both the relevant passages and output which are both important pieces for simulating real-world conversations.

We present  RAGaphene , a chat-based annotation platform for having conversations with an LLM that are grounded in an existing corpora to ensure faithfulness, primarily for building Multi-turn RAG benchmarks. The pipeline of  RAGaphene  is shown in Figure

1  . Our platform adopts the RAG pipeline to enable a user to chat with an LLM agent with real-time retrieval and generation while providing the ability to improve the conversation when the retriever and/or generator fails. We allow the user to easily integrate with their desired retriever and generator and provide the ability to improve the conversation which enables users to create high-quality conversations that can be used to evaluate and improve RAG systems.

It is desirable for a platform for building and evaluating a challenging multi-turn to have the following:

RAG Chat  RAG-based chat with customizable retrieval and generation is an important scenario, particularly in industry when domain specific corpora is likely.

Enhanced Feedback  Enhanced feedback is desired as typical chat feedback of thumbs up/down is limiting in its use. It does not give the user the opportunity to troubleshoot when the agent response is not satisfactory which is necessary for building high-quality conversations.

Evaluation and Analysis  Domain experts and engineers may want the ability to quickly evaluate how different models perform in their domain by building a small benchmark.

Our contributions are as follows:

•

RAGaphene : A RAG annotation chat platform with integration to many retrievers and generators that can be inter-changed for domain specific use cases.

1

1  1 We plan to release the  RAGaphene  code on github following internal approval.

•

Enhanced user feedback including adjusting passage retrieval and improving/repairing responses and a user study which highlights the importance of these contributions.

•

Real-time small-scale evaluation and analysis of the full RAG pipeline on RAG conversations created in  RAGaphene .

2

RAGaphene  Platform

(a)  Customizable retriever and generator settings

(b)  Response editing functionality

(c)  Response-context overlap highlighting

Figure 3:  Screenshots of selected functionalities of  RAGaphene .

The  RAGaphene  platform pipeline is shown in Figure

1  . It allows for integration using a desired corpus with any retriever (e.g. BM25, Elser  2

2  2  https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html

) and generator (e.g. Llama 3  Grattafiori et al. ( 2024 ) , GPT 4  OpenAI et al. ( 2024 ) ). The default settings used in our work are an ELSERv1 (ElasticSearch 8.10  1

1  footnotemark:

1

) retriever populated with the corpora from MTRAG  Katsis et al. ( 2025 )  and Mixtral 8X7b Instruct  Jiang et al. ( 2024 )  as the generator.  RAGaphene  consists of three main modes: Create, Review and Experiment as well as integration with InspectorRAGet for analysis to provide a full benchmarking and evaluation life-cycle. We adopt a stateless approach to ensure privacy; all conversations can be preserved by exporting in a structured json format.

2.1  Create Mode

In the create mode a user or annotator can use  RAGaphene  as a chat interface (Figure

2  ) to interact with an LLM. The user is first provided with configuration settings on the right hand side to choose the desired retriever and generator (Figure

3(c)  ). In the retriever settings, they can pick a specific collection of documents and adjust settings such as how many passages to return, how to formulate the query, and other more sophisticated features such as how to render retrieved results. In the generator settings, they can choose a model to chat with, adjust the prompt and other settings such as decoding and number of tokens. On the left-hand side, the user can chat with the generator chosen in the settings.
We describe the process of conversation creation at each turn and all of its features including adding question enrichments, answer repair (Figure

3(b)  ), and passage search (Figure

2  ) in Section

3.1  . During conversation creation, we also provide tips to assist the user which appear on the