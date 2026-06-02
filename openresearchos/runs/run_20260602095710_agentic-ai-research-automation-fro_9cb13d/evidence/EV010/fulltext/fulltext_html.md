[2501.02842] Foundations of GenIR

\equalcont

These authors contributed equally to this work.

\equalcont

These authors contributed equally to this work.

1] \orgdiv Dept. of Computer Science and Technology,  \orgname Tsinghua University,  \city Beijing,  \country China

Foundations of GenIR

\fnm Qingyao  \sur Ai

aiqy@tsinghua.edu.cn

\fnm Jingtao  \sur Zhan

zhanjt20@mails.tsinghua.edu.cn

\fnm Yiqun  \sur Liu

yiqunliu@tsinghua.edu.cn

[

Abstract

The chapter discusses the foundational impact of modern generative AI models on information access (IA) systems. In contrast to traditional AI, the large-scale training and superior data modeling of generative AI models enable them to produce high-quality, human-like responses, which brings brand new opportunities for the development of IA paradigms. In this chapter, we identify and introduce two of them in details, i.e., information generation and information synthesis. Information generation allows AI to create tailored content addressing user needs directly, enhancing user experience with immediate, relevant outputs. Information synthesis leverages the ability of generative AI to integrate and reorganize existing information, providing grounded responses and mitigating issues like model hallucination, which is particularly valuable in scenarios requiring precision and external knowledge. This chapter delves into the foundational aspects of generative models, including architecture, scaling, and training, and discusses their applications in multi-modal scenarios. Additionally, it examines the retrieval-augmented generation paradigm and other methods for corpus modeling and understanding, demonstrating how generative AI can enhance information access systems. It also summarizes potential challenges and fruitful directions for future studies.

The primary distinction between modern generative models and traditional AI techniques lies in their capability to generate complicated and high-quality output based on human instructions.
As shown by many studies  [ 1 ,  2 ,  3 ] , modern generative AI models possess remarkable abilities to generate responses that closely mimic human interaction.
General speaking, such impressive performance comes from their large-scale training collections and their advanced data modeling algorithms.
Their superior data understanding ability can benefit almost every components of existing information access systems, from document encoding and index construction, to query processing and relevance analysis, etc.
However, when talking about new opportunities or paradigms that are uniquely brought by the generative AI to information access, they can be broadly categorized in two directions.
The first one is to create content that directly addresses user’s information needs.
By understanding and taking user queries as input instructions, generative AI models are able to generate specific answers or products tailored to the individual’s request.
This direct approach to information generation can significantly enhance user experience by providing immediate and relevant responses.
The second direction is to leverage the advanced instruction-following capabilities of generative AI models to synthesize and recombine existing information in innovative ways.
Generative AI such as large language models (LLMs) can take existing data and transform it into new, coherent pieces of information that may not have been explicitly outlined before.
This ability to reinterpret and organize information opens up new possibilities for retrieval system design and applications.
Therefore, in this chapter, we discuss how generative AI models could help information access from two perspectives, namely  information generation  and  information synthesis .

1  Information Generation

Information need is diverse and typically long-tail. Traditional information retrieval systems, such as search engines and recommendation platforms, are designed to present information that already exists. However, these systems often fall short when it comes to fulfilling the less common information needs. This is particularly evident in scenarios requiring creative creation, where users seek not just information but inspiration and novel ideas. The limitations of traditional information systems in addressing these unique demands have paved the way for the emergence of generative models, which hold the promise of creating new information that aligns closely with the long-tail information needs.

In recent years, generative models have made significant developments. For instance, ChatGPT can respond to user questions, Bing enhances its responses with retrieval-augmented generation, and Midjourney generate images based on user prompts, and recommendation systems generate personal contents for different users. The development is mainly driven by the capable model architectures, computational resources, and the large-scale internet data. These elements have facilitated the performance of generative models to new heights. With the continuous efforts on scaling up these elements, the model performance is still rapidly improving. Nowadays, generative models have gradually been integrated into various workflows and everyday life activities.

In this section, we present the foundation of generative models.
This section is organized as follows. Section

1.1

shows the efforts on designing the model architectures for large language models. Section

1.2

discusses how scaling facilitates the development of generative models and its potential future. Section

1.3

presents the different training stages of large language models. Finally, Section

1.4

introduces how large language models are used in multi-modal scenarios.

1.1  Model Architecture

In different generation scenarios like ChatGPT or SoRA, Transformer  [ 4 ]  has emerged as the predominant model structure. It starts with an embedding layer, followed by multiple neural layers. Within each layer, an attention mechanism models the interactions between words, creating contextualized embeddings. The final decision on word generation probabilities is derived by comparing the output embedding with the vocabulary embeddings. We illustrate the model architecture in Figure

1  .
Unlike traditional Recurrent Neural Networks  [ 5 ] , Transformers are capable of modeling long-distance interactions between words directly, which provides a more powerful representational capability. Numerous enhancements to the Transformer architecture have been proposed. In the following, we will explore various modifications to each component of the Transformer, highlighting the advancements that have further improved its efficacy and efficiency.

Figure 1:  Transformer architecture: the overview on the left and the illustration of one layer on the right  [ 4 ] .

1.1.1  Word Embedding

Word embedding module is at the bottom of the Transformer architecture. Initially, a tokenizer breaks down a sentence into tokens, which the Word embedding module then maps into embeddings. These are combined with position embeddings and fed into subsequent neural layers. Recent research on large-scale language models has identified word embeddings as one of the main sources to training instability  [ 6 ] . Particularly in the early stages of training, the gradients of word embeddings are often orders of magnitude larger than those of other parameters. To address this issue,  Le Scao et al. [ 7 ]  introduced a layer normalization immediately after the word embedding layer, stabilizing the distribution effectively. Besides,  Zeng et al. [ 6 ]  opted to scale down the gradients of the word embeddings by an order of magnitude to prevent substantial updates. Both approaches have been proven effective in stabilizing the training of language models at the 100 billion parameter scale. Yet, whether they are still effective for larger models remains to be investigated.

1.1.2  Position Embedding

Position embedding is essential for