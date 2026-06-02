[2402.05859] 1 Introduction

marginparsep has been altered.
 topmargin has been altered.
 marginparpush has been altered.

The page layout violates the ICML style. 
Please do not change the page layout, or include packages like geometry,
savetrees, or fullpage, which change it for you.
We’re not able to reliably undo arbitrary changes to the style. Please remove
the offending package(s), or layout-changing commands and try again.

Learning to Route Among Specialized Experts for Zero-Shot Generalization

Mohammed Muqeeth

1

Haokun Liu

2

3

Yufan Liu

1

Colin Raffel

2

3

†

†  footnotetext:

1 University of North Carolina at Chapel Hill
 2 University of Toronto
 3 Vector Institute.
Correspondence to: Mohammed Muqeeth &lt;muqeeth101@gmail.com&gt;, Haokun Liu &lt;haokunliu412@gmail.com&gt;, Colin Raffel &lt;craffel@gmail.com&gt;.

Abstract

Recently, there has been a widespread proliferation of “expert” language models that are specialized to a specific task or domain through parameter-efficient fine-tuning.
How can we recycle large collections of expert language models to improve zero-shot generalization to unseen tasks?
In this work, we propose  P ost- H oc  A daptive  T okenwise  G ating  O ver an  O cean of  S pecialized  E xperts (PHATGOOSE), which learns to route among specialized modules that were produced through parameter-efficient fine-tuning.
Unlike past methods that learn to route among specialized models, PHATGOOSE explores the possibility that zero-shot generalization will be improved if different experts can be adaptively chosen for each token and at each layer in the model.
Crucially, our method is  post-hoc  - it does not require simultaneous access to the datasets used to create the specialized models and only requires a modest amount of additional compute after each expert model is trained.
In experiments covering a range of specialized model collections and zero-shot generalization benchmarks, we find that PHATGOOSE outperforms past methods for post-hoc routing and, in some cases, outperforms explicit multitask training (which requires simultaneous data access).
To better understand the routing strategy learned by PHATGOOSE, we perform qualitative experiments to validate that PHATGOOSE’s performance stems from its ability to make adaptive per-token and per-module expert choices.
We release all of our code to support future work on improving zero-shot generalization by recycling specialized experts.  1

1  1

https://github.com/r-three/phatgoose

1  Introduction

The availability of performant pre-trained language models has led to a proliferation of fine-tuned “expert” models that are specialized to a particular task or domain.
Many of these expert models are created through parameter-efficient fine-tuning (PEFT) techniques  Ding et al. ( 2022 ); Lialin et al. ( 2023 ); He et al. ( 2021 ) , which produce a fine-tuned model by adding small “modules” (such as Low-Rank Adapters  (Hu et al.,  2021 )  or (IA) 3  vectors  (Liu et al.,  2022 ) ) that only introduce or modify a small number of parameters.
Specialized PEFT modules can be easily shared due to their small size, which has led to the distribution of an ever-growing number of adapters on various platforms – for example, as of writing over 17,000 adapters based on the  peft  library  Mangrulkar et al. ( 2022 )  have been uploaded to the Hugging Face Model Hub.  1

1  1  https://huggingface.co/models?library=peft

The availability of these PEFT modules makes it cheap and easy to modularly adapt a given pre-trained model to a specific task or domain.

Figure 1:  Average performance of different multitask training and expert routing methods when using the same held-in and held-out tasks as T0  (Sanh et al.,  2021 ) . Notably, our proposed method PHATGOOSE outperforms all past methods for recycling experts as well as explicit multitask training (which requires simultaneous data access) and nearly matches the performance of an oracle routing scheme. See

Section

4

for more details.

Figure 2:  Visualization of how PHATGOOSE learns to route among specialized modules. This diagram shows how routing is learned for a single module at a single layer; a PEFT-based model will often introduce many such modules. Left: After a specialized module (here, shown as LoRA  (Hu et al.,  2021 )  modules) has been trained, it is frozen and a sigmoid gate is trained to choose which activations should be fed into the module. Right: During inference, top-

k

𝑘

k

routing is performed by choosing the modules whose gates have the highest dot product with a given activation.

In the meantime, extremely large-scale language models (LLMs) are now being treated as “general-purpose” AI systems that can perform any task without any task-specific training or adaptation.
This approach stems from the observation that LLMs often exhibit strong  zero-shot generalization , i.e. the ability to perform new tasks that they were not explicitly trained on.
Such zero-shot generalization capabilities are often improved through large-scale multitask fine-tuning (also called “instruction tuning”)  (Sanh et al.,  2021 ; Wei et al.,  2021 ; Mishra et al.,  2022 ) .
Relying on zero-shot generalization stands in stark contrast to the aforementioned approach of training specialized models for each task (via PEFT or otherwise).

The allure of general-purpose language models and the proliferation of specialized PEFT-based models raises a natural question:
Can we leverage a large collection of specialized modules to improve zero-shot generalization of a base language model?
Such an approach is attractive for various reasons:
First, it would provide a path to  decentralized development  of generalist language models, which otherwise require a huge amount of centralized compute  (Kaplan et al.,  2020 ; Hoffmann et al.,  2022 ) .
In addition, it would provide a way to recycle the widespread effort and compute already being expended to create specialized models.
We might hope such an approach might be successful given the extensive evidence that multitask training improves zero-shot generalization  (Sanh et al.,  2021 ; Wei et al.,  2021 ; Mishra et al.,  2022 ) , and combining specialized models could be seen as a form of multitask learning that does not require simultaneous data access.

To tackle this problem, most past work  (Jang et al.,  2023 ; Belofsky,  2023 ; Durbin,  2024 ; Maxine,  2023 )  learns a post-hoc routing strategy by comparing an embedding of the input query to the average embedding of examples in each dataset used to train each expert.
Such methods implicitly assume there is a single expert well-suited for the query and hope that the retrieval algorithm can accurately identify this best expert.
However,  Jang et al. ( 2023 )  showed that such approaches lag behind an “oracle” router that always chooses the best expert for a given query.
To explore alternative routing approaches, we first note that many PEFT methods typically insert small trainable modules in many places across the model (e.g. at each weight matrix  (Hu et al.,  2021 ) ).
Meanwhile, many sparsely gated Mixture-of-Experts models make routing decisions separately for each token  (Shazeer et al.,  2016 ; Fedus et al.,  2022 ; Du et al.,  2022 ) .
In this work, we therefore explore the angle of improving zero-shot generalization through adaptive  per-token  and  per-module  routing.
In doing so, the aggregate model might be better able to generalize to new tasks by using different expert capabilities at different stages and/or for different tokens.
In addition, zero-shot performance would not restrained by that of the single best specialized model and the ability to correctly retrieve it.

Building on this reasoning, we introduce  P ost-Hoc  A daptive  T okenwise  G ating  O ver an  O cean of  S pecialized  E xperts (PHATGOOSE), a post-hoc method that enables zero-shot generalization among specialized models.
PHATGOOSE recycles PEFT modules by i