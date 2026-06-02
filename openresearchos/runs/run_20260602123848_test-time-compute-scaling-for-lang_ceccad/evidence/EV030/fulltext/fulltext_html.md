[2506.08552] Efficient Post-Training Refinement of Latent Reasoning in Large Language Models

Efficient Post-Training Refinement of Latent Reasoning

in Large Language Models

Xinyuan Wang 1  ,
 Dongjie Wang 2  ,
 Wangyang Ying 1  ,
 Haoyue Bai 1  ,

Nanxu Gong 1  ,
 Sixun Dong 1  ,
 Kunpeng Liu 3  ,
 Yanjie Fu 1

1 Arizona State University,
 2 University of Kansas,
 3 Portland State University

{xwang735, wying4, haoyuebai, ngong6, sixundong, yanjie.fu}@asu.edu

wangdongjie@ku.edu, kunpeng@pdx.edu

Abstract

Reasoning is a key component of language understanding in Large Language Models.
While Chain-of-Thought prompting enhances performance via explicit intermediate steps, it suffers from sufficient token overhead and a fixed reasoning trajectory, preventing step-wise refinement.
Recent advances in latent reasoning address these limitations by refining internal reasoning processes directly in the model’s latent space, without producing explicit outputs.
However, a key challenge remains: how to effectively update reasoning embeddings during post-training to guide the model toward more accurate solutions.
To overcome this challenge, we propose a lightweight post-training framework that refines latent reasoning trajectories using two novel strategies:
1) Contrastive reasoning feedback, which compares reasoning embeddings against strong and weak baselines to infer effective update directions via embedding enhancement;
2) Residual embedding refinement, which stabilizes updates by progressively integrating current and historical gradients, enabling fast yet controlled convergence.
Extensive experiments and case studies are conducted on five reasoning benchmarks to demonstrate the effectiveness of the proposed framework.
Notably, a +5% accuracy gain on MathQA without additional training.
Code and data are publicly available at  this link .

Efficient Post-Training Refinement of Latent Reasoning

in Large Language Models

Xinyuan Wang 1 ,
Dongjie Wang 2 ,
Wangyang Ying 1 ,
Haoyue Bai 1 ,

Nanxu Gong 1  ,
 Sixun Dong 1  ,
 Kunpeng Liu 3  ,
 Yanjie Fu 1

1 Arizona State University,
 2 University of Kansas,
 3 Portland State University

{xwang735, wying4, haoyuebai, ngong6, sixundong, yanjie.fu}@asu.edu

wangdongjie@ku.edu, kunpeng@pdx.edu

1  Introduction

Reasoning serves as a fundamental capability in Large Language Models (LLMs), enabling them to comprehend prompts and effectively solve complex tasks.
Existing approaches, such as Chain-of-Thought (CoT)  Wei et al. ( 2022b )  and ReAct  Yao et al. ( 2023b ) , guide models toward correct answers by explicitly generating intermediate textual reasoning steps.
While these methods have shown effectiveness, they suffer from: 1) the explicit reasoning steps cause substantial token overhead, leading to increased computational cost; 2) the reasoning trajectory becomes fixed once the template is generated, preventing step-by-step refinement during the generation process.

Recent advances have partially addressed them by converting explicit reasoning steps into latent embeddings, enabling latent reasoning in models, such as Coconut  Hao et al. ( 2024 ) .
They represent the reasoning state using the LLM’s hidden state (i.e., “continuous thought”) and recursively feed it back into the model in the latent space to enable more effective reasoning.
However, there are two critical challenges:
1) the reasoning trajectory in the latent space lacks explicit directional guidance, making it difficult to ensure consistent progression toward more accurate reasoning states;
2) the recursive embedding updates tend to be unstable, especially across multiple reasoning steps, which may compromise both robustness and accuracy.
These challenges motivate us to explore how reasoning embeddings can be effectively and efficiently updated during post-training to guide the model toward more accurate solutions.

To this end, we draw inspiration from two complementary lines of research.
For the challenge of providing directional guidance in reasoning embedding updates, we are inspired by reinforcement learning from human feedback (RLHF)  Ouyang et al. ( 2022 ) , where learning from relative performance comparisons has demonstrated superior efficiency and effectiveness compared to relying solely on absolute supervision.
For the challenge of stabilizing recursive updates, we take inspiration from the success of momentum-based optimization techniques in deep learning  Qian ( 1999 ) , which demonstrate the importance of adaptively integrating historical and current information to achieve smoother and more stable convergence.

Thus, we propose a lightweight post-training framework to refine the latent reasoning embeddings, built upon two novel strategies:

•

Contrastive Reasoning Feedback Search. 
To infer updated directions in the latent reasoning space, we pass the current reasoning embedding through both a strong and a weak LLM to obtain enhanced embeddings.
We derive a contrastive direction by comparing the outputs of strong and weak models, and use its gradient with respect to the current embedding to guide the reasoning embedding update.

•

Residual Embedding Refinement. 
To ensure stable updates in the latent reasoning space, we blend the current reasoning embedding with its previous state using a residual weighting parameter. This interpolation smooths the transition between steps and prevents abrupt shifts in the reasoning process. As a result, the model achieves more consistent convergence across multi-step latent reasoning.

We empirically evaluate our method through comprehensive experiments and case studies, highlighting its effectiveness, efficiency, and scalability across diverse settings. These experiments demonstrate that our strategies significantly enhance reasoning performance compared to latent-only and explicit token-based reasoning baselines. Notably, on the MathQA task, our approach improves accuracy by over 5% compared to the original latent reasoning method. We further conduct case studies to illustrate how the latent embedding evolves step by step. The results show that the embedding progresses toward more accurate reasoning solutions.

These empirical findings not only validate the effectiveness of our approach but also highlight its practical value.
Our framework offers three key advantages:  Efficiency and Cost-Effectiveness.  The proposed method enhances reasoning performance via a lightweight post-training refinement process. It does not require any modification to the model architecture or parameters, enabling consistent improvements with minimal cost.  Dynamic Post-Training Adaptation.  Both components operate after training to refine the reasoning process. By preserving informative latent states and exploring better latent representations, the model dynamically adjusts its internal reasoning trajectories without requiring additional training.  Training-Free Deployment.  Our refinement procedure is entirely training-free: it relies solely on forward computation in the latent space and avoids any backpropagation or parameter updates. This makes the method easy to integrate into existing models as a plug-and-play component at the post-training stage.

2  Preliminary

2.1  Problem Definition

We focus on complex reasoning tasks where a large language model (LLM) generates a correct answer

y  y

from an input question

x  x

, such as in math word problems, multi-hop question answering, and commonsense reasoning.
These tasks typically require multiple inference steps, even if such steps are not explicitly annotated.
Formally, the objective is to learn a function

f  :

x  →  y

f:x\rightarrow y

, where intermediate cognitive states are latent and only the final answer is observed.
While optional intermediate steps can be included during training, they are often unavailable during inference.
Building on the latent reasoning framework, we represent each reasoning step as an embedding in a latent space.
Our key cont