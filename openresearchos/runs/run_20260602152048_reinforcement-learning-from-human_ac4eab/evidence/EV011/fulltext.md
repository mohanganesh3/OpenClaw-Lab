[2410.16714] Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Models Alignment

Magnetic Preference Optimization: Achieving Last-iterate Convergence for Language Models Alignment

Mingzhi Wang  1,2  , Chengdong Ma  1  , Qizhi Chen  1  ,
Linjian Meng  3  , Yang Han  4

Jiancong Xiao  5  , Zhaowei Zhang  1  , Jing Huo  3  , Weijie J. Su  5  ,
Yaodong Yang  1

1 Institute for Artificial Intelligence, Peking University

2 Beijing Academy of Artificial Intelligence

3 National Key Laboratory for Novel Software Technology, Nanjing University

4 China Telecom,  5 University of Pennsylvania

Correspondence to: Yaodong Yang

&lt;

&lt;

yaodong.yang@pku.edu.cn

&gt;

&gt;

Abstract

Self-play methods have demonstrated remarkable success in enhancing model capabilities across various domains. In the context of Reinforcement Learning from Human Feedback (RLHF), self-play not only boosts Large Language Model (LLM) performance but also overcomes the limitations of traditional Bradley-Terry (BT) model assumptions by finding the Nash equilibrium (NE) of a preference-based, two-player constant-sum game. However, existing methods either guarantee only average-iterate convergence, incurring high storage and inference costs, or converge to the NE of a regularized game, failing to accurately reflect true human preferences. In this paper, we introduce Magnetic Preference Optimization (MPO), a novel approach capable of achieving last-iterate convergence to the NE of the original game, effectively overcoming the limitations of existing methods. Building upon Magnetic Mirror Descent (MMD), MPO attains a linear convergence rate, making it particularly suitable for fine-tuning LLMs. To ensure our algorithm is both theoretically sound and practically viable, we present a simple yet effective implementation that adapts the theoretical insights to the RLHF setting. Empirical results demonstrate that MPO can significantly enhance the performance of LLMs, highlighting the potential of self-play methods in alignment.

1  Introduction

Self-play has emerged as an effective method for improving model performance, particularly in domains that require strategic decision-making and complex problem-solving  (Silver et al.,  2017 ; Vinyals et al.,  2019 ; Perolat et al.,  2021 ) . By allowing models to iteratively refine their strategies through self-competition, self-play enables them to discover optimal policies. In the realm of Reinforcement Learning from Human Feedback (RLHF)  (Ouyang et al.,  2022 ; Peng et al.,  2023 ; Achiam et al.,  2023 ) , self-play not only has proven effective in enabling Large Language Models (LLMs) to better align with human preferences  (Chen et al.,  2024 ; Wu et al.,  2024 ; Zhang et al.,  2024 ) , but also offers unique advantages by addressing the limitations of traditional preference modeling methods  (Munos et al.,  2023 ; Swamy et al.,  2024 ) .

Conventional RLHF methods typically rely on the Bradley-Terry (BT)  (Bradley &amp; Terry,  1952 )  assumption for preference modeling, which presumes transitivity in human preferences—if response  A  is preferred over  B , and  B  over  C , then  A  should also be preferred over  C . While this may hold for individuals in specific contexts, generalizing transitive preferences across broader populations often fails due to the presence of non-transitive preferences  (Swamy et al.,  2024 ) . This limitation undermines the ability of existing RLHF methods to capture the complexity of human preferences. Self-play, however, offers a solution by finding the Nash equilibrium (NE) of a two-player constant-sum game based on human preferences  (Munos et al.,  2023 ; Swamy et al.,  2024 ) .

Despite its promise, self-play in the context of LLM alignment presents unique challenges. Most existing methods, such as Self-Play Preference Optimization (SPO)  (Swamy et al.,  2024 ) , rely on Mirror Descent (MD)  (Beck &amp; Teboulle,  2003 )  based Deep RL methods like PPO  (Schulman et al.,  2017 )  and SAC  (Haarnoja et al.,  2018 )  to learn the NE of the preference-based game. However, from a theoretical perspective, MD only guarantees average-iterate convergence to the NE, while the last-iterate policy tends to oscillate around the NE  (Mertikopoulos et al.,  2018b ;  a ; Perolat et al.,  2021 ) . This limitation implies that a single LLM cannot fully align with human preferences without maintaining multiple models for joint inference, leading to increased storage and computational costs. As shown in Figure

1  , where the duality gap measures the distance between the current policy and the NE, classic Deep RL methods exhibit poor last-iterate convergence, even in a simple Kuhn Poker game. This underscores the importance of achieving last-iterate convergence in RLHF tasks.

Figure 1:  Kuhn Poker Experiments.

On the other hand, Nash Learning from Human Feedback (NLHF)  (Munos et al.,  2023 )  also leverages MD but achieves last-iterate convergence by employing a geometric mixture of the current policy and a reference poliy, commonly referred to as a first-order approximation of the reference policy  (Munos et al.,  2023 ) . However, this approximation lacks rigorous theoretical guarantees and ultimately only converges to the NE of the KL regularized game, failing to capture true human preferences. In summary, existing methods fail to obtain a single LLM policy that aligns with human preferences in the original game. The reliance on multiple LLMs as proxies leads to inefficiency and high cost  (Swamy et al.,  2024 ; Wu et al.,  2024 ; Rosset et al.,  2024 ) , while various approximation methods result in misalignment  (Munos et al.,  2023 ; Calandriello et al.,  2024 ; Zhang et al.,  2024 ) . These limitations collectively represent the core challenges in preference alignment of LLMs.

Figure 2:  An illustration of MPO and its background. Non-transitive preferences are prevalent across diverse populations, necessitating a more generalized preference model that frames the alignment problem as a two-player constant-sum game. Existing methods either converge to the NE of a regularized game or require maintaining multiple models. In contrast, MPO achieves last-iterate convergence to the original NE, aligning with diverse human preferences using only a single model.

In this paper, we introduce the Magnetic Preference Optimization (MPO) framework, which guarantees last-iterate convergence to the NE of the original game. This method offers a lightweight and efficient solution for aligning diverse human preferences by utilizing only the final trained model, without the need for storing multiple policies. Specifically, we adapt the insight of Magnetic Mirror Descent (MMD)  (Sokota et al.,  2022 )  to the RLHF context to derive MPO and further established theoretical guarantees for convergence to the original NE. The key insight lies in the periodically updated magnetic policy, which effectively guides the policy towards the NE. Our results show that MPO achieves last-iterate convergence at a significantly faster rate than standard Mirror Descent (MD), with empirical evaluations demonstrating substantial improvements in LLM performance, further emphasizing the potential of self-play methods for preference alignment.

2  Preliminaries

We consider a Large Language Model (LLM) denoted by

π  ∈  Π

𝜋  Π

\pi\in\Pi

and parametrized by

θ  ∈  Θ

𝜃  Θ

\theta\in\Theta

. The model receives a prompt

𝐱  =

[

x  1

,  …  ,

x  n

]

𝐱

subscript  𝑥  1

…

subscript  𝑥  𝑛

\mathbf{x}=[x_{1},\ldots,x_{n}]

, and generates a corresponding output sequence

𝐲  =

[

y  1

,  …  ,

y  m

]

𝐲

subscript  𝑦  1

…

subscript  𝑦  𝑚

\mathbf{y}=[y_{1},\ldots,y_{m}]

. The output

𝐲

𝐲

\mathbf{y}

is sampled from the conditional probability distribution

π

(  ⋅  ∣  𝐱  )

\pi(\cdot\mid\mathbf{x})

. In LLMs,

x  i

subscript  𝑥  𝑖

x_{i}

and

y  i

subscript  𝑦  𝑖

y_{i}

represent individual