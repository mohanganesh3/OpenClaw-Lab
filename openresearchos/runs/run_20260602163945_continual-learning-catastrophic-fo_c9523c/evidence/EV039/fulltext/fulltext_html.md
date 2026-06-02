[2410.07812] Temporal-Difference Variational Continual Learning

Temporal-Difference

Variational Continual Learning

Luckeciano C. Melo  1,2

Alessandro Abate  2

Yarin Gal  1

1 OATML, University of Oxford

2 OXCAV, University of Oxford

luckeciano.carvalho.melo@cs.ox.ac.uk

Abstract

A crucial capability of Machine Learning models in real-world applications is the ability to continuously learn new tasks. This adaptability allows them to respond to potentially inevitable shifts in the data-generating distribution over time. However, in Continual Learning (CL) settings, models often struggle to balance learning new tasks (plasticity) with retaining previous knowledge (memory stability). Consequently, they are susceptible to Catastrophic Forgetting, which degrades performance and undermines the reliability of deployed systems. Variational Continual Learning methods tackle this challenge by employing a learning objective that recursively updates the posterior distribution and enforces it to stay close to the latest posterior estimate. Nonetheless, we argue that these methods may be ineffective due to compounding approximation errors over successive recursions. To mitigate this, we propose new learning objectives that integrate the regularization effects of multiple previous posterior estimations, preventing individual errors from dominating future posterior updates and compounding over time. We reveal insightful connections between these objectives and Temporal-Difference methods, a popular learning mechanism in Reinforcement Learning and Neuroscience. We evaluate the proposed objectives on challenging versions of popular CL benchmarks, demonstrating that they outperform standard Variational CL methods and non-variational baselines, effectively alleviating Catastrophic Forgetting.

1  Introduction

Figure 1:

Average accuracy across observed tasks in the PermutedMNIST benchmark . The TD-VCL approach, proposed in this work, leads to a substantial improvement against standard VCL and non-variational approaches.

A fundamental aspect of robust Machine Learning (ML) models is to learn from non-stationary sequential data. In this scenario, two main properties are necessary: first, models must learn from new incoming data –- potentially from a different task -– with satisfactory asymptotic performance and sample complexity. This capability is called plasticity. Second, they must retain the knowledge from previously learned tasks, know as memory stability. When this does not happen, and the performance of previous tasks degrades, the model suffers from Catastrophic Forgetting  (Goodfellow et al.,  2015 ; McCloskey &amp; Cohen,  1989 ) . These two properties are the central core of Continual Learning (CL)  (Schlimmer &amp; Fisher,  1986 ; Abraham &amp; Robins,  2005 ) , being strongly relevant for ML systems susceptible to test-time distributional shifts.

Given the critical importance of this topic, extensive literature addresses the challenges of CL in traditional ML methods  (Schlimmer &amp; Fisher,  1986 ; Sutton &amp; Whitehead,  1993 ; McCloskey &amp; Cohen,  1989 ; French,  1999 )  and, more recently, for overparameterized models  (Hadsell et al.,  2020 ; Goodfellow et al.,  2015 ; Serra et al.,  2018 ) . Particularly in this work, we focus on the Bayesian setting, as we argue that it provides a principled framework for learning in online or low-data regimes. We investigate the Variational Continual Learning (VCL) approach  (Nguyen et al.,  2018 ) . As detailed in Section

3  , VCL identifies a recursive relationship between subsequent posterior distributions over tasks. A variational optimization objective then leverages this recursion, which regularizes the updated posterior to stay close to the very latest posterior approximation. Nevertheless, we argue that solely relying on a single previous posterior estimate for building up the next optimization target may be ineffective, as the approximation error propagates to the next update and compounds after successive recursions. If a particular estimation is especially poor, the error will be carried over to the next step entirely, which can dramatically degrade model‘s performance.

In this work, we show that the same optimization objective can be represented as a function of a sequence of previous posterior estimates and task likelihoods. We thus propose a new Continual Learning objective, n-Step KL VCL, that explicitly regularizes the posterior update considering several past posterior approximations. By considering multiple previous estimates, the objective dilutes individual errors, allows correct posterior approximates to exert a corrective influence, and leverages a broader global context to the learning target, reducing the impact of compounding errors over time. Figure

2

illustrates the underlying mechanism.

We further generalize this unbiased optimization target to a broader family of CL objectives, namely Temporal-Difference VCL, which constructs the learning target by prioritizing the most recent approximated posteriors. We reveal a link between the proposed objective and Temporal-Difference (TD) methods, a popular learning mechanism in Reinforcement Learning  (Sutton,  1988 )  and Neuroscience  (Schultz et al.,  1997 ) . Furthermore, we show that TD-VCL represents a spectrum of learning objectives that range from vanilla VCL to n-Step KL VCL. Finally, we present experiments on challenging versions of popular CL benchmarks, demonstrating that they outperform standard VCL and non-variational baselines (as shown in Figure

1  ), effectively alleviating Catastrophic Forgetting.

Figure 2:

An intuitive illustration of how TD-VCL functions in comparison to vanilla VCL . At each timestep

t

𝑡

t

, a new task dataset

𝒟  t

subscript  𝒟  𝑡

\mathcal{D}_{t}

arrives. Both methods aim to learn variational parameters

q  t

​

(  𝜽  )

subscript  𝑞  𝑡

𝜽

q_{t}(\bm{\theta})

over a family of distributions

𝒬

𝒬

\mathcal{Q}

that approximates the true posterior

p  ​

(

𝜽  ∣

𝒟

1  :  t

)

𝑝

conditional  𝜽

subscript  𝒟

:  1  𝑡

p(\bm{\theta}\mid\mathcal{D}_{1:t})

via minimizing the KL divergence

𝒟

K  ​  L

(

q  t

(  𝜽  )

∣  ∣  p

(  𝜽  ∣

𝒟

1  :  t

)

)

\mathcal{D}_{KL}(q_{t}(\bm{\theta})\mid\mid p(\bm{\theta}\mid\mathcal{D}_{1:t}))

. VCL optimization (left) is only constrained by the most recent posterior, which compounds approximation errors from previous estimations and potentially deviates far from the true posterior. TD-VCL (right) is regularized by a sequence of past estimations, alleviating the impact of compounded errors.

2  Related Work

Continual Learning  has been studied throughout the past decades, both in Artificial Intelligence  (Schlimmer &amp; Fisher,  1986 ; Sutton &amp; Whitehead,  1993 ; Ring,  1997 )  and in Neuro- and Cognitive Sciences  (Flesch et al.,  2023 ; French,  1999 ; McCloskey &amp; Cohen,  1989 ) . More recently, the focus has shifted towards overparameterized models, such as deep neural networks  (Hadsell et al.,  2020 ; Goodfellow et al.,  2015 ; Serra et al.,  2018 ; Adel et al.,  2020 ) . Given their powerful predictive capabilities, recent literature approaches CL from a wide range of perspectives. For instance, by regularizing the optimization objective to account for old tasks  (Kirkpatrick et al.,  2016 ; Zenke et al.,  2017 ; Chaudhry et al.,  2018 ) ; by replaying an external memory composed by a set of previous tasks  (Lopez-Paz &amp; Ranzato,  2017 ; Bang et al.,  2021 ; Rebuffi et al.,  2016 ) ; or by modifying the optimization procedure or manipulating the estimated gradients  (Zeng et al.,  2018 ; Javed &amp; White,  2019 ; Liu &amp; Liu,  2022 ) . We refer to  (Wang et al.,  2024 )  for an extensive review of recent approaches. The proposed method in this work is placed between regularization-based and replay-based methods.

Bayesian CL.  In the Bayesian framework, prior methods exploit the recursive relation