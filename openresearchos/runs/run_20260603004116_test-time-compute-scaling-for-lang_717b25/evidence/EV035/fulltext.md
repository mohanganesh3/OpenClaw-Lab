[2510.03199] Best-of-Majority: Minimax-Optimal Strategy for Pass@𝑘 Inference Scaling

Best-of-Majority: Minimax-Optimal
Strategy for Pass@

k  k

Inference Scaling

Qiwei Di ∗

Kaixuan Ji ∗

Xuheng Li ∗

Heyang Zhao

Quanquan Gu

Equal contributionDepartment of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail:  qiwei2000@cs.ucla.edu Department of Computer Science, University of California, Los Angeles, CA 90095, USA; email:  kaixuanji@cs.ucla.edu Department of Computer Science, University of California, Los Angeles, CA 90095, USA; email:  xuheng.li@cs.ucla.edu Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail:  hyzhao@cs.ucla.edu Department of Computer Science, University of California, Los Angeles, CA 90095, USA; e-mail:  qgu@cs.ucla.edu

Abstract

LLM inference often generates a batch of candidates for a prompt and selects one via strategies like majority voting or Best-of-

N  N

(BoN). For difficult tasks, this single-shot selection often underperforms. Consequently, evaluations commonly report Pass@

k  k

: the agent may submit up to

k  k

responses, and only the best of them is used when computing regret.
Motivated by this, we study inference scaling in the more general Pass@

k  k

inference setting, and prove that neither majority voting nor BoN exhibits the desirable scaling with

k  k

and the sampling budget

N  N

.
Combining the advantages of majority voting and BoN, we propose a new inference strategy called Best-of-Majority (BoM), with a pivotal step that restricts the candidates to the responses with high frequency in the

N  N

samples before selecting the top-

k  k

rewards.
We prove that when the sampling budget is

N  =

Ω  ~

​

(

C  ∗

)

N=\widetilde{\Omega}(C^{*})

, the regret of BoM is

O  ​

(

ϵ  opt

+

ϵ  RM  2

​

C  ∗

/  k

)

O(\epsilon_{\mathrm{opt}}+\sqrt{\epsilon_{\mathrm{RM}}^{2}C^{*}/k})

, where

C  ∗

C^{*}

is the coverage coefficient,

ϵ  RM

\epsilon_{\mathrm{RM}}

is the estimation error of the reward model, and

ϵ  opt

\epsilon_{\mathrm{opt}}

is the estimation error of reward at the optimal response.
We further establish a matching lower bound, certifying that our algorithm is minimax optimal. Beyond optimality, BoM has a key advantage: unlike majority voting and BoN, its performance does not degrade when increasing

N  N

.
Experimental results of inference on math problems show BoM outperforming both majority voting and BoN.

1  Introduction

Scaling law serves as a powerful tool for guiding the  training  of large language models (LLMs), providing insight into how increased training compute, data, and model size contribute to performance improvements.
Originating in the early days of deep neural networks  (Hestness et al.,  2017 ; Rosenfeld et al.,  2019 ) , the concept has since demonstrated remarkable predictive power across a variety of domains, including strategic board games  (Jones,  2021 ) , image generation  (Henighan et al.,  2020 ; Yu et al.,  2022 ; Peebles and Xie,  2023 ) , video modeling  (Brooks et al.,  2024 ) , language generation  (Kaplan et al.,  2020 ; Hoffmann et al.,  2022 ; Achiam et al.,  2023 ) , retrieval systems  (Fang et al.,  2024 ; Cai et al.,  2025 ) , and reward modeling  (Gao et al.,  2023 ; Rafailov et al.,  2024 ) . While training-time scaling has proven effective, it is also highly resource-intensive. As a result, increasing attention has been directed toward a complementary paradigm:  inference , which examines how model performance can be improved after training.
This relationship between additional compute at inference time and performance improvement is known as the inference scaling law  (Brown et al.,  2024 ; Snell et al.,  2024 ; Wu et al.,  2024b ; Guo et al.,  2025 ) .

Compared to training-time scaling, inference scaling allows for increasing computational cost in several distinct ways, including expanding the generation input via chain-of-thought prompting  (Wei et al.,  2022 ; Li et al.,  2024 ) , incorporating iterative self-improvement,  (Zheng et al.,  2023 ; Wu et al.,  2024a ) , and applying search-based algorithms  (Yao et al.,  2023 ; Feng et al.,  2023 ; Gao et al.,  2024 ; Zhang et al.,  2024 ) . It can also be realized through repeated sampling, using strategies such as majority voting  (Wang et al.,  2022 ; Lewkowycz et al.,  2022 ; Li et al.,  2023 )  or Best-of-

N  N

(BoN)  (Lightman et al.,  2023 ) . In parallel, a growing line of works has sought to establish theoretical guarantees for inference strategies.  Wu et al. ( 2024b )  provided convergence bounds and rates for the scaling of majority voting algorithms.  Huang et al. ( 2024 )  showed that BoN can achieve self-improvement via a special mechanism called sharpening.  Huang et al. ( 2025 )  analyzed the sample complexity of BoN and proposed a pessimistic inference algorithm with provable benefits.

While most existing analyses focus on inference algorithms that output a single response, there are tasks that allow for multiple candidate outputs, where it is considered solved if any one of them is correct. This setting is captured by the Pass@

k  k

metric  (Li et al.,  2022 ) . Building on this metric, we propose a novel  Pass@

k  k

inference framework, in which the inference algorithm is allowed to generate

N  N

responses and return up to

k  k

of them.
Since

N  &gt;  k

N&gt;k

, the performance depends not only on generating a diverse set of candidates but also on the algorithm’s ability to effectively select the

k  k

outputs that are most likely to be correct.
 Brown et al. ( 2024 )  conducted empirical studies on this inference framework and observed the relationship between the coverage and the performance of the algorithm. However, this work is restricted to the majority voting and BoN inference strategies, and failed to theoretically justify the inference scaling law.

As there have been few works on understanding the scaling of the Pass@

k  k

inference problem, we are motivated to investigate the following fundamental question:

Q1: What is the optimal scaling of the Pass@

k  k

inference problem?

To answer this question, we derive a minimax lower bound as a function of

k  k

that characterizes the fundamental limits of any Pass@

k  k

inference strategy, establishing the theoretical scaling behavior for Pass@

k  k

inference problems.

Going one step further, we also aim to evaluate existing inference strategies for the Pass@

k  k

inference problem and find a strategy that achieves the optimal scaling. Beyond standard metrics like regret and sample complexity, we further introduce a formal definition of  scaling-monotonicity

(Huang et al.,  2025 ) , which captures whether an inference algorithm maintains (or improves) its performance as the number of samples

N  N

increases. This leads to our second question:

Q2: What inference strategies are scaling-monotonic and optimal in the Pass@

k  k

inference setting?

Unfortunately, our analysis reveals that majority voting and BoN are not scaling-monotonic. Furthermore, these methods face fundamental limitations that make it difficult, if not impossible, to attain the optimal regret scaling with respect to

k  k

. To address this issue, we propose a new inference strategy, Best-of-Majority (BoM), which integrates the core ideas of both majority voting and BoN. We establish a regret upper bound for BoM that matches the minimax lower bound, thereby demonstrating that our algorithm is minimax optimal. Please refer to Table

1

for detailed results.

Table 1:  Comparison of Pass@

k  k

inference strategies. Our algorithm BoM is the first minimax-optimal Pass@

k  k

inference strategy. Compared with majority voting and BoN, BoM is scaling-monotonic, indicating that the optimal performance can be achieved with large sampling budget

N  N

, making it preferable when scaling up

N  N

to achieve better performance. Additionally, th