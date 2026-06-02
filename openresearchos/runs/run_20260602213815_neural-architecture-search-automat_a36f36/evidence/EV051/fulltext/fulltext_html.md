[2001.08437] Multi-objective Neural Architecture Search via Non-stationary Policy Gradient

Multi-objective Neural Architecture Search
 via Non-stationary Policy Gradient

Zewei Chen, Fengwei Zhou, George Trimponias, Zhenguo Li
 Huawei Noah’s Ark Lab
 {chen.zewei, zhoufengwei, g.trimponias, li.zhenguo}@huawei.com

Abstract

Multi-objective Neural Architecture Search (NAS) aims to discover novel architectures in the presence of multiple conflicting objectives.
Despite recent progress, the problem of approximating the full Pareto front accurately and efficiently remains challenging.
In this work, we explore the novel reinforcement learning (RL) based paradigm of non-stationary policy gradient (NPG). NPG utilizes a non-stationary reward function, and encourages a continuous adaptation of the policy to capture the entire Pareto front efficiently. We introduce two novel reward functions with elements from the dominant paradigms of scalarization and evolution.
To handle non-stationarity, we propose a new exploration scheme using cosine temperature decay with warm restarts.
For fast and accurate architecture evaluation, we introduce a novel pre-trained shared model that we continuously fine-tune throughout training.
Our extensive experimental study with various datasets shows that our framework can approximate the full Pareto front well at fast speeds. Moreover, our discovered cells can achieve supreme predictive performance compared to other multi-objective NAS methods, and other single-objective NAS methods at similar network sizes. Our work demonstrates the potential of NPG as a simple, efficient, and effective paradigm for multi-objective NAS.

1  Introduction

Neural Architecture Search (NAS) automatically designs neural architectures, which is otherwise a time-consuming and labor-intensive process ( ? ;  ? ;  ? ;  ? ;  ? ;  ? ;  ? ;  ? ;  ? ).
Traditionally, NAS searches for architectures with maximal predictive performance.
However, in real applications, additional objectives such as inference time and energy consumption must be considered.
The trade-off among different objectives is typically captured by the Pareto front, i.e., the set of Pareto-optimal architectures with the property that no objective can be improved without harming the other objectives.

Figure 1:  Multi-objective NAS for maximizing accuracy (

O  2

subscript  𝑂  2

O_{2}

) and minimizing number of parameters (

O  1

subscript  𝑂  1

O_{1}

).
ADF expands the Pareto front  (left)  from left to right.
ADC expands it in bands  (right) .

Obtaining the full Pareto front accurately and efficiently is challenging for NAS, because the number of possible architectures is typically very large and architecture evaluation is expensive. Most works on multi-objective NAS approximate the Pareto front by relying on the two paradigms of scalarization and evolution. Scalarization combines the multiple objectives into a single one (e.g., by using a weighted sum or product).
However, multiple runs with different scalarizations are required to obtain multiple Pareto-optimal solutions, which is computationally-intensive.
Evolution involves the use of genetic algorithms
 ( ? ;  ? ),
which iteratively update the current population to create a new generation via crossovers, mutations and elitist selection. A recent genetic approach is LEMONADE ( ? ), which can approximate the Pareto front well but at a relatively high search cost. NSGA-Net ( ? ), a different evolutionary method, achieves comparatively lower cost by training sampled architectures for only 25 epochs, which is nevertheless a low fidelity estimate with high bias ( ? ).
This, in turn, can take a toll on the ability to approximate the true Pareto front well.

A critical observation on the two aforementioned para-digms is that it is in principle possible to continuously adapt a single policy to efficiently generate Pareto-optimal architectures. For scalarization, we can adapt the optimal policy for a given scalarization to obtain an optimal policy for a similar scalarization. For evolution, we can
continuously change the policy to produce architectures that Pareto-dominate existing ones.
Inspired by this observation, our work explores the novel paradigm of non-stationary policy gradient (NPG) for approximating efficiently and accurately the full Pareto front.
NPG is built upon an RL framework, and is compatible with existing RL-based NAS ( ? ).
We opt for RL, since it is well-established in traditional NAS and because the reward is easy to customize.
In detail, a controller learns a policy to generate architectures. Unlike traditional RL though, the reward function is non-stationary;
this allows us to efficiently produce the entire Pareto front by continuously adapting the policy according to the changing reward.

NPG-NAS, our proposed framework, introduces two new reward functions.
The first one (ADF) utilizes a target-based desirability function ( ? ) to scalarize the objectives.
By slowly annealing target values for all but one objectives, it gradually uncovers the entire Pareto front in a single run.
The second one (ADC) is based on the Pareto dominance concept: an architecture receives a reward based on whether it dominates or is dominated by other architectures in the current Pareto front.
As shown in Fig.

1  , ADF traverses the entire Pareto front, whereas ADC expands it in bands like evolution.
ADF relies on scalarization, while ADC is based on Pareto dominance which is ubiquitous in evolutionary methods.
But while evolution typically creates a new generation via genetic operations on the best-fit individuals of the current generation as measured by a fitness function, NPG-NAS expands the band by sampling new individuals that are likely to dominate solutions in the current Pareto front using a non-stationary policy that learns from observed non-stationary rewards.

To deal with non-stationarity, we propose a new exploration scheme with a changing Boltzmann temperature to better balance exploration and exploitation, which is inspired by prior work on multi-armed bandits ( ? ).
Furthermore, a common challenge for NAS concerns architecture evaluation.
Techniques such as parameter sharing ( ? ), continuous relaxation ( ? ), and network morphisms ( ? ) have addressed this.
For fast and accurate evaluation, our work introduces a pre-trained shared model which is continuously fine-tuned during search whenever an architecture is sampled. Our pre-trained shared model leads to fast convergence in just 50 steps, as well as high correlation between the validation accuracy during search versus when architectures are trained from scratch.

Our extensive experimental study on CIFAR-10 and CIFAR-100 based on a cell-based search space and with a strong high-budget baseline shows that NPG-NAS can approximate the full Pareto front accurately at fast speeds. Furthermore, by stacking our discovered cells we obtain networks with very high predictive performance.
For LEMONADE and NSGA-Net, the performance from transferring cells searched on CIFAR-10 to other datasets (ImageNet or CIFAR-100) is less striking than their performance on CIFAR-10.
For NPG-NAS, we transfer architecture cells found for CIFAR-10 and CIFAR-100 to ImageNet, yielding higher classification accuracies than most multi-objective methods, or competitive accuracies at a fraction of the GPU cost.
Importantly, both our searched and transferred cells are often on par with or even outperform many state-of-the-art NAS methods with similar network sizes.
This points to the high quality of our discovered cells.
Our experiments unveil fundamental differences in the way architectures are sampled for ADF and ADC versus random search. We show that random search performs poorly in the multi-objective setting, contrary to single-objective NAS ( ? ).

NPG-NAS is a fully gradient-based scheme, as it uses gradient descent to optimize both the network weights and the controller policy. It favors a