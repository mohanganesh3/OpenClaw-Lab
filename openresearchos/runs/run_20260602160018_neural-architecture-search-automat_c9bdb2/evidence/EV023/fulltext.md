[2105.09356] Generative Adversarial Neural Architecture Search

Generative Adversarial Neural Architecture Search

Seyed Saeed Changiz Rezaei 1

*

*  * Equal Contribution. Correspondence to: Di Niu ¡dniu@ualberta.ca¿, Seyed Rezaei ¡seyeds.rezaei@huawei.com¿.

Fred X. Han  1∗

Di Niu 2

Mohammad Salameh 1

Keith Mills 2

Shuo Lian 3

Wei Lu 1 &amp;Shangling Jui 3

1 Huawei Technologies Canada Co., Ltd.

2 Department of Electrical and Computer Engineering, University of Alberta

3 Huawei Kirin Solution, Shanghai, China

Abstract

Despite the empirical success of neural architecture search (NAS) in deep learning applications, the optimality, reproducibility and cost of NAS schemes remain hard to assess. In this paper, we propose Generative Adversarial NAS (GA-NAS) with theoretically provable convergence guarantees, promoting stability and reproducibility in neural architecture search. Inspired by importance sampling, GA-NAS iteratively fits a generator to previously discovered top architectures, thus increasingly focusing on important parts of a large search space. Furthermore, we propose an efficient adversarial learning approach, where the generator is trained by reinforcement learning based on rewards provided by a discriminator, thus being able to explore the search space without evaluating a large number of architectures. Extensive experiments show that GA-NAS beats the best published results under several cases on three public NAS benchmarks. In the meantime, GA-NAS can handle ad-hoc search constraints and search spaces. We show that GA-NAS can be used to improve already optimized baselines found by other NAS methods, including EfficientNet and ProxylessNAS, in terms of ImageNet accuracy or the number of parameters, in their original search space.

1  Introduction

Neural architecture search (NAS) improves neural network model design by replacing the manual trial-and-error process with an automatic search procedure, and has achieved state-of-the-art performance on many computer vision tasks  Elsken  et al.  ( 2018 ) .
Since the underlying search space of architectures grows exponentially as a function of the architecture size, searching for an optimum neural architecture is like looking for a needle in a haystack.
A variety of  search algorithms  have been proposed for NAS, including random search (RS)  Li and Talwalkar ( 2020 ) , differentiable architecture search (DARTS)  Liu  et al.  ( 2018 ) , Bayesian optimization (BO)  Kandasamy  et al.  ( 2018 ) , evolutionary algorithm (EA)  Dai  et al.  ( 2020 ) , and reinforcement learning (RL)  Pham  et al.  ( 2018 ) .

Despite a proliferation of NAS methods proposed, their sensitivity to random seeds and reproducibility issues concern the community  Li and Talwalkar ( 2020 ) ,  Yu  et al.  ( 2019 ) ,  Yang  et al.  ( 2019 ) .
Comparisons between different search algorithms, such as EA, BO, and RL, etc., are particularly hard, as there is no shared search space or experimental protocol followed by all these NAS approaches.
To promote fair comparisons among methods, multiple NAS benchmarks have recently emerged,
including NAS-Bench-101  Ying  et al.  ( 2019 ) , NAS-Bench-201  Dong and Yang ( 2020 ) , and NAS-Bench-301  Siems  et al.  ( 2020 ) , which contain collections of architectures with their associated performance. This has provided an opportunity for researchers to fairly benchmark search algorithms (regardless of the search space in which they are performed) by evaluating how many queries to architectures an algorithm needs to make in order to discover a top-ranked architecture in the benchmark set  Luo  et al.  ( 2020 ); Siems  et al.  ( 2020 ) .
The number of queries converts to an indicator of how many architectures need be evaluated in reality, which often forms the bottleneck of NAS.

In this paper, we propose Generative Adversarial NAS (GA-NAS), a provably converging and efficient search algorithm to be used in NAS based on adversarial learning.
Our method is first inspired by the  Cross Entropy  (CE) method  Rubinstein and
Kroese ( 2013 )  in importance sampling, which iteratively retrains an architecture generator to fit to the distribution of winning architectures generated in previous iterations so that the generator will increasingly sample from more important regions in an extremely large search space. However, such a generator cannot be efficiently trained through back-propagation, as performance measurements can only be obtained for discretized architectures and thus the model is not differentiable. To overcome this issue, GA-NAS uses RL to train an architecture generator network based on RNN and GNN. Yet unlike other RL-based NAS schemes, GA-NAS does not obtain rewards by evaluating generated architectures, which is a costly procedure if a large number of architectures are to be explored. Rather, it learns a discriminator to distinguish the winning architectures from randomly generated ones in each iteration. This enables the generator to be efficiently trained based on the rewards provided by the discriminator, without many true evaluations. We further establish the convergence of GA-NAS in a finite number of steps, by connecting GA-NAS to an importance sampling method with a symmetric Jensen–Shannon (JS) divergence loss.

Extensive experiments have been performed to evaluate GA-NAS in terms of its convergence speed, reproducibility and stability in the presence of random seeds, scalability, flexibility of handling constrained search, and its ability to improve already optimized baselines. We show that GA-NAS outperforms a wide range of existing NAS algorithms, including EA, RL, BO, DARTS, etc., and state-of-the-art results reported on three representative NAS benchmark sets, including NAS-Bench-101, NAS-Bench-201, and NAS-Bench-301—it consistently finds top ranked architectures within a lower number of queries to architecture performance. We also demonstrate the flexibility of GA-NAS by showing its ability to incorporate ad-hoc hard constraints and its ability to further improve existing strong architectures found by other NAS methods.
Through experiments on ImageNet, we show that GA-NAS can enhance EfficientNet-B0  Tan and Le ( 2019 )  and ProxylessNAS  Cai  et al.  ( 2018 )  in their respective search spaces, resulting in architectures with higher accuracy and/or smaller model sizes.

2  Related Work

A typical NAS method includes a  search phase  and an  evaluation phase .
This paper is concerned with the search phase, of which the most important performance criteria are robustness, reproducibility and search cost.
DARTS  Liu  et al.  ( 2018 )  has given rise to numerous optimization schemes for NAS  Xie  et al.  ( 2018 ) . While the objectives of these algorithms may vary, they all operate in the same or similar search space.
However,  Yu  et al.  ( 2019 )  demonstrates that DARTS performs similarly to a random search and its results heavily dependent on the initial seed.
In contrast, GA-NAS has a convergence guarantee under certain assumptions and its results are reproducible.

NAS-Bench-301  Siems  et al.  ( 2020 )  provides a formal benchmark for all

10  18

superscript  10  18

10^{18}

architectures in the DARTS search space.
Preceding NAS-Bench-301 are NAS-Bench-101  Ying  et al.  ( 2019 )  and NAS-Bench-201  Dong and Yang ( 2020 ) .
Besides the cell-based searches, GA-NAS also applies to macro-search  Cai  et al.  ( 2018 ); Tan  et al.  ( 2019 ) , which searches for an ordering of a predefined set of blocks.

On the other hand, several RL-based NAS methods have been proposed.
ENAS  Pham  et al.  ( 2018 )  is the first Reinforcement Learning scheme in weight-sharing NAS.
TuNAS  Bender  et al.  ( 2020 )  shows that guided policies decisively exceed the performance of random search on vast search spaces.
In comparison, GA-NAS proves to be a highly efficient RL solution to NAS, since the rewards used to train the actor come from the discriminator instead of from cost