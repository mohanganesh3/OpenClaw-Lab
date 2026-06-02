[2402.06734] Corruption Robust Offline Reinforcement Learning with Human Feedback

Corruption Robust Offline Reinforcement Learning

with Human Feedback

Debmalya Mandal
 Dept. of Computer Science
 University of Warwick, UK

Debmalya.Mandal@warwick.ac.uk

Andi Nika
 Max-Planck Institute for

Software Systems, Germany

andinika@mpi-sws.org

Parameswaran Kamalaruban
 Independent Researcher
 London, UK

pkamalaruban@gmail.com

Adish Singla
 Max-Planck Institute for

Software Systems, Germany

adishs@mpi-sws.org

Goran Radanović
 Max-Planck Institute for

Software Systems, Germany

gradanovic@mpi-sws.org

Abstract

We study data corruption robustness for reinforcement learning with human feedback (RLHF) in an offline setting. Given an offline dataset of pairs of trajectories along with feedback about human preferences, an

ε

𝜀

\varepsilon

-fraction of the pairs is corrupted (e.g., feedback flipped or trajectory features manipulated), capturing an adversarial attack or noisy human preferences. We aim to design algorithms that identify a near-optimal policy from the corrupted data, with provable guarantees. Existing theoretical works have separately studied the settings of corruption robust RL (learning from scalar rewards directly under corruption) and offline RLHF (learning from human feedback without corruption); however, they are inapplicable to our problem of dealing with corrupted data in offline RLHF setting. To this end, we design novel corruption robust offline RLHF methods under various assumptions on the coverage of the data-generating distributions. At a high level, our methodology robustifies an offline RLHF framework by first learning a reward model along with confidence sets and then learning a pessimistic optimal policy over the confidence set. Our key insight is that learning optimal policy can be done by leveraging an offline corruption-robust RL oracle in different ways (e.g., zero-order oracle or first-order oracle), depending on the data coverage assumptions. To our knowledge, ours is the first work that provides provable corruption robust offline RLHF methods.

\doparttoc  \faketableofcontents

1  Introduction

Reinforcement Learning from Human Feedback (RLHF) has emerged as a powerful paradigm for addressing complex tasks across diverse domains, ranging from large language models (LLMs) to robotics and game-playing  [ CL+17 ,  Zie+19 ,  SO+20 ,  OW+22 ,  BJ+22 ,  SDB23 ] . At the core of RLHF is its unique ability to model reward functions solely from preference data, making it particularly well-suited for scenarios where explicit reward signals are challenging to define. Following reward model estimation, traditional RLHF approaches employ online reinforcement learning algorithms for subsequent policy optimization. However, the integration of offline RL within the RLHF pipeline holds promise for alleviating limitations inherent to online RL, notably in terms of sample efficiency and safety concerns  [ Lev+20 ,  Kid+20 ] . By incorporating offline RL algorithms, RLHF becomes more adaptable to scenarios where online data collection proves prohibitive, facilitating the reuse of valuable pre-existing datasets  [ SDB23 ] .

The real-world deployment of RLHF faces substantial challenges rooted in the reliability of the preference data, which is integral to its effectiveness. These challenges primarily arise from two sources: adversarial corruption and inherent noise  [ Cas+23 ,  Xue+23 ,  CNL24 ] . Adversarial entities, acting with malicious intent, may deliberately manipulate feedback labels or trajectory features, introducing potential biases in the reward model. Simultaneously, inherent human subjectivity within crowd-sourced preference data can contribute substantial noise, impeding accurate reward estimation. In light of these challenges, a pivotal research question emerges:  Can we devise a robust variant of RLHF that efficiently learns from adversarially corrupted or noisy preference data, exhibiting graceful scalability amidst increasing corruption levels?

Type of Coverage

Suboptimality Gap

Robust RL Oracle

# Oracle Calls

Uniform (

ξ

𝜉

\xi

)

O  ​

(

H  3

+

H  ​  d

ξ

​

ε

1  −

o  ​

(  1  )

)

𝑂

superscript  𝐻  3

𝐻  𝑑

𝜉

superscript  𝜀

1

𝑜  1

O\left(\frac{H^{3}+\sqrt{Hd}}{\xi}\varepsilon^{1-o(1)}\right)

R-LSVI

[ Zha+22 ] , zero-order access

1

Relative Condition Number (

α

𝛼

\alpha

)

O  ~

​

(

H  2

​  d  ​  κ  ​

α  ​  ε

)

~  𝑂

superscript  𝐻  2

𝑑  𝜅

𝛼  𝜀

\widetilde{O}\left(H^{2}d\kappa\sqrt{\alpha\varepsilon}\right)

+

O  ~

​

(

H

5  /  4

​

d

3  /  4

​

(

α  ​  ε

)

1  /  4

)

~  𝑂

superscript  𝐻

5  4

superscript  𝑑

3  4

superscript

𝛼  𝜀

1  4

+\ \widetilde{O}\left(H^{5/4}d^{3/4}(\alpha\varepsilon)^{1/4}\right)

R-LSVI

[ Zha+22 ] , zero-order access

O  ~

​

(

H

3  /  2

​

d  5

ε  3

)

~  𝑂

superscript  𝐻

3  2

superscript  𝑑  5

superscript  𝜀  3

\widetilde{O}\left(\frac{H^{3/2}d^{5}}{\varepsilon^{3}}\right)

Generalized Coverage Ratio (

ν

𝜈

\nu

)

O  ​

(

ν  ​  κ  ​

ε

​

H  2

​

d

3  /  2

)

𝑂

𝜈  𝜅

𝜀

superscript  𝐻  2

superscript  𝑑

3  2

{O}\left(\nu\kappa\sqrt{\varepsilon}H^{2}d^{3/2}\right)

Algorithm

7

( Our method ), first-order access

O  ​

(

1

ε  ​  ν

)

𝑂

1

𝜀  𝜈

O\left(\frac{1}{\varepsilon\nu}\right)

Table 1 :

We design  corruption robust RLHF  through reduction to  corruption robust offline RL  problem. Under uniform coverage and low relative condition number, we use  R-LSVI

[ Zha+22 ]  as an oracle, and obtain suboptimality gap of

O  ​

(

ε

1  −

o  ​

(  1  )

)

𝑂

superscript  𝜀

1

𝑜  1

O(\varepsilon^{1-o(1)})

and

O  ​

(

ε

1  /  4

)

𝑂

superscript  𝜀

1  4

O(\varepsilon^{1/4})

respectively, in terms of

ε

𝜀

\varepsilon

(fraction of corrupted data). Calls to  R-LSVI  are zero-order i.e. we only obtain a robust policy and an estimate of the value function. Under bounded generalized coverage ratio, we design a new robust offline RL method (algorithm (  7  )) that also returns an estimate of the sub-gradient (first order access). Using algorithm (  7  ), we can improve the dependence on

ε

𝜀

\varepsilon

to

O  ​

(

ε

)

𝑂

𝜀

O(\sqrt{\varepsilon})

and also significantly reduce the number of oracle calls.

In this paper, we initiate the study of  corruption-robust offline reinforcement learning from human feedback . Although there are several works on corruption robust offline reinforcement learning  [ Zha+22 ,  Ye+23a ] , and provable preference based reinforcement learning  [ Zha+23 ,  ZJJ23 ] , ours is the first work to combine these two threads and provide provable corruption robust offline RLHF methods. We design corruption robust offline RLHF methods through reduction to corruption robust offline RL methods. In particular, we modify the existing RLHF framework through three steps – (1) Robustly learn a reward model by solving a robust logistic regression problem, (2) Construct a confidence set around the learned model, and (3) learn a pessimistic optimal policy over the confidence set through reduction to offline RL. We instantiate this general framework for datasets with various types of coverage assumptions, and as is often the case in offline RL, different coverage assumptions require different algorithms. For example, under  uniform coverage , the solution to the robust logistic regression (step 1) gives a reward estimate that is

O

(

ε

1  −

o  ​

(  1  )

O(\varepsilon^{1-o(1)}

) close to the true parameter. In this case, just one call to a robust offline RL method is sufficient to obtain an

O

(

ε

1  −

o  ​

(  1  )

O(\varepsilon^{1-o(1)}

)-optimal policy.

However, the problem is significantly harder under weaker coverage assumptions e.g.  low relative condition number . In this case, the reward estimate might not be close to the true parameter. However, we show that their difference in likelihood is bounded, and this observation