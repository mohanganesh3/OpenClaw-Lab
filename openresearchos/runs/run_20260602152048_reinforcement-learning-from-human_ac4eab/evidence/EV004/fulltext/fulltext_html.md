[2503.22480] Probabilistic Uncertain Reward Model: A Natural Generalization of Bradley-Terry Reward Model

Probabilistic Uncertain Reward Model: A Natural Generalization of Bradley-Terry Reward Model

Wangtao Sun  1,2  , Xiang Cheng  3  , Xing Yu  3  , Haotian Xu  3  ,

Zhao Yang  4

,  Shizhu He  1,2

,  Jun Zhao  1,2

,  Kang Liu  1,2,5

1

The Laboratory of Cognition and Decision Intelligence for Complex Systems,

Institute of Automation, Chinese Academy of Sciences, Beijing, China

2

School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China

3

Xiaohongshu Inc

4

Meituan Inc

5

Shanghai Artificial Intelligence Laboratory

Corresponding author: kliu@nlpr.ia.ac.cn

Abstract

Reinforcement Learning from Human Feedback (RLHF) has emerged as a critical technique for training large language models. However, reward hacking—a phenomenon where models exploit flaws in the reward model—remains a significant barrier to achieving robust and scalable intelligence through long-term training. Existing studies have proposed uncertain reward model to address reward hacking, however, they often lack systematic or theoretical foundations,  failing to model the uncertainty intrinsically emerging from preference data . In this paper, we propose the  Probabilistic Uncertain Reward Model (PURM) , a natural generalization of the classical Bradley-Terry reward model. PURM learns reward distributions directly from preference data and quantifies per-sample uncertainty via the average overlap area between reward distributions. To mitigate reward hacking, we further introduce an uncertainty-aware penalty into Proximal Policy Optimization (PPO), which leverages the learned uncertainty to dynamically balance reward optimization and exploration. We propose a lightweight and easy-to-use implementation of PURM. Experiments demonstrate that PURM significantly delays the onset of reward hacking while improving final reward performance, outperforming baseline methods in both stability and effectiveness.

1  Introduction

Reinforcement learning from human feedback (RLHF) has emerged as a critical pathway for aligning LLMs with human values  Hu et al. ( 2024 ) . While reinforcement fine-tuning the LLMs with ground-truth signals (e.g., correct answers in mathematical reasoning, the execution results of codes, rule-based reward in games like  Go ) has demonstrated remarkable success  Guo et al. ( 2025 )  in specialized domains, most real-world alignment tasks lack explicit ground-truth supervision. For these scenarios, reward models (RMs) trained on preference data serve as the primary proxy for guiding policy optimization  Skalse et al. ( 2022 ) .
However, conventional RMs based on deterministic Bradley-Terry reward models suffer from overconfidence when evaluating inconsistently labeled data or out-of-distribution (OOD) samples,
leading to  reward hacking —a pathological divergence where policy optimization blindly maximizes proxy rewards while degrading true performance (BTRM in Figure

1  ).
This failure mode fundamentally limits the potential for sustained capability scaling through RLHF in open-ended domains.

There are existing reward modeling methods that handle reward hacking, such as data-augmentation  Liu et al. ( 2025 ) , reward shaping  Chen et al. ( 2024 ) , model-ensembling  Yan et al. ( 2024 ) , and rely on explicit reward annotation  Lou et al. ( 2024 ) .
Among these methods, uncertain reward model has emerged as a critical way to mitigate reward hacking through quantifying  how much we can trust the reward given by reward model .
However, current approaches to handling uncertainty lack systematic or theoretical underpinnings, such as simplistic averaging or manually designed ad-hoc balancing strategies, therefor fail to  model the uncertainty that intrinsically emerged from the preference data .

Yet currently, the standard Bradley-Terry reward model (BTRM) is only to produce point value reward, collapses the underlying uncertainty into deterministic scalar values, forcing the policy to treat all reward signals as equally reliable. Consequently, the agent will inevitably overfit to the spurious correlations in flawed proxy rewards, terminating exploration prematurely. To enable long-term exploration and robust scaling in RLHF, it is imperative to equip RMs with principled uncertainty quantification.

To solve the above challenges, this work proposes the Probabilistic Uncertain Reward Model (PURM), which is a probabilistic reformulation of reward modeling that intrinsically captures uncertainty from standard preference data. Our key insight is to generalize the Bradley-Terry reward model to characterize reward distributions rather than point estimates. Though comparing PURM with other existing reward modeling methods on preference datasets, we first verify that PURM is an effective reward model.
Based on the modeled reward distribution of PURM, we then introduce the Bhattacharyya Coefficient, to compute the overlap between the reward distributions, thereby enabling the uncertainty estimation for individual prompt-response pair

x  ,  y

x,y

.
To validate the effectiveness of this uncertainty measure, we simulate scenarios that induce aleatoric uncertainty (from inconsistent labeling) and epistemic uncertainty (from OOD samples), demonstrating that PURM indeed generates higher uncertainty values in these critical cases.

To finally allative the problem of reward hacking, we propose to leverage uncertainty to dynamically balance exploration and exploitation during policy training in Proximal Policy Optimization (PPO,  Schulman et al.  2017  ) by penalizing uncertain rewards. We find that this Uncertainty-Aware Reinforcement Learning significantly helps mitigate the reward hacking behavior. As shown in Figure

1  , our proposed PURM guides the policy model to continuously obtain increasing rewards in a longer training period, delaying the reward hacking phenomenon and obtaining a higher maximum reward.
We also implement this framework with minimal code modifications: converting a deterministic BTRM to PURM, the calculation of Bhattacharyya-based uncertainty, and leveraging uncertainty in PPO requires fewer than 10 lines of code, respectively.

Figure 1:  Traditional Bradley-Terry reward model can easily get hacked. The proposed PURM delays reward hacking by 2-3× compared to standard BTRMs while obtaining higher final rewards.

In summary, this paper makes the following foundational contributions:

Probabilistic Uncertain Reward Model : We generalize the Bradley-Terry reward model to Probabilistic Uncertain Reward Model (PURM), and derive the closed-form training objective through the maximum likelihood estimation of BTRM, enabling RMs to learn reward distributions directly from pairwise preferences without any additional annotations or training phases/objectives.

Uncertainty-Aware Reinforcement Learning : We define the trajectory-level uncertainty of a single prompt-response pair through computing the Bhattacharyya coefficient between reward distributions. During training the policy model with PPO, we propose to penalize the reward based on its uncertainty.

Lightweight Implementation : PURM can be implemented through concise modifications to standard Bradley-Terry reward modeling (  A.4  ): 1) Converting traditional reward models to PURMs, 2) leveraging closed-form statistical measures to quantify uncertainty for reward distributions, and 3) adapting uncertainty into PPO, require fewer than 10 lines of code changes, respectively.

Empirical Validation : Our experimental results shows that (1) generalized probabilistic formulation does not compromise the core reward prediction capability of PURM; (2) PURM can effectively detect the aleatoric uncertainty (from inconsistent labeling) and epistemic uncertainty (from OOD samples), and (3) using PURM in PPO training delays reward hacking by 2-3× compared to standard BTRMs while obtaini