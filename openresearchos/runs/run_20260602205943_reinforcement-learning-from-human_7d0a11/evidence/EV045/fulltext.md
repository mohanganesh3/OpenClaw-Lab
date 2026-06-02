[2410.05782] Reinforcement Learning from Imperfect Corrective Actions and Proxy Rewards

Reinforcement Learning from Imperfect Corrective Actions and Proxy Rewards

Zhaohui Jiang, Xuening Feng &amp; Yifei Zhu
 The University of Michigan-Shanghai Jiao Tong University Joint Institute (UM-SJTU JI)

{jiangzhaohui}@sjtu.edu.cn

&amp;Paul Weng
 Electrical and Computer Engineering, Duke Kunshan University

{paul.weng}@dukekunshan.edu.cn

\AND Yan Song, Tianze Zhou, Yujing Hu, Tangjie Lv &amp; Changjie Fan
 NetEase Fuxi AI Lab

Abstract

In practice, reinforcement learning (RL) agents are often trained with a possibly imperfect proxy reward function, which may lead to a human-agent alignment issue
(i.e., the learned policy either converges to non-optimal performance with low cumulative rewards, or achieves high cumulative rewards but in undesired manner).
To tackle this issue, we consider a framework where a human labeler can provide additional feedback in the form of corrective actions,
which expresses the labeler’s action preferences although this feedback may possibly be imperfect as well.
In this setting, to obtain a better-aligned policy guided by both learning signals, we propose a novel value-based deep RL algorithm called  I terative learning from  Co rrective actions and  Pro xy rewards (ICoPro)  1

1  1 We have open-sourced its implementation:  https://github.com/JiangZhaoh/RLHF_CF .

, which cycles through three phases:
(1) Solicit sparse corrective actions from a human labeler on the agent’s demonstrated trajectories;
(2) Incorporate these corrective actions into the Q-function using a margin loss to enforce adherence to labeler’s preferences;
(3) Train the agent with standard RL losses regularized with a margin loss to learn from proxy rewards and propagate the Q-values learned from human feedback.
Moreover, another novel design in our approach is to integrate pseudo-labels from the target Q-network to reduce human labor and further stabilize training.
We experimentally validate our proposition on a variety of tasks (Atari games and autonomous driving on highway).
On the one hand, using proxy rewards with different levels of imperfection, our method can better align with human preferences and is more sample-efficient than baseline methods.
On the other hand, facing corrective actions with different types of imperfection, our method can overcome the non-optimality of this feedback thanks to the guidance from proxy rewards.

1  Introduction

While reinforcement learning (RL) has proved its effectiveness in numerous application domains  (Mnih et al.,  2015 ; Silver et al.,  2017 ; Levine et al.,  2016 ) ,
its impressive achievements are only possible if a high-quality reward signal for the RL agent to learn from is available.
In practice, correctly defining such reward signal is often very difficult (e.g., in autonomous driving).
If rewards are misspecified, the RL agent would generally learn behaviors that are unexpected  (Amodei et al.,  2016b )  and unwanted  (Clark &amp; Amodei,  2016 ; Russell &amp; Norvig,  2016 )  by the system designer, notably due to overoptimization  (Gao et al.,  2023 )  or specification gaming  (Randlov &amp; Alstrom,  1998 ) .
This important issue has been well-recognized in the research community  (Amodei et al.,  2016a )  and is an active research direction  (Ouyang et al.,  2022 ; Skalse et al.,  2022 ) .

Various solutions have been proposed to avoid having to define a reward function, for instance,
behavior cloning  (Pomerleau,  1989 ; Bain &amp; Sammut,  1995 ) , inverse reinforcement learning  (Ng &amp; Russell,  2000 ; Russell,  1998 ) , (inverse) reward design  (Hadfield-Menell et al.,  2017 ) , or reinforcement learning from human feedback  (Busa-Fekete et al.,  2014 ; Christiano et al.,  2017 ) .
However, these approaches could be impractical and inefficient since they may require a perfect demonstrator, a reliable expert to provide correct labels, or assume that the agent only learns from human feedback, which would subsequently require too many (often hard-to-answer) queries for a human to consider.

As an alternative intermediate approach, we propose the following framework in which the RL agent has access to two sources of signals to learn from: proxy rewards and corrective actions.
A  proxy reward  function is an imperfect reward function, that approximately specifies the task to be learned.
A  corrective action  is provided by a (possibly unreliable) human labeler when s/he is queried by the RL agent:
a trajectory segment is shown to the labeler, who can then choose to correct an action performed by the agent by demonstrating another supposedly-good action.

This framework is practical and easily implementable.
Regarding proxy rewards, they are generally easy for system designers to provide.
For instance, they can
(1) learn proxy rewards from (possibly imperfect) demonstration,
(2) manually specify them to roughly express their intention (e.g., supposedly-good actions are rewarded and expected bad actions are penalized), or
(3) only define sparse rewards (e.g., positive reward for reaching a destination and penalty for a crash).
Regarding corrective actions, in contrast to typical demonstrations of whole trajectories, this feedback is usually much easier for the labeler to provide, since humans may not be able to complete a task themselves but can readily offer action preference on some states.

While only learning with one of those two sources of signals has its own limitations, by proposing our framework, we argue (and experimentally demonstrate) that
learning simultaneously from both of them, even though both may be imperfect, can be highly beneficial.
More specifically, on the one hand, solely learning from proxy rewards would either lead to very slow learning (e.g., when proxy rewards are well-aligned with ground-truth rewards but are very sparse) or yield a policy whose performance is not acceptable to the system designer (e.g., when proxy rewards are dense, but misspecified).
On the other hand, solely learning from corrective actions would require too many queries to the human labeler and may lead to suboptimal learned behaviors since the human may be suboptimal.
In contrast, our key insight is that the two sources of signals can complement each other.
Since they are generally imperfect in different state-space regions, bad decisions learned from proxy rewards can be corrected by the human labeler, while the effects of suboptimal corrective actions may be weakened by proxy rewards.
Therefore, learning simultaneously from the two imperfect signals can achieve better behaviors more aligned to the system designer’s goals than using any one of the two alone and can be more data efficient (in terms of both environmental transitions and human queries).

As a proof of concept, we design a novel value-based RL algorithm (see

Figure

1  ), called Iterative learning from Corrective action and Proxy reward (ICoPro).
ICoPro alternates between three steps.
In the first step (

𝚍𝚊𝚝𝚊𝚌𝚘𝚕𝚕𝚎𝚌𝚝𝚒𝚘𝚗

𝚍𝚊𝚝𝚊𝚌𝚘𝚕𝚕𝚎𝚌𝚝𝚒𝚘𝚗

{\tt datacollection}

-phase),
the RL agent interacts with the environment to collect transition data, and then the labeler provides corrective actions on them.
In the second step (

𝚏𝚒𝚗𝚎𝚝𝚞𝚗𝚎

𝚏𝚒𝚗𝚎𝚝𝚞𝚗𝚎

{\tt finetune}

-phase), the RL agent learns to select actions according to this feedback via a margin loss, which can be interpreted as an imitation learning (IL) loss.
In the third step (

𝚙𝚛𝚘𝚙𝚊𝚐𝚊𝚝𝚒𝚘𝚗

𝚙𝚛𝚘𝚙𝚊𝚐𝚊𝚝𝚒𝚘𝚗

{\tt propagation}

-phase), the RL agent is trained to maximize the expected cumulative proxy rewards while further enforcing a margin loss.
This latter loss is expressed on both observed labels (i.e., corrective actions given by the human labeler) and pseudo-labels generated by a trained model (target Q-network).
Imitating such pseudo-labels can be interpreted in two ways: either to reduce the number of queries or to stay