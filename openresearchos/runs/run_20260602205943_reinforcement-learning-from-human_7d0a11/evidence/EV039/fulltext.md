[2409.20016] Personalisation via Dynamic Policy Fusion

1

1  institutetext:  A2I2, Deakin University, Geelong, Australia

2

2  institutetext:  School of IT, Deakin University, Geelong, Australia

2

2  email:  a.palattuparambil@deakin.edu.au

Personalisation via Dynamic Policy Fusion

Ajsal Shereef Palattuparambil

11

Thommen George Karimpanal

22

Santu Rana

11

Abstract

Deep reinforcement learning (RL) policies, although optimal in terms of task rewards, may not align with the personal preferences of human users. To ensure this alignment, a naïve solution would be to retrain the agent using a reward function that encodes the user’s specific preferences. However, such a reward function is typically not readily available, and as such, retraining the agent from scratch can be prohibitively expensive. We propose a more practical approach - to adapt the already trained policy to user-specific needs with the help of human feedback. To this end, we infer the user’s intent through trajectory-level feedback and combine it with the trained task policy via a theoretically grounded dynamic policy fusion approach. As our approach collects human feedback on the very same trajectories used to learn the task policy, it does not require any additional interactions with the environment, making it a zero-shot approach. We empirically demonstrate in a number of environments that our proposed dynamic policy fusion approach consistently achieves the intended task while simultaneously adhering to user-specific needs.

Keywords:  Reinforcement Learning, Personalisation, Dynamic Policy Fusion

1  Introduction

Reinforcement learning (RL) has demonstrated its effectiveness in various real-world scenarios, including computer games  [ 16 ] , inventory management, autonomous driving, robotics  [ 8 ] , healthcare  [ 30 ] , recommendation systems  [ 25 ]  etc. In RL, the learning agent typically learns to maximise its reward by learning an optimal task policy through interactions with the environment. However, in certain scenarios, a user may desire a policy that subtly deviates from this optimal policy to accommodate their personal preference or style. For example, in a navigation setting, users may want to avoid toll roads, or they may prefer to drive along a scenic coastal route, even if this translates to slightly longer travel times compared to the route determined to be the optimal one by an RL agent trained to minimise travel time.

A simple solution to accommodate such preferences in a policy would be to retrain an RL agent from scratch, using a reward function that takes users’ preferences into account. However, such a reward function could be challenging to obtain. In addition, the poor sample efficiency of RL could render this approach infeasible, as it would require the agent to interact with the environment again, which could be expensive in terms of time, energy, etc., We posit that a more practical approach would be to adapt the originally trained policy by personalising it based on a user’s specific preferences on the trajectories already collected during training. This approach to personalisation eliminates the need for additional agent-environment interactions, thereby making it a practically feasible option for adapting already trained policies.

In this work, we present a policy fusion-based approach to learn such a  Personalised policy  that respects human preferences while also satisfying the task objective. To infer human preferences, the trajectories used to initially learn the task (we refer to the corresponding policy as the  Task-specific policy ) are numerically scored to reflect the degree to which they align with the human’s preferred behaviour. These trajectory-score pairs are fed to an LSTM  [ 10 ]  to construct an  Intent-specific policy . We choose an LSTM-based approach as it is well-suited to model temporal patterns within the trajectories. Once the user’s intent is inferred, the key idea is then to fuse the task-specific policy with this intent-specific policy, such that the resulting personalised policy completes the task at hand while simultaneously respecting user-specific needs. We design the described policy fusion method to theoretically ensure that the divergence of the personalised policy with respect to the task-specific policy remains bounded implying that the personalised policy upon fusion cannot change arbitrarily. Since the inference of the user’s intent (learning the intent-specific policy) is done by reusing the same trajectories used for training the task policy/task-specific policy, our approach obviates the need for any additional environment interactions, making it a zero-shot scheme.

Despite the mentioned advantages of the described static policy fusion-based approach, we show that this policy fusion approach is insufficient, as it does not account for whether one policy is over/under-represented, which can lead to undesirable effects. For example, in a navigation scenario, the preference for visiting a particular state may cause the agent to repeatedly visit that state, thereby ignoring the primary navigation objective. In other words, static policy fusion may cause one of the policies to dominate the other,
leading to unwanted agent behaviours. We address this problem through a novel dynamic policy fusion approach that automatically regulates the dominance of the intent-specific policy by regulating the associated temperature parameter of the Boltzmann distribution  [ 23 ]  used to represent the policy. This helps control the relative contribution of the intent-specific policy in the personalised policy, thereby allowing the agent to complete the task while respecting user-specific needs/preferences.

We note that the problem of respecting additional user-specific objectives could also be formulated as a multi-objective reinforcement learning  [ 22 ,  9 ,  15 ]  problem, although such approaches are applicable only when rewards corresponding to each objective are available during learning. As such, the present work aims to deal with how one can adapt a trained policy after it has learned an initial task. Overall, the contributions of this work can be summarised as follows:

•

A technique to achieve zero-shot personalisation by adapting an already trained task policy without additional environmental interactions.

•

Theoretical analysis establishing the boundedness of the personalised policy with respect to the task policy.

•

Demonstration of the failure of static policy fusion methods, motivating the need for dynamic fusion techniques.

•

A modulation mechanism for dynamic policy fusion to balance task performance with user preference.

2  Related Work

Learning from human feedback is of particular interest in the RL community as it leverages human knowledge during the learning process, offering several benefits. Firstly, it improves the efficiency of the system in terms of sample requirements as well as overall performance

[ 6 ] . Secondly, leveraging human feedback has been shown to enable RL agents to solve complex tasks that are otherwise challenging to manually specify through conventional reward functions  [ 12 ,  5 ] . Consequently, there exist a number of approaches aimed at leveraging human feedback for learning.

A number of works in interactive reinforcement learning  [ 28 ,  14 ]  examine how agents can learn from state-action level human feedback. However, such methods are limited, as providing state-action-level feedback is non-intuitive and cognitively taxing. Preference-based reinforcement learning  [ 5 ,  12 ] , uses trajectory-level human preference feedback, using which a corresponding reward function is learned. However, following the learning of the reward function, this approach may require the agent to interact with the environment again to learn the corresponding policy. As our present work focuses on developing a zero-shot approach, to estimate human intent, we instead use RUDDER