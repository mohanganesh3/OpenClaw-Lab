[2211.06527] Rewards Encoding Environment Dynamics Improves Preference-based Reinforcement Learning

Rewards Encoding Environment Dynamics Improves Preference-based Reinforcement Learning

Katherine Metcalf  Miguel Sarabia  Barry-John Theobald
 { kmetcalf ,  miguelsdc ,  barryjohn_theobald }@apple.com
 Apple, California, USA.

Abstract

Preference-based reinforcement learning (RL) algorithms help avoid the pitfalls of hand-crafted reward functions by distilling them from human preference feedback, but they remain impractical due to the burdensome number of labels required from the human, even for relatively simple tasks. In this work, we demonstrate that encoding environment dynamics in the reward function (REED) dramatically reduces the number of preference labels required in state-of-the-art preference-based RL frameworks. We hypothesize that REED-based methods better partition the state-action space and facilitate generalization to state-action pairs not included in the preference dataset. REED iterates between encoding environment dynamics in a state-action representation via a self-supervised temporal consistency task, and bootstrapping the preference-based reward function from the state-action representation. Whereas prior approaches train only on the preference-labelled trajectory pairs, REED exposes the state-action representation to all transitions experienced during policy training. We explore the benefits of REED within the PrefPPO  [ 1 ]  and PEBBLE  [ 2 ]  preference learning frameworks and demonstrate improvements across experimental conditions to both the speed of policy learning and the final policy performance. For example, on quadruped-walk and walker-walk with 50 preference labels, REED-based reward functions recover 83% and 66% of ground truth reward policy performance and without REED only 38% and 21% are recovered. For some domains, REED-based reward functions result in policies that outperform policies trained on the ground truth reward.

1  Introduction

The quality of a reinforcement learned (RL) policy depends directly on the quality of the reward function used to train it. However, hand-crafting reward functions is very challenging, even for experts. A poorly specified reward can result in sub-optimal behaviors, with reward exploitation frequently leading to undesired behaviors  [ 3 ,  4 ,  5 ] . Therefore, methods for specifying robust, non-hackable reward functions are critical to correctly and efficiently train policies with RL. One such method is to teach an agent to align its policy with human preferences by distilling reward functions from feedback on sets of trajectories  [ 1 ,  2 ,  6 ,  7 ,  8 ,  9 ,  10 ,  11 ,  12 ] .

Learning from preference labels over sets of trajectories is promising as it empowers non-experts to provide such feedback, especially compared to providing corrections (e.g. DAgger  [ 13 ]  and related approaches) or providing real-valued rewards (e.g. TAMER  [ 14 ]  and related approaches). Still, the majority of existing preference-based deep RL methods require either a dataset of demonstrations  [ 6 ]  or thousands of preference labels to recover optimal policy performance  [ 1 ,  2 ,  12 ] .

In this paper, we target efficient reward function learning by introducing  R ewards  E ncoding  E nvironment  D ynamics (REED) (Section

4.1  ). Given the difficulty people face when providing feedback for a single state-action pair (e.g.  [ 14 ] ), and the importance of defining preferences over transitions instead of single state-action pairs  [ 2 ] ,  it is likely that people’s internal reward functions are defined over outcomes not state-action pairs .

We hypothesize that: (1) modelling the relationship between state, action, and next-state triplets is essential for learning preferences over transitions, (2) encoding environment dynamics with a temporal consistency objective will allow the reward function to better generalize over states and actions with similar outcomes, and (3) exposing the reward model to all transitions experienced by the policy during training will result in more stable reward estimations during reward and policy learning. Therefore, we incorporate a self-supervised temporal consistency task using self-predictive representations (SPR)  [ 15 ]  into preference-based RL frameworks. REED reward functions lead to faster policy training and reduce the number of preference samples needed (as we will show in Section

5  ).

We demonstrate the benefits of REED using the current state-of-the-art in preference learning ( [ 1 ] ,  [ 2 ] ). In our experiments, which follow those outlined in  [ 11 ] , REED reward functions outperform non-REED reward functions across different preference dataset sizes, quality of preference labelling strategy, and tasks (Section

5.2  ). The improvements in policy learning and preference sample efficiency represent a significant step towards enabling end-users to naturally adjust the behavior of agents in their environment with their own feedback.

2  Related Work

Learning from Human Feedback.  Learning reward functions from preference-based feedback  [ 16 ,  17 ,  18 ,  19 ,  20 ,  21 ,  22 ,  8 ]  has been used to address the limitations of learning policies directly from human feedback  [ 23 ,  24 ,  25 ]  by inferring reward functions from either task success  [ 26 ,  27 ,  28 ]  or real-valued reward labels  [ 29 ,  30 ] . Learning policies directly from human feedback is time inefficient for the human as near constant supervision is frequently assumed. Inferring reward functions from task success feedback requires examples of task success, which can be difficult to acquire in complex and multi-step task domains. Finally, people are not able to reliably provide real-valued reward labels. Preference-based RL was extended to deep RL domains in  [ 1 ]  then made more efficient and improved in  [ 2 ]  and  [ 12 ] . To reduce the feedback complexity of preference-based RL,  Lee et al. [ 2 ]  sped up policy learning via (1) intrinsically-motivated exploration and (2) relabelling the experience replay buffer. These two techniques aimed to improve both the sample complexity of the policy, and the trajectories generated by the policy, which are then used to seek feedback.  Park et al. [ 12 ]  reduced feedback complexity by incorporating augmentations and pseudo-labelling into the reward model learning. Additionally, preference-learning has also been incorporated into data-driven skill extraction and execution in the absence of a known reward function  [ 31 ] .

Encoding Environment Dynamics.  Prior work has demonstrated the benefits of encoding environment dynamics in the state-action representation of a policy  [ 15 ,  32 ,  33 ] . We instead encode environment dynamics in the reward function which, as far as we are aware, has not been done before. It is nonetheless common for dynamics models to predict both the next state and the environment’s reward  [ 32 ]  which suggests it is important to imbue the reward function with some understanding of the dynamics. The primary self-supervised approach to learning a dynamics model is to predict the latent next state  [ 15 ,  32 ,  34 ,  35 ] . Indeed, the current state-of-the-art in data efficient RL  [ 15 ,  33 ]  for encoding dynamics via latent next-state predictions uses a self-predictive representation (SPR)  [ 15 ]  task, an approach we follow in this work.

3  Preference-based Reinforcement Learning

Reinforcement learning (RL) trains an agent to achieve tasks via environment interactions and reward signals  [ 36 ] . For each time step

t

𝑡

t

the environment provides a state

s  t

subscript  𝑠  𝑡

s_{t}

used by the agent to select an action according to its policy

a  t

∼

π  ϕ

​

(

a  |

s  t

)

similar-to

subscript  𝑎  𝑡

subscript  𝜋  italic-ϕ

conditional  𝑎

subscript  𝑠  𝑡

a_{t}\sim\pi_{\phi}(a|s_{t})

. Then

a  t

subscript  𝑎  𝑡

a_{t}

is applied to the environment, which re