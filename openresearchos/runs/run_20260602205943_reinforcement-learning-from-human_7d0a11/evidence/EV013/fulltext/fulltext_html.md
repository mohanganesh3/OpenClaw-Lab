[2602.02488] RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System

\uselogo  \correspondingauthor

yangling0818@163.com

RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System

Yinjie Wang

Tianbao Xie

Ke Shen

Mengdi Wang

Ling Yang

Abstract

Code:  https://github.com/Gen-Verse/Open-AgentRL

Models:  Policy &amp; Reward

We propose  RLAnything , a reinforcement learning framework that dynamically forges environment, policy, and
reward models through closed-loop optimization, amplifying learning signals and strengthening the overall RL system for  any LLM or agentic scenarios . Specifically, the policy is trained with integrated feedback from step-wise and outcome signals, while the reward model is jointly optimized via consistency feedback, which in turn further improves policy training. Moreover, our theory-motivated automatic environment adaptation improves training for both the reward and policy models by leveraging critic feedback from each, enabling learning from experience. Empirically, each added component consistently improves the overall system, and RLAnything yields substantial gains across various representative LLM and agentic tasks, boosting Qwen3-VL-8B-Thinking by

9.1  %

9.1\%

on OSWorld and Qwen2.5-7B-Instruct by

18.7  %

18.7\%

and

11.9  %

11.9\%

on AlfWorld and LiveBench, respectively. We also that optimized reward-model signals outperform outcomes that rely on human labels.

1. Jointly optimizing the reward model and the environment, in turn, benefits the policy’s learning curve, yielding higher converged accuracy.

2. Step-wise signals from optimized reward model outperform human-labeled outcome signals. Moreover, integrated feedback is vital for long-trajectory tasks.

3. We demonstrate effectiveness of RLAnything across diverse real-world applications, including computer control, coding, and text-based games.

4. New environment tasks scale linearly, and the reward model gets stronger at evaluating both current-step correctness and outcome influence.

Figure 1 :

Summarized experimental results and key insights from our RLAnything framework.

1  Introduction

Reinforcement learning with verifiable rewards (RLVR) is an effective approach for improving the reasoning capabilities of large language models  [ o1 ,  deepseekmath ,  guo2025deepseek ] . However, as real-world applications extend beyond single-turn question answering, especially when policies interact with environments iteratively over long trajectories, binary outcome rewards alone provide insufficient supervision  [ xiong2024watch ,  lightman2023let ,  xi2025agentprm ] . Step-wise signals are typically provided by generative reward models, which often outperform scalar-based models by leveraging the reasoning capabilities of language models  [ zhang2024generative ,  liu2025inference ] . However, training these models usually requires collecting high-quality, task-specific supervision  [ xi2025agentprm ,  zhang2025generative ] , motivating the need for more automated methods and scalable supervision.

Figure 2 :

Motivation and takeaways of our RLAnything framework. First, in complex real-world applications, reinforcement learning benefits from integrating step-wise rewards with outcome rewards. Second, the reward model can be jointly optimized with the policy via outcome supervision and self-consistency signals. Third, we show that adapting environment task difficulty to the policy’s capability not only facilitates policy learning but also improves reward model training within our framework. Environment tasks leverage critic feedback from both the policy and the reward model to drive automatic, targeted adaptation, further enabling active learning from experience.

Beyond reward design, the quality of the environment is also vital for scaling reinforcement learning. Aligning task difficulty with a model’s current capabilities is known to improve training dynamics  [ yu2025dapo ,  yang2025qwen3 ] . In RLVR, it has been shown that adapting task difficulty during optimization can improve policy training  [ zeng2025rlve ] . In real-world environments, such as computers for GUI agents  [ xie2024osworld ,  wang2025ui ]  or the physical world for robots  [ kober2013reinforcement ] , the scope of exploration is largely defined by the task. Moreover, scaling the environment by increasing task diversity can further promote policy generalization in broader scenarios  [ cobbe2020leveraging ,  team2021open ,  fang2025towards ,  cai2025autoforge ,  song2026envscaler ,  chen2025scaling ] .

If there exists an RL system that jointly optimizes the environment, policy, and reward model to amplify learning signals and strengthen the overall system?

In this work, we propose  RLAnything , a dynamic RL framework that forges the environment, policy, and reward model in a closed-loop system, where each component continuously receives feedback from the others to amplify learning signals across various complex LLM or agentic scenarios.
First, the policy is trained with integrated feedback that combines verifiable outcome rewards with step-wise signals provided by the reward model. Second, the reward model is jointly optimized via consistency feedback based on outcome and self-consistency, producing reliable step-wise supervision that in turn improves policy learning. Finally, motivated by our theoretical results, we show that balancing task difficulty benefits not only policy training but also reward model training in our RL system. Accordingly, we adapt environment tasks using critic feedback from both the policy and reward model, enabling precise and automatic task adjustment. In particular, we feed the reward model’s summarized information, which captures the policy’s failures, into a language model to perturb the task, providing concrete guidance on how to modify it.
To demonstrate the generality of our framework, we conduct empirical studies in three representative scenarios on computer use setting  [ xie2024osworld ] , text-based interactive games  [ cote2018textworld ,  shridhar2020alfworld ] , and coding LLMs. We summarize our main contributions as follows:

•

We propose RLAnything, a fully dynamic RL system that forges the environment, policy, and reward model through closed-loop optimization to amplify learning signals and strengthen the overall system, guided by our theoretical insights.

•

Across computer-use agents, text-based LLM agents, and coding LLMs, we show that each added dynamic component consistently benefits the overall system and improves out-of-distribution performance.

•

We achieve significant gains in practical applications: Qwen3-VL-8B-Thinking improves by

9.1  %

9.1\%

on OSWorld, and Qwen2.5-7B-Instruct improves by

18.7  %

18.7\%

and

11.9  %

11.9\%

on AlfWorld and LiveBench, respectively.

•

We show broad applicability: optimized reward-model signals outperform outcomes that rely on human labels, enabling active learning from experience and potential environment scaling.

2  RLAnything

Figure 3 :

Examples of environment task adaptation based on critic feedback across computer use agent, text-game agent, and coding LLM in our experiments. The critic feedback is summarized from the reward model’s evaluations and is used to automatically adapt tasks.

Our framework (see Algorithm

1  ) tightly couples the policy model, reward model, and environment to achieve joint optimization. Specifically, we train the policy using integrated feedback (Equation

1

in Section

2.1  ), which combines step-wise signals from reward model with the trajectory-level outcome signal. We train the reward model by treating the policy’s trajectories as environment tasks and assigning consistency feedback via Equation

2  ; we will prove that this objective also improves the reward model’s accuracy in predicting final outcomes in Section

2.3  . As the reward model becomes more accurate, it in tur