[2503.09730] Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving

Local Look-Ahead Guidance via Verifier-in-the-Loop for
Automated Theorem Proving

Sara Rajaee

s.rajaee@uva.nl

University of Amsterdam

&amp;Kumar Pratik ∗ , Gabriele Cesa, Arash Behboodi

{kpratik, gcesa, behboodi}@qti.qualcomm.com

Qualcomm AI Research , Amsterdam

Equal contribution.
Work done during an internship at Qualcomm AI Research, Amsterdam.

Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.

©2025 Qualcomm Technologies, Inc. and/or its affiliated companies. All Rights Reserved.

Abstract

The most promising recent methods for AI reasoning require applying variants of reinforcement learning (RL) either on rolled out trajectories from the model, even for the step-wise rewards, or large quantities of human annotated trajectory data. The reliance on the rolled-out trajectory renders the compute cost and time prohibitively high. In particular, the correctness of a reasoning trajectory can typically only be judged at its completion, leading to sparse rewards in RL or requiring expensive synthetic data generation in expert iteration-like methods.
In this work, we focus on the Automatic Theorem Proving (ATP) task and propose a novel verifier-in-the-loop design, which unlike existing approaches that leverage feedback on the entire reasoning trajectory, employs an automated verifier to give intermediate feedback at each step of the reasoning process. Using Lean as the verifier, we empirically show that the step-by-step local verification produces a global improvement in the model’s reasoning accuracy and efficiency.

1  Introduction

As the new applications of modern machine learning are emerging in various scientific and engineering domains, automated mathematical theorem proving has garnered interests from both machine learning researchers and mathematicians. Many ongoing efforts leverage reinforcement learning and expert iteration, inspired by the success of methods like AlphaZero, to build models that search the proof space and provide step wise or holistic solutions  (Lample et al.,  2022 ; Xin et al.,  2024a ; Gloeckle et al.,  2024 ; Anthony et al.,  2017 ; Silver et al.,  2018 ) . These solutions are usually verified by formal proof verification systems like Lean  (Moura and Ullrich,  2021 )  or Coq  (Coq Development Team,  2024 ) . Relying on

Reinforcement Learning

(  RL  ) is advantageous in terms of data efficiency but comes with high computational and training costs  (Gloeckle et al.,  2024 ) . Part of this complexity is related to the necessity of rolling out the proofs and computing rewards from successful episodes.

In contrast, ReProver  (Yang et al.,  2023 )  takes a simpler supervised training approach, specifically imitation learning, paired with premise retrieval methods. The key components of this proof system are as follows: theorem that we would like to prove, tactics that are actions toward the final proof and itself consist of set of goals to be proven, premises that are used to prove goals, and state of the proof which includes the set of goals that are still unproven. The approach consists of retrieving the relevant premises from a database given the final theorem and the state of the proof, and then using ReProver to provide tactics for getting to the next state. The proof terminates when all the goals are proven. The method achieves competitive performance with an order of magnitude smaller complexity and training time  (Gloeckle et al.,  2024 ) . While the computational cost and simplicity of ReProver are appealing, we empirically observed that many failure cases of ReProver are due to syntactically incorrect tactics or tactics that are not applicable to the current state of a proof. This has detrimental effects on the beam search performed at inference time, as many beams result in invalid tactics that need to be verified by Lean, thus taking away time from exploring more promising tactics.

To preserve the desirable computational efficiency of ReProver and simultaneously address this problem, the natural choice is to fine-tune the model to remove syntactic errors and increase the number of useful tactics for the proof at each state. Recently, feedback-based alignment has been gaining traction in various other similar fields such as automated code generation and various preference optimization methods where the rewards come either from human feedback (RLHF) or other reward models  (Ouyang et al.,  2022 ; Ziegler et al.,  2019 ; Rafailov et al.,  2024 ) . Since applying many such reinforcement learning methods for training large models can be complex and expensive, various methods have been introduced in the literature with moderate complexity, among which we can refer to

Direct Preference Optimization

(  DPO  )  (Rafailov et al.,  2024 )  and

Group Reward Preference Optimization

(  GRPO  )  (Shao et al.,  2024 ) . In context of mathematical reasoning and theorem proving, many works emphasized the importance of trajectory-level preferences in mathematical problem solving with large language models (LLMs) (e.g. see  Xiong et al. ( 2024 )  and Preference Optimization paragraph in Sec.

2  ). Even when the complexity is saved in the

RL

training algorithm, computing these trajectory level preferences can incur additional complexity. This discussion extends to more general episodic reasoning tasks with stepwise verification where the model needs to provide outputs that are both syntactically and semantically correct and useful for solving the problem at hand.

Our contribution.

In this work, we aim to address the above issues by fine-tuning a pre-trained model which listens and uses the feedback from the tool, in this case Lean, during the training. At each step, our framework, called LeanListener, obtains feedback on the generated tactics directly via its interaction with the Lean software, and performs policy optimization with a reward that is designed based on Lean feedback. Given the pre-trained ReProver model from  Yang et al. ( 2023 ) , we sample different tactics from its output for each proof state in the training set and use Lean feedback to compute the reward. The reward consists of a negative return for invalid tactics, a positive one for applicable tactics, and a return based on the number of remaining unsolved goals. The

RL

training is done using

GRPO  . First note that the sampling step for ReProver’s output can yield applicable and new tactics that differ from the provided human label. Therefore, it helps the data efficiency of the method by exploring and adding new tactics like what we see in expert iteration. Second, unlike methods like

Process-supervised Reward Model

(  PRM  ), we do not compute the step-wise reward based on the full trajectory information and only rely on  local look-ahead  feedback from the number of remaining and unsolved goals in the next immediate steps, and therefore, we address the complexity of the trajectory based preference association. Thanks to this fine-tuning strategy, we expect the model to rank valid and effective tactics higher than invalid ones, even if they were not previously observed in the human-labeled trajectories. As a result, our fine-tuned model can make better use of the beam search used at inference time.
Besides, as we will show in our numerical results, the local look ahead, online training with Lean in loop, and GRPO are crucial components in improving the final performance of the model. To summarize, our contributions are as follows. We propose a framework for efficient training of the ReProver to leverage the feedback provided by the external tool, in this case, the Lean solver. We use

GRPO

for training the model using the reward that is based on the applicable tactics and the number of unsolved goals. The

GRPO

training is particularly beneficial compared to

DPO

training in our case. We show that the training based on o