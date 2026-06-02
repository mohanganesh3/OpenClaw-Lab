# EV020: Human-in-the-loop RL with an EEG wearable headset: on effective use of brainwaves to accelerate learning

URL: https://www.semanticscholar.org/paper/0467cc2ffb8deb03b3a9417f14db564b2fd71770
Year: 2020
Source: semantic_scholar
Arxiv: n/a

## Abstract

Intrinsic Human-In-The-Loop Reinforcement Learning (HITL-RL) is an approach to obtain the human feedback implicitly by capturing brain waves through the use of wearable electroencephalogram (EEG) headsets. It can significantly accelerate the training convergence of RL algorithms while reducing the burden placed on the humans involved in the training loop. While a human naturally observes the performance of an RL agent, any erroneous behavior of the agent can be recognized through the error-potentials (ErrP) in the EEG signal. This information can then be incorporated into the reward function of the RL algorithm to accelerate its learning. The detection accuracy of the error-potentials thus significantly affects the convergence time of the RL algorithm. The focus of this work is the reliable detection of error-potentials using the brain waves of the user detected using only an off-the-shelf EEG wearable. We first present a new error-potential decoding algorithm that leverages the spatial, temporal, and frequency-domain characteristics of the EEG signals. We develop three Atari-like game environments and recruit 25 volunteers for evaluation. The proposed algorithm achieves an accuracy performance of 73.71% (an improvement of 8.11% over the current state-of-the-art algorithm). We then show that a modified algorithm that intelligently discards low-confidence estimates is capable of boosting the accuracy to 79.51% (16.63% improvement).
