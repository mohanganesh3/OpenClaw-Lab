# EV003: Visuo-tactile feedback policies for terminal assembly facilitated by reinforcement learning

URL: https://www.semanticscholar.org/paper/0047ce3055cc40da85927369601b7cd6de5debca
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Industrial terminal assembly tasks are often repetitive and involve handling components with tight tolerances that are susceptible to damage. Learning an effective terminal assembly policy in real-world is challenging, as collisions between parts and the environment can lead to slippage or part breakage. In this paper, we propose a safe reinforcement learning approach to develop a visuo-tactile assembly policy that is robust to variations in grasp poses. Our method minimizes collisions between the terminal head and terminal base by decomposing the assembly task into three distinct phases. In the first grasp phase,a vision-guided model is trained to pick the terminal head from an initial bin. In the second align phase, a tactile-based grasp pose estimation model is employed to align the terminal head with the terminal base. In the final assembly phase, a visuo-tactile policy is learned to precisely insert the terminal head into the terminal base. To ensure safe training, the robot leverages human demonstrations and interventions. Experimental results on PLC terminal assembly demonstrate that the proposed method achieves 100% successful insertions across 100 different initial end-effector and grasp poses, while imitation learning and online-RL policy yield only 9% and 0%.
