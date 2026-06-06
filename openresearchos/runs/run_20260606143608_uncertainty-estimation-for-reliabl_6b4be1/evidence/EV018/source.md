# EV018: Learning-Based Tracking Control of Unknown Robot Systems with Online Parameter Estimation

URL: https://www.semanticscholar.org/paper/003f1e33b8406c5815534d240f29cba7663f67eb
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

This paper proposes a novel learning-based optimal control approach for the tracking control problem of a robot manipulator, which is allowed to have uncertainty in the model parameters. We first employ neural network technique to design an identifier for the system parameters estimation. Then, an optimal tracking controller is proposed with a critic network. A novel online weight adaptation law is designed with the dynamic regression extension and mixing technique, for updating both the unknown parameters and the critic network's weight during the control process. With such setup, our approach can develop an important capability of relaxing the persistent excitation condition, leading to improved parameter-convergence accuracy and control applicability, which can be distinguished from the existing methods that use gradient-descent based weight adaptation laws. Rigorous theoretical analysis is conducted based on the Lyapunov stability theory and demonstrates the the uniform ultimate boundedness stability of the closed-loop systems. Effectiveness of the proposed method is validated through simulation study by using a two degree-of-freedom robot system.
