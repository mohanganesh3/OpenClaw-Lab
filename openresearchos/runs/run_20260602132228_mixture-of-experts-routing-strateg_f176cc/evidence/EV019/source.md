# EV019: Scene-Adaptive Motion Planning with Explicit Mixture of Experts and Interaction-Oriented Optimization

URL: https://www.semanticscholar.org/paper/549c6df94f5eca49643056652e124402470beecc
Year: 2025
Source: semantic_scholar
Arxiv: 2505.12311

## Abstract

Despite over a decade of development, autonomous driving trajectory planning in complex urban environments continues to encounter significant challenges. These challenges include the difficulty in accommodating the multi-modal nature of trajectories, the limitations of single expert model in managing diverse scenarios, and insufficient consideration of environmental interactions. To address these issues, this paper introduces the EMoE-Planner, which incorporates three innovative approaches. Firstly, the Explicit MoE (Mixture of Experts) dynamically selects specialized experts based on scenario-specific information through a shared scene router. Secondly, the planner utilizes scene-specific queries to provide multi-modal priors, directing the model's focus towards relevant target areas. Lastly, it enhances the prediction model and loss calculation by considering the interactions between the ego vehicle and other agents, thereby significantly boosting planning performance. Comparative experiments were conducted on the Nuplan dataset against the state-of-the-art methods. The simulation results demonstrate that our model consistently outperforms SOTA models across nearly all test scenarios. Our model is the first pure learning model to achieve performance surpassing rule-based algorithms in almost all Nuplan closed-loop simulations.
