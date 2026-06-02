[2510.01519] Online Hierarchical Policy Learning using Physics Priors for Robot Navigation in Unknown Environments

Online Hierarchical Policy Learning using Physics Priors for Robot Navigation in Unknown Environments

Wei Han Chen*, Yuchen Liu*, Alexiy Buynitsky, and Ahmed H. Qureshi

Wei Han Chen, Yuchen Liu, Alexiy Buynitsky, and Ahmed H. Qureshi are with the Department of Computer Science, Purdue University, West Lafayette, IN, USA, 47907. Email

{  \{

chen3189, liu3853, abuynits, ahqureshi

}  @

\}@

purdue.edu * denotes equal contribution

Abstract

Robot navigation in large, complex, and unknown indoor environments is a challenging problem. The existing approaches, such as traditional sampling-based methods, struggle with resolution control and scalability, while imitation learning-based methods require a large amount of demonstration data. Active Neural Time Fields (ANTFields) have recently emerged as a promising solution by using local observations to learn cost-to-go functions without relying on demonstrations. Despite their potential, these methods are hampered by challenges such as spectral bias and catastrophic forgetting, which diminish their effectiveness in complex scenarios. To address these issues, our approach decomposes the planning problem into a hierarchical structure. At the high level, a sparse graph captures the environment’s global connectivity, while at the low level, a planner based on neural fields navigates local obstacles by solving the Eikonal PDE. This physics-informed strategy overcomes common pitfalls like spectral bias and neural field fitting difficulties, resulting in a smooth and precise representation of the cost landscape. We validate our framework in large-scale environments, demonstrating its enhanced adaptability and precision compared to previous methods, and highlighting its potential for online exploration, mapping, and real-world navigation.  https://sites.google.com/view/mntfields/home

I

Introduction

Navigating large, unknown environments presents significant challenges in robotics, where both mapping and planning are crucial yet difficult tasks. Traditional mapping approaches often generate occupancy or signed distance field (SDF) maps that require additional processing, such as grid search or optimization, to extract navigable paths  [ 1 ,  2 ,  3 ,  4 ,  5 ] . This extra step not only increases computational overhead but also complicates the transition from a raw map to an actionable navigation plan.

Alternatively, some methods directly build probabilistic roadmaps (PRMs) from sensor data  [ 6 ,  7 ] . While these techniques can yield efficient paths in simpler settings, they tend to become unwieldy in expansive, complex environments. The sheer number of nodes required to accurately represent intricate spaces often leads to memory-intensive computations and difficulties in maintaining and controlling the roadmap structure over large scales.

More recently, methods like Neural Time Fields (NTFields)  [ 8 ]  have been introduced to infer cost-to-go functions directly by solving the Eikonal equation. These approaches aim to bypass the intermediate mapping step by providing an implicit representation of the navigation cost. However, NTFields encounter significant hurdles when scaled up to large scenes. Their reliance on neural network architectures introduces issues such as spectral bias, catastrophic forgetting, and poor convergence. Moreover, the inherent scaling challenge of solving a partial differential equation (PDE) further complicates their application in diverse, cluttered environments.

Inspired by the hierarchical planning strategies we use in everyday navigation—such as how mapping applications outline broad routes while vehicles make local maneuvers—we propose a modular, hierarchical approach called Modular-NTFields (mNTFields) to address navigation challenges.

At a high level, our method constructs an online sparse navigation graph from local observations, capturing the connectivity between different subparts of a large environment. This high-level strategy is informed by our low-level method and yields a compact and efficient representation, allowing for rapid global planning without the burden of excessive detail. On a local level, we integrate NTFields to develop detailed cost-to-go maps for each subpart based on local observations. By solving the Eikonal equation locally, our approach effectively manages obstacles and complex geometries while alleviating the scalability issues that often affect traditional NTFields. Additionally, we enhance local planning with Temporal Difference Metric Learning (TDM)  [ 9 ]  to further improve convergence to an accurate Eikonal PDE solution.

We validate mNTFields across several challenging scenarios, demonstrating that it outperforms existing methods—particularly in large, complex, unknown indoor environments where standard approaches often struggle. Our experiments show that mNTFields achieves significantly faster navigation with higher success rates. We also showcase its practical deployment on a quadruped robot navigating through multiple rooms and narrow passages. This work highlights the potential of hierarchical planning frameworks for advancing robust and scalable robot navigation in unknown environments.

II

Related Work

Efficient navigation in partially or fully unknown environments remains a fundamental challenge in robotics, often requiring reliance on local sensor observations  [ 10 ] . Traditional motion planning methods include sampling-based approaches  [ 11 ,  12 ,  6 ] , which iteratively build a roadmap of feasible paths, and grid-based methods  [ 3 ,  4 ] , which discretize the environment into occupancy maps for search algorithms. While these techniques perform well in lower-dimensional or structured environments, their reliance on explicitly maintaining large graphs or grids makes them computationally expensive in complex, high-dimensional spaces. In contrast, our hierarchical representation enables efficient pathfinding for arbitrary start-goal pairs without requiring exhaustive search over large maps.

Recent data-driven approaches such as reinforcement learning (RL)  [ 13 ,  14 ,  15 ]  and imitation learning (IL)  [ 16 ,  17 ,  18 ,  19 ]  attempt to directly learn control policies from high-dimensional sensor inputs (e.g., images), often leveraging photorealistic simulations. However, these methods typically require extensive interaction data or expert demonstrations, which can be costly to obtain for real-world applications. Our method sidesteps this issue by leveraging physics-based principles, reducing the dependency on large expert datasets and improving adaptability across different environments.

Another class of solutions focuses on ego-centric local planning  [ 20 ,  21 ] , where navigation is based solely on onboard sensing, without global information or expert supervision. These approaches are well-suited for dynamic environments as they continuously update their plans based on new observations. However, their reliance on local information often leads to suboptimal routes or getting trapped in local minima. By maintaining a hierarchical global representation, our approach preserves adaptability while incorporating long-range planning, reducing the risk of poor local decisions.

More recently, foundation models  [ 22 ,  23 ] , trained on large-scale datasets from expert demonstrations or powerful planners, have shown promising generalization across diverse environments. However, their heavy computational demands and reliance on vast amounts of high-quality training data limit their scalability. These models may also struggle in highly complex or novel environments where their pre-trained knowledge does not generalize well. In contrast, our approach offers a lightweight, scalable solution that can be efficiently deployed in large, complex workspaces without requiring extensive pre-training.

A growing bo