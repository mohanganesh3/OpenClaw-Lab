# EV014: Towards Aircraft Autonomy Using a POMDP-Based Planner

URL: https://www.semanticscholar.org/paper/003cfcca68fbd5aebca5f98c0243d62ec5073b2e
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

The autonomy aircraft guidance problem has been an important and challenging issue that has received significant attention in recent years. In this paper, we propose a novel controller to plan the trajectory of an aircraft under uncertainty by providing optimal commands to reach the target while avoiding no-fly zones and optimizing various performance metrics (e.g., fuel consumption and travel distance). In particular, we introduce a two-layer controller, where a Partially Observable Markov Decision Process (POMDP) is formalized as the high-level controller (outer loop), and an inverse dynamics controller serves as the low-level controller (inner loop). The POMDP provides the best local reference values to the low-level controller, which then commands the aircraft actuators. By leveraging a linearized dynamic model obtained through dynamics inversion, the POMDP can efficiently compute optimal reference values. We tested this approach in a simulated scenario where the aircraft avoids no-fly zones to reach a target position.
