# EV010: Predictive direct yaw moment control with active steering based on polytopic linear parameter-varying model

URL: https://www.semanticscholar.org/paper/00a8311c52ad30df55a5642e07efdd8ffd6a6c88
Year: 2022
Source: semantic_scholar
Arxiv: n/a

## Abstract

In this paper, stabilizing predictive direct yaw moment control with active steering is proposed. The prediction model used in the model predictive control algorithm is a linear time-varying (LTV) bicycle model that depends on the velocity. To ensure stability and recursive feasibility regardless of the velocity change, the LTV model is transformed into a polytopic linear parameter-varying (LPV) model using the tensor product model transformation. This model is used to offline solve the robust LQR problem and form the terminal set and terminal cost for the online optimization problem. Furthermore, the same model is used to compute a robust N -step controllable set to the terminal set. In the online optimization problem, the states of the system are constrained in this set to guarantee recursive feasibility. The proposed control algorithm is tested in simulation and experimentally on a laboratory-scale car.
