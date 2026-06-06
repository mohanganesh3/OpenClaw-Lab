# EV008: Adaptive Outer-Loop Control of Quadrotors via Reinforcement Learning

URL: https://www.semanticscholar.org/paper/05e530365a37e4f4863a82ba43db605348bf24a5
Year: 2026
Source: semantic_scholar
Arxiv: 2605.16015

## Abstract

Deep Reinforcement Learning (DRL) for quadrotor flight control typically relies on Domain Randomization (DR) for sim-to-real transfer, resulting in overly conservative policies that struggle with dynamic disturbances. To overcome this, we propose a novel adaptive control architecture that actively perceives and reacts to instantaneous perturbations. First, we train an optimal outer-loop policy, then replace its reliance on ground-truth disturbance data with a Residual Dynamics Predictor (RDP). The RDP estimates the external forces and moments acting on the aircraft in flight online using only the history of states and control actions. For seamless hardware transfer, we introduce a data-efficient linear calibration bridge and an online thrust correction mechanism that align the simulated latent space with reality using mere seconds of flight data. Real-world validations on a Crazyflie micro-quadrotor demonstrate that our adaptive controller significantly outperforms baselines, maintaining precise trajectory tracking under severe uncertainties including mass variations, asymmetric payloads, and dynamic slung loads
