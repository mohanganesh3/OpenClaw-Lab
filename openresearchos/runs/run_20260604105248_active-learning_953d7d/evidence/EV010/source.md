# EV010: FLEX: A Backbone for Diffusion-Based Modeling of Spatio-temporal Physical Systems

URL: https://www.semanticscholar.org/paper/003cf1ddd21cba7f9843a7bd7df9d45df7724391
Year: 2025
Source: semantic_scholar
Arxiv: 2505.17351

## Abstract

We introduce FLEX (FLow EXpert), a backbone architecture for generative modeling of spatio-temporal physical systems using diffusion models. FLEX operates in the residual space rather than on raw data, a modeling choice that we motivate theoretically, showing that it reduces the variance of the velocity field in the diffusion model, which helps stabilize training. FLEX integrates a latent Transformer into a U-Net with standard convolutional ResNet layers and incorporates a redesigned skip connection scheme. This hybrid design enables the model to capture both local spatial detail and long-range dependencies in latent space. To improve spatio-temporal conditioning, FLEX uses a task-specific encoder that processes auxiliary inputs such as coarse or past snapshots. Weak conditioning is applied to the shared encoder via skip connections to promote generalization, while strong conditioning is applied to the decoder through both skip and bottleneck features to ensure reconstruction fidelity. FLEX achieves accurate predictions for super-resolution and forecasting tasks using as few as two reverse diffusion steps. It also produces calibrated uncertainty estimates through sampling. Evaluations on high-resolution 2D turbulence data show that FLEX outperforms strong baselines and generalizes to out-of-distribution settings, including unseen Reynolds numbers, physical observables (e.g., fluid flow velocity fields), and boundary conditions.
