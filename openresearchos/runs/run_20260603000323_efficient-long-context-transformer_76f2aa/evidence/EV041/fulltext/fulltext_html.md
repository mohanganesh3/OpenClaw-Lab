[2509.04154] Attention as an Adaptive Filter

Attention as an Adaptive Filter

Peter Racioppo

Independent Researcher, formerly UCLA

pcracioppo@ucla.edu

(October 6, 2025)

Contents

1  Introduction

2  Preliminaries

2.1  Model Setup and Definitions

2.2  Propagation of Uncertainty

2.3  Batch Maximum Likelihood Trajectory Estimation

2.4  Recursive (Kalman Filter) Formulation

2.5  Closed-form Precision Matrix Calculation for Diagonalizable Matrices

2.6  Robust Maximum Likelihood Trajectory Estimation

2.6.1  Variational Formulation of Robust Estimation

2.6.2  Iteratively-Reweighted Least Squares

2.6.3  Iteratively-Reweighted Kalman Filter (IRKF)

3  Adaptive Filter Attention

3.1  Generalizing the Adaptive Filter to a Tensor Form of Attention

3.2  Complexity

3.3  Convolutional Representation

3.4  Broadcasting the Estimates

3.5  Factorizing the Residuals

3.6  Isotropic Decay and Noise

3.7  Unitary Dynamics

3.8  Implementation

3.8.1  Adaptive Filter Attention

3.8.2  Unrolled Adaptive Filter Attention

3.9  Basic Experiments

4  Radial-Tangential Model

4.1  Itô SDE with Radial–Tangential Dynamics and Noise

4.2  Propagation of Uncertainty through the RT-SDE

4.3  Radial–Tangential Kalman IRLS

4.4  Implementation

4.5  Interpreting the Transformer as an MLE on a Hypersphere

5  Related Work

5.1  Classical Filtering Methods

5.2  Robust and Adaptive Filtering

5.3  Neural Networks for Filtering Dynamical Systems

5.4  Neural State Space Models

5.5  Attention as a Structured State Space Computation

5.6  Dynamical Systems Perspectives on Attention

5.7  Diffusion, Score-Based Models, and Neural SDEs

5.8  Structuring Attention with Priors and Positional Biases

5.9  Modeling Uncertainty in Neural Networks

5.10  Structured Low-Rank Covariances for Efficient Propagation

5.11  Unrolling Optimization Algorithms and Connections to Attention

6  Discussion and Conclusion

A  Background

A.1  Linear Systems

A.2  Stochastic Processes

A.3  The Kalman Filter

A.4  Adaptive Filtering

A.5  The Attention Mechanism

A.6  State Space Models

B  Additional Material

B.1  Separable Approximation of the Full Precision Kernel

B.2  Approximating the Batch MLE with a Recursive Filter

B.3  Normal plus Rank-1 Noise Covariance

B.4  Expressivity of Radial-Tangential SDEs

Abstract

We introduce Adaptive Filter Attention (AFA), a novel attention mechanism that incorporates a learnable dynamics model directly into the computation of attention weights. Rather than comparing queries and keys directly, we model the input sequence as discrete observations of a linear stochastic differential equation (SDE). By imposing a linear dynamics model with simultaneously diagonalizable state matrices and noise covariances, we can make use of a closed-form solution to the differential Lyapunov equation to efficiently propagate pairwise uncertainties through the dynamics. Attention naturally arises as the maximum likelihood solution for this linear SDE, with attention weights corresponding to robust residual-based reweightings of the propagated pairwise precisions. Imposing an additional constraint on the state matrix’s eigenvalues leads to a simplified variant with the same computational and memory complexity as standard attention. In the limit of vanishing dynamics and process noise, and using a small-angle approximation, we recover ordinary dot-product attention.

1  Introduction

Self-attention has become the dominant paradigm for sequence modeling due to its parallelism and scalability. However, this parallelism comes at the cost of losing the recursive structure that RNNs naturally impose, which provides an implicit temporal regularization that encourages consistency over time. State space models (SSMs) offer a complementary approach: they retain temporal structure through shared linear dynamics while supporting parallel training via convolutional implementations. However, most SSMs assume deterministic dynamics, limiting their ability to capture stochasticity and uncertainty.

Our goal is to bridge these two paradigms. We view self-attention as a form of filtering: it selectively averages across inputs, denoising by pooling information from similar states. However, unlike Kalman Filtering, standard attention does not explicitly propagate states through time. This motivates incorporating explicit temporal structure—through known or learned dynamics—into attention, restoring the inductive bias that standard attention loses by forgoing recurrence.

This technical report is structured as follows:

Section

2  : Preliminaries

We introduce the model setup and review standard results for propagating covariances through linear SDEs. We then discuss the batch maximum likelihood estimator and its connection to the Kalman Filter. Next, we derive a closed-form solution of the differential Lyapunov equation for diagonalizable systems with simultaneously diagonalizable process and measurement noise covariances. Finally, we discuss an adaptively reweighted MLE that reweights propagated covariances using the pair-wise residuals, and its relationship to the iteratively reweighted least squares (IRLS) algorithm and a robustified form of the Kalman Filter.

Section

3  : Adaptive Filter Attention

We generalize the robustified MLE to a tensor form of attention, and briefly analyze its time and memory complexity. We show that the algorithm can be simplified to a form with the same time and memory complexity as standard attention, through the use of convolutional kernels for the (diagonalized) matrix exponential and pair-wise propagated covariances, and by imposing additional structure on the linear SDE. We show that this formulation reduces to a complex-valued variant of standard attention in the case of a skew-symmetric state matrix and normalized inputs. We then describe an implementation of this algorithm, and an unrolled, recursive approximation for inference. We describe basic experiments on simple simulated linear dynamical systems.

Section

4  : Radial-Tangential Model

We generalize the SDE model from  Section

3

, which assumed isotropic noise, to allow for separate noise covariances in the tangential and radial directions. We show that the propagated covariances in this case can be written in closed-form for the case of diagonal plus rank-1 covariances that co-rotate with the dynamics. We derive a form of attention that solves the MLE while respecting the geometric constraints imposed by the assumed dynamics model, and compare it to the standard Transformer architecture.

Section

5  : Related Work

We overview related work, including classical methods for filtering and adaptive filtering; the use of neural networks for modeling and filtering dynamical systems; neural state space models; connections between attention and structured state space models; related architectures; and various other topics.

Section

6  : Discussion and Conclusion

Appendix A

: Background

We briefly overview basic background on linear systems, stochastic processes, the Kalman Filter, adaptive filtering, the attention mechanism, and neural state space models.

Appendix B

: Additional Material

We cover some additional topics, including a separable approximation of the propagated precision kernel, discussion of normal plus rank-1 noise covariance, and the expressivity of our structured SDE model.

Note:  This technical report describes ongoing work. Evaluation on real-world datasets is in progress and will be reported in future work.

2  Preliminaries

We review standard results on vector SDEs and linear Gaussian filtering. The reader familiar with these topics may skip to  Section

3

. Additional background material can be found in

Appendix A  .

2.1  Model Setup and Definitions

Consider a linear time-invariant (LTI) Itô stochastic differential equation (SDE):

d  ​  𝐱  ​

(  t  )

=

𝐀𝐱  ​

(  t  )

​  d  ​  t

+

𝐁  ​  d  ​  𝐰  ​

(  t  )

,

𝐳  ​

(

t  k

)

=

𝐂𝐱  ​

(

t