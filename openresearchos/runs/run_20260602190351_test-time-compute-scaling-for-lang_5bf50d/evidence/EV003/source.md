# EV003: Low-synchronization Arnoldi Methods for the Matrix Exponential with Application to Exponential Integrators

URL: https://www.semanticscholar.org/paper/0028f860bdc296506b7d2e75c7963d4079875856
Year: 2024
Source: semantic_scholar
Arxiv: 2410.14917

## Abstract

High order exponential integrators require computing linear combination of exponential like $\varphi$-functions of large matrices $A$ times a vector $v$. Krylov projection methods are the most general and remain an efficient choice for computing the matrix-function-vector-product evaluation when the matrix is $A$ is large and unable to be explicitly stored, or when obtaining information about the spectrum is expensive. The Krylov approximation relies on the Gram-Schmidt (GS) orthogonalization procedure to produce the orthonormal basis $V_m$. In parallel, GS orthogonalization requires \textit{global synchronizations} for inner products and vector normalization in the orthogonalization process. Reducing the amount of global synchronizations is of paramount importance for the efficiency of a numerical algorithm in a massively parallel setting. We improve the parallel strong scaling properties of exponential integrators by addressing the underlying bottleneck in the linear algebra using low-synchronization GS methods. The resulting orthogonalization algorithms have an accuracy comparable to modified Gram-Schmidt yet are better suited for distributed architecture, as only one global communication is required per orthogonalization-step. We present geophysics-based numerical experiments and standard examples routinely used to test stiff time integrators, which validate that reducing global communication leads to better parallel scalability and reduced time-to-solution for exponential integrators.
