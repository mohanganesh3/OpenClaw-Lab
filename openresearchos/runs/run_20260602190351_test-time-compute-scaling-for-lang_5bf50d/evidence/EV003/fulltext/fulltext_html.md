[2410.14917] Low-synchronization Arnoldi Methods for the Matrix Exponential with Application to Exponential Integrators

Low-synchronization Arnoldi Methods for the Matrix Exponential with Application to Exponential Integrators

Tanya V. Tafolla, Stéphane Gaudreault, Mayya Tokman

Abstract

High order exponential integrators require computing linear combination of exponential like

φ

𝜑

\varphi

-functions of large matrices

A

𝐴

A

times a vector

v

𝑣

v

. Krylov projection methods are the most general and remain an efficient choice for computing the matrix-function-vector-product evaluation when the matrix is

A

𝐴

A

is large and unable to be explicitly stored, or when obtaining information about the spectrum is expensive. The Krylov approximation relies on the Gram-Schmidt (GS) orthogonalization procedure to produce the orthonormal basis

V  m

subscript  𝑉  𝑚

V_{m}

. In parallel, GS orthogonalization requires  global synchronizations  for inner products and vector normalization in the orthogonalization process. Reducing the amount of global synchronizations is of paramount importance for the efficiency of a numerical algorithm in a massively parallel setting. We improve the parallel strong scaling properties of exponential integrators by addressing the underlying bottleneck in the linear algebra using low-synchronization GS methods. The resulting orthogonalization algorithms have an accuracy comparable to modified Gram-Schmidt yet are better suited for distributed architecture, as only one global communication is required per orthogonalization-step. We present geophysics-based numerical experiments and standard examples routinely used to test stiff time integrators, which validate that reducing global communication leads to better parallel scalability and reduced time-to-solution for exponential integrators.

1  Introduction

Many problems in science and engineering require simulating complex systems with dynamics governed by various forces acting over a wide range of time scales. Numerical solutions of such systems typically involve spatial discretization of the partial differential equations (PDEs) and their transformation into large systems of ordinary differential equations (ODEs) using the method of lines. Because these systems span wide temporal scales, the resulting ODEs are typically stiff, necessitating numerical solutions over intervals far exceeding the characteristic time scale of the fastest modes. Therefore, selecting an appropriate time integration method is crucial for ensuring both accuracy and efficiency in computing solutions.

Explicit time integrators are unsuitable for such problems due to strict stability limitations on the timestep and implicit methods are often the preferred choice. With implicit schemes, stiffness manifests in the increased computational complexity of both nonlinear and linear solvers involved in approximating solution at each timestep. A common numerical approach to implementing an implicit method is to use the Jacobian-Free Newton-Krylov algorithm  [ 30 ] . In this method the nonlinear system of equations resulting from the implicit discretization is handled with a Newton iteration which, in turn, involves solution of linear systems using Krylov subspace solvers. For stiff systems, the Krylov method can converge slowly, and a preconditioner is needed to make it efficient. However, constructing an effective preconditioner is often challenging and is the subject of considerable research.

Over the past decades, exponential integrators have emerged as an efficient alternative for numerically integrating large-scale stiff systems. Exponential time integrators express the numerical solution as a linear combination of products of exponential-like functions of the Jacobian, or its approximations, and vectors  [ 33 ,  11 ,  55 ,  56 ,  26 ] . Similar to implicit methods, exponential schemes have good stability properties that allow integration with a large timestep. For problems where a fast preconditioner is not available, exponential methods can offer computational savings per timestep compared to implicit integrators  [ 34 ,  56 ] . This is particularly true when Krylov-projection based algorithms are used for both implicit and exponential integrators. For many large-scale applications of interest, evaluating exponential-like functions of matrices can be more efficient and accurate than computing a rational function of a Jacobian, which is part of an implicit timestep  [ 34 ] .

The dominating cost of exponential integrators is the computation of the exponential-like

φ

𝜑

\varphi

-functions times a vector. Krylov methods have shown to be efficient for problems involving large matrices that are difficult to compute explicitly or store, or when obtaining information about the spectrum is prohibitively expensive  [ 36 ] . The main idea of Krylov subspace methods is that matrix functions can be efficiently computed using a projection of the matrix onto a low-dimensional subspace. For the general case where the matrix is non-symmetric, this is generally accomplished through the Arnoldi iteration. At the base of the Arnoldi method is essentially a Gram-Schmidt orthogonalization process which requires a projection of consecutive Krylov vectors onto an orthogonal basis computed during previous iterations. Thus, the Arnoldi method does not scale well in parallel due to the cost of global communication needed to compute the inner products involved in the orthogonalization procedure. This global communication can become a bottleneck as the number of processors increases  [ 2 ] .

Various techniques have been proposed in recent work on the General Minimal Residual (GMRES) method to reduce global communication by reformulating the orthogonalization process into a projection onto the orthogonal complement  [ 4 ,  51 ,  53 ] . Such algorithms aim to decrease the amount of the global communication required by the Arnoldi iteration to a single call per Arnoldi-step, improving the strong scaling properties of linear solvers on distributed memory parallel systems.

The aim of this paper is to explore Krylov methods that minimize global communication and to extend their application to computing the matrix exponential, with a primary focus on enhancing the overall performance of exponential time integrators. We demonstrate the performance advantages of the new methods using a variety of large-scale problems, including some standard benchmarks routinely used by the numerical weather prediction community  [ 19 ,  58 ] . Our numerical experiments indicate that these low-synchronization orthogonalization methods are strong competitors to current techniques when using a large number of processors. The methods also exhibit a clear trend of improved strong scaling behavior as the number of processors increases, resulting in enhanced efficiency for exponential time integrators. The paper explores the potential of these methods to address the communication bottleneck and improve the overall performance of exponential integrators for large-scale simulations.

The remainder of this paper is organized as follows: Section 2 provides background information and highlights parallel scaling limitations. Section 3 presents current low-synchronization variants and introduces a new hybrid low-synchronization Arnoldi algorithm. Section 4 shows numerical evidence of the improvement in parallel scalability achieved by different exponential time integrators when using low-synchronization Arnoldi methods. Finally, Section 5 draws conclusions and outlines future research directions.

2  Background and Motivation

Exponential time integrators are a family of numerical methods for solving initial value problems of the form

d

d  ​  t

​  u  ​

(  t  )

=

F  ​

(

u  ​

(  t  )

)

,

u  ​

(  0  )

=

u  0

,

u  ​

(  t  )

∈

ℝ  N

formulae-sequence

𝑑

𝑑  𝑡

𝑢  𝑡

𝐹

𝑢  𝑡

formulae-sequence

𝑢  0

subscript  𝑢  0

𝑢