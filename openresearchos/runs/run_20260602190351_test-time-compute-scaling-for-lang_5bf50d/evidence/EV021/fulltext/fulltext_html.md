[2312.12282] Parallel iterative solvers for discretized reduced optimality systems

1

1  institutetext:

Ulrich Langer

2

2  institutetext:  Institute of Numerical Mathematics,
JKU Linz,
Austria,

2

2  email:  ulanger@numa.uni-linz.ac.at

3

3  institutetext:  Richard Löscher

4

4  institutetext:  Institut für Angewandte Mathematik,
TU Graz,
Austria,

4

4  email:  loescher@math.tugraz.at

5

5  institutetext:  Olaf Steinbach

6

6  institutetext:  Institut für Angewandte Mathematik, TU Graz, Austria,

6

6  email:  loescher@math.tugraz.at

7

7  institutetext:  Huidong Yang

8

8  institutetext:  Faculty of Mathematics, University of Vienna, Austria,

8

8  email:  huidong.yang@univie.ac.at

Parallel iterative solvers for discretized reduced optimality systems

Ulrich Langer

Richard Löscher

Olaf Steinbach

Huidong Yang

Abstract

We propose, analyze, and test new iterative solvers
for large-scale systems of linear algebraic equations arising from
the
finite element discretization of reduced optimality systems
defining the finite element approximations to the solution of
elliptic
tracking-type distributed optimal control problems
with both the standard

L  2

subscript  𝐿  2

L_{2}

and the more general energy regularizations.
If we aim at
an
approximation of the given desired state

y  d

subscript  𝑦  𝑑

y_{d}

by the computed finite element state

y  h

subscript  𝑦  ℎ

y_{h}

that asymptotically differs from

y  d

subscript  𝑦  𝑑

y_{d}

in the order of the best

L  2

subscript  𝐿  2

L_{2}

approximation
under acceptable
costs for the control,
then the optimal choice of the regularization parameter

ϱ

italic-ϱ

\varrho

is
linked
to the mesh-size

h

ℎ

h

by the relations

ϱ  =

h  4

italic-ϱ

superscript  ℎ  4

\varrho=h^{4}

and

ϱ  =

h  2

italic-ϱ

superscript  ℎ  2

\varrho=h^{2}

for the

L  2

subscript  𝐿  2

L_{2}

and the energy regularization, respectively.
For this setting, we can construct efficient parallel iterative solvers for the reduced finite element optimality systems.
These results can be generalized to
variable regularization parameters
adapted to the local behavior
of the mesh-size that can heavily change in case of adaptive
mesh refinement.
Similar results can be obtained for the space-time finite element discretization
of the corresponding parabolic and hyperbolic optimal control problems.

1  Introduction

Let us first consider an abstract tracking-type, distributed Optimal Control Problem (OCP)
of the form: Find the state

y  ϱ

∈  Y

subscript  𝑦  italic-ϱ

𝑌

y_{\varrho}\in Y

and the control

u  ϱ

∈  U

subscript  𝑢  italic-ϱ

𝑈

u_{\varrho}\in U

minimizing the cost functional

J  ϱ

​

(

y  ϱ

,

u  ϱ

)

:=

1  2

​

‖

y  ϱ

−

y  d

‖

H  2

+

ϱ  2

​

‖

u  ϱ

‖

U  2

=

1  2

​

‖

y  ϱ

−

y  d

‖

H  2

+

1  2

​

‖

ϱ

​

u  ϱ

‖

U  2

assign

subscript  𝐽  italic-ϱ

subscript  𝑦  italic-ϱ

subscript  𝑢  italic-ϱ

1  2

superscript

subscript

norm

subscript  𝑦  italic-ϱ

subscript  𝑦  𝑑

𝐻

2

italic-ϱ  2

superscript

subscript

norm

subscript  𝑢  italic-ϱ

𝑈

2

1  2

superscript

subscript

norm

subscript  𝑦  italic-ϱ

subscript  𝑦  𝑑

𝐻

2

1  2

superscript

subscript

norm

italic-ϱ

subscript  𝑢  italic-ϱ

𝑈

2

J_{\varrho}(y_{\varrho},u_{\varrho}):=\frac{1}{2}\|y_{\varrho}-y_{d}\|_{H}^{2}+\frac{\varrho}{2}\|u_{\varrho}\|_{U}^{2}=\frac{1}{2}\|y_{\varrho}-y_{d}\|_{H}^{2}+\frac{1}{2}\|\sqrt{\varrho}\,u_{\varrho}\|_{U}^{2}

(1)

subject to (s.t.) the state equation

B  ​

y  ϱ

=

u  ϱ

​  in  ​  U

⊆

P  ∗

,

𝐵

subscript  𝑦  italic-ϱ

subscript  𝑢  italic-ϱ

in

𝑈

superscript  𝑃

By_{\varrho}=u_{\varrho}\;\;\mbox{in}\;\;U\subseteq P^{*},

(2)

where

y  d

∈  H

subscript  𝑦  𝑑

𝐻

y_{d}\in H

denotes the given desired state (target),

ϱ  &gt;  0

italic-ϱ  0

\varrho&gt;0

is a suitably chosen
regularization parameter that also
affects
the energy cost

‖

u  ϱ

‖

U  2

superscript

subscript

norm

subscript  𝑢  italic-ϱ

𝑈

2

\|u_{\varrho}\|_{U}^{2}

for the control

u  ϱ

subscript  𝑢  italic-ϱ

u_{\varrho}

as source term in (  2  ), and

X  =

Y  ,  P  ,  U  ,  H

𝑋

𝑌  𝑃  𝑈  𝐻

X=Y,P,U,H

are Hilbert spaces equipped with the corresponding norms

∥  ⋅

∥  X

\|\cdot\|_{X}

and scalar products

(  ⋅  ,  ⋅  )

X

subscript

⋅  ⋅

𝑋

(\cdot,\cdot)_{X}

.
We assume that

Y  ⊂  H  ⊂

Y  ∗

𝑌  𝐻

superscript  𝑌

Y\subset H\subset Y^{*}

and

P  ⊂  H  ⊂

P  ∗

𝑃  𝐻

superscript  𝑃

P\subset H\subset P^{*}

form Gelfand triples of
Hilbert spaces, and that

B  :

Y  →

P  ∗

:  𝐵

→  𝑌

superscript  𝑃

B:Y\rightarrow P^{*}

is an isomorphism,
where

X  ∗

superscript  𝑋

X^{*}

denotes the dual space of

X

𝑋

X

with the duality product

⟨  ⋅  ,  ⋅  ⟩

:

X  ∗

×  X

→  ℝ

:

⋅  ⋅

→

superscript  𝑋

𝑋

ℝ

\langle\cdot,\cdot\rangle:X^{*}\times X\rightarrow\mathbb{R}

that is nothing but the extension of the scalar product

(  ⋅  ,  ⋅  )

H

subscript

⋅  ⋅

𝐻

(\cdot,\cdot)_{H}

in

H

𝐻

H

for

X  =  Y

𝑋  𝑌

X=Y

and

X  =  P

𝑋  𝑃

X=P

.
We are interested in the cases

U  =  H

𝑈  𝐻

U=H

and

U  =

P  ∗

𝑈

superscript  𝑃

U=P^{*}

.
Optimal control problems of the form (  1  )-(  2  )
with many applications were already investigated in
the classical monograph

DD28:Lions:1968a

by Lions and
the more recent book

DD28:Troeltzsch:2010a

by Tröltzsch,
where additional constraints of the form

u  ϱ

∈

U  ad

⊂  U

subscript  𝑢  italic-ϱ

subscript  𝑈

ad

𝑈

u_{\varrho}\in U_{\text{ad}}\subset U

imposed on the control

u  ϱ

subscript  𝑢  italic-ϱ

u_{\varrho}

are permitted. The unique solvability of such kind of OCPs is based on
the unique solvability of the state equation, i.e.

y  ϱ

=

B

−  1

​

u  ϱ

subscript  𝑦  italic-ϱ

superscript  𝐵

1

subscript  𝑢  italic-ϱ

y_{\varrho}=B^{-1}u_{\varrho}

,
the strong convexity of the quadratic cost functional and the assumption that the admissible set

U  ad

subscript  𝑈

ad

U_{\text{ad}}

is a non-empty, convex, and closed subset of

U

𝑈

U

;
cf. Theorem 2.16 in

DD28:Troeltzsch:2010a

.
Here we only consider the case

U  ad

=  U

subscript  𝑈

ad

𝑈

U_{\text{ad}}=U

. Then the unique solution

(

y  ϱ

,

u  ϱ

)

∈

Y  ×  U

subscript  𝑦  italic-ϱ

subscript  𝑢  italic-ϱ

𝑌  𝑈

(y_{\varrho},u_{\varrho})\in Y\times U

of the
OCP (  1  ) - (  2  )
can also be extracted from
the unique solution

(

y  ϱ

,

p  ϱ

,

u  ϱ

)

∈

Y  ×  P  ×  U

subscript  𝑦  italic-ϱ

subscript  𝑝  italic-ϱ

subscript  𝑢  italic-ϱ

𝑌  𝑃  𝑈

(y_{\varrho},p_{\varrho},u_{\varrho})\in Y\times P\times U

of the first-order optimality system (OS)

B  ​

y  ϱ

=

u  ϱ

,

B  ∗

​

p  ϱ

=

y  ϱ

−

y  d

,

p  ϱ

+

A

1  /  ϱ

−  1

​

u  ϱ

=  0

,

formulae-sequence

𝐵

subscript  𝑦  italic-ϱ

subscript  𝑢  italic-ϱ

formulae-sequence

superscript  𝐵

subscript  𝑝  italic-ϱ

subscript  𝑦  italic-ϱ

subscript  𝑦  𝑑

subscript  𝑝  italic-ϱ

superscript

subscript  𝐴

1  italic-ϱ

1

subscript  𝑢  italic-ϱ

0

By_{\varrho}=u_{\varrho},\,B^{*}p_{\varrho}=y_{\varrho}-y_{d},\,p_{\varrho}+A_{1/\varrho}^{-1}u_{\varrho}=0,

(3)

where the self-adjoint and elliptic regularization operator

A

1  /  ϱ

:

P  →

P  ∗

:

subscript  𝐴

1  italic-ϱ

→  𝑃

superscript  𝑃

A_{1/\varrho}:P\rightarrow P^{*}

is defined by the regularization via
the Riesz representation of the control. For

U  =

P  ∗

𝑈

superscript  𝑃

U=P^{*}

, we have

A

1  /  ϱ

=

ϱ

−  1

​  A

subscript  𝐴

1  italic-ϱ

superscript  italic-ϱ

1

𝐴

A_{1/\varrho}=\varrho^{-1}A

and

‖  u  ‖

P  ∗

2

=

⟨  u  ,

A

−  1

​  u

⟩

superscript

subscript

norm  𝑢

superscript  𝑃

2

𝑢

superscript  𝐴

1

𝑢

\|u\|_{P^{*}}^{2}=\langle u,A^{-1}u\rangle

,
whereas

A  =  I

𝐴  𝐼

A=I

(canonical embedding operator) for

U  =  H

𝑈  𝐻

U=H

.
Formally, we will write

‖

ϱ

​  u

‖

U  2

=

⟨  u  ,

A

1  /  ϱ

−  1

​  u

⟩

superscript

subscript

norm

italic-ϱ

𝑢

𝑈