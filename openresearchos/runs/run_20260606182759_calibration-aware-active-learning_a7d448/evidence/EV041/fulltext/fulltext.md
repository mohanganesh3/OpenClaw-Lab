<!-- page 1 -->
Probabilistic Analysis of Least Squares, Orthogonal Projection,
and QR Factorization Algorithms Subject to Gaussian Noise
Ali Lotfi1, Julien Langou2, and Mohammad Meysami3
1Department of Computer Science, University of Saskatchewan; Saskatoon, SK S7N 5A2, Canada
2Department of Applied Mathematics, University of Colorado Denver; Denver, Colorado, 80201, USA
3Department of Mathematics, Clarkson University; Potsdam, NY 13699, USA
October 7, 2024
Abstract
In this paper, we extend the work of Liesen et al. [7], which analyzes how the condition number
of an orthonormal matrix Q changes when a column is added ([Q, c]), particularly focusing on the
perpendicularity of c to the span of Q. Their result, presented in Theorem 2.3 of [7], assumes exact
arithmetic and orthonormality of Q that is a strong assumption when applying these results to numerical
methods such as QR factorization algorithms. In our work, we address this gap by deriving bounds on the
condition number increase for a matrix B without assuming perfect orthonormality even when a column
is not perfectly orthogonal to the span of B. This framework allows us to analyze QR factorization
methods where orthogonalization is imperfect and subject to Gaussian noise. We also provide results on
the performance of orthogonal projection and least squares under Gaussian noise, further supporting the
development of this theory.
1
Introduction
1.1
History
Stability analyses have been conducted for various QR algorithms. For example, in the case of Modified
Gram-Schmidt (MGS) and GSXO (Gram-Schmidt based algorithm studied in [8]), stability and loss of
orthogonality have been investigated in [10], [1], [12] and [8]. These results are based on the assumption that
the number of columns is at most O(1/u), where u is the machine precision. However, algorithms such as
MGS and GSXO appear to generate well-conditioned matrices Q even when the number of columns exceeds
the bounds required for these analyses to hold. Our work is among the few attempts to derive results for
QR algorithms that are independent of machine precision. These methods could potentially be applied to
address the open problem of MGS analysis without any assumptions on the condition number of the input
matrix.
There has been related work on the condition number of random matrices ([3], [13], [5], [14]) and on
matrices subject to noise ([19], [21], [18]). For those interested in an abstract analysis of numerical complexity
under noise using differential geometry, see [2]. Additionally, probabilistic bounds on singular values, such
as those in [15], lead to bounds on the condition number though they are not as strong as those in [3].
These works do not address how well QR factorization algorithms perform when the constraints on matrix
dimension and machine precision are violated, as they typically assume randomness in the matrix components
as columns are added.
Despite accurate normalization in finite precision arithmetic, the crucial step of
orthogonalization requires further investigation. Therefore, there is a need for a theory that models the
loss of orthogonality through Gaussian noise to provide more realistic bounds on the performance of QR
factorization algorithms.
1
arXiv:2409.18905v2  [math.NA]  3 Oct 2024


<!-- page 2 -->
1.2
Overview
As widely recognized in the community, numerical instability is typically not caused by the accumulation of
small errors from different steps in algorithms but often results from a single critical step. In QR algorithms,
the most crucial step is orthogonalizing one column against all the previously generated columns. Therefore,
it is logical to introduce Gaussian noise into the orthogonalization step of algorithms that involve least squares
and orthogonalization, such as QR factorization algorithms. To understand how noise affects computations
and metrics such as the condition number, the problem needs to be broken down into a sequence of steps with
the effects of Gaussian noise analyzed at each stage. This approach reveals that the fundamental challenge
is in bounding the norm of a vector subject to noise. Once a probabilistic analysis of the norm of a vector,
orthogonal projection, and least squares under noise is conducted, one can extend this analysis to methods
such as QR factorization algorithms under the influence of noise.
1.3
Motivation and Problem Solving Strategies
This work builds on the machinery developed in [8] and expands upon the seminal work presented by Liesen,
Rozložník, and Strakoš in [7], which establishes the relationship between the condition number and the least
squares problem. Specifically, the following theorem, which is Theorem 2.3 of [7], is central to this paper:
Theorem 1.1. Suppose that [c, B] ∈RN,n+1 has full column rank, and r ̸= 0 is the residual of the least
squares problem (∥r∥2 = minu ∥c−Bu∥2). Let B = QR be a QR-factorization of the matrix B, and let γ > 0
be a real parameter. Then:
∥r∥2
=
σmin([B, cγ])
γ
n
Y
j=1
σj([B, cγ])
σj(B)
=
1
γ σmin([Q, cγ])σ1([Q, cγ]),
where ∥r∥2 = ∥c −By∥2. Furthermore,
κ([Q, cγ]) = α +
p
α2 −4γ2∥r∥2
2
2γ∥r∥2
,
∥r∥2 = α
γ
κ([Q, cγ])
κ([Q, cγ])2 + 1,
(1)
where α = 1 + γ2∥c∥2
2.
By examining Equation 1 from the above theorem, we can observe the relationship between the condition
number of an orthonormal matrix and the effect of adding a column. However, in finite precision arith-
metic, the assumption that Q is orthonormal may be too strong. When analyzing different QR factorization
algorithms in finite precision arithmetic, one might observe that the crucial step is orthogonalization, not
normalization. Normalization is generally performed well, whereas orthogonalization may introduce signif-
icant error and thus result in a large condition number. This observation motivates us to generalize the
results of [7] to matrices like B that are not necessarily orthogonal but with columns of norm ones.
The analysis is expected to focus on a few key factors, such as how well the least squares step for the
new column q is performed relative to the already computed columns and the condition number of B. Our
strategy is to begin by addressing the problem of bounding the condition number of [B, b] in terms of κ(B)
without assuming anything about B or b. We then introduce more realistic assumptions, similar to those
encountered in QR-factorization algorithms under finite precision arithmetic.
For example, a weak but
realistic assumption might be that the added column b can be normalized while a stronger assumption would
be that b is orthogonalized with respect to span(B). To model this, we consider the introduction of Gaussian
noise to b.
Considering Theorem 2.3 of [7], we expect to encounter situations involving least squares and orthogonal
projection under Gaussian noise and we will further explore in this paper.
The structure of this paper is as follows: In Section 2, we explain the notations used throughout the
paper. In Section 3, we present results on how the condition number of a matrix grows when a column is
added, under the most general conditions with no assumptions. We then develop the necessary tools for
analyzing Gaussian noise in the norm of a vector, orthogonal projection, and least squares in Sections 4, 5,
2


<!-- page 3 -->
and 6, respectively. Finally, in Section 7, we apply our findings to QR factorization algorithms and analyze
their probabilistic behavior when the orthogonalization step is subject to noise.
2
Notation
In this section, we define the notation used throughout this paper.
• Random Variables and Vectors:
X:
A random variable or random vector.
E(X):
The expected value of the random variable X.
σ2 or V ar(X):
The variance of the random variable X.
Cov(X, Y ):
The covariance of the random variables X and Y .
E(X):
The vector with the i-th component as E(Xi), for 1 ≤i ≤n.
• Matrices:
I or Ik×k:
The identity matrix of size k × k.
Mm×n(R):
The set of all m × n matrices with real entries.
A[i1 : i2, j1 : j2]:
The submatrix of A containing rows from i1 to i2 and columns from j1 to j2.
A[:, j]:
The j-th column of matrix A.
A[i, :]:
The i-th row of matrix A.
span(A):
The subspace spanned by the columns of matrix A.
AT :
The transpose of matrix A, where (AT )ij = Aji.
λmin(A), λmax(A):
The minimum and maximum eigenvalue of the matrix A respectivly.
eig(A):
The set eigen-values of A.
σmin(A):
The minimum singular value of the matrix A, defined as λmin(AT A).
σj(A):
The j-th largest singular value of matrix A.
σmax(A):
The maximum singular value of matrix A.
κ(A):
The condition number of matrix A, defined as σmax(A)
σmin(A) .
• Norms:
∥a∥2:
The Euclidean norm of vector a, calculated as
pPm
i=1 a2
i for a vector of size m.
∥A∥2:
The spectral norm of matrix A, equivalent to the largest singular value σmax(A).
∥A∥F :
The Frobenius norm of matrix A, defined as
qPm
i=1
Pn
j=1 a2
ij.
⟨x, y⟩:
The inner product of the vector x and y.
• Projections:
PA(−):
The orthogonal projection onto the subspace spanned by A if A is a subspace,
or onto the column space of A if A is a matrix.
P ⊥
A (−):
The orthogonal projection onto the orthogonal complement A⊥,
where A⊥is defined as the complement of A in the space.
3
Condition Number Growth Induced by Adding a Column Under
No Assumption
In this section, we provide general bounds on the growth of the condition number as columns are added to
a matrix without making any assumptions about the matrix or the added column. These general bounds
can serve as a foundation that can be refined when specific assumptions are introduced. For instance, if we
assume that the columns of the matrix have a norm of one, as in the previous section, tighter bounds can
be obtained.
Theorem 3.2 extends Theorem 2.3 in [7]. To prove this theorem, which bounds the growth of the condition
number as columns are added, we begin by proving the following lemma:
3


<!-- page 4 -->
Lemma 3.1. Let a ∈Rn−1, b ∈R, and A =
 
0
aT
a
b
 
, then:
eig(A) ⊆
(
0, b ±
p
b2 + 4∥a∥2
2
2
)
.
Proof.
chA(x) = det(A −xI)
=


−x
0
· · ·
0
a1
0
−x
· · ·
0
a2
...
...
−x
an−1
a1
a2
· · ·
an−1
b −x


Clearly when ∥a∥2 = 0, then:
eig(A) = {0, b}
= {0, b ±
p
b2 + 4∥a∥2
2
2
}
as desired. Therefore, we assume that ∥a∥2 ̸= 0. Since rank(A) ≤2, therefore, a solution to chA(x) is zero.
Therefore, we assume that x ̸= 0. Let Rn+ ai
x (B) be the operator of multiplying row i of B by ai
x and adding
it to the n-th row. Then:
chA(x) = det (A −xI)
= det
  n−1
Y
i=1
Rn+ ai
x
!
(A −xI)
!
=


−x
0
· · ·
0
a1
0
−x
· · ·
0
a2
...
...
−x
an−1
0
0
· · ·
0
b −x +
Pn−1
i=1 a2
i
x

.
Therefore, chA(x) = (−1)nxn(b−x+
Pn−1
i=1 a2
i
x
). This means that nontrivial solution to chA(x) = 0 is obtained
from the quadratic:
0 = x2 + bx + ∥a∥2
2.
which has roots:
b ±
p
b2 + 4∥a∥2
2
2
as desired.
With Lemma 3.1 established, we are now ready to prove the main result on condition number growth as
a column is added:
Theorem 3.2. Assume B is a matrix, and let x, y, and b be three vectors, such that x ∈span(B⊥), and
b = (x + y)γ, where γ ∈R, then:
1.
κ([B, b]) ≤
s
2σ2max(B) + γ2∥x + y∥2
2 +
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
2σ2
min(B) + γ2∥x + y∥2
2 −
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
(2)
2. If for some ε > 0, ∥BT y∥2
∥x+y∥2 ≤ε2−1
4
, then:
4


<!-- page 5 -->
κ
 
[B,
x + y
∥x + y∥2
]
 
≤κ(B)
p
1 + f(B, ε).
where:
f(B, ε) =
ε(1 +
1
κ(B)2 ) −(1 −
1
κ(B)2 )
(2σ2
min(B) + 1 −ε)
.
Proof. In order to prove item 1, we start by the following:
[B, (x + y)γ]T [B, (x + y)γ]
=
 
BT B
γBT (x + y)
γ(x + y)T B
γ2(x + y)T (x + y)
 
=
 
BT B
0
0
γ2∥x + y∥2
2
 
+
 
0
γBT y
γyT B
0
 
σmax ([B, (x + y)γ])
=
q
λmax ([B, (x + y)γ]T [B, (x + y)γ])
=
s
λmax
  
BT B
0
0
1
 
+
 
0
γBT y
γyT B
γ2∥x + y∥2
2
  
≤
s
λmax
  
BT B
0
0
1
  
+ λmax
  
0
γBT y
γyT B
γ2∥x + y∥2
2
  
≤
s
σ2max(B) + λmax
  
0
γBT y
γyT B
γ2∥x + y∥2
2
  
Let A =
 
0
γBT y
γyT B
γ2∥x + y∥2
2
 
, then by Lemma 3.1, we obtain:
eig(A) = {0, γ2∥x + y∥2
2 ±
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
2
}
σmax([B, (x + y)γ])
=
q
λmax ([B, (x + y)γ]T [B, (x + y)γ])
≤
s
σ2max(B) + γ2∥x + y∥2
2 +
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
2
Since
 
BT B
0
0
1
 
and
 
0
γBT y
γyT B
γ2∥x + y∥2
2
 
are Hermitian matrices, then:
σmin([B, (x + y)γ])
=
q
λmin ([B, (x + y)γ]T [B, (x + y)γ])
=
s
λmin
  
BT B
0
0
1
 
+
 
0
γBT y
γyT B
γ2∥x + y∥2
2
  
≥
s
λmin
  
BT B
0
0
1
  
+ λmin
  
0
γBT y
γyT B
γ2∥x + y∥2
2
  
≥
s
σ2
min(B) + γ2∥x + y∥2
2 −
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
2
.
Therefore, we have:
κ([B, b]) ≤
s
2σ2max(B) + γ2∥x + y∥2
2 +
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
2σ2
min(B) + γ2∥x + y∥2
2 −
p
γ4∥x + y∥4
2 + 4γ2∥BT y∥2
2
.
5


<!-- page 6 -->
We will finish by the proof of item 2. Take γ =
1
∥x+y∥2 in item 1, then
κ([B,
x + y
∥x + y∥2
]) ≤
s
2σ2max(B) + 1 +
p
1 + 4γ2∥BT y∥2
2
2σ2
min(B) + 1 −
p
1 + 4γ2∥BT y∥2
2
.
By assumption
q
1 + 4 ∥BT y∥2
2
∥x+y∥2
2 ≤ε, therefore:
κ
 
[B,
x + y
∥x + y∥2
]
 
≤
s
2σ2max(B) + 1 + ε
2σ2
min(B) + 1 −ε .
Furthermore,
κ([B, b])
≤
s
2σ2max(B) + 1 + ε
2σ2
min(B) + 1 −ε
=
s
2σ2max(B) + 1 + ε
2σ2
min(B) + 1 −ε
(3)
Furthermore,
2σ2
max(B) + 1 + ε
2σ2
min(B) + 1 −ε
=
2σ2
max(B) + 1 + ε
2σ2
min(B) + 1 −ε −2σ2
max(B)
2σ2
min(B) + κ(B)2
=
ε
 σ2
max(B) + σ2
min(B)
 
−
 σ2
max(B) −σ2
min(B)
 
σ2
min(B) (2σ2
min(B) + 1 −ε)
+ κ(B)2
=

1 +
ε
 
1 +
1
κ(B)2
 
−
 
1 −
1
κ(B)2
 
(2σ2
min(B) + 1 −ε)

κ(B)2
=
(1 + f(B, ε)) κ(B)2.
Substituting the last Equation in Eq. 3, we obtain:
κ
 
[B,
x + y
∥x + y∥2
]
 
≤κ(B)
p
1 + f(B, ε).
4
Probabilistic Analysis of the Norm of a Vector Subject to Gaus-
sian Noise
In this section, we provide results related to orthogonal projection and least squares under Gaussian noise.
We derive an exact expression for P(∥X + Y ∥2 > ε). This is accomplished by analyzing the modified Bessel
function and two variations of the Marcum Q-function, and their connection to the noncentral chi-squared
distribution.
The cumulative distribution function (CDF) of the noncentral chi-squared distribution, which we will
show is the CDF of ∥X + Y ∥2, relies on these two functions: (i) the modified Bessel function and (ii) the
second is the Marcum-Q function.
Definition 4.1 (Bessel function). For a real number ν, the modified Bessel function of the first kind of order
ν, denoted Iν, is defined by:
Iν(t) =
X
n≥0
(t/2)2n+ν
n!Γ(ν + n + 1).
6


<!-- page 7 -->
Definition 4.2 (Generalized Marcum-Q function). For β ≥0 and α, M > 0, the generalized Marcum-Q
function, denoted Qma
M , is defined by:
Qma
M (α, β) =
1
αM−1
Z ∞
β
xMe−x2+α2
2
IM−1 (αx) dx.
Furthermore, for α = 0 (Equation 4.22 of [16]), define:
Qma
M (0, β) = Γ(M, β2)
Γ(M)
.
where Γ(−) and Γ(−, −) are Gamma and complementary Gauss incomplete gamma functions respectively.
We are now present the main result of this section:
Theorem 4.3. If X ∈Rm and Y is a random vector such that Yis are i.i.d. and Yi ∼N(0, σ2), then:
P(∥X + Y ∥2 > ε) = Qma
m/2
 ∥X∥2
σ
, ε
σ
 
.
Proof. Define Zi = Xi + Yi. Thus, Zi
σ ∼N
  Xi
σ , 1
 
. This implies:
∥X + Y ∥2
2 = σ2
m
X
i=1
 Zi
σ
 2
.
Note that Pm
i=1
  Zi
σ
 2 follows a non-central chi-squared distribution. This distribution, as described in [9],
has two parameters, the degrees of freedom m and the noncentrality parameter λ, which in our case is:
λ =
m
X
i=1
 Xi
σ
 2
= ∥X∥2
2
σ2
.
Therefore, ∥X+Y ∥2
2
σ2
has a non-central chi-squared distribution with parameters m and λ = ∥X∥2
2
σ2 . Using the
CDF of the non-central chi-squared distribution [9], we obtain:
P(∥X + Y ∥2 > ε) = P
 ∥X + Y ∥2
2
σ2
> ε2
σ2
 
= 1 −P
 ∥X + Y ∥2
2
σ2
≤ε2
σ2
 
= Qma
m/2
 ∥X∥2
σ
, ε
σ
 
.
Corollary 4.4. If Y ∈Rm is a random vector such that Yi ∼N(0, σ2), then:
P(∥Y ∥2 ≤ε) = 1 −Qma
m/2(0, ε
σ ).
Proof. This result follows directly from Theorem 4.3 by setting X = 0 and applying basic probability
theory.
4.1
Numerical Test of Theorem 4.3
We validate Theorem 4.3 through numerical experiments implemented in Matlab. We begin by considering
the case where ∥X∥2 = 1 and ε = 0.9 (Figure 1). The results show perfect agreement between the empirical
data (red curve) and the theoretical prediction (blue curve). A similar test is conducted for ∥X∥2 = 1 and
ε = 1.5, with results presented in Figure 2. Again, the numerical results align perfectly with the theory. This
demonstrates that when noise with small variance is added, ∥X + Y ∥2 is unlikely to be large, as expected.
7


<!-- page 8 -->
0
2
4
6
8
10
0.99999999965
0.9999999997
0.99999999975
0.9999999998
0.99999999985
0.9999999999
0.99999999995
1
Prob( || X + Y || > 
 |X|) 
n = 100, epsilon = 0.9000, M =  1e+05, ||X||=1e+00
Figure 1: Comparison of numerical versus theoretical results to test Theorem 4.3 (ε = 0.9). The red
triangles represent the empirical data obtained from numerical experiments and the blue line corresponds
to the theoretical predictions.
5
Orthogonal Projection Subject to Gaussian Noise
In this section, we present a theorem regarding the behavior of orthogonal projections under the influence
of Gaussian noise.
Theorem 5.1. Let Q be an orthonormal m × n matrix with m ≥n, X ∈span(Q⊥) ⊂Rm = span(Q⊥) ⊕
span(Q), and Y ∈Rm be a random vector with i.i.d.
components Yi ∼N(0, σ2).
Define ¯Y such that
Q⊥¯Y = P ⊥
Q (Y ). Then:
• The components ¯Yi are i.i.d. and ¯Yi ∼N(0, σ2).
• The probability that the norm of X+P ⊥
Q (Y ) exceeds ε is given by Qma
(m−n)/2
 
∥X∥2
σ
, ε
σ
 
, where Qma
−(−, −)
is defined in Definition 4.2.
Proof. Since P ⊥
Q Y ∈span(Q⊥), there exist ai ∈R and bj ∈R (for 1 ≤j ≤n and 1 ≤i ≤m −n, as the
dimensions of Q⊥and Q are m −n and n, respectively) such that:
Y =
m−n
X
i=1
aiQ⊥[:, i] +
n
X
j=1
bjQ[:, j].
(4)
Applying P ⊥
Q to both sides of Equation 4, we have:
P ⊥
Q (Y ) =
m−n
X
i=1
aiQ⊥[:, j]
=
m−n
X
i=1
⟨Y, Q⊥[:, i]⟩Q⊥[:, i].
8

[CAPTION] Figure 1: Comparison of numerical versus theoretical results to test Theorem 4.3 (ε = 0.9). The red


<!-- page 9 -->
0
2
4
6
8
10
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Prob( || X + Y || > 
 |X|) 
n = 100, epsilon = 1.5000, M =  1e+05, ||X||=1e+00
Figure 2: Comparison of numerical versus theoretical results to test Theorem 4.3 (ε = 1.5).
Furthermore, by one of the assumptions, we know that:
P ⊥
Q (Y ) = Q⊥¯Y
=
m−n
X
i=1
¯YiQ⊥[:, i].
(5)
Therefore, combining the last two equations, we have:
m−n
X
i=1
¯YiQ⊥[:, i] =
m−n
X
i=1
⟨Y, Q⊥[:, i]⟩Q⊥[:, i].
(6)
Given the last equation and the fact that the columns of Q⊥are linearly independent, we obtain:
¯Yi = ⟨Q⊥[:, i], Y ⟩.
Claim 5.2. The ¯Yi are independent, and ¯Yi ∼N(0, σ2).
Proof of Claim.
We showed that ¯Yi = ⟨Q⊥[:, i], Y ⟩, and E( ¯Yi) = ⟨Q⊥[:, i], E(Y )⟩= 0.
Therefore, for
1 ≤i, j ≤m −n, we have:
Cov( ¯Yi, ¯Yj) = E( ¯Yi ¯Yj)
= E
 ⟨Q⊥[:, i], Y ⟩⟨Q⊥[:, j], Y ⟩
 
= E


m−n
X
i1=1
m−n
X
j1=1
Q⊥[i1, i]Yi1Q⊥[j1, j]Yj1

.
9

[CAPTION] Figure 2: Comparison of numerical versus theoretical results to test Theorem 4.3 (ε = 1.5).


<!-- page 10 -->
Since the Yi are i.i.d., then:
E


m−n
X
i1=1
m−n
X
j1=1
Q⊥[i1, i]Yi1Q⊥[j1, j]Yj1

= E
 m−n
X
i1=1
Q⊥[i1, i]Yi1Q⊥[i1, j]Yi1
!
=
m−n
X
i1=1
Q⊥[i1, i]Q⊥[i1, j]E(Y 2
i1)
= ⟨Q⊥[:, i], Q⊥[:, j]⟩σ2.
Combining the last set of equations and noting that Q⊥is orthonormal, we get:
Cov( ¯Yi, ¯Yj) = 0 ⇐⇒i ̸= j
and
Var( ¯Yi) = ⟨Q⊥[:, i], Q⊥[:, i]⟩σ2
= σ2.
Therefore, combining the last two sets of equations, and considering that an affine combination of random
normals is normal, it follows that:
¯Yi ∼N(0, σ2),
completing the proof of the claim.
□
Notice that:
∥P ⊥
Q (X + Y )∥2 = ∥P ⊥
Q (X) + P ⊥
Q (Y )∥2
= ∥Q⊥( ¯X) + Q⊥( ¯Y )∥2
= ∥¯X + ¯Y ∥2,
where the last equality holds since Q⊥is orthonormal.
Since X ∈span(Q⊥), then ¯Xi = Xi for 1 ≤i ≤n. Thus, P(∥X + P ⊥
Q (Y )∥2 > ε) = P(∥¯X + ¯Y ∥2 > ε),
and by Theorem 4.3, we have:
P(∥X + P ⊥
Q (Y )∥2 > ε) = Qma
(m−n)/2
 ∥X∥2
σ
, ε
σ
 
.
(7)
We can now derive corollaries for practical cases such as tall matrices.
Corollary 5.3. Let B be an m × n matrix with m ≥n, and Y ∈Rm be a random vector with i.i.d.
components Yi ∼N(0, σ2). Then:
P(∥P ⊥
B (Y )∥2 ≤ε) = 1 −Qma
(m−n)/2(0, ε
σ ).
The following corollary illustrates how substantial the probability can be for tall matrices.
Corollary 5.4. Under the assumptions of Theorem 5.1, with m −n ≥100, σ = 2−8, and ε =
1
10, we have:
P(∥X + P ⊥
Q (Y )∥2 > 1/10) ≥0.9734.
10


<!-- page 11 -->
6
Least Squares Subject to Gaussian Noise
In this section, we aim to develop an intuitive understanding of how Gaussian noise affects the least squares
problem. Suppose we are given a vector q and a matrix B whose columns span a subspace. The goal is to
find a vector r such that ∥r∥2 = minu ∥q −Bu∥2. In exact arithmetic, this problem is solved by projecting
q onto span(B⊥), ensuring that r lies precisely in this subspace and ∥r∥2 = minu ∥q −Bu∥2.
However, in finite precision arithmetic, the projection onto span(B⊥) is not perfect, resulting in a vector
that does not exactly belong to span(B⊥). This deviation can be modeled by introducing Gaussian noise to
the projection of q onto span(B⊥). With this intuition in mind, we now present a theorem that calculates
the probability that ∥r∥2 exceeds a certain threshold, given that the projection is subject to Gaussian noise.
Theorem 6.1. Assume A is an m × n full-rank matrix with m > n, and Q is obtained from the QR
factorization of A. Let X ∈span(Q⊥) ⊂Rm = span(Q⊥) ⊕span(Q) and Y be a random vector with i.i.d.
components Yi ∼N(0, σ2). If r =
P ⊥
Q (X+Y )
√
∥P ⊥
Q (X+Y )∥2
2+∥PQY ∥2
2 , then for all ε1 ≥0 and ε2 > 0, we have:
P



∥r∥2 ≥
1
r
1 +
 
ε1
ε2
 2



= F ′
m−n,n
 
nε2
(m −n)ε1
; ∥X∥2
2
σ2
 
Proof. We start by simplifying ∥r∥2 as follows:
∥r∥2
=
∥P ⊥
Q (X + Y )∥2
q
∥P ⊥
Q (X + Y )∥2
2 + ∥PQY ∥2
2
=
1
r
1 +
∥PQ(Y )∥2
2
∥P ⊥
Q (X+Y )∥2
2
=
1
r
1 + (
∥PQY ∥2
∥X+P ⊥
Q (Y )∥2 )2
.
Define ¯Q = [Q, Q⊥], therefore, span( ¯Q) = Rm. Take ¯Y such that:
¯Q ¯Y = P ¯
Q(Y ),
then by Theorem 5.1 (in Theorem 5.1, take Q⊥= ¯Q and n = 0), we have that ¯Yis are i.i.d. with ¯Yi ∼N(0, σ2).
Define:
Y1
=
¯Y [1 : n]
Y2
=
¯Y [n + 1 : m],
since ¯Yis are i.i.d., then Y1 and Y2 are two independent random vectors in Rn and Rm−n respectively.
Furthermore, define:
Z1
=
∥PQ(Y )∥2
Z2
=
∥X + P ⊥
Q (Y )∥2,
Therefore,
Z1 = ∥QY1∥2 = ∥Y1∥2
Z2 = ∥X + P ⊥
Q (Y )∥2
= ∥Q⊥¯X + Q⊥Y2∥2
= ∥Q⊥¯X + Q⊥Y2∥2
= ∥¯X + Y2∥2.
11


<!-- page 12 -->
Therefore,
P

∥r∥2 ≥
1
q
1 + ( ε1
ε2 )2


=
P


∥P ⊥
Q (X + Y )∥2
q
∥P ⊥
Q (X + Y )∥2
2 + ∥PQY ∥2
2
≥
1
q
1 + ( ε1
ε2 )2


=
P




1
r
1 +
∥PQ(Y )∥2
2
∥P ⊥
Q (X+Y )∥2
2
≥
1
q
1 + ( ε1
ε2 )2




=
P




1
r
1 + (
∥PQ(Y )∥2
∥X+P ⊥
Q (Y )∥2 )2
≥
1
q
1 + ( ε1
ε2 )2




=
P


1
q
1 + ( Z1
Z2 )2
≥
1
q
1 + ( ε1
ε2 )2


=
P
 Z1
Z2
≤ε1
ε2
 
=
P
 ε2
ε1
≤Z2
Z1
 
=
1 −P
 Z2
Z1
< ε2
ε1
 
(8)
By Corollary 5.3 and Theorem 5.1, we respectively have:
P(Z1 ≤ε1)
=
1 −Qma
n/2(0, ε1
σ )
P(Z2 ≤ε2)
=
1 −Qma
(m−n)/2(∥X∥2
σ
, ε2
σ ).
Since Z1 ∼χ2
n, Z2 ∼χ′2
m−n
 
∥X∥2
2
σ2
 
, therefore, by Chapter 6, equation 40 of [11], we have:
Z2/(m −n)
Z1/n
∼F ′
m−n,n
 ∥X∥2
2
σ2
 
Therefore,
P



∥r∥2 ≥
1
r
1 +
 
ε1
ε2
 2



= 1 −P
 Z2
Z1
< ε2
ε1
 
= 1 −P
 
Z2
m−n
Z1
n
<
ε2
m−n
ε1
n
!
= F ′
m−n,n
 
nε2
(m −n)ε1
; ∥X∥2
2
σ2
 
concluding the proof.
7
Imperfect Orthogonalization and Perfect Normalization in QR
Factorization Algorithms
We now analyze how the addition of a column affects the condition number, particularly in the context
of QR factorization algorithms. We consider the case where orthogonalization might not be perfect, but
normalization is assumed to be accurate. This analysis is important for understanding the numerical stability
12


<!-- page 13 -->
of these algorithms under realistic conditions where Gaussian noise may affect the orthogonalization process.
The following theorem builds on the work of Liesen et al. [7]. We will extend their results to analyze the
impact on the condition number when a column is added to a matrix. This will be done without assuming
perfect orthogonality that is important for applications in numerical methods where finite precision arithmetic
is used.
Theorem 7.1. Assume B ∈Mm×n(R), c ∈Rm, γ ∈R, B = QR, and α = 1 + γ2∥c∥2
2. Then:
1. The maximum singular value of the matrix after adding the column is bounded by:
σmax([B, cγ]) ≤max{σmax(B), γ}σmax ([Q, c]) .
2. The minimum singular value of the matrix after adding the column satisfies:
σmin([B, cγ]) ≥min{σmin(B), γ}σmin ([Q, c]) .
3. The condition number of the matrix after adding the column is bounded by:
κ([B, cγ]) ≤max{κ(B), ∥B∥2, ∥B†∥2}κ([Q, cγ])
= max{κ(B), ∥B∥2
γ
, γ∥B†∥2}κ([Q, c]).
Proof. The proof relies on the analysis of the matrix product [B, cγ]T [B, cγ], which can be expressed as:
[B, cγ]T [B, cγ] =
 
RT
0
0
γ
 
[Q, c]T [Q, c]
 R
0
0
γ
 
.
Taking the maximum eigenvalue λmax on both sides leads to:
σmax([B, cγ]) =
q
λmax ([B, cγ]T [B, cγ]),
which is further bounded by:
σmax([B, cγ]) ≤max{σmax(R), γ}σmax ([Q, c])
= max{σmax(B), γ}σmax ([Q, c]) .
A similar argument applies to the minimum singular value σmin, leading to:
σmin([B, cγ]) ≥min{σmin(R), γ}σmin ([Q, c])
= min{σmin(B), γ}σmin ([Q, c]) .
Finally, the condition number κ([B, cγ]) is obtained by dividing the maximum singular value by the minimum
singular value:
κ([B, cγ]) ≤max{σmax(B), γ}
min{σmin(B), γ} κ([Q, c])
= max{κ(B), ∥B∥2
γ
, γ∥B†∥2}κ([Q, c]).
This concludes the proof.
This theorem provides a method to bound the condition number of a matrix as a new column is added,
even when the orthogonalization step is imperfect. Such results are crucial for understanding the stability
of numerical algorithms, particularly in the context of QR factorization under realistic conditions.
The
following theorem, which is our final result, offers an upper bound for the growth of the condition number
when a column is added, assuming that the norms of the existing columns of the matrix are one.
13


<!-- page 14 -->
Theorem 7.2. Assume B ∈Mm×n(R) with columns of norm 1, c ∈Rm, γ ∈R, and α = 1 + γ2∥c∥2
2. If
∥r∥2 = minz{cγ −Bz}, then:
κ([B, cγ]) ≤κ(B)α +
p
(α2 −4γ2∥r∥2
2)
2γ∥r∥2
.
Proof. We will start by the following claim.
Claim 7.3. If columns of B have norm 1, then σmax(B) ≥1 and σmin(B) ≤1.
Proof of Claim. Take e1 = [1, 0, · · · , 0], then:
σmin(B) = inf{∥Bx∥2 : ∥x∥2 = 1|}
≤∥Be1∥2
≤sup{∥Bx∥2 : ∥x∥2 = 1|}
= σmax(B).
Since columns of B have norm 1, then ∥Be1∥2 = 1, therefore,
σmin(B) ≤1 ≤σmax(B).
□
Thus,
max{σmax(B), 1}
min{σmin(B), 1} = σmax(B)
σmin(B)
= κ(B).
Furthermore, by Theorem 7.1, item 3, it yields:
κ([B, cγ]) ≤κ(B)κ([Q, cγ]).
From Theorem 2.3 [7] Equation 2.8, we have:
κ([Q, cγ]) = α +
p
(α2 −4γ2∥r∥2)
2γ∥r∥
.
Therefore,
κ([B, cγ]) ≤κ(B)α +
p
(α2 −4γ2∥r∥2
2)
2γ∥r∥2
finishing the proof.
Corollary 7.4. Assume B ∈Mm×n(R) with columns of norm 1, q ∈Rm, such that ∥q∥2 = 1. If ∥r∥2 =
minz{q −Bz}, then:
κ([B, q]) ≤κ(B)1 +
p
(1 −∥r∥2
2)
∥r∥2
.
Proof. This corollary follows directly by setting c = q and γ = 1 in Theorem 7.2.
Corollary 7.5. Assume B is an m × n full rank matrix (m > n) and X ∈span(B⊥) ⊂Rm =
span(B⊥) L span(B) and Y is a random vector such that Yi’s are i.i.d. with Yi ∼N(0, σ2). Then for
all ε1 ≥0 and ε2 > 0, we have:
κ([B, q]) ≤
 
ε1
r
1 + (ε1
ε2
)2
 
κ(B)
with probability at least:
F ′
m−n,n
 
nε2
(m −n)ε1
; ∥X∥2
2
σ2
 
.
14


<!-- page 15 -->
Proof. The proof follows by recognizing that
1+√
1−∥r∥2
2
∥r∥2
is a decreasing function of ∥r∥2, and then applying
Corollary 7.4 and Theorem 6.1.
The following corollary, which is the last, will analyze QR factorization algorithms under the model
considered in this paper.
Specifically, suppose we are analyzing a QR factorization under the model in
which normalization is done perfectly, but orthogonalization is subject to Gaussian noise with mean zero
and variance σ2.
Corollary 7.6. Assume A is a full rank m × n matrix (m ≥n). Furthermore, when the suppose a QR
factorization algorithm is run on A and computes ˆQ, at iteration i for all 1 ≤i ≤n, orthogonalization is
subject to noise. Meaning ˆqi =
ai+ei
∥ai+ei∥2 , and ai ∈span( ˆQ[:, 1 : i −1])⊥and ei is a random vector in Rm
with ei i.i.d. and ei ∼N(0, σ2I). Then for all for i, 2 ≤i ≤n, and 0 < ϵ2,i, ϵ1,i, with probability at least:
1 −
n
X
i=2
 
1 −F ′
m−i,i
 
iε2
(m −i)ε1
; ∥ai∥2
2
σ2
  
,
we have:
κ( ˆQ) ≤
n
Y
i=2
 
ε1,i
r
1 + (ε1,i
ε2,i
)2
 
Proof. Define the event Ei for 2 ≤i ≤n, as follows:
Ei =
 
κ( ˆQ[:, i]) ≤κ( ˆQ[:, i −1])
 
ε1,i
r
1 + (ε1,i
ε2,i
)2
  
.
By exclusion principle and Corollary 7.5, we achieve that:
P
 n
\
i=2
Ei
!
=
1 −P
 n
[
i=2
Ec
i
!
≥1 −
n
X
i=2
P(Ec
i )
=
1 −
n
X
i=2
 
1 −F ′
m−i,i
 
iε2
(m −i)ε1
; ∥wi∥2
2
σ2
  
.
(9)
Therefore, with probability at least (9), we have Ei holds for 2 ≤i ≤n. If all Eis hold, we have:
κ( ˆQ) ≤
n
Y
i=2
 
ε1,i
r
1 + (ε1,i
ε2,i
)2
 
finishing the proof.
8
Conclusion and Future Research
In this paper, we extended the work presented in [7], by proving a new theorem that bounds the condition
number of a matrix as a new column is added, even in the presence of orthogonalization noise. This result
is important for understanding the behavior of QR factorization algorithms under realistic conditions that
the orthogonalization step is not perfect due to finite precision arithmetic. We also developed a method to
probabilistically bound the norm of vectors subjected to Gaussian noise. This is crucial for analyzing least
squares solutions under these conditions
We then analyzed the effects of Gaussian noise on orthogonal projections and applied this to QR factor-
ization algorithms. We investigated how the condition number can grow due to noise in the orthogonalization
step with specific implications for the Modified Gram-Schmidt (MGS) algorithm. This offers a new perspec-
tive on understanding MGS without the usual restrictions on machine precision or the number of columns
in the input matrix.
15


<!-- page 16 -->
Our findings not only have applications in numerical linear algebra, but they can also be utilized in
machine learning and deep learning, where QR factorization is critical. The stability of algorithms under
noisy conditions in training deep neural networks is important because limited floating-point precision or
noise in data collection can lead to large numerical errors ([6]). The probabilistic bounds on the condition
number under noise can be used for developing more robust training algorithms, especially, in distributed
learning environments where numerical errors can accumulate ([17]).
The framework we developed could be applied to other dimension reduction factorization methods that
are in machine learning, such as singular value decomposition (SVD) and principal component analysis
(PCA)([20]). Extending our probabilistic analysis to these methods could potentially lead to more reliable
algorithms for high-dimensional data processing that noise can adversely impact stability and accuracy.
Future research could focus on extending the analysis to other numerical algorithms or to non-Gaussian
noise models. One could also extend these results to algorithms that involve the approximation of gradients,
which in exact arithmetic should be orthogonal to the tangent plane. However, in finite precision arithmetic,
they are subject to noise, in which case our results become relevant. This study could also provide new
insights into the convergence and stability analysis of these methods under noisy conditions in the context of
stochastic optimization algorithms commonly used in deep learning([4]). More specifically, in algorithms that
cannot be analyzed under traditional models due to their large dimensional input matrices and/or machine
precision not being small enough, our model could potentially provide probabilistic bounds on performance.
Acknowledgments
The first author would like to express his sincere gratitude to Professor Julien Langou and Professor Sean
O’Rourke, who were the previous coadvisors of the first author, for their invaluable guidance and support
during the development of the foundation for this paper. Their mentorship and encouragement were instru-
mental in the completion of this work. The authors would like to thank Professor Miroslav Rozložník for
his constructive feedback. The first author would like to thank Professor Vahid Lotfi for his guidance and
support in this project.
References
[1] Åke Björck and Christopher C Paige. Loss and recapture of orthogonality in the modified gram–schmidt
algorithm. SIAM journal on matrix analysis and applications, 13(1):176–190, 1992.
[2] Peter Bürgisser, Felipe Cucker, and Martin Lotz. The probability that a slightly perturbed numerical
analysis problem is difficult. Mathematics of Computation, 77(263):1559–1583, 2008.
[3] Zizhong Chen and Jack J Dongarra. Condition numbers of gaussian random matrices. SIAM Journal
on Matrix Analysis and Applications, 27(3):603–620, 2005.
[4] Pierre Collet and Jean-Philippe Rennard. Stochastic optimization algorithms. In Intelligent information
technologies: Concepts, methodologies, tools, and applications, pages 1121–1137. IGI Global, 2008.
[5] Alan Edelman.
Eigenvalues and condition numbers of random matrices.
SIAM journal on matrix
analysis and applications, 9(4):543–560, 1988.
[6] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.
Explaining and harnessing adversarial
examples. arXiv preprint arXiv:1412.6572, 2014.
[7] Jörg Liesen, Miroslav Rozložník, and Zdeněk Strakoš. Least squares residuals and minimal residual
methods. SIAM Journal on Scientific Computing, 23(5):1503–1525, 2002.
[8] Ali Lotfi. Numerical Stability of the GSXO Orthogonalization Scheme. PhD thesis, University of Col-
orado at Boulder, 2022.
[9] A. Nuttall. Some integrals involving the< tex > qm < /tex >function (corresp.). IEEE Transactions
on Information Theory, 21(1):95–96, 1975.
16


<!-- page 17 -->
[10] Christopher C Paige, Miroslav Rozložník, and Zdeněk Strakoš. Modified Gram-Schmidt (MGS), least
squares, and backward stability of MGS-GMRES. SIAM Journal on Matrix Analysis and Applications,
28(1):264–284, 2006.
[11] PB Patnaik. The non-central χ 2-and f-distribution and their applications. Biometrika, 36(1/2):202–232,
1949.
[12] Weslley da Silva Pereira, Ali Lotfi, and Julien Langou. Numerical analysis of givens rotation. arXiv
preprint arXiv:2211.04010, 2022.
[13] Tharmalingam Ratnarajah, Rémi Vaillancourt, and M Alvo. Eigenvalues and condition numbers of
complex random matrices. SIAM Journal on Matrix Analysis and Applications, 26(2):441–456, 2004.
[14] M Rudelson and R Vershynin. The littlewood-offord problem and the condition number of random
matrices. Advances in Mathematics, 218:600–633, 2008.
[15] Anirvan M Sengupta and Partha P Mitra. Distributions of singular values for some random matrices.
Physical Review E, 60(3):3389, 1999.
[16] Marvin K Simon and Mohamed-Slim Alouini. Digital communications over fading channels (mk simon
and ms alouini; 2005)[book review]. IEEE Transactions on Information Theory, 54(7):3369–3370, 2008.
[17] Samuel Smith, Erich Elsen, and Soham De. On the generalization benefit of noise in stochastic gradient
descent. In International Conference on Machine Learning, pages 9058–9067. PMLR, 2020.
[18] Stanislaw J Szarek. Condition numbers of random matrices. Journal of Complexity, 7(2):131–149, 1991.
[19] Terence Tao and Van Vu.
Smooth analysis of the condition number and the least singular value.
Mathematics of computation, 79(272):2333–2352, 2010.
[20] Michael E Tipping and Christopher M Bishop. Probabilistic principal component analysis. Journal of
the Royal Statistical Society Series B: Statistical Methodology, 61(3):611–622, 1999.
[21] Van H Vu and Terence Tao. The condition number of a randomly perturbed matrix. In Proceedings of
the thirty-ninth annual ACM symposium on Theory of computing, pages 248–255, 2007.
17