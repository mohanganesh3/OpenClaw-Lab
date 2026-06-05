[2602.08681] The Theory and Practice of MAP Inference over Non-Convex Constraints

The Theory and Practice of MAP Inference over Non-Convex Constraints

Leander Kurscheidt

Gabriele Masina

Roberto Sebastiani

Antonio Vergari

Abstract

In many safety-critical settings, probabilistic ML systems have to make predictions subject to algebraic constraints, e.g., predicting the most likely trajectory that does not cross obstacles.
These real-world constraints are rarely convex, nor the densities considered are (log-)concave.
This makes computing this constrained maximum a posteriori (MAP) prediction efficiently and reliably extremely challenging.
In this paper, we first investigate under which conditions we can perform constrained MAP inference over continuous variables exactly and efficiently and devise a scalable message-passing algorithm for this tractable fragment.
Then, we devise a general constrained MAP strategy that interleaves partitioning the domain into convex feasible regions with numerical constrained optimization.
We evaluate both methods on synthetic and real-world benchmarks, showing our approaches outperform constraint-agnostic baselines, and scale to complex densities intractable for SoTA exact solvers.

Machine Learning, ICML

1  Intro

Making predictions with probabilistic machine learning (ML) models can be mapped to performing  maximum a posteriori  (MAP;  Bishop &amp; Nasrabadi  2006  )  inference , i.e., computing the output configuration with the highest likelihood according to the distribution learned by the model.
However, in several real-world scenarios, from
physics applications  (Hansen et al.,  2023 ; Cheng et al.,  2024 )  to fair predictions  (Ghandi et al.,  2024 )  and “what-if” time-series analysis  (Narasimhan et al.,  2024 ) ,
distributions are  constrained , i.e., some configurations are infeasible and should never be predicted nor sampled  (Grivas et al.,  2024 ) .
This is mandatory if ML models are deployed in safety-critical scenarios  (Giunchiglia et al.,  2023 ; Bortolotti et al.,  2024 ) .

Figure 1 :

Our structure-aware solver  PaMap  is able to correctly and efficiently perform MAP inference over non-convex constraints and non-log-concave densities  while classical optimizers (Adam), even when being constrained-aware ( PCAdam ;

Sec.

6  ), are imprecise and slower and when exact solvers (OptiMathSAT  (Sebastiani &amp; Trentin,  2020 )  and CDCL-OCAC  (Jia et al.,  2025 ) ) timeout.
If the density factorizes as a tree, our  MpMap  solver (  Sec.

4  ) can be exact and even faster (see

Fig.

5  ).

Constrained MAP inference is well understood from an optimization perspective when it comes to discrete variables  (Marinescu &amp; Dechter,  2004 ; Martins et al.,  2011 )  or when the problem has a simple form, i.e., constraints are convex  (Dantzig,  2002 ; Jaggi,  2013 )  and distributions are log-concave  (Doss &amp; Wellner,  2019 ) , as we discuss in

Sec.

3  .
However, these assumptions are generally not met in real-world applications  (De Smet et al.,  2023 ; Kurscheidt et al.,  2025 ; Stoian &amp; Giunchiglia,  2025 )  and understanding how to perform efficient MAP inference over non-convex constraints and non-log-concave distributions is an open and challenging problem.
In this paper, we reduce this gap by advancing a number of contributions, discussed next.

First,  C1)  we theoretically trace a non-trivial fragment of tractable constrained MAP problems over non-convex constraints and distributions represented as tree-factorized piecewise (exponentiated) polynomials.
We then prove that it can be solved by an efficient message passing scheme ( MpMap ;

Sec.

4  ).
Our  MpMap  is inspired by message passing schemes to compute the probability of non-convex constraints  (Zeng et al.,  2020a ,  b ) , but differently from them, performing MAP inference yields different challenges and different complexity results.

Second,  C2)  we investigate
how to approximate constrained MAP via optimization for general constraints and distributions for which  MpMap  is not applicable.
To this end, we design  PaMap , a general scheme that decomposes global optimization into a series of MAP inference problems over convex constraints which can be efficiently solved by calling local optimizers and can provide approximation guarantees for polynomial densities  (Powers &amp; Wörmann,  1998 ; Lasserre,  2001 ) .
Lastly,  C3)  we rigorously evaluate  MpMap  and  PaMap  over a set of synthetic and real-world benchmarks, reporting that our custom optimizers, are able to outperform a number of SoTA optimizers  (Sebastiani &amp; Trentin,  2020 ; Jia et al.,  2025 )  both in terms of approximation quality and time to achieve it (see

Fig.

1  ).
As such, we set the first milestone to tackle the challenging problem of constrained optimization from both theory and practice.

2  Maximum A Posteriori Inference Under Non-Convex Algebraic Constraints

Notation.

We denote random variables by uppercase letters (e.g.,

X  ,  Y

X,Y

), and their assignments with lowercase ones (e.g.,

x  ,  y

x,y

). Bold symbols denote sets of variables (e.g.,

𝐗  ,  𝐘

\bm{\mathrm{X}},\bm{\mathrm{Y}}

), and their joint assignments (e.g.,

𝐱  ,  𝐲

\bm{\mathrm{x}},\bm{\mathrm{y}}

). Greek letters such as

Δ  ,  Φ

\Delta,\Phi

or

Δ  \Delta

denote logical formulas that map real values to binary values (false, true). We say that assignment

𝐱  \bm{\mathrm{x}}

satisfies the constraint

Δ  \Delta

, and denote it as

𝐱  ⊧  Δ

\bm{\mathrm{x}}\models\Delta

, if substituting

𝐱  \bm{\mathrm{x}}

into

Δ  \Delta

makes

Δ  \Delta

true.
So, the indicator function

⟦

𝐱  ⊧  Δ

⟧

\llbracket{\bm{\mathrm{x}}\models\Delta}\rrbracket

is 1 when

𝐱  \bm{\mathrm{x}}

satisfies

Δ  \Delta

, 0 otherwise.

Figure 2 :

An example of

MAP(

ℒ  ​  ℛ  ​  𝒜

\mathcal{LRA}

)

inference  over non-convex constraints and non-log-concave density. Left: an unconstrained density in 2D. Center: non-convex constraints.
Right: constrained density. Orange stars indicate the solutions for the MAP problem for the unconstrained and constrained densities.

SMT(

ℒ  ​  ℛ  ​  𝒜

\mathcal{LRA}

) Constraints.

We consider algebraic constraints over

𝐗  \bm{\mathrm{X}}

representing a collection of non-convex polytopes.
We express them in the language of  satisfiability modulo theory over linear real arithmetic  (SMT(

ℒ  ​  ℛ  ​  𝒜

\mathcal{LRA}

);  Barrett et al.  2021  ), hereafter just SMT for short.
We consider quantifier-free SMT-formulas over continuous variables

𝐗  \bm{\mathrm{X}}

, where linear (in)equalities

∑  i

a  i

​

X  i

⋈  b

\sum_{i}a_{i}X_{i}\bowtie{}b

, with

⋈  ∈

{  ≤  ,  =  }

\bowtie\,\in\{\leq,=\}

, are connected via Boolean connectives —i.e., conjunctions (

∧  \wedge

), disjunctions (

∨  \vee

), and negations (

¬  \neg

).
As we show in

Fig.

1  , SMT is flexible enough to represent rich real-world constraints.
We now provide a simpler example.

Example 2.1

(SMT formula) .

Consider the following SMT formula over variables

𝐗  =

{

X  1

,

X  2

}

\bm{\mathrm{X}}=\{X_{1},X_{2}\}

:

Δ  ​

(  𝐗  )

\displaystyle\Delta(\bm{\mathrm{X}})

=

X  1

∈

[  0  ,  2  ]

∧

X  2

∈

[  0  ,  2  ]

∧

\displaystyle=X_{1}\in[0,2]\wedge X_{2}\in[0,2]\wedge

(

X  2

≤

1  ∨

X  2

&gt;

2  ​

X  1

∨

X  2

&gt;

4.75  −

2  ​

X  1

)

.

\displaystyle\left(X_{2}\leq 1\vee X_{2}&gt;2X_{1}\vee X_{2}&gt;4.75-2X_{1}\right).

It denotes a square feasible area with an inner infeasible triangle area, as shown in

Fig.

2

(center).

MAP under SMT constraints.

Given a probability density function

p  ​

(  𝐗  )

p(\bm{\mathrm{X}})

over continuous random variables

𝐗  =

{

X  1

,  …  ,

X  n

}

\bm{\mathrm{X}}=\{X_{1},\ldots,X_{n}\}

and some SMT constraints

Δ  \Delta

over

𝐗  \bm{\mathrm{X}}

, the goal of constrained MAP inference is to find the assignment

𝐱  ∗

\bm{\mathrm{x}}^{*}

that maximizes

p  ​

(  𝐗  )

p(\bm{\mathrm{X}}