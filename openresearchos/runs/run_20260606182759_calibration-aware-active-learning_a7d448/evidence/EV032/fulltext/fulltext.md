<!-- page 1 -->
On Exact Sampling in the Two-Variable Fragment
of First-Order Logic
Yuanhong Wang1,2, Juhua Pu1,2, Yuyi Wang3,4, and Ondˇrej Kuˇzelka5
1State Key Laboratory of Software Development Environment,
Beihang University, China
2Beihang Hangzhou Innovation Institute Yuhang, China
3CRRC Zhuzhou Institute, China
4ETH Zurich, Switzerland
5Czech Technical University in Prague, Czech Republic
May 9, 2023
Abstract
In this paper, we study the sampling problem for ﬁrst-order logic proposed recently by
Wang et al.—how to eﬃciently sample a model of a given ﬁrst-order sentence on a ﬁnite do-
main? We extend their result for the universally-quantiﬁed subfragment of two-variable logic
FO2 (UFO2) to the entire fragment of FO2. Speciﬁcally, we prove the domain-liftability
under sampling of FO2, meaning that there exists a sampling algorithm for FO2 that runs in
time polynomial in the domain size. We then further show that this result continues to hold
even in the presence of counting constraints, such as ∀x∃=ky : ϕ(x, y) and ∃=kx∀y : ϕ(x, y),
for some quantiﬁer-free formula ϕ(x, y). Our proposed method is constructive, and the result-
ing sampling algorithms have potential applications in various areas, including the uniform
generation of combinatorial structures and sampling in statistical-relational models such as
Markov logic networks and probabilistic logic programs.
1
Introduction
Let Γ denote a function-free ﬁrst-order sentence formed over a vocabulary P, and let ∆be a
ﬁnite domain. A model of Γ interprets each predicate in P over ∆such that the interpretation
satisﬁes Γ. We use MΓ,∆to denote the set of all models of Γ over ∆. The uniform ﬁrst-order
model sampling problem on Γ over ∆is to uniformly generate a model µ of Γ according to the
probability P[µ] = 1/|MΓ,∆|. The weighted variant of this problem adds nonnegative weights
to atomic facts and their negations in the models; the total weight of a model is the product
of its facts’ weights. The problem is then to sample a model according to a probability strictly
proportional to its weight.
We investigate the symmetric weighted ﬁrst-order model sampling problem (WFOMS) for the
two-variable fragment FO2 of ﬁrst-order logic. The term “symmetric” refers to the property that
the weights are determined solely by the relation symbol. In this paper, we focus on studying
the data complexity of WFOMS—the complexity of sampling a model when the Γ and w and ¯w
are ﬁxed, and the domain is considered as an input. In particular, we are interested in designing
1
arXiv:2302.02730v2  [cs.AI]  6 May 2023


<!-- page 2 -->
a domain-lifted weighted model sampler for FO2, which runs in time polynomial in the size of
the domain.
The WFOMS was ﬁrst considered by Wang et al. [1] who showed that the data complexity
of WFOMS is in polynomial time for formulas of the universally-quantiﬁed subfragment UFO2
of FO2. The subfragment UFO2, comprising of sentences of the form ∀x∀y : ψ(x, y) with some
quantiﬁer-free formula ψ(x, y), is proved to admit a lifted weighted model sampler, and then
identiﬁed to be domain-liftable under sampling.
Symmetric weighted model sampling problems have a wide range of practical applications.
For example, many problems related to the generation of combinatorial structures can easily
be formulated as WFOMS and solved using the techniques developed for this problem. There
are also applications of WFOMS in the realm of statistical-relational learning (SRL) [2]. It is
known that probabilistic inference in many SRL models is reducible to weighted ﬁrst-order model
counting (WFOMC) [3, 4], and the same reduction can also be applied to the corresponding
sampling problems.
Among the various applications of WFOMS, the input ﬁrst-order sentences are usually com-
plex and go beyond the fragment of UFO2. For instance, even the very simple problem of uni-
formly generating graphs with no isolated vertices necessitates the utilization of the existentially-
quantiﬁed formula ∀x∃y : E(x, y) to encode the constraint that every vertex must have at least
one incident edge.
However, directly extending the approach described in [1] to FO2 is in-
feasible. As the authors showed, their technique would at some point need to solve #P-hard
problems: “. . . applying our sampling algorithm on an FO2 sentence with existential quanti-
ﬁers is intractable (not domain-lifted) unless FP=#P,. . . ”. We stress here that they did not
show the intractability of WFOMS for the FO2 fragment, but rather the infeasibility of their
speciﬁc method, indicating that a distinct sampling approach is required. Moreover, the stan-
dard Skolemization techniques used in automated reasoning [5] and WFOMC [4] to eliminate
existential quantiﬁers beforehand are not applicable to WFOMS, as they introduce either func-
tions or negative weights, which make the resulting sampling problem ill-deﬁned. This further
complicates the extension of the WFOMS approach to more complex formulas beyond UFO2.
1.1
Our Contribution
In this paper, we present a novel sampling algorithm for the full FO2. The algorithm employs
a completely diﬀerent approach than Skolemization, based on the domain recursion scheme.
The basic idea is to consider one object from the domain at a time, and then sample the value
of all related atomic facts, resulting in a new WFOMS over a smaller domain with the object
removed. The new WFOMS has an identical form to the original one but possibly contains
fewer existentially-quantiﬁed formulas.
The algorithm then runs recursively on the reduced
sampling problems until the domain becomes singleton or all existentially-quantiﬁed formulas
are eliminated. We prove that the data complexity of our algorithm is in PTIME, meaning that
the entire fragment of FO2 is domain-liftable under sampling.
We also show how to further extend the result to the cases, where we include counting
constraints. Speciﬁcally, our generalized algorithm can be applied to the FO2 sentences with
additional counting constraints of the form ∀x∃=ky : ϕ(x, y) and ∃=kx∀y : ϕ(x, y), where ϕ
is a quantiﬁer-free formula and k is a natural number. This extension, originally proposed by
Kuusisto and Lutz [6] and Kuzelka [7] for ﬁrst-order counting problems, is mainly motivated by
the connection of WFOMS to the uniform generation of combinatorial structures. For example,
our algorithm can be applied to eﬃciently solve the uniform sampling problem of k-regular
2


<!-- page 3 -->
graphs, a problem that has been widely studied in the combinatorics community [8, 9]. This
problem can be formulated as a WFOMS on the following sentence:
∀x∀y : (E(x, y) ⇒E(y, x)) ∧∀x : ¬E(x, x) ∧∀x∃=ky : E(x, y),
where ∀x∃=ky : E(x, y) expresses that every vertex x has exactly k connected edges.
1.2
Related Work
The symmetric weighted ﬁrst-order model sampling problem was ﬁrst proposed and studied
in [1]. The approach, as well as the formal liftability notions considered in that study, were
derived from the literature on lifted inference [10, 11, 3].
In lifted inference, the goal is to
perform probabilistic inference in SRL models in a way that takes advantage of the symmetries
in the high-level structure of the models. The symmetry also exists in WFOMS and is a vital
property leveraged by this paper to prove the liftability under sampling of FO2. We note here
that the importance of symmetry for lifted inference (and its reduced WFOMC) has also been
extensively discussed by Beame et al. [12].
The domain recursion approach adopted in this paper is similar to the domain recursion
rule used in weighted ﬁrst-order model counting [13, 14, 15, 16]. The domain recursion rule
for WFOMC is a technique that utilizes a gradual grounding process on the input ﬁrst-order
sentence, where only one element of the domain is grounded at a time. As each element is
grounded, the partially grounded sentence is simpliﬁed until the element is entirely removed,
resulting in a new WFOMC problem with a smaller domain. With the domain recursion rule,
one can apply the principle of induction on the domain size, and compute WFOMC by dynamic
programming. A closely related work to this paper is the approach presented by Kazemi et al.
[14], where they used the domain recursion rule to compute WFOMC without Skolemization [4],
which introduces negative weights. However, it is important to note that their approach can
be only applied to some speciﬁc ﬁrst-order formulas, whereas the domain recursion scheme
presented in this paper, mainly designed for eliminating the existentially-quantiﬁed formulas,
supports the entire FO2 fragment.
It is also worth mentioning that sampling from propositional logic formula (Boolean for-
mula) is a relatively well-studied area [17, 18, 19]. However, many real-world problems can
be represented more naturally and concisely in ﬁrst-order logic, and suﬀer from a signiﬁcant
increase in formula size when grounded out to propositional logic. For example, a formula of
the form ∀x∃y : ϕ is encoded as a Boolean formula of the form Vn
i=1
Wn
j=1 li,j, whose length is
quadratic in the domain size n. Since even ﬁnding a solution to a such large ground formula
is challenging, most sampling approaches for propositional logic instead focus on designing ap-
proximate samplers. We also note that these approaches are not polynomial-time in the length
of the input formula, and rely on access to an eﬃcient SAT solver. An alternative strand of
research [20, 21, 22] on combinatorial sampling, focuses on the development of near-uniform
and eﬃcient sampling algorithms. However, these approaches can only be employed for spe-
ciﬁc Boolean formulas that satisfy a particular technical requirement known as the Lov´asz Local
Lemma. The WFOMS problems studied in this paper do not typically meet the requisite criteria
for the application of these techniques.
3


<!-- page 4 -->
2
Preliminaries
In this section, we brieﬂy review the main necessary technical concepts that we will use in the
paper
2.1
Symmetric weighted ﬁrst-order model sampling
We consider the function-free fragment of ﬁrst-order logic. An atom of arity k takes the form
P(x1, . . . , xk) where P/k is from a vocabulary of predicates (also called relations), and x1, . . . , xk
are logical variables from a vocabulary of variables. A literal is an atom or its negation. A
formula is formed by connecting one or more literals together using conjunction or disjunction.
A formula may optionally be surrounded by one or more quantiﬁers of the form ∀x or ∃x, where
x is a logical variable. A logical variable in a formula is said to be free if it is not bound by any
quantiﬁer. A formula with no free variables is called a sentence. The vocabulary of a formula α
is taken to be Pα.
Given a vocabulary P, a P-structure A interprets each predicate in P over a given domain.
We often interchangeably view a structure as a set of ground literals and their conjunction. Given
a P-structure A and P′ ⊆P, we write ⟨A⟩P′ for the P′-reduct of A. We follow the standard
semantics of ﬁrst-order logic for determining whether a structure is a model of a formula. We
denote the set of all models of a sentence Γ over the domain ∆by MΓ,∆. The two-variable
syntactic fragment of ﬁrst-order logic (FO2) is obtained by restricting the variable vocabulary
to {x, y}.
The ﬁrst-order model counting problem [3] asks, when given a domain ∆and a sentence Γ,
how many models Γ has over ∆. The weighted ﬁrst-order model counting problem (WFOMC)
adds a pair of weighting functions (w, ¯w) to the input, that both map the set of all predicates
in Γ to a set of weights: PΓ →R. Given a set L of literals, the weight of L is deﬁned as
⟨w, ¯w⟩(L) :=
Y
l∈LT
w(pred(l)) ·
Y
l∈LF
¯w(pred(l))
where LT (resp. LF ) denotes the set of true ground (resp. false) literals in L, and pred(l) maps
a literal l to its corresponding predicate name. The value of WFOMC(Γ, ∆, w, ¯w) is then the
sum of the weight ⟨w, ¯w⟩(µ) over all models of Γ over ∆.
Recently, the model counting problem was extended to the sampling regime by [1], and the
symmetric weighted ﬁrst-order model sampling problem (WFOMS) deﬁned therein is the main
focus of this paper.
Deﬁnition 1 (Symmetric weighted ﬁrst-order model sampling). Let (w, ¯w) be a pair of
weighting function: PΓ →R≥0 1. The symmetric weighted ﬁrst-order model sampling problem
on Γ over a domain ∆under (w, ¯w) is to generate a model G(Γ, ∆, w, ¯w) of Γ over ∆such that
P[G(Γ, ∆, w, ¯w) = µ] =
⟨w, ¯w⟩(µ)
WFOMC(Γ, ∆, w, ¯w)
(1)
for every µ ∈MΓ,∆.
Following the terminology in [1], we call a probabilistic algorithm that realizes a solution
to the WFOMS a weighted model sampler (WMS). A WMS is domain-lifted (or simply lifted)
1The non-negative weights ensures that the sampling probability of a model is well-deﬁned.
4


<!-- page 5 -->
if the model generation algorithm runs in time polynomial in the size of the domain ∆.
A
sentence, or class of sentences, is domain-liftable (or simply liftable) under sampling if it admits
a domain-lifted WMS.
Example 1. The WMS of the sentence
(∀x∀y : (E(x, y) ⇒E(y, x)) ∧¬E(x, x)) ∧(∀x∃yE(x, y))
over a domain of size n under the weighting w(E) = ¯w(E) = 1 uniformly samples undirected
graphs with no isolated vertices.
For technical purposes, when the domain is ﬁxed, we allow the input sentence of the WFOMC
(and WFOMS) to contain some ground literals, e.g., Γ = (∀x∀y : fr(x, y) ∧sm(x) ⇒sm(y)) ∧
sm(e1) ∧¬sm(e3) over a ﬁxed domain of {e1, e2, e3}. The WFOMC problem on such sentences
is also known as conditional WFOMC [23, 24].
We deﬁne the probability of a sentence Φ
conditional on another sentence Γ over a domain ∆under (w, ¯w) as
P[Φ | Γ; ∆, w, ¯w] := WFOMC(Φ ∧Γ, ∆, w, ¯w)
WFOMC(Γ, ∆, w, ¯w)
.
With a slight abuse of notation, we also write the probability of a set L of ground literals
conditional on a sentence Γ over a domain ∆under (w, ¯w) in the same form:
P[L | Γ; ∆, w, ¯w] := P
"^
l∈L
l | Γ; ∆, w, ¯w
#
.
Then, the required sampling probability of G(Γ, ∆, w, ¯w) in the WFOMS can be written as
P[G(Γ, ∆, w, ¯w) = µ] = P[µ | Γ; ∆, w, ¯w]. When the context is clear, we omit ∆and (w, ¯w) in
the conditional probability.
We call a set L of ground literals valid in a WFOMS (Γ, ∆, w, ¯w), if there exists a model
µ ∈MΓ,∆that includes L. A skeleton of a WFOMS (Γ, ∆, w, ¯w) is a subset P of the vocabulary
PΓ, such that the interpretation for P fully determines PΓ \ P in the models of Γ, and for any
predicate P ∈PΓ \ P, w(P) = ¯w(P) = 1. Using the notion of skeleton, a WFOMS (Γ, ∆, w, ¯w)
can be reduced to randomly generating a valid P-structure G(Γ, ∆, w, ¯w) such that
P[G(Γ, ∆, w, ¯w) = ⟨µ⟩P] = P[⟨µ⟩P | Γ; ∆, w, ¯w]
for every µ ∈MΓ,∆, where P is a skeleton of the problem.
In this paper, we often convert complicated WFOMS problems into simpler ones, which are
commonly referred to as reductions. The essential property of such reductions is soundness.
Deﬁnition 2 (Soundness). A reduction of a WFOMS of (Γ, ∆, w, ¯w) to (Γ′, ∆′, w′, ¯w′) is sound
iﬀthere exists a polynomial-time deterministic function f mapping from MΓ′,∆′ to MΓ,∆, and
for every model µ ∈MΓ,∆,
P[µ | Γ; ∆, w, ¯w] =
X
µ′∈MΓ′,∆′:
f(µ′)=µ
P[µ′ | Γ′; ∆′, w′, ¯w′].
(2)
A general mapping function f used most in this paper is the projection f(µ′) = ⟨µ′⟩PΓ, where
PΓ is a skeleton of (Γ′, ∆′, w′, ¯w′). In this case, the mapping function is bijective and preserves
the weight of the mapped models. Through a sound reduction, we can easily transform a WMS
G′ of (Γ′, w′, ¯w′, ∆′) to a WMS G of (Γ, w, ¯w, ∆) by G(Γ, ∆, w, ¯w) = f(G′(Γ′, ∆′, w′, ¯w′)). Note
that the soundness is transitive, i.e., if the reductions from a WFOMS S1 to S2 and from S2
to S3 are both sound, the reduction from S1 to S3 is also sound.
5


<!-- page 6 -->
2.2
Types and Tables
We deﬁne a 1-literal as an atomic predicate or its negation using only the variable x, and a
2-literal as an atomic predicate or its negation using both variables x and y. An atom like
R(x, x) or its negation is considered a 1-literal, even though R is a binary relation. A 2-literal
is always of the form R(x, y) and R(y, x), or their respective negations.
Let P be a ﬁnite vocabulary. A 1-type over P is a maximally consistent set of 1-literals
formed by P. Denote the set of all 1-types over P as UP. The size of UP is ﬁnite and only
depends on the size of P. We often view a 1-type τ as a conjunction of its elements, whence
τ(x) is simply a formula in the single variable x.
Let A be a structure over P. A domain element e ∈dom(A) realizes the 1-type τ if A |= τ(e).
Note that every element of A realizes exactly one 1-type over P, which we call the 1-type of the
element. The cardinality of a 1-type is the number of elements realizing it.
A 2-table over P is a maximally consistent set of 2-literals formed by P. We often identify
a 2-table π with a conjunction of its elements and write it as a formula π(x, y). Denote TP the
set of all 2-tables over P, whose size also only depends on the size of P. Given a P-structure A
over a domain ∆, the 2-table of an element tuple (a, b) ∈∆2 is the unique 2-table π that (a, b)
satisﬁes in A: A |= π(a, b). It is worth noting that the 1-types together with the 2-tables fully
characterize a structure.
Example 2. Consider the vocabulary P = {F/2, G/1} and the structure
{F(a, a), G(a), ¬F(b, b), G(b), F(a, b), ¬F(b, a)}
over the domain {a, b}. The 1-type of the elements a and b are F(x, x)∧G(x) and ¬F(x, x)∧G(x)
respectively. The cardinalities of these two 1-types are both 1, while that of the other 1-types
F(x, x) ∧¬G(x) and ¬F(x, x) ∧¬G(x) are both 0. The 2-table of the element tuples (a, b) and
(b, a) are F(x, y) ∧¬F(y, x) and ¬F(x, y) ∧F(y, x) respectively.
2.3
Universally Quantiﬁed FO2 is Liftable under Sampling
As an elementary attempt to the symmetric weighted ﬁrst-order model sampling problem, Wang
et al. [1] provided a positive result of the data complexity for the universally quantiﬁed fragment
of FO2 (UFO2) of the form ∀x∀y : ψ(x, y), where ψ(x, y) is a quantiﬁer-free formula2.
The proof of this result established a general framework for designing a WMS. Therefore,
We summarize the main ideas of their argument here and refer the reader to their paper for
the complete proof and technical details. We note that the approach presented here is slightly
diﬀerent from the original one in [1]. The main divergence is that, instead of using the notion of
count distribution [7], we perform the sampling of 1-types by a random partition on the domain,
which keeps in line with our sampling algorithm for FO2.
Theorem 1 (Proposition 1 in [1]). The fragment UFO2 is domain-liftable under sampling.
Proof sketch. Suppose that we wish to randomly sample models from some input UFO2 sentence
Γ = ∀x∀y : ψ(x, y) over a domain ∆= {e1, e2, . . . , en} under weights (w, ¯w).
Given a PΓ-
structure A over ∆, we denote τi the 1-type of the ith element and πi,j the 2-table of the tuple
2They went a bit beyond this fragment, e.g., UFO2 with cardinality constraints, which we also handle later
in this paper.
6


<!-- page 7 -->
of the ith and jth elements. The structure A is fully characterized by the ground 1-types τi(ei)
and 2-tables πi,j(ei, ej). We can write the sampling probability of A as
P[A | Γ] = P

^
i∈[n]
τi(ei) ∧
^
i,j∈[n]:i<j
πi,j(ei, ej) | Γ


= P

^
i∈[n]
τi(ei) | Γ


|
{z
}
P1
· P


^
i,j∈[n]:
i<j
πi,j(ei, ej) | Γ ∧
^
i∈[n]
τi(ei)


|
{z
}
P2
,
where [n] denotes the set of {1, 2, . . . , n}. This decomposition naturally gives rise to a two-phase
sampling algorithm:
1. sample 1-types τi according to the probability P1, and
2. randomly assign 2-table πi,j to each element tuple according to P2.
The sampling of 1-types can be achieved through a random partition of the domain ∆, resulting
in |UPΓ| disjoint subsets of ∆; each subset contains the elements assigned to its corresponding 1-
type. The symmetry property of the weighting function ensures that the satisfaction and weight
of the models are not aﬀected by permutations of the domain elements. Therefore, any partitions
of the domain with the same partition size have the same probability to be sampled.
This
further decomposes the sampling problem of 1-types into two stages: the stochastic generation
of partition size and the random partitioning of the domain according to the sampled size.
Randomly partitioning the domain according to the sampled size is straightforward, and we will
demonstrate that sampling a partition size can be done in time polynomial in the domain size.
Recall that the number |UPΓ| of all 1-types only depends on the input sentence, and thus
enumerating all possible partition sizes is computationally tractable.
For any partition size
(n1, n2, . . . , n|UPΓ|), there are totally
 n
n1,n2,...,n|UPΓ |
 
partitions of the domain with the same
sampling probability. It will turn out that the sampling probability, which is of the form P1, can
be computed in time polynomial in the domain size. The reason for this is: we expand P1 into
WFOMC(Γ ∧Vn
i=1 ηi(ei), ∆, w, ¯w)/WFOMC(Γ, ∆, w, ¯w); the numerator WFOMC can be viewed
as a WFOMC of Γ conditional on the unary facts in all ηi(ei), whose complexity is polynomial
in the domain size by [23]; and the denominator WFOMC can be also eﬃciently computed due
to the liftability (in terms of WFOMC) of Γ by [3]. Finally, the sampling probability of the
partition size (n1, n2, . . . , n|UPΓ|) is given by P1 ·
 n
n1,n2,...,n|UPΓ|
 
.
For sampling πi,j according to P2, we ﬁrst ground out Γ over the domain ∆:
^
i,j∈[n]:i<j
ψ(ei, ej) ∧ψ(ej, ei).
Let ψ′
i,j(x, y) be the simpliﬁed formula of ψ(x, y) ∧ψ(y, x) obtained by replacing the unary
ground literals with their truth value given by the 1-types τi and τj. Then the probability P2
can be written as
P[
^
i,j∈[n]:i<j
πi,j(ei, ej) |
^
i,j∈[n]:i<j
ψ′
i,j(ei, ej)].
7


<!-- page 8 -->
Note that in this probability, each ground 2-tables πi,j(ei, ej) are independent in the sense that
they do not share any ground literals. The independence also holds for the ground formulas
ψ′
i,j(ei, ej). It follows that the probability P2 can be factorized into
Y
i,j∈[n]:i<j
P[πi,j(ei, ej) | ψ′
i,j(ei, ej)].
Hence, the sampling of each πi,j can be solved separately by randomly choosing a model of its
respective ground formula ψ′(ei, ej) according to the probability P[πi,j(ei, ej) | ψ′
i,j(ei, ej)]. The
overall computational complexity is clearly polynomial in the domain size.
The procedure for both sampling τi and πi,j is polynomial in the domain size, which completes
the liftability under sampling of UFO2.
Extending the approach above to the case of FO2 would requires a novel and more so-
phisticated strategy, especially for the sampling of 2-tables, as decoupling the grounding of
∀x∃y : ϕ(x, y) to the form of V
i,j∈[n]:i<j ψi,j(ei, ej) is impossible even conditioning on the sam-
pled 1-types.
2.4
Notations
We will use [n] to denote the set of {1, 2, . . . , n}. The notation {xi}i∈[n] represents the set of
terms {x1, x2, . . . , xn}, and (xi)i∈[n] the vector of (x1, x2, . . . , xn). We also use the bold symbol
x to denote a vector (xi)i∈[n], and denote by xy the product over element-wise power of two
vectors xy = Q
i∈[n] xyi
i . The notation ⊕is used to denote the concatenation of two vectors.
Using the vector notation, we often write the multinomial coeﬃcient
 N
x1,x2,...,xn
 
as
 N
x
 
.
3
Sampling Algorithm for FO2
We now show the domain-liftability under sampling of the FO2 fragment by providing a lifted
WMS for it. It is common for logical algorithms to operate on normal form representations
instead of arbitrary sentences. The normal form of FO2 used in our sampling algorithm is the
Scott normal form (SNF) [25]; an FO2 sentence is in SNF, if it is written as:
Γ = ∀x∀y : ψ(x, y) ∧
^
k∈[m]
∀x∃y : ϕk(x, y),
(3)
where ψ and ϕk are quantiﬁer-free formulas. It is well-known that one can convert any FO2 sen-
tence Γ in polynomial time into a formula ΓS in SNF such that Γ and ΓS are equisatisﬁable [26].
The principal idea is to substitute, starting from the atomic level and working upwards, any
subformula ϕ(x) = Qy : φ(x, y), where Q ∈{∀, ∃} and φ is quantiﬁer-free, with an atomic
formula Aϕ, where Aϕ is a fresh predicate symbol. This novel atom Aϕ(x) is then separately
“axiomatized” to be equivalent to ϕ(x). The weight of Aϕ is set to be w(Aϕ) = ¯w(Aϕ) = 1. It
follows from reasoning similar to one by Kuusisto and Lutz [6] that such reduction is not only
equisatisﬁable but also sound (according to Deﬁnition 2).
Lemma 1. For any WFOMS of S = (Γ, ∆, w, ¯w) where Γ is an FO2 sentence, there exists a
WFOMS S′ = (Γ′, ∆, w′, ¯w′), where Γ′ is in SNF and independent of ∆, such that the reduction
from S to S′ is sound.
8


<!-- page 9 -->
Figure 1: A sampling step for an undirected graph with no isolated vertices: (a) begin with
an initial graph that has no edges, and in the more general sampling problem, V∀= V∃=
{v1, v2, v3, v4}; (b) sample edges for the vertex v1; (c) remove the vertex v1 with its sampled
edges; (d) and obtain a graph with some vertices already non-isolated (v2 and v3), resulting in
a new sampling problem with V ′
∀= {v2, v3, v4} and V ′
∃= {v4}.
The proof is clear, as every novel predicate (e.g., Pϕ) introduced in the SNF transformation
is axiomatized to be equivalent to the subformula (ϕ(x)), whose quantiﬁers are to be eliminated,
and thus the interpretation of the predicate is fully determined by the subformula in every model
of the resulting SNF sentence (see the details in Appendix A.1).
3.1
An Intuitive Example
We start with an intuitive example of how to generate an undirected graph of size n without
any isolated vertex uniformly at random, to illustrate the basic idea of our sampling algorithm.
This graph structure can be expressed by an FO2 sentence in SNF,
ΓG := (∀x∀y : (E(x, y) ⇒E(y, x)) ∧¬E(x, x)) ∧(∀x∃y : E(x, y)) ,
and the sampling problem corresponds to a WFOMS on ΓG under w(E) = ¯w(E) = 1 over
a domain of vertices V = {vi}i∈[n]. In this sentence, the only realizable 1-type is ¬E(x, x),
and the realizable 2-tables are π1(x, y) = E(x, y) ∧E(y, x) and π2(x, y) = ¬E(x, y) ∧¬E(y, x)
representing the connectedness of two vertices.
We ﬁrst apply the following transformation on ΓG resulting in ΓGT :
1. introduce an auxiliary Tseitin predicate Z/1 that indicates the non-isolation of vertices,
and append ∀x : Z(x) ⇔∃y : E(x, y) to ΓG, and
2. remove ∀x∃y : E(x, y),
and set the weight of the predicate Z to w(Z) = ¯w(Z) = 1. Then we consider a bit more
general WFOMS of (ΓGT ∧V
v∈V∃Z(v), V∀, w, ¯w), where V∃⊆V∀⊆V and V∃represents the set
of vertices that should be non-isolated in the graph induced by V∀. The original WFOMS on ΓG
can be clearly reduced to the more general problem by setting V∃= V∀= V , and the reduction
is sound with the mapping function f(µ′) = ⟨µ′⟩{E}.
Let bΓG denote the sentence ΓGT ∧V
v∈V∃Z(v). For the more general WFOMS, since {E} is
its skeleton, the problem is equivalent to sampling an {E}-structure A over V∀according to the
probability P[A | bΓG]. Given an {E}-structure A over V∀, denote by Ai the substructure of A
concerning the vertex vi ∈V∀, which consists of the 2-tables of all vertex tuples containing vi
and other vertices in V∀:
Ai :=
[
vj∈V∀:j̸=i
πi,j(vi, vj),
9

[CAPTION] Figure 1: A sampling step for an undirected graph with no isolated vertices: (a) begin with


<!-- page 10 -->
where πi,j is the 2-table of (vi, vj). Following the domain recursion scheme, we choose a vertex
vt from V∀and decompose the sampling probability of A into
P[A | bΓG] = P
h
A | bΓG ∧At
i
· P[At | bΓG].
The decomposition leads to two successive subproblems of the general WFOMS: the ﬁrst one is
to sample a valid substructure At from bΓG; the other can be viewed as a new WFOMS on bΓG
given the sampled At.
We ﬁrst show that the new WFOMS can be also reduced to the general WFOMS but with
the smaller domain V ′
∀= V∀\ {vt} and
V ′
∃= {vi | vi ∈V∃: vi ̸= vt ∧πt,i = π2}.
The reduction is obviously sound because every model of the WFOMS (ΓGT ∧V
v∈V ′
∃Z(v), V ′
∀, w, ¯w)
can be mapped to a unique model of the WFOMS (bΓG ∧At, V∀, w, ¯w), and vice versa, without
aﬀecting the weight of the models. Thus, the decomposition can be performed recursively on
any WFOMS on bΓG over V∀. Speciﬁcally, the algorithm takes V∀and V∃as input,
1. selects a vertex vt from V∀,
2. samples its substructure At according to the probability P[At | bΓG], and
3. obtains a new problem with updated V ′
∀and V ′
∃for recursion.
The recursion procedure terminates when all substructures Ai are sampled (V∀contains a single
vertex), or the problem degenerates to a WFOMS on UFO2 sentence (V∃is empty). The number
of recursions is less than |V |, the total number of vertices. An example of a recursion step is
shown in Figure 1.
The remaining problem is to sample the substructure At according to P[At | bΓG]. Recall
that At determines the edges between vt and vertices in V ′
∀. Let V1 = V ′
∀\ V∃and V2 = V ′
∀\ V1.
The sampling of At can be realized by performing two random binary partitions on V1 and
V2 respectively, resulting in {V11, V12} and {V21, V22}, where the vertices in V11 and V21 will
be connected to vt, while the vertices in V12 and V22 will not be connected to it. It can be
demonstrated that the sampling probability of a substructure At only depends on the size
(|V11|, |V12|, |V21|, |V22|). The proof of this claim can be found in Section 3.4, where the more
general case of FO2 sampling is addressed. As a result, the sampling of At can be accomplished
by a random generation of the partition size, followed by two random partitions of the sampled
size on V1 and V2 respectively. We use the enumerative sampling method to generate a partition
size. The number of all possible partition sizes is clearly polynomial in |V∀|, and it will be shown
in Section 3.4 that the sampling probability of each partition size can be computed in time
polynomial in |V ′
∀|. Therefore, the complexity of the sampling algorithm is polynomial in the
number of vertices. This, together with the complexity of the recursion procedure, which is also
polynomial in the number of vertices, implies that the whole sampling algorithm is lifted.
3.2
A More General Sampling Problem
W.l.o.g.3 , we suppose that each formula ϕk(x, y) in the SNF sentence (3) is an atomic formula
Rk(x, y), where Rk is a binary predicate in Pψ(x,y), and its weights w(Rk) = ¯w(Rk) = 1. We
3Any SNF sentence can be transformed into such form by introducing an auxiliary predicate Rk with weights
w(Rk) = ¯w(Rk) = 1 for each ϕk(x, y), append ∀x∀y : Rk(x, y) ⇔ϕk(x, y) to the sentence, and replacing ϕk(x, y)
with Rk(x, y). The transformation is obviously sound when viewing it as a reduction in WFOMS.
10


<!-- page 11 -->
ﬁrst construct the following sentence from the SNF one:
ΓT := ∀x∀y : ψ(x, y) ∧
^
k∈[m]
∀x : Zk(x) ⇔(∃y : Rk(x, y)),
(4)
where each Zk/1 is a Tseitin predicate with the weight w(Zk) = ¯w(Zk) = 1.
We then consider a more general WFOMS problem on the following sentence
bΓ := ΓT ∧
^
i∈[n]
Ci
(5)
over a domain of {ei}i∈[n], where each Ci is a conjunction over a subset of the ground atoms
{Zk(ei)}k∈[m]. We call Ci the existential constraint on the element ei and allow Ci = ⊤, which
means ei is not existentially quantiﬁed. The more general WFOMS can be regarded as a con-
ditional sampling problem, where the existential constraint serves as unary facts that condition
the problem. The original WFOMS on Γ can be reduced to a more general problem by setting
all existential constraints to be V
k∈[m] Zk(x). On the other hand, the WFOMS on the UFO2
sentence ∀x∀y : ψ(x, y) is also reducible to the problem with Ci = ⊤for all i ∈[n]. It is easy
to check that these two reductions are both sound. The main idea of our sampling algorithm
is to use the domain recursion scheme to gradually remove the existential constraints until we
eventually end up with a WFOMS problem on a UFO2 sentence.
3.3
Partitioning the Domain
Unless stated otherwise, in the rest of this section, 1-types and 2-tables are deﬁned over PΓ,
where Γ is the sentence in SNF. Please bear in mind that the Tseitin predicates Zk are not in
these 1-types.
We introduce the concepts of block and cell types as extensions of 1-types. These types are
utilized in a manner akin to 1-types in the sampling algorithm for UFO2. Consider a sentence bΓ
of the form (5) with Tseitin predicates Zk. A block type β is a subset of the atoms {Zk(x)}k∈[m].
The number of the block types is 2m, where m is the number of existentially-quantiﬁed formulas.
We often represent a block type as β(x) and view it as a conjunctive formula over the atoms
within the block. With the notion of block type, we can write bΓ = ΓT ∧V
i∈[n] βi(ei), where
the grounding βi(ei) is exactly the existential constraint Ci imposed on ei. We call βi the block
type of ei. We ﬁx the order of all block types and denote by βi the ith block type. The domain
∆is then partitioned by the blocks {Bβi}i∈[2m], where each subset Bβi ⊆∆contains precisely
the domain elements with the block type βi. It is important to note that the block types only
indicate which Tseitin atoms should hold for a given element, and the Tseitin atoms not covered
by the block types are left unspeciﬁed. In contrast, the 1-types explicitly determine the truth
values of all unary and reﬂexive atoms, excluding the Tseitin atoms.
The blocks are further partitioned into cells. A cell type η = (β, τ) is a pair of a block type β
and a 1-type τ. We also write a cell type as a conjunctive formula of η(x) = β(x) ∧τ(x). Given
a PΓ-structure A, the cell type of an element e is the combination of its block type (which is
given by the sentence bΓ) and its realizing 1-type in A. Each block Bβ is partitioned by the cells
{CA
η | η = (β, τ), τ ∈UPΓ}, where each cell CA
η ⊆Bβ contains precisely the domain elements
that are of cell type η.
For brevity, we denote by Nu = |UPΓ|, the number of all 1-types and Nc = 2m × Nu, the
number of all cell types. We ﬁx a linear order of 1-types as well as cell types, and let τ i and ηj
11


<!-- page 12 -->
be the ith 1-type and jth cell type respectively. Given a PbΓ-structure A with the cell partition
{CA
ηi}i∈[Nc], we call the size
 
|CA
ηi|
 
i∈[Nc] of the cell partition the cell conﬁguration of A, and
 
|CA
(β,τ i)|
 
i∈[Nu] the cell conﬁguration of A in the block β. A PbΓ-structure will have a unique
cell conﬁguration (in a block).
We will often care about the set of all cell conﬁgurations over a set of elements, which is
deﬁned as the conﬁguration space.
Deﬁnition 3 (Conﬁguration Space). Given a nonnegative integer M and a positive integer
m, we deﬁne the conﬁguration space TM,m as
TM,m =


(ni)i∈[m] |
X
i∈[m]
ni = M, n1, n2, . . . , nm ∈N


.
The size of TM,m is given by
 M+m−1
m−1
 
, which is clearly polynomial in M (while exponential
in m).
3.4
The Sampling Algorithm
We now describe our algorithm for the WFOMS of (bΓ, ∆, w, ¯w) where bΓ is of the form (5) and
∆= {ei}i∈[n], and prove that the algorithm is domain-lifted (i.e. runs in time polynomial in the
domain size). It can be easily veriﬁed that PΓ is a skeleton of the WFOMS problem. Hence, this
WFOMS problem is equivalent to sampling a valid PΓ-structure A according to the probability
P[A | bΓ].
Given a PΓ-structure A over ∆, let τi be the 1-type of the element ei, and denote by
ηi = (βi, τi) the cell type of ei. Using the notation of conditional probability, we decompose the
sampling probability as
P[A | bΓ] = P

^
i∈[n]
τi(ei) | bΓ

· P

A | bΓ ∧
^
i∈[n]
τi(ei)


Following a similar idea to the one used for sampling models from UFO2 sentences, our proposed
algorithm is divided into two phases—we ﬁrst sample the 1-type for each element, and then
handle the sampling of the structure conditional on the sampled 1-types.
3.4.1
Sampling 1-Types
Let us ﬁrst consider the sampling of 1-types according to P[V
i∈[n] τi(ei) | bΓ]. Note that the block
type of each element has been determined by the sentence bΓ, and thus the problem is equivalent
to sampling the cell type of each element 4. Due to the symmetry of the WFOMS problem,
the sampling probability of a cell partition is completely determined by its corresponding cell
conﬁguration. Therefore, the problem of randomly partitioning cells is further reduced to a
problem of sampling cell conﬁgurations.
4In the special case where there is a single block, e.g., bΓ is of the standard SNF or in UFO2, the sampling
problem can be simpliﬁed.
12


<!-- page 13 -->
The sampling algorithm for cell conﬁgurations is outlined in Algorithm 1. The algorithm
begins by sampling a random cell conﬁguration in Line 1-14, which is then used to randomly
partition each block into cells in Line 15-21. While the partitioning process is relatively straight-
forward, in the following discussion, we will focus speciﬁcally on how to sample a cell conﬁgu-
ration.
Algorithm 1 OneTypeSampler(bΓ, ∆, w, ¯w)
1: W ←WFOMC(bΓ, ∆, w, ¯w)
2: Obtain the blocks Bβ1, Bβ2, . . . , Bβ2m from bΓ
3: for
 nβi
 
i∈[2m] ∈Prod
 
T|Bβ1|,Nu, . . . , T|Bβ2m |,Nu
 
do
4:
n ←L
i∈[2m] nβi
5:
Compute Wn by (6)
6:
W ′ ←Wn · Q2m
t=1
 |Bβt|
nβt
 
7:
// Uniform(0, 1) produces a uniformly random number over [0, 1]
8:
if Uniform(0, 1) < W ′
W then
9:
n∗←n
10:
break
11:
else
12:
W ←W −W ′
13:
end if
14: end for
15: for i ∈[2m] do
16:
Fetch the cell conﬁguration n∗
βi in βi from n∗
17:
Randomly partition Bβi into {Cβi,τ j}j∈[Nu] according to n∗
βi
18:
for j ∈[Nu] do
19:
Assign the 1-type τ j to all elements in Cβi,τ j
20:
end for
21: end for
The basic idea is again based on enumerative sampling. Let Bβ1, Bβ2, . . . , Bβ2m be the blocks
deﬁned by bΓ. Any cell conﬁguration n can be viewed as a concatenation of 2m cell conﬁgurations
nβ1, nβ2, . . . , nβ2m in the blocks, and each nβi is from the conﬁguration space T|Bβi|,Nu. Hence, in
the algorithm, the enumeration of all cell conﬁgurations is performed by applying the Cartesian
product function Prod on the conﬁguration spaces T|Bβ1|,Nu, T|Bβ2|,Nu, . . . , T|Bβ2m |,Nu.
For the computation of the sampling probability, we ﬁrst observe that any cell partitions that
produce the same conﬁguration n will have the same sampling probability. We use Wn to denote
the sampling weight (the numerator WFOMC of the probability) of any such cell partition. Then
the sampling probability of a cell conﬁguration n can be derived from Wn ·Q
i∈[2m]
 |Bβi|
nβi
 
, where
the latter product equates to the total number of the partitions giving rise to n.
The value of Wn will play a crucial role in the remaining sampling algorithm.
In this
context, we provide its formal deﬁnition. Given a nonnegative integer vector n of size Nc, let
en = P
i∈[Nc] ni, and Wn be deﬁned as
Wn := WFOMC(ΓT ∧
^
i∈[en]
eηi(ei), e∆, w, ¯w),
(6)
13


<!-- page 14 -->
where {ηi}i∈[en] is a set of cell types that gives rise to the conﬁguration n, e∆is a domain of size
en, and ei is the ith element in e∆. Recall that a cell type can be represented by a conjunction of
unary atoms, and thus each eηi(ei) in Wn can be interpreted as a set of unary facts. The value
of Wn is then equal to the WFOMC of the FO2 sentence ΓT conditional on a set of unary facts.
Such conditional counting problems have been thoroughly studied by Van den Broeck and Davis
[23], and the computational complexity has been shown to be polynomial in the domain size en
and the number of facts. Since the number of facts is clearly polynomial in en, computing Wn
can be done in time polynomial in en.
Lemma 2. The complexity of OneTypeSampler(·, ·, ·, ·) in Algorithm 1 is polynomial in the size
of the input domain.
Proof. For each block Bβi, there are totally |T|Bβi|,Nu| possible cell conﬁgurations in Bβi. Travers-
ing over all blocks in the loop at line 3 in Algorithm 1 will enumerate a total of Q2m
i=1 |T|Bβi|,Nu|
possible cell conﬁgurations. Even though this number may appear daunting, it is polynomial in
the domain size by the deﬁnition of conﬁguration space. The remaining complexity of the algo-
rithm is derived from the computation of Wn, which has been shown to be polynomial-time in
P
i∈[Nc] ni. Finally, the value of P
i∈[Nc] ni is equal to the domain size, completing the proof.
3.4.2
Domain Recursive Sampling
Now, let us consider the sampling problem of PΓ-structures conditional on the sampled 1-types
ηi. We rewrite the sampling probability P[A | bΓ ∧V
i∈[n] τi(ei)] as P[A | ΓT ∧V
i∈[n] ηi(ei)] for
better presentation.
Let πi,j be the 2-table of the element tuple (ei, ej), and denote by Ai the set of ground
2-tables over all element tuples involved in the element ei:
Ai :=
[
j∈[n]:j̸=i
πi,j(ei, ej).
Following the idea of domain recursion, we select an element et from ∆and decompose the
sampling probability as
P

A | ΓT ∧
^
i∈[n]
ηi(ei)

= P

A | ΓT ∧
^
i∈[n]
ηi(ei) ∧At

· P

At | ΓT ∧
^
i∈[n]
ηi(ei)

.
We ﬁrst demonstrate that for any valid substructure At of the sampling problem, the WFOMS
speciﬁed by the probability P[A | ΓT ∧V
i∈[n] ηi(ei) ∧At] can be reduced to a WFOMS of the
same form as the original problem on ΓT ∧V
i∈[n] ηi(ei), but over a smaller domain ∆′ = ∆\ et.
Given a 2-table π and a block type β, let β|π be a new block type:
β|π = β \ {Zk(x) | k ∈[m] : Rk(y, x) ∈π(x, y)}
We call β|π the relaxed block type of β under π, as it removes a part of the existential constraint
that is already satisﬁed by the relations in π. We can also apply the relaxation under π on a
cell type η = (β, τ), resulting in η|π = (β|π, τ). Let
eΓ = ΓT ∧
^
i∈[n]
ηi(ei) ∧At,
(7)
14


<!-- page 15 -->
and
eΓ′ = ΓT ∧
^
i∈[n]\{t}
ηi|πt,i(ei).
(8)
We have the following lemma.
Lemma 3. If At is valid w.r.t. the WFOMS of (ΓT ∧V
i∈[n] ηi(ei), ∆, w, ¯w), i.e., eΓ is satisﬁable,
the reduction from the WFOMS of (eΓ, ∆, w, ¯w) to (eΓ′, ∆′, w, ¯w) is sound.
Proof. Let S1 and S2 be the WFOMS of (eΓ, ∆, w, ¯w) and (eΓ′, w, ∆′, ¯w) respectively. Both of S1
and S2 have PΓ as their skeleton. Thus, the problem of S1 (resp. S2) is equivalent to sampling
a PΓ-structure over the domain ∆(resp. ∆′). The mapping function in the reduction can be
deﬁned on PΓ-structures over ∆′. We argue that the mapping function is f(A′) = A′∪At∪τt(et).
The function f is clearly deterministic and polynomial-time.
To simplify the rest arguments of the proof, we will ﬁrst show that f is bijective, i.e., for any
valid PΓ-structure A of S1, there exists a unique valid structure A′ of S2 such that f(A′) = A.
Let the respective structure to be A′ = A \ At \ {τt(et)}, and the uniqueness is clear. Next, we
demonstrate that A′ is valid w.r.t. S2. Ground out eΓ into ∆:
At∧
^
i∈[n]
ηi(ei) ∧
^
i,j∈[n]
ψ(ei, ej) ∧Λ,
where Λ = V
k∈[m]
V
i∈[n]
 
Zk(ei) ⇔W
j∈[n] Rk(ei, ej)
 
. By replacing the ground Tseitin atoms
in cell types ηi(ei) in Λ with their corresponding truth assignments and then discarding Λ, we
obtain a ground formula without Tseitin atoms:
At ∧
^
i∈[n]
τi(ei) ∧
^
i,j∈[n]
ψ(ei, ej) ∧
^
i∈[n]
^
k∈[m]:
Zk(x)∈βi
_
j∈[n]
Rk(ei, ej).
(9)
It can be easily shown that A is valid w.r.t. S1, iﬀA satisﬁes the formula (9). It follows that
the structure A′ satisﬁes the following formula
^
i∈[n]\{t}
τi(ei) ∧
^
i,j∈[n]\{t}
ψ(ei, ej) ∧
^
i∈[n]\{t}
^
k∈[m]:
Zk(x)∈βi|πt,i
_
j∈[n]\{t}
Rk(ei, ej),
(10)
which is obtained from (9) by substituting the ground atoms in At and τt(et). The formula (10)
is nothing else but the grounding of eΓ′ over ∆′ followed by the same replacement of ground
Tseitin atoms in the relaxed cell types ηi|πt,i. So we can conclude that A′ is also valid w.r.t. S2.
Now, we are prepared to demonstrate the consistency of sampling probability through the
mapping function. Since f is bijective, it remains to be shown that
P[f(A′) | eΓ; ∆, w, ¯w] = P[A′ | eΓ′; ∆′, w, ¯w]
for any valid structure A′ of S2. By the deﬁnition of the mapping function f, we have
⟨w, ¯w⟩(f(A′)) = ⟨w, ¯w⟩(A′) · ⟨w, ¯w⟩(At) · ⟨w, ¯w⟩(τt).
15


<!-- page 16 -->
Moreover, due to the bijection of f and the fact that PΓ is a skeleton of S1 and S2, we have
WFOMC(eΓ, ∆, w, ¯w) =
X
µ∈MeΓ,∆
⟨w, ¯w⟩(⟨µ⟩PΓ)
=
X
µ′∈MeΓ′,∆′
⟨w, ¯w⟩(f(⟨µ′⟩PΓ))
= ⟨w, ¯w⟩(At) · ⟨w, ¯w⟩(τt) ·
X
µ′∈MeΓ′,∆′
⟨w, ¯w⟩(⟨µ′⟩PΓ)
= ⟨w, ¯w⟩(At) · ⟨w, ¯w⟩(τt) · WFOMC(eΓ′, ∆′, w, ¯w).
(11)
Finally, by the deﬁnition of conditional probability, we can write
P[f(A′) | eΓ; ∆, w, ¯w] =
⟨w, ¯w⟩(f(A′))
WFOMC(eΓ, ∆, w, ¯w)
=
⟨w, ¯w⟩(A′) · ⟨w, ¯w⟩(At) · ⟨w, ¯w⟩(τt)
⟨w, ¯w⟩(A′) · ⟨w, ¯w⟩(At) · WFOMC(eΓ′, ∆′, w, ¯w)
=
⟨w, ¯w⟩(A′)
WFOMC(eΓ′, ∆′, w, ¯w)
= P[A′ | eΓ′; ∆′, w, ¯w],
(12)
and thus complete the proof.
With the sound reduction presented above, what remains to the algorithm is the sampling
of At given the probability P
h
At | ΓT ∧V
i∈[n] ηi(ei)
i
.
Recall that At consists of the ground 2-tables of all tuples comprising et and the elements
in ∆′. We follow a similar approach as in the cell type sampling and accomplish the sampling
of At through random partitions on cells. Let Cη1, Cη2, . . . , CηNc be the cell partition of ∆′
corresponding to the sampled cell types η1, η2, . . . , ηn−1. Let Nb be the number of all 2-tables,
and ﬁx the linear order of 2-tables π1, π2, . . . , πNb.
Any substructure At can be viewed as
partitions on each cell into Nb disjoint subsets; each subset corresponds to a 2-table πj and
precisely contains the elements that realize πj in combination with et.
Given a substructure At, we use
n
GAt
ηi,πj
o
j∈[Nb] to denote the reﬁned partition on Cηi, and
gAt
ηi
=
 
|GAt
ηi,πj|
 
j∈[Nb] its corresponding cardinality vector.
Let gAt = L
i∈[Nc] gAt
ηi
be the
concatenation of cardinality vectors over all cells, which is called the 2-table conﬁguration of At.
We ﬁrst assume that At is valid in the sampling problem (ΓT ∧V
i∈[n] ηi(ei), ∆, w, ¯w). It
will turn out that the sampling probability of At is completely determined by its corresponding
2-table conﬁguration gAt. To begin with, as stated by (11), we can write the sampling weight
WFOMC(ΓT ∧V
i∈[n] ηi(ei) ∧At, ∆, w, ¯w) as
WFOMC(eΓ′, ∆′, w, ¯w) · ⟨w, ¯w⟩(τt) · ⟨w, ¯w⟩(At)
(13)
where eΓ′, deﬁned as (8), is the reduced sentence by the 2-tables in At. Let nAt be the cell
conﬁguration corresponding to the ground cells in eΓ′.
The value of WFOMC(eΓ′, ∆′, w, ¯w) is
16


<!-- page 17 -->
exactly WnAt, which is formally deﬁned in Section 3.4.1. Denote by w =
 ⟨w, ¯w⟩(πi)
 
i∈Nb, the
weight vector of 2-tables. We can then write (13) as
WnAt · ⟨w, ¯w⟩(τt) ·
Y
i∈[Nc]
w
gAt
ηi .
(14)
In the equation above, τt has already been decided in the cell type ηt (by OneTypeSampler), and
the last term only depends on the 2-table conﬁgurations gAt. It is easy to check that the cell
conﬁguration of nAt is also fully determined by gAt. To illustrate this, let nAt
η
be the cardinality
of cell type η in nAt, and gAt
η,π the cardinality of π in gAt
η , i.e., |GAt
η,π|. For any cell type η, the
value of nAt
η
be can computed by
nAt
η
=
X
i∈[Nc],j∈[Nb]:ηi
|πj =η
gAt
ηi,πj.
(15)
By the argument above, the sampling probability (14) of a valid substructure At is com-
pletely determined by gAt. Thus, we can sample At, in the same spirit of sampling 1-types
in Section 3.4.1, by ﬁrst sampling a 2-table conﬁguration gAt, and then partitioning the cells
accordingly. We can then simply apply the enumerative sampling method, as the number of
possible 2-table conﬁgurations is clearly polynomial in the domain size. For any 2-table conﬁg-
uration gAt, its sampling weight can be computed by multiplying (14) by Q
i∈[Nc]
 nηi
gAt
ηi
 
, where
nηi is the size of Cηi.
So far in our discussion, we have been always assuming that the substructure At is valid
in the WFOMS (ΓT ∧V
i∈[n] ηi(ei), ∆, w, ¯w), or it should not be sampled. We guarantee this
assumption by imposing some constraints on the 2-table conﬁguration gAt. We call a 2-table π
coherent with a 1-types tuple (τ, τ ′) if, for some domain elements a and b, the interpretation of
τ(a) ∪π(a, b) ∪τ ′(b) satisﬁes the formula ψ(a, b) ∧ψ(b, a). Then, the ﬁrst constraint is that any
2-table πt,i in At must be coherent with τt and τi. This translates to a requirement on 2-table
conﬁguration that, when partitioning a cell ηi, the cardinality of 2-tables that are not coherent
with τt and τi is restricted to be 0. The second constraint is that, for any index k ∈{i | Zi ∈βt},
the substructure At must contain at least one ground atom of the form Rk(et, a), where a is a
domain element from ∆, to make At satisfy the existential formula ∃y : Rk(et, y). This means
that there must be at least one nonzero cardinality in the 2-table conﬁguration such that its
corresponding 2-table π satisﬁes Rk(x, y) ∈π.
By combining all the ingredients discussed above, we now present our sampling algorithm
for the sentence ΓT conditionally on the cell types ηi, as shown in Algorithm 2. The overall
structure of the algorithm follows a recursive approach, where a recursive call with a smaller
domain and relaxed cell types is invoked at Line 33. The algorithm terminates when the input
domain contains a single element (at Line 1) or there are no existential constraints on the
elements (at Line 4). In Lines 10-23, all possible 2-table conﬁgurations are enumerated. For
each conﬁguration, we compute its corresponding weight in Lines 13-15 and decide whether
it should be sampled in Lines 16-21. When the 2-table conﬁguration has been sampled, we
randomly partition the cells in Lines 25-32, and then update the sampled structure and the cell
type of each element respectively at Line 29 and 30. The function ExSat(g, η) at Line 12 is used to
check whether the 2-table conﬁguration g guarantees the validity of the sampled substructures,
as discussed above. The pseudo-code for this function is presented in Appendix A.5.2.
17


<!-- page 18 -->
Algorithm 2 DRSampler(ΓT , ∆, w, ¯w, (ηi)i∈[n])
1: if n = 1 then
2:
return ∅
3: end if
4: if only the block of type β = ⊤is nonempty then
5:
return a model µ sampled by a UFO2 WMS from ∀x∀y : ψ(x, y) ∧V
i∈[n] τi(ei) over ∆
under (w, ¯w)
6: end if
7: Choose t ∈[n]; µ ←τt(et); ∆′ ←∆\ {et}
8: Get the cell conﬁguration n =
 nηi
 
i∈[Nc] of (ηi)i∈[n]\{t}
9: W ←Wn
10: for
 gηi
 
i∈[Nc] ←Prod(Tnη1,Nb, . . . , TnηNc ,Nb) do
11:
g ←L
i∈[Nc] gηi
12:
if ExSat (g, ηt) then
13:
Get the new cell conﬁguration n′ w.r.t g by (15)
14:
Compute Wn′ by (6)
15:
W ′ ←Wn′ · ⟨w, ¯w⟩(τt) · Q
i∈[Nc]
 nηi
gηi
 
wgηi
16:
if Uniform(0, 1) < W ′
W then
17:
g∗←g
18:
break
19:
else
20:
W ←W −W ′
21:
end if
22:
end if
23: end for
24: Obtain the cell partition {Cηi}i∈[Nc] from (ηi)i∈[n]\{t}
25: for i ∈[Nc] do
26:
Fetch the cardinality vector g∗
ηi of ηi from g∗
27:
Randomly partition the cell Cηi into
 
Gηi,πj
	
j∈[Nb] according to g∗
ηi
28:
for j ∈[Nb] do
29:
µ ←µ ∪
 
πj(et, e)
	
e∈Gηi,πj
30:
∀es ∈Gηi,πj, η′
s ←ηs|πj
31:
end for
32: end for
33: µ ←µ ∪DRSampler(ΓT , ∆′, w, ¯w, (η′
i)i∈[n−1])
34: return µ
Lemma 4. The complexity of DRSampler(·, ·, ·, ·) in Algorithm 2 is polynomial in the size of the
input domain.
Proof. The algorithm DRSampler is called at most n times, where n is the size of the domain.
The main computation of each recursive call is for the loop, where we need to iterate over all
Q
i∈[Nc] |T nηi, Nb| possible conﬁgurations. The size of a conﬁguration space TM,m is polynomial
in M, and thus the complexity of this loop is also polynomial in the domain size. The other
complexity of computing Wn′ has been shown to be polynomial in the summation over the vector
18


<!-- page 19 -->
n′, which is clearly smaller than the domain size.
3.4.3
A Lifted WMS for FO2
We present our WMS for FO2 in Algorithm 3. Given a FO2 sentence Γ in SNF, the algorithm
ﬁrst obtains the sentence ΓT of the general form (5). Then the algorithms OneTypeSampler and
DRSampler then applied successively to sample a skeleton structure of bΓ. It is easy to verify
that the skeleton structure is also a model of Γ, as it can be regarded as the output of the
mapping function in the sound reduction from the WFOMS problem on Γ to bΓ. Since both
OneTypeSampler and DRSampler have been proved to be polynomial-time in the domain size by
Lemma 2 and 4, the WMS in Algorithm 3 is clearly lifted.
Algorithm 3 WMS(Γ, ∆, w, ¯w)
INPUT: An FO2 sentence Γ of the form (3), a domain ∆= {ei}i∈[n] of size n, a weighting
(w, ¯w)
OUTPUT: A model µ of Γ over ∆
1: Construct ΓT from Γ by (4)
2: ∀i ∈[n], βi(x) ←V
k∈[m] Zk(x)
3: bΓ ←ΓT ∧V
i∈[n] βi(ei)
4: (τi)i∈[n] ←OneTypeSampler(bΓ, ∆, w, ¯w)
5: ∀i ∈[n], ηi ←(βi, τi)
6: µ ←DRSampler(ΓT , ∆, w, ¯w, (ηi)i∈[n])
7: return µ
Theorem 2. The fragment FO2 is domain-liftable under sampling.
Proof of Theorem 2. The proof is directly following from the above and from Lemma 1.
Remark 1. We note that there are several optimizations to our WMS, e.g., heuristically se-
lecting the domain element in DRSampler so that the algorithm can quickly reach the terminal
condition. However, the current algorithm is clear and eﬃcient enough to prove our main result,
so that we leave the discussion on some of the optimizations to Appendix A.5.1.
4
A Genralization to FO2 with Cardinality Constraints
In this section, we extend our results to FO2 with cardinality constraints. A single cardinality
constraint is a statement of the form |P| ▷◁q, where ▷◁is a comparison operator (e.g., =, ≤,
≥, <, >) and q is a natural number. These constraints are imposed on the number of distinct
positive ground literals in a structure A formed by the predicate P. For example, a structure
A satisﬁes the constraint |P| ≤q if there are at most q literals for P that are true in A.
For illustration, we allow cardinality constraints as atomic formulas in the FO formulas, e.g.,
(|E| = 2) ∧(∀x∀y : E(x, y) ⇒E(y, x)) (its models can be interpreted as undirected graphs with
exactly one edge) and the satisfaction relation |= is extended naturally.
The cardinality constraints are not necessarily expressible in FO logic without grounding
out the constraint over the domain, and have a strong connection to the fragment of C2, which
19


<!-- page 20 -->
will be introduced in the next section. In [1], the authors also extended their WMS (which was
originally developed for UFO2) to handle cardinality constraints. However, their method was
relatively straightforward whereas the extension to FO2 is more complicated.
Let Γ be an FO2 sentence and
Υ := ϕ(|P1| ▷◁q1, . . . , |PM| ▷◁qM),
(16)
where ϕ is a Boolean formula, {Pi}i∈[M] ⊆PΓ, and ∀i ∈[M], qi ∈N. Consider the WFOMS
problem on Γ ∧Υ over the domain ∆under (w, ¯w).
The overall structure of the sampling
algorithm for Γ ∧Υ remains unchanged from Algorithm 3.
The algorithm still begins with
obtaining the general sentence bΓ from Γ. Then the formula bΓ ∧Υ is fed into OneTypeSampler
and DRSampler successively to sample the PΓ-structure. Please refer to Appendix A.5.3 for the
detailed algorithm. We only describe its modiﬁcations to the original one below.
The algorithm of OneTypeSampler for bΓ∧Υ is similar to Algorithm 1, with the main diﬀerence
being that the computation of WFOMC problems, speciﬁcally WFOMC(bΓ, ∆, w, ¯w) and Wn, now
include cardinality constraints Υ in their input sentences. To account for this change, we slightly
modify the deﬁnition of Wn in (6) by taking ΓT ∧V
i∈[en] eηi ∧Υ as input, and denote the new
term by Wn,Υ. According to Proposition 5 in [7], the addition of cardinality constraints to a
liftable sentence does not aﬀect the liftability of the resulting formula (in terms of WFOMC
problems). Therefore, the computation of Wn,Υ remains polynomial-time in the domain size, as
the original sentence in Wn was already proven to be liftable.
For the sampling problem conditional on the sampled cell types ηi, the domain recursive
property still holds as we will show in turn. Given a set L of ground literals and a predicate P,
let N(P, L) denote the number of positive ground literals for P in L. Given a valid substructure
At of the element et, denote the 1-type of et by τt as usual, let q′
i = qi −N(Pi, At)−N(Pi, τt(et))
for every i ∈[M], and deﬁne
Υ′ = ϕ(|P1| ▷◁q′
1, . . . , |PM| ▷◁q′
M).
(17)
Let eΓC = eΓ ∧Υ and eΓ′
C = eΓ′ ∧Υ′, where eΓ and eΓ′ are deﬁned as (7) and (8) respectively. Then
the reduction from the WFOMS problem on eΓ to eΓ′ is sound.
Lemma 5. If At is valid w.r.t. the WFOMS of (V
i∈[n] ηi(ei) ∧ΓT ∧Υ, ∆, w, ¯w), i.e., eΓC is
satisﬁable, the reduction from the WFOMS of (eΓC, ∆, w, ¯w) to (eΓ′
C, ∆′, w, ¯w′) is sound.
Proof of Lemma 5. The proof follows the same argument for Lemma 3. The only statement
that needs to be argued again is the bijection of the mapping function f(A′) = A′ ∪At ∪τt(et).
Let SC
1 and SC
2 be the WFOMS of (eΓC, ∆, w, ¯w) and (eΓ′
C, ∆′, w, ¯w). For any valid A of SC
1 , A
must satisfy both eΓ and Υ. It follows that A′ = A \ At \ {τt(et)} satisﬁes eΓ′ and Υ′, meaning
that A′ is also valid w.r.t.
SC
2 .
This establishes the bijection of f.
The remainder of the
proof, including the consistency of sampling probability, proceeds exactly the same as Lemma 3,
speciﬁcally follows (11) and (12).
The core structure of DRSampler remains the same as the sound reduction still holds. How-
ever, the recursive call is now made with the reduced sentence ΓT ∧Υ′.
The other slight
modiﬁcations include:
• WnAt in (14) has been replaced with WnAt,Υ′, and
20


<!-- page 21 -->
• the validity check for the sampled 2-table conﬁguration in ExSat now includes an additional
check for the well-deﬁnedness of the reduced cardinality constraints Υ′, returning False if
any q′
i /∈N for i ∈[M].
As discussed above, the extension of our sampling algorithm to handle cardinality constraints
in FO2 only slightly increases the complexity of the procedure. Furthermore, the computation
of Wn,Υ remains polynomial-time in the domain size, meaning that the generalized algorithm is
still lifted, and thus proving the liftability under sampling of FO2 with cardinality constraints 5.
Theorem 3. Let Γ be an FO2 sentence and Υ of the form (16). Then Γ ∧Υ is domain-liftable
under sampling.
Proof. The proof follows from the discussion above.
5
A Further Generalization to SC2
With the lifted WMS for FO2 with cardinality constraints, we can further extend our result to
the case involving the counting quantiﬁers ∃=k [27]. Here, we study the sentences of the form
Γ ∧(∀x∃=k1y : φ1(x, y)) ∧· · · ∧(∀x∃=kM′ : φM′(x, y))
∧(∃=k′
1x∀y : φ′
1(x, y)) ∧· · · ∧(∃=k′
M′′x∀y : φ′
M′′(x, y)),
where Γ is an FO2 sentence and ∃=k is the counting quantiﬁer that speciﬁes the exact number of
elements in the domain that satisfy a given formula. For instance, a structure A over a domain ∆
satisﬁes the sentence ∃=kx : ψ(x), if there are exactly k distinct elements t1, . . . , tk ∈∆such that
A |= ψ(ti) for all i ∈[k]. We call this fragment two-variable logic with counting in SNF SC2, as
its extended conjunction to FO2 sentences resembles SNF. The presence of counting quantiﬁers
signiﬁcantly enhances the expressiveness of SC2, e.g., k-regular graphs can be encoded in SC2,
as demonstrated in the introduction.
Recently, Kuzelka [7] showed that the liftability of FO2 can be generalized to the fragment
of two-variable logic with counting C2, a superset of SC2, by reducing the WFOMC problem on
C2 sentences to FO2 sentences with cardinality constraints. We demonstrate that this reduction
can be also applied to the sampling problem and it is sound, when the fragment is restricted to
be SC2.
Lemma 6. For any WFOMS S = (Φ, ∆, w, ¯w) where Φ is a SC2 sentence, there exists a
WFOMS S′ = (Γ′∧Υ, ∆, w′, ¯w′), where Γ′ is an FO2 sentence, Υ denotes cardinality constraints
of the form (16) and both Γ′ and Υ are independent of ∆, such that the reduction from S to S′
is sound.
The proof follows a similar technique used in [7], and the details are deferred to Appendix A.2.
We note here that further generalizing this result to the general C2 sentences is infeasible,
since the original reduction used in [7] for C2 sentences introduced some negative weights on
predicates. However, it can be established that the domain-liftability under sampling of C2 can
5It is worth noting that the computational complexity of Wn,Υ is independent of the values q1, . . . , qM in the
cardinality constraints Υ, so that the reduction on these constraints does not aﬀect the liftability of the reduced
WnAt ,Υ′.
21


<!-- page 22 -->
be demonstrated by directly applying our domain recursion sampling method without resorting
to the reduction to cardinality constraints.
For a more detailed discussion, please refer to
Appendix A.3.
Since FO2 with cardinality constraints has been proved to be liftable under sampling, it is
easy to prove the liftability under sampling of the SC2 fragment.
Theorem 4. The fragment of SC2 is domain-liftable under sampling.
Proof. The proof follows from Lemma 6 and Theorem 3.
Moreover, one can further introduce additional cardinality constraints into SC2 without
degrading its liftability under sampling.
Corollary 1. Let Φ be a SC2 sentence and Υ of the form (16). Then Φ ∧Υ is domain-liftable
under sampling.
6
Experimental Results
We conducted several experiments to evaluate the performance and correctness of our sampling
algorithms. All algorithms were implemented in Python and the experiments were performed
on a computer with an 8-core Intel i7 3.60GHz processor and 32 GB of RAM 6.
Many sampling problems can be expressed as WFOMS problems. Here we consider two
typical ones.
• Sampling combinatorial structures: the uniform generation of some combinatorial
structures can be directly reduced to a WFOMS, e.g., the uniform generation of graphs
with no isolated vertices and k-regular graphs in Examples 1 and the introduction. We
added four more combinatorial sampling problems to these two for evaluation: functions,
functions w/o ﬁx-points (i.e., the functions f satisfying f(x) ̸= x), permutations and per-
mutations w/o ﬁx-points. The details of these problems are described in Appendix A.4.1.
• Sampling from MLNs: our algorithms can be also applied to sample possible worlds
from MLNs. An MLN deﬁnes a distribution over structures (i.e., possible worlds in SRL
literature), and its respective sampling problem is to randomly generate possible worlds
according to this distribution. There is a standard reduction from the sampling problem
of an MLN to a WFOMS problem (see Append A.4.1 and also [1]). We used two MLNs
in our experiments: 1) A variant of the classic friends-smokers MLN with the constraint
that every person has at least one friend:
{(∞, ¬fr(x, x)), (∞, fr(x, y) ⇒fr(y, x)), (0, sm(x)),
(0.2, fr(x, y) ∧sm(x) ⇒sm(y)), (∞, ∃y : fr(x, y))}.
2) The employment MLN used in [4]:
{(1.3, ∃y : workfor(x, y) ∨boss(x))},
which states that with high probability, every person either is employed by a boss or is
a boss. The details about the reduction from sampling from MLNs to WFOMS and the
corresponding WFOMS problems of these two MLNs can be found in Appendix A.4.1.
6The code can be found in https://github.com/lucienwang1009/lifted_sampling_fo2
22


<!-- page 23 -->
Figure 2: Uniformity comparison between an ideal sampler (IS) and our WMS.
(a) friends-smokers
(b) employment
Figure 3: Conformity testing for the count distribution of MLNs.
6.1
Correctness
We ﬁrst examine the correctness of our implementation on the uniform generation of combi-
natorial structures over small domains, where exact sampling is feasible via enumeration-based
techniques; we choose the domain size of 5 for evaluation. To serve as a benchmark, we imple-
mented a simple ideal uniform sampler, denoted by IS, by enumerating all the models and then
drawing samples uniformly from these models. For each combinatorial structure encoded into
an FO2 sentence Γ, a total of 100 × |MΓ,∆| models were generated from both IS and our WMS.
Figure 2 depicts the model distribution produced by these two algorithms—the horizontal axis
represents models numbered lexicographically, while the vertical axis represents the generated
frequencies of models. The ﬁgure suggests that the distribution generated by our WMS is in-
distinguishable from that of IS. Furthermore, a statistical test on the distributions produced by
WMS was performed, and no statistically signiﬁcant diﬀerence from the uniform distribution
was found. The details of this test can be found in Appendix A.4.2.
For sampling problems from MLNs, enumerating all the models is infeasible even for a domain
23

[CAPTION] Figure 2: Uniformity comparison between an ideal sampler (IS) and our WMS.

[CAPTION] Figure 3: Conformity testing for the count distribution of MLNs.

[CAPTION] Figure 2 depicts the model distribution produced by these two algorithms—the horizontal axis


<!-- page 24 -->
Figure 4: Performance of WMS versus UniGen.
of size 5, e.g., there are 225+5 = 237 models in the employment MLN. That is why we test the
count distribution of vocabulary for these two MLNs. Instead of specifying the probability of
each model, the count distribution only tells us how probably a certain number of predicates
are interpreted to be true in the models. An advantage of testing count distributions is that
they can be eﬃciently computed for our MLNs. Please refer to [7] for more details about count
distributions. We also note that the conformity of count distribution is a necessary condition
for the correctness of algorithms. We keep the domain size to be 5 and sampled 105 models
from friends-smokers and employment MLNs respectively. The empirical distributions of count-
statistics, along with the true count distributions, are shown in Figure 3. It is easy to check the
conformity of the empirical distribution to the true one from the ﬁgure. The statistical test was
also performed on the count distribution, and the results conﬁrm the conclusion drawn from the
ﬁgure (also see Appendix A.4.2).
6.2
Performance
To evaluate the performance, we compared our weighted model samplers with Unigen [18, 28],
the state-of-the-art approximate sampler for Boolean formulas. A WFOMS problem can be
reduced to a sampling problem of the Boolean formula by grounding the input sentence over
the given domain. Since Unigen only works for uniform sampling, we employed the technique
in [29] to encode the weighting function in the WFOMS problem into a Boolean formula.
For each sampling problem, we randomly generated 1000 models by our WMS and Unigen
respectively and computed the average sampling time of one model. The performance com-
parison is shown in Figure 4. In most cases, our approach is much faster than UniGen. The
exception in the employment MLN, where UniGen performed better than WMS, is likely due
24

[CAPTION] Figure 4: Performance of WMS versus UniGen.


<!-- page 25 -->
to the simplicity of this speciﬁc instance for its underlying SAT solver. This coincides with the
theoretical result that our WMS is polynomial-time in the domain size, while UniGen usually
needs amounts of expensive SAT calls on the grounding formulas.
7
Conclusion and Future Work
In this paper, we prove the domain-liftability under sampling of FO2 by presenting a novel and
eﬃcient approach to its symmetric weighted ﬁrst-order model sampling problems. The result
is further extended to the fragment of SC2 with the presence of counting constraints.
The
widespread applicability of WFOMS renders the proposed approach a promising candidate to
serve as a universal paradigm for a plethora of sampling problems.
A potential avenue for further research is to expand the methodology presented in this paper
to encompass more expressive ﬁrst-order languages. Speciﬁcally, the utilization of the domain
recursion scheme employed in this study could be extended beyond the conﬁnes of the FO2 and
SC2, as its analogous counterpart in WFOMC has been demonstrated to be eﬀective in proving
the domain-liftability of the fragments S2FO2 and S2RU [15].
In addition to extending the input logic, other potential directions for future research include
incorporating elementary axioms, such as tree axiom [30] and linear order axiom [16], as well
as more general weighting functions that involve negative weights. However, it is important to
note that these extensions would likely require a more advanced and nuanced approach than the
one proposed in this paper, and may present signiﬁcant challenges.
Finally, the lower complexity bound of WFOMS is also an interesting open problem. A
direct implication from the infeasibility of WFOMC in [31] suggests that there is unlikely for an
(even approximate) lifted WMS to exist for full ﬁrst-order logic. However, the establishment of
a tighter lower bound for fragments of FO, such as FO3, remains an unexplored and challenging
area that merits further investigation.
Acknowledgement
The authors would like to thank the anonymous reviewers for their helpful comments. Yuanhong
Wang and Juhua Pu are supported by the National Key R&D Program of China (2021YFB2104800)
and the National Science Foundation of China (62177002). Ondˇrej Kuˇzelka’s work is supported
by the Czech Science Foundation project 20-19104Y and partially also 23-07299S (most of the
work was done before the start of the latter project)
25


<!-- page 26 -->
A
Appendix
A.1
Scott Normal Forms
We brieﬂy describe the transformation of FO2 formulas to SNF and prove the soundness of
its corresponding reduction on the WFOMS problems. The process is well-known, so we only
sketch the related details.
Let Γ be a sentence of FO2. To put it into SNF, consider a subformula ϕ(x) = Qy : φ(x, y),
where Q ∈{∀, ∃} and φ is quantiﬁer-free. Let Aϕ be a fresh unary predicate7 and consider the
sentence
∀x : (Aϕ(x) ⇔(Qy : φ(x, y)))
which states that ϕ(x) is equivalent to Aϕ(x). Let Q′ denote the dual of Q, i.e., Q′ = {∀, ∃}\{Q},
this sentence can be seen equivalent to
Γ′ :=∀xQy : (Aϕ(x) ⇒φ(x, y))
∧∀xQ′y : (φ(x, y) ⇒Aϕ(x)).
Let
Γ′′ = Γ′ ∧Γ[ϕ(x)/Aϕ(x)],
where Γ[ϕ(x)/Aϕ(x)] is obtained from Γ by replacing ϕ(x) with Aϕ(x). For any domain ∆,
every model of Γ′′ over ∆can be mapped to a unique model of Γ over ∆. The bijective mapping
function is simply the projection ⟨·⟩PΓ. Let both the positive and negative weights of Aϕ be 1 and
denote the new weighting functions as w′ and ¯w′. It is clear that the reduction from (Γ, ∆, w, ¯w)
to (Γ′′, ∆, w′, ¯w′) is sound. Repeat this process from the atomic level and work upwards until
the sentence is in SNF. The whole reduction remains sound due to the transitivity of soundness.
A.2
A Sound Reduction from SC2 to FO2 with Cardinality Constraints
In this section, we show the sound reduction from a WFOMS problem on SC2 sentence to a
WFOMS problem on FO2 sentence with cardinality constraints.
We ﬁrst need the following two lemmas.
Lemma 7. Let Γ be a ﬁrst-order logic sentence, and let ∆be a domain. Let Φ be a ﬁrst-order
sentence with cardinality constraints, deﬁned as follows:
Π :=(|P| = k · |∆|)
∧(∀x∀y : P(x, y) ⇔(RP
1 (x, y) ∨· · · ∨RP
k (x, y)))
∧
^
i∈[k]
(∀x∃y : RP
i (x, y))
∧
^
i,j∈[k]:i̸=j
(∀x∀y : ¬RP
i (x, y) ∨¬RP
j (x, y)),
where RP
i
are auxiliary predicates not in PΓ with weight w(RP
i ) = ¯w(RP
i ) = 1.
Then the
reduction from the WFOMS (Γ ∧∀x∃=ky : P(x, y), ∆, w, ¯w) to (Γ ∧Π, ∆, w, ¯w) is sound.
7If ϕ(x) has no free variables, e.g., ∃x : φ(x), the predicate Aϕ is nullary.
26


<!-- page 27 -->
Proof. Let f(·) = ⟨·⟩PΓ∪{P} be a mapping function. We ﬁrst show that f is from MΓ∧Π,∆to
MΓ∧∀x∃=k:P(x,y),∆: if A |= Γ ∧Π then f(A) |= Γ ∧∀x∃=ky : P(x, y).
The sentence Π means that for every c1, c2 ∈∆such that P(c1, c2) is true, there is exactly
one i ∈[k] such that RP
i (c1, c2) is true.
Thus we have that P
i∈[k] |RP
i | = |P| = k · |∆|,
which together with V
i∈[k] ∀x∃y : RP
i (x, y) implies that |RP
i | = k for i ∈[k]. We argue that
each RP
i
is a function predicate in the sense that ∀x∃=1y : RP
i (x, y) holds in any model of
Γ ∧Π. Let us suppose, for contradiction, that (∀x∃y : RP
i (x, y)) ∧(|RP
i | = k) holds but there
is some a ∈∆such that RP
i (a, b) and RP
i (a, b′) are true for some b ̸= b′ ∈∆.
We have
|{(x, y) ∈∆2 | RP
i (x, y) ∧x ̸= a}| ≥|∆| −1 by the fact ∀x∃y : RP
i (x, y).
It follows that
|RP
i | ≥|{(x, y) ∈∆2 | RP
i (x, y) ∧x ̸= a}| + 2 > |∆|, which leads to a contradiction. Since all of
RP
i are function predicates, it is easy to check ∀x∃=ky : P(x, y) must be true in any model µ of
Γ ∧Π, i.e., f(µ) |= Γ ∧∀x∃=ky : P(x, y).
To ﬁnish the proof, one can easily show that, for every model µ ∈MΓ∧∀x∃=ky:P(x,y),∆, there
are exactly (k!)|∆| models µ′ ∈MΓ∧Π,∆such that f(µ′) = µ. The reason for this is that 1) if,
for any a ∈∆, we permute b1, b2, . . . , bk in RP
1 (a, b1), RP
2 (a, b2), . . . , RP
k (a, bk) in the model µ′,
we get another model of Γ ∧Π, and 2) up to these permutations, the predicates RP
i in µ′ are
determined uniquely by µ. Finally, the weights of all these µ′s are the same as those of µ, and
we can write
X
µ′∈MΓ∧Π,∆:
f(µ′)=µ
P[µ′ | Γ ∧Π] =
P
µ′∈MΓ∧Π,∆:
f(µ′)=µ
µ′
WFOMC(Γ ∧Π, ∆, w, ¯w)
=
(k!)|∆| · ⟨w, ¯w⟩(µ)
(k!)|∆| · WFOMC(Γ ∧∀x∃=ky : P(x, y), ∆, w, ¯w)
=
⟨w, ¯w⟩(µ)
WFOMC(Γ ∧∀x∃=ky : P(x, y), ∆, w, ¯w)
= P[µ | Γ ∧∀x∃=ky : P(x, y)],
which completes the proof.
Lemma 8. Let Γ be a ﬁrst-order logic sentence, ∆be a domain, and P be a predicate. Then
the WFOMS (Γ ∧∀=k∀y : P(x, y), ∆, w, ¯w) can be reduced to (Γ ∧(|U| = k) ∧(∀x : U(x) ⇔(∀y :
P(x, y))), ∆, w, ¯w), where U is an auxiliary unary predicate with weight w(U) = ¯w(U) = 1, and
the reduction is sound.
Proof. The proof is straightforward.
Proof of Lemma 6. We can ﬁrst get rid of all formulas of the form ∃=kx∀y : P(x, y) by repeatedly
using Lemma 8. Then we can use Lemma 7 repeatedly to eliminate the formulas of the form
∀x∃=ky : P(x, y). The whole reduction is sound due to the transitivity of soundness.
A.3
Applying Domain Recursion Scheme on C2 is Possible
The C2 sentences that we need to handle are of the form
Γ ∧
^
k∈[q]
(∀x : Ak(x) ⇔(∃=mky : Rk(x, y))),
(18)
27


<!-- page 28 -->
where Γ is a FO2 sentence, each Rk(x, y) is an atomic formula, and each Ak is an auxiliary
Tseitin predicate. Any WFOMS problem on C2 can be reduced to a new one, whose input
sentence is of the above form and maxk∈[q] mk ≤|∆|, by the following steps:
• Convert each counting-quantiﬁed formula of the form ∃≥my : ϕ(x, y) to ¬(∃≤m−1y :
ϕ(x, y)).
• Decompose each ∃≤my : ϕ(x, y) into (∀y : ¬ϕ(x, y)) ∧W
i∈[m](∃=iy : ϕ(x, y)).
• Replace each subformula ∃=my : ϕ(x, y), where m > |∆|, with False.
• Starting from the atomic level and working upwards, replace any subformula ∃=mϕ(x, y),
where ϕ(x, y) is a formula that does not contain any counting quantiﬁer, with A(x); and
append ∀x∀y : R(x, y) ⇔ϕ(x, y) and ∀x : A(x) ⇔(∃=my : R(x, y)), where R is an
auxiliary binary predicate, to the original sentence.
It is easy to check that the reduction presented above is sound and independent of the domain
size if the domain size is greater than the maximum counting parameter m in the input sentence.8
We ﬁrst sample the 1-types of each element from the sentence (18) so that all the predicates
Ak will be eliminated. The resulting WFOMS is then deﬁned on the following sentence:
Γ0 ∧
^
k∈[q]

^
e∈∆∃
k
∃=mky : Rk(e, y) ∧
^
e∈∆∄
k
¬(∃=ky : Rk(e, y))

,
where Γ0 is the simpliﬁed sentence of Γ by replacing all its unary literals with their truth values,
∆∃
k contains precisely the elements with positive sampled literals Ak(e), and ∆∄
k = ∆\ ∆∃
k.
We need to consider a more general WFOMS problem to apply our domain recursion scheme.
For each counting quantiﬁed formula ∃=ky : Rk(x, y), we introduce 2mk new unary predicates
Z∃
k,1, Z∃
k,2, . . . , Z∃
k,mk, Z∄
k,1, Z∄
k,2, . . . , Z∄
k,mk, and append the conjunction of
∀x :
 
Z∃
k,t(x) ⇔(∃=ty : Rk(x, y))
 
∧
 
Z∄
k,t(x) ⇔¬(∃=ty : Rk(x, y))
 
over t ∈[mk] to Γ0, resulting in a new sentence Γ1. The more general WFOMS is then deﬁned
on
Γ1 ∧
^
i∈[n]
νi(ei),
(19)
where each νi(x) is a quantiﬁer-free conjunction over a subset of {Z∃
k,t(x)}t∈[mk]∪{Z∄
k,t(x)}t∈[mk].
It is easy to check that the original WFOMS of (18) is reducible to the more general WFOMS
problem, and the reduction is sound and independent of the domain size.
We show that the domain recursion scheme is still applicable to the WFOMS of (19). We
only provide an intuition here while leaving the details for the future version of this paper.
Additionally, we hope to discover a more practical and eﬃcient solution in the future that would
introduce fewer unary predicates, despite the current approach being domain-lifted.
The intuition is that we can view νi(x) as the “block type” similar to what we have done in
the WMS of FO2. Then the domain recursion strategy is applied, ﬁrst sampling the substructure
8This condition does not change the data complexity of the problem, as all the counting parameters in the
sentence are considered constants but not the input of the problem.
28


<!-- page 29 -->
of an element, and then updating each block type accordingly. The updated block types can still
be represented by the unary predicates Z∃
k,t and Z∄
k,t, and the new sampling problem is reducible
to a new WFOMS of the general form. Following the similar argument for sampling FO2, the
corresponding WMS for C2 is also lifted, which means that the full fragment of C2 is liftable
under sampling.
A.4
Missing Details of Experiments
A.4.1
Experiment Settings
Sampling Combinatorial Structures
The corresponding WFOMS problems for the uni-
form generation of combinatorial structures used in our experiments are presented as follows.
The weighting functions w and ¯w map all predicates to 1.
• Functions:
∀x∃=1y : f(x, y).
• Functions w/o ﬁx points:
(∀x∃=1y : f(x, y)) ∧(∀x : ¬f(x, x)).
• Permutations:
(∀x∃=1y : Per(x, y)) ∧(∀y∃=1x : Per(x, y)).
• Permutation without ﬁx-points:
(∀x∃=1y : Per(x, y)) ∧(∀y∃=1x : Per(x, y)) ∧(∀x : ¬Per(x, x)).
Sampling from MLNs
An MLN is a ﬁnite set of weighted ﬁrst-order formulas {(wi, αi)}i∈[m],
where each wi is either a real-valued weight or ∞, and αi is a ﬁrst-order formula. Let P be
the vocabulary of α1, α2, . . . , αm. An MLN Φ paired with a domain ∆induces a probability
distribution over P-structures (also called possible worlds):
pΦ,∆(ω) :=
(
1
ZΦ,∆exp
 P
(α,w)∈ΦR w · #(α, ω)
 
if ω |= Φ∞
0
otherwise
where ΦR and Φ∞are the real-valued and ∞-valued formulas in Φ respectively, and #(α, ω)
is the number of groundings of α satisﬁed in ω. The sampling problem on an MLN Φ over a
domain ∆is to randomly generate a possible world ω according to the probability pΦ,∆(ω).
The reduction from the sampling problems on MLNs to WFOMS can be performed as follows.
For every real-valued formula (αi, wi) ∈ΦR, where the free variables in αi are x, we introduce
a novel auxiliary predicate ξi and create a new formula ∀x : ξi(x) ⇔αi(x). For formula αi
with inﬁnity weight, we instead create a new formula ∀x : αi(x). Denote the conjunction of
the resulting set of sentences by Γ, and set the weighting function to be w(ξi) = exp(wi) and
¯w(ξi) = 1, and for all other predicates, we set both w and ¯w to be 1. Then the sampling problem
on Φ over ∆is reduced to the WFOMS (Γ, ∆, w, ¯w).
By the reduction above, we can write the two MLNs used in our experiments to WFOMS
problems. The weights of predicates are all set to be 1 unless otherwise speciﬁed.
29


<!-- page 30 -->
• Friends-smokers MLN: the reduced sentence is
(∀x : ¬fr(x, x) ∧sm(x))
∧(∀x∀y : fr(x, y) ⇔fr(y, x))
∧(∀x∀y : ξ(x, y) ⇔(fr(x, y) ∧sm(x) ⇒sm(y)))
∧(∀x∃y : fr(x, y)),
and the weight of ξ is set to be w(ξ) = exp(0.2).
• Employment MLN: the corresponding sentence is
∀x : ξ(x) ⇔(∃y : workfor(x, y) ∨boss(x)),
and the weight of ξ is set to be exp(1.3).
A.4.2
More Experimental Results
The Kolmogorov–Smirnov Test
We utilized the Kolmogorov-Smirnov (KS) test [32] to
validate the conformity of the (count) distributions produced by our algorithm to the reference
distributions. The KS test used here is based on the multivariate Dvoretzky–Kiefer–Wolfowitz
(DKW) inequality recently proved by [33].
Let X1 = (X1i)i∈[k], X2 = (X2i)i∈[k], . . . , Xn = (Xni)i∈[k] be n real-valued independent and
identical distributed multivariate random variables with cumulative distribution function (CDF)
F(·). Let Fn(·) be the associated empirical distribution function deﬁned by
Fn(x) := 1
n
X
i∈[n]
1Xi1≤x1,Xi2≤x2,...,Xik≤xk,
x ∈Rk.
The DKW inequality states
P
"
sup
x∈Rk |Fn(x) −F(x)| > ϵ
#
≥(n + 1)ke−2nϵ2
(20)
for every ϵ, n, k > 0. When the random variables are univariate, i.e., k = 1, we can replace
(n + 1)k in the above probability bound by a tighter constant 2.
Table 1: The Kolmogorov-Smirnov Test
Problem
Maximum deviation
Upper bound
graphs w/o isolated vertices
0.0036
0.0049
2-regular graphs
0.0065
0.0069
functions
0.0013
0.0024
functions w/o ﬁx-points
0.0027
0.0042
permutations
0.0071
0.0124
permutations w/o ﬁx-points
0.019
0.02
friends-smokers
0.0021
0.0087
employmenet
0.0030
0.0087
30


**[Table p30.1]**
| Problem | Maximum deviation | Upper bound |
| --- | --- | --- |
| graphs w/o isolated vertices | 0.0036 | 0.0049 |
| 2-regular graphs | 0.0065 | 0.0069 |
| functions | 0.0013 | 0.0024 |
| functions w/o fix-points | 0.0027 | 0.0042 |
| permutations | 0.0071 | 0.0124 |
| permutations w/o fix-points | 0.019 | 0.02 |
| friends-smokers | 0.0021 | 0.0087 |
| employmenet | 0.0030 | 0.0087 |

[CAPTION] Table 1: The Kolmogorov-Smirnov Test


<!-- page 31 -->
In the KS test, the null hypothesize is that the samples X1, X2, . . . , Xn are distributed ac-
cording to some reference distribution, whose CDF is F(·). Then by (20), with probability 1−α,
the maximum deviation supx∈Rk |Fn(x) −F(x)| between empirical and reference distributions
is bounded by ϵ =
p
ln(k(n + 1)/α)/2n (
p
ln(2/α)/2n for the univariate case). If the actual
value of the maximum deviation is larger than ϵ, we can reject the null hypothesis at the conﬁ-
dence level α. Otherwise, we cannot reject the null hypothesis, i.e., the empirical distribution of
the samples is not statistically diﬀerent from the reference one. In our experiments, we choose
α = 0.05 as a signiﬁcant level.
For the uniform generation of combinatorial structures, we assigned each model a lexico-
graphical number and treated the model index as a random variable with a discrete uniform
distribution. For the sampling problems of MLNs, we test their count distributions against the
true count distributions. Table 1 shows the maximum deviation between the empirical and refer-
ence cumulative distribution functions, along with the upper bound set by the DKW inequality.
As shown in Table 1, all maximum deviations are within their respective upper bounds. There-
fore, we cannot reject any null hypotheses, i.e., there is no statistically signiﬁcant diﬀerence
between the two sets of distributions.
A.5
Missing Details of WMS
A.5.1
Optimizations for WMS
There exist several optimizations to make it more practical. Here, we present some of them that
are used in our implementation.
• The complexity of DRSampler heavily depends on the recursion depth. In our implementa-
tion, when selecting a domain element et for sampling its substructure, we always chose the
element with the “strongest” existential constraint that contains the most Tseitin atoms
Zk(x). It would help DRSampler fast reach the condition that the existential constraint for
all elements is ⊤. In this case, DRSampler will invoke the more eﬃcient WMS for UFO2
to sample the remaining substructures.
• Let P∃be the union of vocabularies of the existentially quantiﬁed formulas
P∃:= Pϕ1(x,y) ∪Pϕ2(x,y) ∪· · · ∪Pϕm(x,y).
We further decomposed the sampling probability P[A | ΓT ∧V
i∈[n] ηi(ei)] into
P

A | ΓT ∧
^
i∈[n]
ηi(ei) ∧A∃

· P

A∃| ΓT ∧
^
i∈[n]
ηi(ei)

,
where A∃is a P∃-structure over ∆.
It decomposes the conditional sampling problem
of A into two subproblems—one is to sample A∃and the other sample the remaining
substructures conditional on A∃. The advantage of this decomposition is that the lat-
ter subproblem can be reduced into a sampling problem on a UFO2 sentence, since all
existentially-quantiﬁed formulas have been satisﬁed with A∃. The ﬁrst subproblem for
sampling A∃can be solved by a similar algorithm of DRSampler. In this algorithm, the
2-tables used to partition cells are now deﬁned over P∃, whose size is exponentially smaller
than the one in the original algorithm. As a result, the enumeration of 2-table conﬁgura-
tions in Algorithm 2 will be exponentially faster.
31


<!-- page 32 -->
• We cached Wn, the weight ⟨w, ¯w⟩(τ i) of all 1-types and the weight ⟨w, ¯w⟩(πi) of all 2-tables,
which are widely used in our WMS.
A.5.2
The Function ExSat(·, ·)
The pseudo-code of ExSat is presented in Algorithm 4.
Algorithm 4 ExSat(g, η)
1: Decompose g into {gηi,πj}i∈[Nc],j∈[Nb]
2: (β, τ) ←(η)
3: // Check the coherence of 2-tables
4: for i ∈[Nu] do
5:
for j ∈[Nb] do
6:
// τ(ηi) is the 1-type in ηi
7:
if πj is not coherent with τ(ηi) and τ and gηi,πj > 0 then
8:
return False
9:
end if
10:
end for
11: end for
12: // Check the satisfaction of existentially-quantiﬁed formulas
13: ∀j ∈[Nb], hπj ←P
i∈[Nc] gηi,πj
14: for Zk(x) ∈β do
15:
for j ∈[Nb] do
16:
if Rk(x, y) ∈πj and hπj > 0 then
17:
return True
18:
end if
19:
end for
20: end for
21: return False
A.5.3
A Lifted WMS for FO2 with Cardinality Constraints
We present the modiﬁed sampling algorithm for FO2 with cardinality constraints. The changes
from the original WMS for FO2 are highlighted by the blue lines.
32


<!-- page 33 -->
Algorithm 5 WMS(Γ, Υ, ∆, w, ¯w)
INPUT: An FO2 sentence Γ of the form (3), a set of cardinality constraints Υ of the
form (16), a domain ∆= {ei}i∈[n] of size n, a weighting (w, ¯w)
OUTPUT: A model µ of Γ ∧Υ over ∆
1: Construct Γ to ΓT by (4)
2: ∀i ∈[n], βi(x) ←V
k∈[m] Zk(x)
3: bΓ ←ΓT ∧V
i∈[n] βi(ei)
4: (τi)i∈[n] ←OneTypeSampler(bΓ, Υ, ∆, w, ¯w)
5: ∀i ∈[n], ηi ←(βi, τi)
6: µ ←DRSampler(ΓT , Υ, ∆, w, ¯w, (ηi)i∈[n])
7: return µ
Algorithm 6 OneTypeSampler(bΓ, Υ, ∆, w, ¯w)
1: W ←WFOMC(bΓ ∧Υ, ∆, w, ¯w)
2: Obtain the blocks Bβ1, Bβ2, . . . , Bβ2m from bΓ
3: for
 nβi
 
i∈[2m] ∈Prod
 
T|Bβ1|,Nu, . . . , T|Bβ2m |,Nu
 
do
4:
n ←L
i∈[2m] nβi
5:
Compute Wn,Υ
6:
W ′ ←Wn,Υ · Q2m
t=1
 |Bβt|
nβt
 
7:
if Uniform(0, 1) < W ′
W then
8:
n∗←n
9:
break
10:
else
11:
W ←W −W ′
12:
end if
13: end for
14: for i ∈[2m] do
15:
Fetch the cell conﬁguration n∗
βi in βi from n∗
16:
Randomly partition Bβi into {Cβi,τ j}j∈[Nu] according to n∗
βi
17:
for j ∈[Nu] do
18:
Assign the 1-type τ j to all elements in Cβi,τ j
19:
end for
20: end for
33


<!-- page 34 -->
Algorithm 7 DRSampler(ΓT , Υ, ∆, w, ¯w, (ηi)i∈[n])
1: if n = 1 then
2:
return ∅
3: end if
4: if only the block of type β = ⊤is nonempty then
5:
return a model µ sampled by a UFO2 WMS from ∀x∀y : ψ(x, y) ∧V
i∈[n] τi(ei) ∧Υ over
∆under (w, ¯w)
6: end if
7: Choose t ∈[n]; µ ←τt(et); ∆′ ←∆\ {et}
8: Get the cell conﬁguration n =
 nηi
 
i∈[Nc] of (ηi)i∈[n]\{t}
9: W ←Wn,Υ
10: ∀i ∈[Nc], Ti ←Tnηi,Nb
11: for
 gηi
 
i∈[Nc] ←Prod(Tnη1,Nb, . . . , TnηNc ,Nb) do
12:
g ←L
i∈[Nc] gηi
13:
if ExSat (g, ηt, Υ) then
14:
Get the new cell conﬁguration n′ w.r.t. g by (15)
15:
Get the new cardinality constraints Υ′ w.r.t. g by (17)
16:
Compute Wn′,Υ
17:
W ′ ←Wn′,Υ · ⟨w, ¯w⟩(τt) · Q
i∈[Nc]
 nηi
gηi
 
· wgηi
18:
if Uniform(0, 1) < W ′
W then
19:
g∗←g
20:
break
21:
else
22:
W ←W −W ′
23:
end if
24:
end if
25: end for
26: Obtain the cell partition {Cηi}i∈[Nc] from (ηi)i∈[n]\{t}
27: for i ∈[Nc] do
28:
Fetch the cardinality vector g∗
ηi of ηi from g∗
29:
Randomly partition the cell Cηi into
 
Gηi,πj
	
j∈[Nb] according to g∗
ηi
30:
for j ∈[Nb] do
31:
µ ←µ ∪
 
πj(et, e)
	
e∈Gηi,πj
32:
∀es ∈Gηi,πj, η′
s ←ηs|πj
33:
end for
34: end for
35: Obtain the reduced cardinality constraints Υ′ w.r.t. g∗by (17)
36: µ ←µ ∪DRSampler(ΓT , Υ′, ∆′, w, ¯w, (η′
i)i∈[n−1])
37: return µ
34


<!-- page 35 -->
References
[1] Y. Wang, T. Van Bremen, Y. Wang, and O. Kuˇzelka, “Domain-lifted sampling for universal
two-variable logic and extensions,” in Proceedings of the AAAI Conference on Artiﬁcial
Intelligence, vol. 36, no. 9, 2022, pp. 10 070–10 079.
[2] L. D. Raedt, K. Kersting, S. Natarajan, and D. Poole, “Statistical relational artiﬁcial intel-
ligence: Logic, probability, and computation,” Synthesis lectures on artiﬁcial intelligence
and machine learning, vol. 10, no. 2, pp. 1–189, 2016.
[3] G. Van den Broeck, N. Taghipour, W. Meert, J. Davis, and L. De Raedt, “Lifted proba-
bilistic inference by ﬁrst-order knowledge compilation,” in Proceedings of the Twenty-Second
international joint conference on Artiﬁcial Intelligence, 2011, pp. 2178–2185.
[4] G. Van den Broeck, W. Meert, and A. Darwiche, “Skolemization for weighted ﬁrst-order
model counting,” in Fourteenth International Conference on the Principles of Knowledge
Representation and Reasoning, 2014.
[5] A. J. Robinson and A. Voronkov, Handbook of automated reasoning.
Elsevier, 2001, vol. 1.
[6] A. Kuusisto and C. Lutz, “Weighted model counting beyond two-variable logic,” in Pro-
ceedings of the 33rd Annual ACM/IEEE Symposium on Logic in Computer Science, 2018,
pp. 619–628.
[7] O. Kuzelka, “Weighted ﬁrst-order model counting in the two-variable fragment with count-
ing quantiﬁers,” Journal of Artiﬁcial Intelligence Research, vol. 70, pp. 1281–1307, 2021.
[8] C. Cooper, M. Dyer, and C. Greenhill, “Sampling regular graphs and a peer-to-peer net-
work,” Combinatorics, Probability and Computing, vol. 16, no. 4, pp. 557–593, 2007.
[9] P. Gao and N. Wormald, “Uniform generation of random regular graphs,” in 2015 IEEE
56th Annual Symposium on Foundations of Computer Science. IEEE, 2015, pp. 1218–1230.
[10] D. Poole, “First-order probabilistic inference,” in IJCAI, vol. 3, 2003, pp. 985–991.
[11] M. Richardson and P. Domingos, “Markov logic networks,” Machine learning, vol. 62, no. 1,
pp. 107–136, 2006.
[12] P. Beame, G. Van den Broeck, E. Gribkoﬀ, and D. Suciu, “Symmetric weighted ﬁrst-order
model counting,” in Proceedings of the 34th ACM SIGMOD-SIGACT-SIGAI Symposium
on Principles of Database Systems, 2015, pp. 313–328.
[13] G. Van den Broeck, “On the completeness of ﬁrst-order knowledge compilation for lifted
probabilistic inference,” Advances in Neural Information Processing Systems, vol. 24, 2011.
[14] S. M. Kazemi, A. Kimmig, G. Van den Broeck, and D. Poole, “New liftable classes for ﬁrst-
order probabilistic inference,” Advances in Neural Information Processing Systems, vol. 29,
2016.
[15] S. M. Kazemi, A. Kimmig, G. V. d. Broeck, and D. Poole, “Domain recursion for lifted
inference with existential quantiﬁers,” arXiv preprint arXiv:1707.07763, 2017.
35


<!-- page 36 -->
[16] J. T´oth and O. Kuˇzelka, “Lifted inference with linear order axiom,” in Proceedings of the
AAAI Conference on Artiﬁcial Intelligence, 2023, to appear.
[17] C. P. Gomes, A. Sabharwal, and B. Selman, “Model counting,” in Handbook of satisﬁability.
IOS press, 2021, pp. 993–1014.
[18] S. Chakraborty, K. S. Meel, and M. Y. Vardi, “A scalable and nearly uniform generator
of sat witnesses,” in International Conference on Computer Aided Veriﬁcation.
Springer,
2013, pp. 608–623.
[19] S. Chakraborty, D. J. Fremont, K. S. Meel, S. A. Seshia, and M. Y. Vardi, “On parallel scal-
able uniform sat witness generation,” in International Conference on Tools and Algorithms
for the Construction and Analysis of Systems.
Springer, 2015, pp. 304–319.
[20] H. Guo, M. Jerrum, and J. Liu, “Uniform sampling through the lov´asz local lemma,”
Journal of the ACM (JACM), vol. 66, no. 3, pp. 1–31, 2019.
[21] K. He, X. Sun, and K. Wu, “Perfect sampling for (atomic) lov\’asz local lemma,” arXiv
preprint arXiv:2107.03932, 2021.
[22] W. Feng, K. He, and Y. Yin, “Sampling constraint satisfaction solutions in the local lemma
regime,” in Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Com-
puting, 2021, pp. 1565–1578.
[23] G. Van den Broeck and J. Davis, “Conditioning in ﬁrst-order knowledge compilation and
lifted probabilistic inference,” in Twenty-Sixth AAAI Conference on Artiﬁcial Intelligence,
2012.
[24] G. Van den Broeck and A. Darwiche, “On the complexity and approximation of binary
evidence in lifted inference,” Advances in Neural Information Processing Systems, vol. 26,
2013.
[25] D. Scott, “A decision method for validity of sentences in two variables,” Journal of Symbolic
Logic, vol. 27, no. 377, p. 74, 1962.
[26] E. Gr¨adel, P. G. Kolaitis, and M. Y. Vardi, “On the decision problem for two-variable
ﬁrst-order logic,” Bulletin of symbolic logic, vol. 3, no. 1, pp. 53–69, 1997.
[27] E. Gradel, M. Otto, and E. Rosen, “Two-variable logic with counting is decidable,” in
Proceedings of Twelfth Annual IEEE Symposium on Logic in Computer Science.
IEEE,
1997, pp. 306–317.
[28] M. Soos, S. Gocht, and K. S. Meel, “Tinted, detached, and lazy cnf-xor solving and its
applications to counting and sampling,” in International Conference on Computer Aided
Veriﬁcation.
Springer, 2020, pp. 463–484.
[29] S. Chakraborty, D. Fried, K. S. Meel, and M. Y. Vardi, “From weighted to unweighted model
counting,” in Twenty-Fourth International Joint Conference on Artiﬁcial Intelligence, 2015.
[30] T. Van Bremen and O. Kuˇzelka, “Lifted inference with tree axioms,” in Proceedings of
the International Conference on Principles of Knowledge Representation and Reasoning,
vol. 18, no. 1, 2021, pp. 599–608.
36


<!-- page 37 -->
[31] M. Jaeger, “Lower complexity bounds for lifted inference,” Theory and Practice of Logic
Programming, vol. 15, no. 2, pp. 246–263, 2015.
[32] F. J. Massey Jr, “The kolmogorov-smirnov test for goodness of ﬁt,” Journal of the American
statistical Association, vol. 46, no. 253, pp. 68–78, 1951.
[33] M. Naaman, “On the tight constant in the multivariate dvoretzky–kiefer–wolfowitz inequal-
ity,” Statistics & Probability Letters, vol. 173, p. 109088, 2021.
37