<!-- page 1 -->
The Hardest Explicit Construction
Oliver Korten ∗
May 2021
Abstract
We investigate the complexity of explicit construction problems, where the goal is to produce
a particular object of size n possessing some pseudorandom property in time polynomial in n.
We give overwhelming evidence that APEPP, deﬁned originally by Kleinberg et al. [22], is the
natural complexity class associated with explicit constructions of objects whose existence follows
from the probabilistic method, by placing a variety of such construction problems in this class.
We then demonstrate that a result of Jeˇr´abek [18] on provability in Bounded Arithmetic, when
reinterpreted as a reduction between search problems, shows that constructing a truth table
of high circuit complexity is complete for APEPP under PNP reductions.
This illustrates
that Shannon’s classical proof of the existence of hard boolean functions is in fact a universal
probabilistic existence argument: derandomizing his proof implies a generic derandomization
of the probabilistic method.
As a corollary, we prove that EXPNP contains a language of
circuit complexity 2nΩ(1) if and only if it contains a language of circuit complexity 2n
2n. Finally,
for several of the problems shown to lie in APEPP, we demonstrate direct polynomial time
reductions to the explicit construction of hard truth tables.
∗Department of Computer Science, Columbia University. Email: oliver.korten@columbia.edu
1
arXiv:2106.00875v3  [cs.CC]  10 Feb 2022


<!-- page 2 -->
1
Introduction
Explicit construction — the task of replacing a nonconstructive argument for the existence of a
certain type of object with a deterministic algorithm that outputs one — is an important genre
of computational problems, whose history is intertwined with the most fundamental questions in
complexity and derandomization. The primary method of existence argument for these problems
is to show that a random object has a desired property with high probability. This technique,
initiated by Erd¨os [10] and since dubbed the “probabilistic method,” has proven immensely useful
across disparate subﬁelds of combinatorics and computer science. Indeed, the probabilistic method
is currently our sole source of certainty that there exist hard Boolean functions, pseudorandom
number generators, rigid matrices, and optimal randomness extractors, among a variety of other
combinatorial objects.
Explicit construction problems can be phrased, in complexity terms, as sparse search problems:
given the input 1n, output some object of size n satisfying a certain property. In the interesting
case, such problems are also total: we have a reason to believe that for all n, at least one object with
this property exists. In contrast to the fundamental importance of explicit constructions, there has
been surprisingly little work attempting to systematically study their complexity. This gap was
pointed out previously by Santhanam [32], who studied the complexity of explicit construction
problems from the following perspective: we have some property Π which is promised to hold for
almost all strings of length n. Based on the complexity of testing the property Π, what can be
said about the complexity of producing an n-bit string with property Π? Though some interesting
reductions can be shown in this framework, Santhanam notes that this approach does not seem to
yield robust complexity classes with complete explicit construction problems.
This issue is familiar in the study of the class TFNP: when we have only a promise that a
search problem is total, it is seemingly impossible to reduce it to a problem of similar complexity
which has a syntactic guarantee of totality.
This led to the study, initiated by Papadimitriou
[29], of characterizing total search problems based on the combinatorial lemma which guarantees
the existence of a solution.
In recent work of Kleinberg et al.
[22], this method was used to
analyse several total search problems in the polynomial hierarchy beyond NP. One class they
deﬁne is APEPP, which consists of the ΣP
2 total search problems whose totality follows from the
“Abundant Empty Pigeonhole Principle,” which tells us that a function f : {0, 1}n →{0, 1}n+1
cannot be surjective. In this paper, we show that APEPP is the natural syntactic class into which
we can place a vast range of explicit construction problems where a solution is guaranteed by the
probabilistic method.
Given that APEPP is a syntactic class, it is natural to ask whether some explicit construction
problem is complete for it. As it turns out, the answer is positive: constructing a truth table
of length 2n with circuit complexity 2ϵn is in fact complete for APEPP under PNP reductions.
Perhaps surprisingly, this important fact had been known for many years in the universe of Bounded
Arithmetic, essentially proved in Emil Jeˇr´abek’s PhD thesis in 2004. Here Jeˇr´abek shows that the
theorem asserting the empty pigeonhole principle is equivalent, in a particular theory of Bounded
Arithmetic, to the theorem asserting the existence of hard boolean functions. Although his result
is phrased in terms of logical expressibility, we show that when translated to language of search
problems his techniques give a PNP reduction from any problem in APEPP to the problem of
constructing a hard truth table. In Section 4 we give a self-contained proof of this, and generalize
the reduction to hold for arbitrary classes of circuits equipped with oracle gates. Combined with
our results placing a wide range of explicit construction problems in APEPP, this shows that in a
concrete sense, constructing a hard truth table is a universal explicit construction problem. We give
further credence to this claim by showing in addition that several well known explicit construction
2


<!-- page 3 -->
problems in APEPP, including the explicit construction of rigid matrices, can be directly reduced
to the problem of constructing a hard truth table via polynomial time reductions (as opposed to
PNP reductions).
1.1
Our Contributions
We investigate the complexity class APEPP introduced in [22], deﬁned by the following complete
problem Empty: given a circuit C : {0, 1}n →{0, 1}m with m > n, ﬁnd an m-bit string outside
the range of C. In Section 3 we give overwhelming evidence that APEPP is the natural class
associated with explicit constructions from the probabilistic method, by placing a wide range of
well studied problems in this class. In particular, we show that the explicit construction problems
associated with the following objects lie in APEPP:
• Truth tables of length 2n with circuit complexity 2n
2n (Theorem 1)
• Pseudorandom generators (Theorem 3)
• Strongly explicit two-source randomness extractors with 1 bit output for min-entropy log n +
O(log(1/ϵ)), and thus strongly explicit O(log n)-Ramsey graphs in both the bipartite and
non-bipartite case (Theorem 4)
• Matrices with high rigidity over any ﬁnite ﬁeld (Theorem 5)
• Strings of time-bounded Kolmogorov complexity n −1 relative to any ﬁxed polynomial time
bound and any ﬁxed Turing machine (Theorem 6)
• Communication problems outside of PSPACECC (Theorem 13)
• Hard data structure problems in the-bit probe model (Theorem 14)
Since the work of Impagliazzo and Wigderson [17] implies that constructing pseudorandom
generators reduces to constructing hard truth tables, APEPP constructions of PRGs follow im-
mediately from APEPP constructions of hard truth tables. However, we provide a self-contained
and simple proof that PRG construction can be reduced to Empty, without requiring the more
involved techniques of Nisan, Wigderson, and Impagliazzo [27][17]. Together with the result in the
following section that constructing hard truth tables is complete for APEPP under PNP reduc-
tions, this gives an alternative and signiﬁcantly simpliﬁed proof that worst-case-hard truth tables
can be used to derandomize algorithms (although it proves a weaker result, that this derandomiza-
tion can be accomplished with an NP oracle).
In Section 4 we show that constructing a truth table of length 2n with circuit complexity 2ϵn is
complete for APEPP under PNP reductions (for any ﬁxed 0 < ϵ < 1). As discussed earlier, the
core argument behind this result was proven by Jeˇr´abek in [18], where he shows that the theorem
asserting the existence of hard boolean functions is equivalent to the theorem asserting the empty
pigeonhole principle in a certain fragment of Bounded Arithmetic. We show that, when viewed
through the lens of explicit construction problems, this technique yields a reduction from Empty to
the explicit construction of hard truth tables. We also generalize this reduction to arbitrary oracle
circuits, which allows us to prove the following more general statement: constructing a truth table
which requires large ΣP
i -oracle circuits is complete for APEPPΣP
i under ∆P
i+2 reductions (the com-
plete problem for APEPPΣP
i is the variant of Empty where the input circuit can have ΣP
i -oracle
gates). By recasting and generalizing Jeˇr´abek’s theorem in the context of explicit construction
problems, we are able to derive several novel results. First and foremost, we conclude that there is
3


<!-- page 4 -->
a PNP construction of hard truth tables if and only if there is a PNP algorithm for every problem
in APEPP, and so in particular such a construction of hard truth tables would automatically
imply PNP constructions for each of the well-studied problems discussed in Section 3. This tells us
that constructing hard truth tables is, in a deﬁnite sense, a universal explicit construction problem.
Since the existence of a PNP construction of hard truth tables is equivalent to the existence of
a language in ENP with circuit complexity 2Ω(n), this completeness result actually gives an exact
algorithmic characterization of proving 2Ω(n) circuit lower bounds for ENP:
Theorem (Theorem 10). There is a PNP algorithm for Empty if and only if ENP contains a
language of circuit complexity 2Ω(n).
As a corollary we are able to derive the following:
Theorem (Corollaries 2 and 3). ENP (resp. EXPNP) contains a language of circuit complexity
2Ω(n) (resp. 2nΩ(1)) if and only if ENP (resp. EXPNP) contains a language of circuit complexity
2n
2n.
Unpacking the proof of the above corollaries reveals an eﬃcient algorithm to “extract hardness”
from truth tables using an oracle for circuit minimization, a prospect previously considered in [5]:
Theorem (Theorem 11). There is a polynomial time algorithm using a circuit minimization oracle
(or more generally an NP oracle) which, given a truth table x of length M and circuit complexity
s, outputs a truth table y of length N = Ω(
q
s
log M ) and circuit complexity Ω(
N
log N ).
We then argue, based on an observation of Williams [39], that improving this construction in
its current form to extract (
s
log M )
1
2 +ϵ bits of hardness would require a breakthrough for 3SUM.
Finally, in Section 5 we consider P (as opposed to PNP) reductions from particular explicit
construction problems to the problem of constructing hard truth tables.
We show that in the
case of rigidity, bit probe lower bounds, and certain communication complexity lower bounds, such
reductions exist.
These reductions take the following form: we show that the failure of an n-
bit string x to satisfy certain pseudorandom properties implies a smaller than worst case circuit
computing x. This then implies that any n-bit string of suﬃciently high circuit complexity will nec-
essarily possess a variety of pseudorandom properties, including high rigidity, high space-bounded
communication complexity, and high bit-probe complexity. We also make note of an interesting
dichotomy (Theorem 15), which tells us that any explicit construction problem in APEPP is either
APEPP-complete under PNP reductions, or solvable in subexponential time with an NP oracle
(for inﬁnitely many input lengths).
Another concrete takeaway from this work is that we demonstrate, for several well-studied
problems, the weakest known assumptions necessary to obtain explicit constructions of a certain
type (polynomial time constructions in some cases and PNP constructions in others). Perhaps the
most interesting application of this is rigidity, as the complexity of rigid matrix construction has
been studied extensively in both the P and PNP regimes. We obtain the following conditional
constructions of rigid matrices:
Theorem (Theorems 5 and 10). If ENP contains a language of circuit complexity 2Ω(n), then for
any prime power q there is a PNP construction of an n × n matrix over Fq which is Ω(n2)-far (in
hamming distance) from any rank-Ω(n) matrix.
Theorem (Theorem 12). If E contains a language of circuit complexity Ω( 2n
n ), then there is a
polynomial time construction of an n × n matrix over F2 which is Ω(n2)-far from any rank-Ω(n)
matrix.
4


<!-- page 5 -->
In both cases, the rigidity parameters in the conclusion would be suﬃcient to carry out Valiant’s
lower bound program [38]. The weakest hardness assumptions previously known to yield construc-
tions with even remotely similar parameters (in either the PNP or P regimes) require a lower bound
against nondeterministic circuits [25].
1.2
Related Work
A large body of work on the hardness/randomness connection, starting with that of Nisan and
Wigderson [27], has exhibited the usefulness of explicit constructions of hard truth tables. The re-
sults of Impagliazzo and Wigderson [17] give, in particular, a reduction from explicit constructions
of hard truth tables to explicit constructions of pseudorandom generators that fool polynomial
size circuits. As noted by Santhanam [32], this immediately implies that for any “dense” property
Π recognizable in P (dense meaning the fraction of n-bit strings holding this property is at least
1/poly(n)), an eﬃcient construction of a hard truth table immediately implies an eﬃcient construc-
tion of an n-bit string with property Π. But many properties of interest such as Rigidity (or any
of the other properties studied in this work) are only known to be recognizable in the larger class
NP. Under the stronger assumption that we can construct truth tables hard for certain classes of
nondeterministic circuits, constructions for all dense NP properties are known to follow as well
[23] [25], so in particular PNP constructions for every problem in APEPP would follow. However,
constructing truth tables that are hard for nondeterministic circuits appears strictly harder than
constructing truth tables hard for standard circuits, and in particular does not seem to be contained
in APEPP, so although this yields an explicit construction problem which is hard for APEPP,
it does not appear to be complete. In contrast, we show here that constructing a truth table which
is hard for standard circuits is both contained in and hard for APEPP, thus showing that a PNP
construction of a hard truth table is possible if and only if such a construction is possible for every
problem in APEPP.
For several of the problems we study, a long line of work has gone into improving state-of-the-art
explicit constructions. We give a brief overview here of some recent work on rigid matrices and
extractors. Rigidity was ﬁrst introduced by Valiant [38], who showed that any matrix which is
n1+ϵ-far from a rank-δn matrix for some ϵ, δ > 0 cannot be computed by linear size, logarithmic
depth arithmetic circuits. Since then it has been a notorious open problem to provide examples of
an explicit matrix family with rigidity parameters anywhere close to this. A recent breakthrough
was achieved in [1], and improved by [4], which gives PNP constructions of matrices that are
Ω(n2)-far from any rank-2log n/Ω(log log n) matrix, for inﬁnitely many values of n. However, such a
construction is still not suﬃcient to carry out Valiant’s arithmetic circuit lower bound program.
A conditional result of [8] implies that PNP constructions of certain rigid matrices are possible
assuming explicit constructions exist for certain hard data structure problems in the group model.
In terms of polynomial time constructions, the best known construction yields a matrix which is
N2
ρ log N
ρ -far from any ρ-rigid matrix, for any parameter ρ [35][11].
The case of Ramsey graphs and extractors is slightly more complicated. There are two common
deﬁnitions of the explicit construction problems corresponding to these objects. The ﬁrst is often
referred to as the “weakly-explicit” version, where we must output the adjacency matrix (in the
case of Ramsey graphs) or truth table (in the case of extractors) in time polynomial in the size
of the truth table/matrix. The second version, referred to as the “strongly-explicit” version, is
to output a succinct circuit which computes the adjacency relation or extractor function. Clearly
the strongly-explicit case is harder, but in both cases, there is a signiﬁcant gap between what is
achievable by explicit methods and what can be proven possible by the probabilistic method. We
will focus on the strongly explicit case in this work, and in the case of extractors we will focus on
5


<!-- page 6 -->
two-source extractors with one bit of output, which remains the most challenging current frontier
[6]. The state of the art constructions for both two-source extractors and Ramsey graphs are due
to Li [24]. He demonstrates a two-source extractor for min-entropy O(log n log log n), and hence
an n-vertex graph which is (log n)O(log log log n)-Ramsey. Our results show that strongly explicit
extractors for min-entropy log n + O(log(1/ϵ)), and thus n-vertex O(log n)-Ramsey graphs, can be
constructed in APEPP. In both cases these parameters are known to be the best possible. For a
comprehensive survey of recent progress on extractors see [6].
Another line of work in the area of explicit constructions investigates the possibility of pseudo-
deterministic constructions of certain objects. Here, the construction algorithm is allowed to use
randomness, but must output the same object on most computation paths. Originally introduced
in [12], this paradigm was recently applied in [28] to the construction of prime numbers, where a
subexponential time pseudodeterministic construction which works for inﬁnitely many input lengths
is given.
1.3
Proof Sketch of Main Theorem
We give here an informal overview of the proof that Empty can be solved in polynomial time given
access to a hard truth table and an NP oracle. At the core of this proof is a familiar construction
in the theory of computing which dates back to the 1980’s, namely the pseudorandom function
generator of Goldreich, Goldwasser, and Micali [13]. Note that in the following, we will refer to the
“circuit complexity of an n-bit string x” to mean the size of the smallest circuit computing xi given
i in binary; this is well-deﬁned even when n is not a power of 2, as we shall formalize Section 31.
Consider the special case of Empty where our input is a circuit C : {0, 1}n →{0, 1}2n which
exactly doubles its input size. For a moment let us forget our primary goal of ﬁnding a 2n-bit string
outside C’s range, and instead consider C as a cryptographic pseudorandom generator which we are
attempting to break. Since C is a function which extends its input size by a positive number of bits,
it is indeed of the same syntactic form as a cryptographic PRG, so this viewpoint is well-deﬁned.
In [13], Goldreich, Goldwasser and Micali give a procedure2 which, for any ﬁxed 0 < ϵ < 1, takes
C and produces in polynomial time a new circuit C∗: {0, 1}n →{0, 1}m for some m = poly(n),
which satisﬁes the following two properties:
(1) Every string in the range of C∗has circuit complexity at most mϵ.
(2) Given a statistical test breaking C∗, we can construct a statistical test of similar complexity
breaking C.
The construction of C∗is in fact quite simple: for an appropriate choice of k, we recursively apply
C to an n-bit input for k iterations as follows: ﬁrst apply C to an n-bit string to get 2 n-bit strings,
then apply it again to each of those to get 4, and continue k times until we obtain 2k n-bit strings.
A key observation made by Razborov and Rudich [31] is that condition (1) automatically implies
a particular statistical test which breaks C∗, namely the test which accepts precisely those m-bit
strings with circuit complexity exceeding mϵ. But by property (2), C∗inherets the security of
C, which is an arbitrary candidate PRG. This means that determining if an m-bit string has
circuits of size mϵ is in fact a universal test for randomness, capable of simultaneously breaking all
pseudorandom generators.
1See Deﬁnition 4
2[13] and [31] apply the construction described here in a diﬀerent parameter regime, so our statement of the result
diﬀers slightly from its original presentation. The version described here has been noted subsequently in the literature
on MCSP, see for example [33].
6


<!-- page 7 -->
Recall now our original goal for C, which was to ﬁnd a 2n-bit string outside its range. Property
(1) of C∗implies that an explicit construction of a length-m truth table of circuit complexity mϵ
would immediately yield an explicit m-bit string outside the range of C∗. In Section 4, we show
that C∗obeys the following third property:
(3) Given a string outside the range of C∗, we can ﬁnd a string outside the range of C using a
polynomial number of calls to an NP oracle.
The analogue of statement (3) in the context of Bounded Arithmetic was ﬁrst shown by Jeˇr´abek
[18], and a quite similar argument appears even earlier in the work of Paris, Wilkie, and Woods
[30]. Combining properties (1) and (3), we get the desired result: any m-bit string of complexity
mϵ must lie outside the range of C∗, so using such a string together with an NP oracle we can
solve our original instance of Empty.
To summarize, the construction C∗of Goldreich, Goldwasser and Micali shows that the property
of requiring large circuits is a universal pseudorandom property of strings in two concrete senses:
(a) (Original analysis of [13] and [31]) A test determining whether a string requires large circuits
can be eﬃciently boostrapped into a test distinguishing any pseudorandom distribution from
the uniform distribution.
(b) (This work together with [18]) An explicit example of a string requiring large circuits can be
used to generate an explicit example of a string outside the range of any eﬃciently computable
map C : {0, 1}n →{0, 1}2n (in fact any C : {0, 1}n →{0, 1}n+1 as shown in Section 4), and
in particular can be used to construct explicit examples of strings possessing each of the
fundamental pseudorandom properties examined in Section 3.
2
Deﬁnitions
Following [22], we deﬁne the set of total functions in ΣP
2 , denoted TFΣP
2 , as follows:
Deﬁnition 1. A relation R(x, y) is in TFΣP
2 if there exists a polynomial p(n) such that the
following conditions hold:
1. For every x, there exists a y such that |y| ≤p(|x|) and R(x, y) holds
2. There is a polynomial time Turing machine M such that
R(x, y) ⇐⇒∀z ∈{0, 1}p(|x|)M(x, y, z) accepts
The search problem associated with such a relation is: “given x, ﬁnd some y such that R(x, y)
holds.” For the majority of this paper, we will be concerned primarily with sparse TFΣP
2 search
problems, where the only relevant part of the input is its length. We can thus deﬁne the following
“sparse” subclass of TFΣP
2 :
Deﬁnition 2. A relation R(x, y) is in STFΣP
2 if R ∈TFΣP
2 and for any x1, x2 such that |x1| =
|x2|, we have that for all y, R(x1, y) ⇔R(x2, y).
Since the length of x fully determines the set of solutions, the relevant search problem here is:
“given 1n, ﬁnd some y such that R(1n, y) holds.” All explicit construction problems considered in
Section 3 will be in STFΣP
2 (with the exception of Complexity which we brieﬂy mention as it
was studied previously in [22]).
We now deﬁne the search problem Empty, which will be the primary subject of this work:
7


<!-- page 8 -->
Deﬁnition 3. Empty is the following search problem: given a boolean circuit C with n input wires
and m output wires where m > n, ﬁnd an m-bit string outside the range of C.
This problem is total due to the basic lemma, referred to in [22] as the “Empty Pigeonhole
Principle” and in the ﬁeld of Bounded Arithmetic as the “Dual Pigeonhole Principle [18],” which
tells us that a map from a smaller set onto a larger one cannot be surjective. Since verifying a
solution y consists of determining that for all x, C(x) ̸= y, we have:
Observation 1. Empty ∈TFΣP
2
Since for any instance of Empty the number of output bits m is at least n + 1, a random m-bit
string will be a solution with probability at least 1
2. Since verifying a solution can be accomplished
with one call to an NP oracle, this implies the following inclusion:
Observation 2. Empty ∈FZPPNP
As mentioned in the introduction, this fact tells us that suﬃciently strong pseudorandom gener-
ators capable of fooling nondeterministic circuits such as those in [23] would suﬃce to derandomize
the above inclusion and yield a PNP algorithm for Empty. In Section 4, we will show that this
derandomization can be accomplished under a signiﬁcantly weaker assumption, using a reduction
of a very diﬀerent form then the hardness-based pseudorandom generators of [27], [17], and [23].
We can now deﬁne the class APEPP, which is simply the class of search problems polynomial-
time reducible to Empty.
This class was originally deﬁned in [22], and is an abbreviation for
“Abundant Polynomial Empty Pigeonhole Principle.” The term “Abundant” was used to distin-
guish this from the larger class PEPP also studied in [22]. The complete problem for PEPP is to
ﬁnd a string outside the range of a map C : {0, 1}n \ {0n} →{0, 1}n, which appears signiﬁcantly
more diﬃcult (it is at least as hard as NP [22]). The distinction between APEPP and PEPP
also appears in the Bounded Arithmetic literature, where the principle corresponding to APEPP
is referred to as the “Dual weak Pigeonhole Principle,” while the principle corresponding to PEPP
is referred to simply as the “Dual Pigeonhole Principle.” We will be concerned only with the abun-
dant/weak principle in this work. It should be noted that we employ a slight change of notation
from [22] for the sake of simplicity: we use Empty to refer to the search problem associated with
the weak pigeonhole principle, while in [22] Empty refers to the search problem associated with
the full pigeonhole principle.
Inﬁnitely-often vs. almost-everywhere circuit lower bounds:
As a ﬁnal point of clariﬁca-
tion, whenever we make the statement “L requires circuits of size s(n)” for some language L and
size bound s, we mean that circuits of size s(n) are required to compute L on length n inputs for
all but ﬁnitely many n. This is in contrast to the statement “L /∈SIZE(s(n)),” which means the
circuit size lower bound holds for inﬁnitely many input lengths. All circuit lower bounds referred
to in this work will be of the ﬁrst kind.
3
Explicit Constructions in APEPP
In this section, we show that a variety of well-studied explicit construction problems can be reduced
in polynomial time to Empty. Each proof follows roughly the following format: there is some
property of interest Π, and our goal is to construct an n-bit string which holds this property. For
each such Π we consider, whenever an n-bit string x fails to have this property, it indicates that x
is somehow more “structured” than a random n-bit string, and this structure allows us to specify x
8


<!-- page 9 -->
using fewer then n bits. We then actualize this argument in the form of an eﬃciently computable
map C : {0, 1}k →{0, 1}n with k < n, such that any string not having property Π is in the range
of C. This immediately implies that any n-bit string outside the range of C must hold property
Π, and thus any solution to the instance of Empty deﬁned by C will be a solution to our explicit
construction problem. For many of the proofs, we will only show that the reduction is valid for n
suﬃciently large; clearly this is suﬃcient, since explicit constructions can be done by brute force
for ﬁxed input lengths.
A useful coding lemma:
In the proofs to come, it will be helpful to utilize succinct and eﬃciently
computable encodings of low-weight strings (the “weight” of binary string is the number of 1 bits
it contains). We start with the following folklore result, reproduced in [14], which gives an optimal
encoding of weight-k n-bit strings:
Lemma 1. [14] For any k ≤n, there exists a map Φ : {0, 1}log(n
k) →{0, 1}n computable in poly(n)
time such that any n-bit string of weight k is in the range of Φ.
As a useful corollary we get the following:
Lemma 2. For any 0 < ϵ < 1
2, there exists a map Φ : {0, 1}n−ϵ2n+log n →{0, 1}n computable in
poly(n) time such that any n-bit string of weight at most n
2 −ϵn is in the range of Φ.
Proof. By the previous lemma, we are able to eﬃciently encode n-bit strings of weight exactly k for
any k ≤n
2 −ϵn using at most log
 n
n
2 −ϵn
 
bits. We can upper bound log
 n
n
2 −ϵn
 
as follows. Letting
X denote the sum of n independent unbiased variables over {0, 1}, we have:
 
n
n
2 −ϵn
 
≤2nPr[X ≤n
2 −ϵn]
Using a standard Chernoﬀbound we have that for any δ ∈(0, 1):
Pr[X ≤(1 −δ)n
2 ] ≤exp(−nδ2/4) ≤2−nδ2/4
So setting δ = 2ϵ we get log
 n
n
2 −ϵn
 
≤n −ϵ2n.
For a string of weight at most
n
2 −ϵn, we can append an additional log n bits specifying the
weight k of our string, together with the n −ϵ2n bits needed to specify a string of weight exactly
k, to get the desired result.
3.1
Hard Truth Tables
Deﬁnition 4. Given a string x of length N, we say that x is computed by a circuit of size s if there
is a boolean circuit C of fan-in 2 over the basis ⟨∧, ∨, ¬⟩with ⌈log N⌉inputs and s gates, such that
C(i) = xi for all 1 ≤i ≤|x|. If N is not a power of 2, we put no restriction on the value of C(i)
for i > |x|.
Deﬁnition 5. Hard Truth Table is the following search problem: given 1N, output a string x
of length N such that x is not computed by any circuit of size at most
N
2 log N .
In the typical case where N = 2n for some n, this is equivalent to ﬁnding a truth table for an
n-input boolean function requiring circuits of size 2n
2n, which is within a 2 + o(1) factor of the worst
case circuit complexity for any n-input boolean function.
9


<!-- page 10 -->
Theorem 1. Hard Truth Table reduces in polynomial time to Empty.
Proof. This proof follows Shannon’s classical argument for the existence of functions of high circuit
complexity [34]. We construct an instance of Empty in the form of a circuit Φ which maps an
encoding of a circuit to its corresponding truth table. Φ interprets its input as a circuit on ⌈log N⌉
bits (using an encoding of circuits to be described below), tests its value on every possible input to
generate a 2⌈log N⌉bit truth table, and then truncates this truth table to be of length exactly N.
We now describe the encoding of circuits used by Φ, which will guarantee that any N-bit string
x with a circuit of size
N
2 log N is in the range of Φ. Given a circuit of size s on ⌈log N⌉inputs
computing x, for each of its s gates we can use 2 bits to encode whether it is an ∧, ∨, or ¬ gate, and
an additional 2 log s bits to specify its inputs. We can then use an additional log s bits to specify
which gate is the terminal output gate. Overall this requires 2s log s + O(s) bits. It is clear that
from such an encoding, Φ can eﬃciently decode the represented circuit and test it on all possible
input values. For s ≤
N
2 log N , we have:
2s log s + O(s) ≤
N
log N log
 
N
2 log N
 
+ O
 
N
log N
 
= N −Ω
 N log log N
log N
 
+ O
 
N
log N
 
which is strictly less then N for N suﬃciently large. So Φ is indeed a valid instance of Empty,
and any string outside the range of Φ is a solution to Hard Truth Table. It is also clear from
the above description that Φ can be constructed in poly(N) time.
A related problem was studied in [22], referred to there as “Complexity.” Rather then an
explicit construction problem with sparse input, in this problem you are given as input one truth
table x of length N, and asked to produce another truth table y such that y requires large circuits,
even with access to x-oracle gates. More formally:
Deﬁnition 6. The problem Complexity is deﬁned as follows: given an N-bit string x, ﬁnd
another N-bit string y such that y requires x-oracle circuits of size
N
log2 N .
Note that the “x oracle” is not the typical deﬁnition of an oracle gate that can solve arbitrarily
sized instances of a ﬁxed language, but rather an oracle for a ﬁxed boolean function on log N
variables. In [22] the following is shown:
Theorem 2. Complexity reduces in polynomial time to Empty.
It should be noted that Complexity bears resemblance to a problem studied by Ilango [16],
termed the “Minimum Oracle Circuit Size Problem,” or “MOCSP.” In this problem, the input
consists of two truth tables x and y and a size parameter s, and the goal is to determine if y
has x-oracle circuits of size at most s. Ilango demonstrates that MOCSP is NP-complete under
randomized reductions.
3.2
Pseudorandom Generators
Deﬁnition 7. We will say that a sequence R = (x1, . . . , xm) of n-bit strings is a pseudorandom
generator if, for all n-input circuits of size n:
|Prx∼R[C(x) = 1] −Pry∼{0,1}n[C(y) = 1]| ≤1/n
Standard applications of the probabilistic method show that such pseudorandom generators
exist of size polynomial in n. Thus we can deﬁne the following total search problem:
10


<!-- page 11 -->
Deﬁnition 8. PRG is the following search problem: given 1n, output a pseudorandom generator
R = (x1, . . . , xm), xi ∈{0, 1}n.
A polynomial time algorithm for PRG would suﬃce to derandomize BPP [27]. We now show
how to formalize the argument for the totality of PRG using the empty pigeonhole principle. In
particular, we show that a PRG of size n6 can be constructed in APEPP.
As noted in the introduction, the results of Impagliazzo and Wigderson [17] imply that PRG
reduces directly to Hard Truth Table, so a reduction of PRG to Empty follows from Theorem 1.
However, we provide here a much simpler direct proof that PRG reduces to Empty, relying only
on Yao’s next bit predictor lemma, and neither the nearly disjoint subsets construction of Nisan
and Wigderson [27] nor the rather involved worst-case to average-case reductions of Impagliazzo
and Wigderson [17]. Together with our completeness result in Section 4, this gives an alternative,
self-contained proof that worst-case-hard truth tables can be used to construct pseudorandom
generators (although it yields a weaker result, as our derandomization will require an NP oracle).
Theorem 3. PRG reduces in polynomial time to EMPTY
Proof. We start by constructing the following circuit Φ. Φ interprets its input as representing a
list R−= (x1, . . . , xn6) of n6 strings of length n −1, a circuit D of size cn (for a ﬁxed universal
constant c to be deﬁned later) with n −1 input bits and one output bit, a log n-bit index i ∈[n],
and a string S encoding an n6-bit string with weight at most n6
2 −n4. Given these, Φ feeds each
n −1-bit string xj ∈R−through D, then inserts the output as an extra bit at the ith position of
xj to obtain an n-bit string x∗
j. In the ﬁnal step, Φ decodes the n6-bit string represented by S,
and ﬂips ith bit of x∗
j if and only if the jth position of S has a 1, to obtain x′
j. Φ then outputs
R = (x′
1, . . . , x′
n6).
It is clear that Φ can be implemented as a circuit of size polynomial in n, and further that Φ
can be constructed in polynomial time given 1n. We now show there are fewer inputs then outputs.
Note that the input size is equal to the number of bits needed to specify R−, D, i, S, which is
n7 −n6 + ˜O(n) + log n + bits(S), where bits(S) is the number of bits needed to specify S. Since S
an n6-bit string with weight at most n6
2 −n4 = n6(1
2 −1
n2 ), we can apply Lemma 2 with ϵ =
1
n2 to
give an encoding for S using n6(1 −1
n4 ) + log(n6) = n6 −n2 + 6 log n bits. Thus the overall number
of bits of input is at most n7 −n6 + ˜O(n)+log n+n6 −n2 +6 log n = n7 + ˜O(n)+7 log n−n2 which
is strictly less then the n7 bits of output (for suﬃciently large n). So this is indeed a polynomial
time reduction to a valid instance of Empty. It remains to show that any string outside the range
of Φ is a pseudorandom generator.
Let R be a sequence of n-bit strings of size n6 which is not a pseudorandom generator. So there
exists a circuit C of size n such that:
|Prx∼R[C(x) = 1] −Pry∼{0,1}n[C(y) = 1]| > 1/n
For x ∈{0, 1}n, i ∈[n], let x−i be the n −1-bit string obtained by deleting the ith bit of x, and
let xi denote the ith bit of x. By Yao’s next bit predictor lemma [40] [37], the previous inequality
implies the existence of an index i ∈[n] and a circuit D : {0, 1}n−1 →{0, 1} of size cn for some
ﬁxed universal constant c, such that
Prx∼R[D(x−i) = xi] > 1
2 + 1
n2
Since R has size n6, this implies that D correctly guesses the ith bit of x ∈R from the other n −1
bits for at least n6(1
2 + 1
n2 ) = n6
2 + n4 of the elements of R. So if we let S be the n6-bit string with
11


<!-- page 12 -->
a 1 at the indices where D guesses the wrong value of xi, we see that the weight of S is at most
n6
2 −n4 as required. Taking R−= (x−i
1 , . . . , x−i
n6), we have that from R−, D, i, S we can eﬃciently
deduce R.
So overall, have established that for any R which is not a pseudorandom generator, there exists
some R−, D, i, S such that our circuit Φ outputs R on input R−, D, i, S. Thus, any string outside
the range of Φ is a pseudorandom generator.
3.3
Strongly Explicit Randomness Extractors and Ramsey Graphs
A (k, ϵ) two-source extractor with one bit of output is a function f : {0, 1}n ×{0, 1}n →{0, 1} such
that for any pair of distributions X, Y on {0, 1}n of min-entropy at least k, the value of f(xy) for a
random x, y in the product distribution of X, Y is ϵ-close to an unbiased coin ﬂip. By a well-known
simpliﬁcation of [7], to show that a function f is a (k, ϵ) extractor, it suﬃces to show that it satisﬁes
the above condition for every pair of “ﬂat” k-sources X, Y , which are uniform distributions over
subsets of {0, 1}n of size 2k. We will thus use the following deﬁnition of two-source extractors to
deﬁne our explicit construction problem:
Deﬁnition 9. We say that a function f : {0, 1}n × {0, 1}n →{0, 1} is a (k, ϵ) extractor if the
following holds: for any two sets X, Y ⊆{0, 1}n of size 2k, |Prx∼X,y∼Y [f(xy) = 1] −1
2| ≤ϵ.
Deﬁnition 10. For any pair of functions k, ϵ : N →N, (k, ϵ)-Extractor is the following search
problem: given 1n, output a circuit C with 2n inputs such that the function fC : {0, 1}n ×{0, 1}n →
{0, 1} deﬁned by C is a (k(n), ϵ(n)) extractor.
The above problem deﬁnition does not expressly constrain the size of C, though for a construc-
tion to be “explicit” in any useful sense (eﬃciently computable as a function of n), C would have to
have size polynomial in n. The following reduction placing extractor construction in APEPP will
immediately imply that we can construct a (log n + O(1), ϵ) extractor of circuit size approximately
n3 in APEPP for any ﬁxed ϵ.
Theorem 4. For any eﬃciently computable ϵ(n) satisfying
1
nc < ϵ(n) < 1
2 for a constant c and
suﬃciently large n, (log n + 2 log(1/ϵ(n)) + 3, ϵ(n))-Extractor reduces in polynomial time to
Empty.
Proof. Let ϵ = ϵ(n), and let d = ⌈4
ϵ2 ⌉.
We will set up an instance of Empty with at most
2d2n3 + 2dn2 −ϵ2d2n2 + 2 log(2dn) + 1 inputs and exactly 2d2n3 outputs which has the following
property: Let A be any 2d2n3-bit string outside the range of this circuit, viewed as an ordered list
of d2n2 elements of F22n denoted α1, . . . αd2n2, and consider the function f : {0, 1}2n →{0, 1}2n
deﬁned by
f(x) =
d2n2
X
i=1
αixi−1
Then the function g : {0, 1}2n →{0, 1} deﬁned by
g(x) = f(x)
mod 2
is a (log dn, ϵ) extractor. Since log dn = log n+log d ≤log n+log(4/ϵ2 +1) ≤log n+2 log(1/ϵ)+3,
this would give the required result.
Let α1, . . . αd2n2 be any sequence of coeﬃcients in F22n such that the above function g corre-
sponding to the αi is not a (log dn, ϵ) extractor. So there exist two sets of n-bit strings X, Y ,
12


<!-- page 13 -->
each of size 2log dn = dn, and some b ∈{0, 1} such that Prx∼X,y∼Y [f(xy) = b] >
1
2 + ϵ.
Let
R = {xy | x ∈X, y ∈Y } ⊆{0, 1}2n. We have |R| = |X||Y | = d2n2. Let r1, . . . rd2n2 denote
the lexicographical enumeration of R. By assumption, we have that g(ri) = b mod 2 for at least
a 1
2 + ϵ fraction of indices i. So then, if we let βi be the 2n −1-bit preﬁx of f(ri), we can de-
duce the value of f(ri) from βi and b for at least d2n2(1
2 + ϵ) values of i. Thus, there is some
d2n2(1
2 −ϵ)-weight d2n2-bit string S, such that from b, S, and the βi’s, we can deduce f(ri) for all
i. Now, once we are able to deduce f(x) for each of the d2n2 distinct values of x in R, since f is a
degree d2n2 −1 polynomial, we can uniquely and eﬃciently determine the coeﬃcients αi of f using
Gaussian elimination on the corresponding d2n2 × d2n2 Vandermonde matrix.
So we have shown that for any set of αi’s which does not deﬁne a (log dn, ϵ) extractor, there
exists X, Y, b, S, and βi’s, from which we can eﬃciently deduce the αi’s. It is clear that we can
encode X, Y using 2dn2 bits, b using 1 bit, and the βi’s using d2n2(2n −1) = 2d2n3 −d2n2 bits.
Since S is a d2n2(1
2 −ϵ)-weight d2n2-bit string, by Lemma 2, we can encode S using at most
d2n2(1 −ϵ2) + log(d2n2) = d2n2 −ϵ2d2n2 + log(d2n2) bits. So in total the number of input bits is
at most:
2dn2 + 1 + d2n2 −ϵ2d2n2 + log(d2n2) + 2d2n3 −d2n2 = 2d2n3 + (2d −ϵ2d2)n2 + 2 log(dn) + 1
Since we chose 4
ϵ2 +1 ≥d ≥4
ϵ2 , we have that(2d−ϵ2d2) < −1 for ϵ ≤1
2, so the number of input bits
is at most 2d2n3 −n2 + 2 log(dn) + 1. Since we assumed ϵ ≥
1
nc for a constant c, we have ϵ ≥
1
2n2/5
for suﬃciently large n, and so n2 > 2 log(dn) + 1, which is strictly less then the number of output
bits 2d2n3.
So this implies that the circuit Φ mapping X, Y, b, S and the βi’s to a corresponding set of αi’s
is a valid instance of Empty, and that any solution to this instance is a set of coeﬃcients deﬁning
a (log n + 2 log(1/ϵ) + 3, ϵ) extractor. From the coeﬃcients αi we can easily construct an eﬃcient
circuit computing the function g, thus solving the problem (log n + 2 log(1/ϵ) + 3, ϵ)-Extractor.
It is also clear from the construction that this reduction can be carried out in poly(n, 1
ϵ) time, so
poly(n) time under our assumption that ϵ(n) ≥
1
nc for a constant c.
For the typical parameter regime where ϵ is an arbitrarily small constant, this gives a two source
extractor for min-entropy log n + O(1).
Corollary 1. Explicit construction of strongly explicit Ramsey graphs (n-vertex graphs containing
no clique or independent set of size c log n for some constant c), in both the bipartite and non-
bipartite case, reduces to Empty.
Proof. As noted in [3], any two-source extractor in the above sense (with ϵ ﬁxed to any constant
less then one half) is automatically a bipartite Ramsey graph, and from a strongly explicit bipartite
Ramsey graph we can construct a strongly explicit non-bipartite Ramsey graph eﬃciently.
3.4
Rigid Matrices
Deﬁnition 11. [38] We say that n × n matrix M over Fq is (r, s) rigid if for any matrix S ∈Fn×n
q
with at most s non-zero entries, M + S has rank greater than r.
Deﬁnition 12. For any q : N →N such that q(n) is a prime power ∀n, (ϵ, q)-Rigid is the following
search problem: given 1n, output an n × n matrix M over Fq(n) which is (ϵn, ϵn2) rigid.
Theorem 5. For any ϵ ≤1
16, and any eﬃciently computable q(n) satisfying the above, (ϵ, q)-Rigid
reduces in polynomial time to Empty.
13


<!-- page 14 -->
Proof. Let M be any n × n matrix over Fq which is not (r, s) rigid. So there exists an n × r matrix
L, an r×n matrix R, and an n×n matrix S with at most s non-zero entries, such that M = LR+S.
It is clear that from the descriptions of L, R, S we can eﬃciently compute M.
L and R can each be described explicitly using nr log q bits. For S, we encode it by specifying an
n2-bit string T of weight s denoting the entries of S which are nonzero, together with an s log q-bit
string giving the values of the nonzero entries. Applying the encoding scheme in Lemma 1 for T,
overall the number of bits in this encoding is at most log
 n2
s
 
+ (2nr + s) log q. Setting r = ϵn and
s = ϵn2 this is at most:
log
 n2
ϵn
 
+ (3ϵn2) log q ≤
(3
4 + ϵ −ϵ2)n2 + (3ϵn2) log q ≤
(3
4 + ϵ −ϵ2 + 3ϵ)n2 log q
Since we chose ϵ ≤1
16, this is at most 997
1000n2 log q, which is strictly less then the n2 log q bits needed
to specify an arbitrary matrix in Fn×n
q
.
3.5
Strings of High Time-Bounded Kolmogorov Complexity
Deﬁnition 13. Let U be any ﬁxed Turing machine, and let t : N →N be a time bound. For a
string x, Kt
U(x) denotes the length of the smallest string y such that U outputs x on input y in
t(|x|) steps.
Deﬁnition 14. For a Turing machine U and time bound t, we deﬁne the following explicit con-
struction problem Kt
U-Random: given 1n, output an n bit string x such that Kt
U(x) ≥n −1
Theorem 6. For any ﬁxed Turing machine U and ﬁxed polynomial time bound t, Kt
U-Random
reduces in polynomial time to Empty.
Proof. We construct a circuit Φ with n −1 input bits and n output bits as follows. If the input
is of the form 0∗1y, Φ simulates U on y for t(n) steps and outputs the result, padded/truncated
to length n if it is not already. Otherwise, Φ outputs 0n. It is clear that Φ can be produced in
polynomial time given 1n for any ﬁxed U, t. Further, for any n-bit string x with Kt
U(x) ≤n −2,
there is some y of length k ≤n −2 such that U outputs x on input y in t(n) steps, and so Φ will
output x on input 0n−k−21y. So any string x outside the range of Φ must have Kt
U(x) ≥n −1.
3.6
Other Problems
In Section 5, we introduce two more explicit construction problems and show that each of these, in
addition to a variant of the rigidity problem, can be reduced directly to Hard Truth Table in
polynomial time. This also implies that both problems are contained in APEPP. We will postpone
a formal deﬁnition of each of these new problems until Section 5, but give an informal statement
here for completeness:
Explicit communication problems outside PSPACECC:
An explicit 2n×2n communication
matrix which cannot by solved by any o(n)-space protocol can be constructed in APEPP.
14


<!-- page 15 -->
Explicit data structure problems with high bit-probe complexity:
An explicit data
structure problem with nearly maximum complexity in the bit-probe model can be constructed
in APEPP.
4
Constructing Hard Truth Tables is Complete for APEPP
In this section we show that constructing a hard truth table is complete for APEPP under PNP
reductions. As mentioned before, the core of this theorem was originally proven by Jeˇr´abek [18],
and the main construction underlying the reduction dates back further to the work of Goldreich,
Goldwasser, and Micali [13].
Jeˇr´abek’s result is phrased in the language of proof complexity,
stating that the theorem asserting the existence of hard boolean functions is equivalent to the
empty pigeonhole principle in a particular theory of Bounded Arithmetic. We demonstrate below
that when translated to the language of search problems and explicit constructions, his proof yields
a PNP reduction from Empty to the problem of constructing a hard truth table.
We in fact
prove a more general statement here which holds for arbitrary circuit classes equipped with oracle
gates. For a very broad set of circuit classes C, we prove that given a truth table which is hard
for C-circuits, we can ﬁnd an empty pigeonhole of any C-circuit using polynomially many calls to
an oracle for inverting C-circuits (by inverting we mean ﬁnding the preimage of a given string, or
reporting that none exist). In order to make this precise, we ﬁrst deﬁne the type of generalized
circuit classes we will consider:
Deﬁnition 15. A circuit class is deﬁned by a basis C, which is simply a (possibly inﬁnite) set of
boolean-valued boolean functions. A C-circuit is then deﬁned as a circuit composed entirely of gates
computing functions in the basis C. For a language L, we will refer to the “basis L” to mean the
basis {Ln | n ∈N}∪{∧, ∨, ¬}, where Ln is the n-bit boolean function deciding L on length n inputs.
For a complexity class C with a complete language L, we will refer to “the basis C” to mean the
basis corresponding to L.
We say that a basis C is “suﬃciently strong” if there exist C-circuits computing the two-input
∧, ∨functions and the one-input ¬ function.
For any basis C, a “C-inverter oracle” is an oracle which, given a C-circuit C and some string
y, determines whether there exists an x such that C(x) = y, and produces such an x if it exists. A
C-inverter reduction is a polynomial time reduction that uses a C-inverter oracle.
Note that in the special case where C = {∧, ∨, ¬}, a C-inverter reduction is equivalent to a PNP
reduction (since inverting a circuit over the standard boolean basis is NP-complete).
Deﬁnition 16. For any basis C, the class of search problems APEPPC is deﬁned by the following
complete problem EMPTYC: given a C-circuit with more output wires than input wires, ﬁnd a
boolean string whose length is equal to the number of output wires but which is not in the range of
this circuit. For any strictly increasing function f : N →N, we deﬁne the problem EMPTY C
f(n),
which is the special case of EMPTYC where the circuit is required to have f(n) output wires, where
n is the number of input wires.
We start with the following technical lemma, which allows us to restrict our attention to circuits
with exactly twice as many outputs as inputs.
Lemma 3. For any basis C, EMPTY C
2n is complete for APEPPC under C-inverter reductions.
Proof. It is straightforward to see that EMPTY C
n+1 is complete for APEPPC: given a C-circuit C
with n inputs and more than n outputs, we can simply delete all the output bits except for the ﬁrst
15


<!-- page 16 -->
n + 1 to obtain an instant of EMPTY C
n+1. Any n + 1-bit string outside the range of this smaller
circuit can than be padded arbitrarily to the output size of the original circuit, and this padded
string must also be outside the range of C.
We now reduce EMPTY C
n+1 to EMPTY C
2n. Let C be some C-circuit with n inputs and n + 1
outputs. For a positive integer i, we deﬁne Ci : {0, 1}n →{0, 1}n+i inductively as follows. For the
base case, C1 is simply deﬁned as C. For i > 1, we ﬁrst compute Ci−1 on the input x to obtain
a string x′ of length n + i −1, and then compute C on the ﬁrst n bits of x′ and concatenate the
output with the remaining bits of x′. Since C replaces the ﬁrst n bits with n + 1 bits, if Ci−1 has
output length n + i −1 then Ci has output length n + i.
We now claim that for any positive i, if we are given an n + i-bit string outside the range of
Ci, we can ﬁnd an n-bit string outside the range of C in poly(i|C|) time using a C-inverter. In
particular, we prove by induction on i that this can be accomplished with i inverter calls. For i = 1
this is trivially the case, since C1 = C. Now say i > 1. Let y ∈xz be outside the range of Ci for
some x ∈{0, 1}n+1, z ∈{0, 1}i−1. We ﬁrst use the inverter oracle to determine if x has a preimage
under C; if it does not then we have found an empty pigeonhole for C and are done. Otherwise,
we use the inverter to ﬁnd a preimage x′ ∈{0, 1}n for x under C. So then x′z must be outside the
range of Ci−1, since if it were not then xz = y would have a preimage under Ci, contradicting our
initial assumption. So by induction we can use i −1 inverter calls to ﬁnd an empty pigeonhole for
C given x′z, completing the proof of the inductive case.
Now, setting i = n, we get a reduction from EMPTY C
n+1 to EMPTY C
2n as claimed.
We now deﬁne the hard truth table construction problem that will be used in our reduction:
Deﬁnition 17. Let ϵ-HardC denote the following search problem: given 1N, output a string x of
length N such that x cannot be computed by C-circuits of size Nϵ.
In the case where C = {∧, ∨, ¬}, we drop the subscript and refer to this problem simply as ϵ-
Hard. For N = 2n, a solution to ϵ-Hard on input 1N is a truth table of a function on n variables
requiring 2ϵn-sized circuits, the same object used to build the Impagliazzo-Wigderson generator.
Theorem 7. Let C be any suﬃciently strong basis and ϵ > 0 be a constant such that ϵ-HardC
is total for suﬃciently large input lengths. Then EmptyC reduces in polynomial time to ϵ-HardC
under C-inverter reductions.
Proof. By Lemma 3 we know that EmptyC reduces to EMPTYC
2n under C-inverter reductions.
Now, let C be an instance of EMPTYC
2n, and let k = 2⌈log |C|⌉⌈1
ϵ⌉. Consider the following map
C∗: {0, 1}n →{0, 1}2kn, deﬁned informally as follows: given a string x ∈{0, 1}n, apply C once to
get 2 n-bit strings, then apply C to both of those n-bit strings to get four, and continue k times
until we have 2k n-bit strings, or equivalently a 2kn-bit string. This process is illustrated in Figure 1
below:
16


<!-- page 17 -->
Figure 1: Extending a map C : {0, 1}n →{0, 1}2n to a map C∗: {0, 1}n →{0, 1}2kn
. Dotted boxes indicate the number of bits along a wire.
To deﬁne this function more formally, ﬁrst we deﬁne the following maps L, R : {0, 1}2n →
{0, 1}n, where L takes a 2n-bit string and ouputs the ﬁrst n bits, and R takes a 2n-bit string and
outputs the last n bits. Given a nonempty sequence σ1, . . . σt ∈{L, R}∗, let Cσ : {0, 1}n →{0, 1}n
be the function σt ◦C ◦. . . ◦σ1 ◦C. Now, given a binary string, we can associate it with such a
sequence by associating 0 with L and 1 with R, and so we will abuse notation and write Cx for
a binary string x as shorthand for Cσ where σ is the sequence of L, R associated with the binary
string x. We are now ready to formally deﬁne our function C∗. As C∗is a map {0, 1}n →{0, 1}2kn,
we can think of the output as being clumped into 2k blocks, each containing n bits. Given this
terminology, C∗behaves as follows: on input x, the ith block of the output of C∗will be C∥i∥(x),
where ∥i∥denotes the standard representation of i as a k-bit binary string.
From here, the proof proceeds in two steps. First, we show that, by setting m = n2k = poly(|C|),
any solution to ϵ-HardC on input 1m will be a string that is not in the range of C∗. Second, we
will show that given a string outside the range of C∗, we can ﬁnd a string outside the range of C
using only a polynomial number of calls to a C-inverter.
To carry out the ﬁrst of these steps, we will show that any string in the range of C∗, when
interpreted as a truth table of length m = n2k on ⌈log n⌉+ k variables, can be computed by a
circuit of size O(|C|k). Since, by construction of k, we have that m ≥|C|
2
ϵ , a solution to ϵ-Hard
on input 1m will be a truth table of length m not computable by a circuit of size mϵ ≥|C|2, and
thus a circuit of size O(|C|k) = O(|C| log |C|) would be a contradiction for all input lengths greater
than some absolute constant. We construct such a circuit for any string in the range of C∗as
follows: let y be a 2kn-bit string such that for some x ∈{0, 1}n, C∗(x) = y. The circuit computing
y will have x written as advice/constants, and will feed x through k copies of the circuit C in series.
We will split the ⌈log n⌉+ k input variables into a block of k variables we call i, and a block of
⌈log n⌉variables we call j. We then use i to determine whether to apply L or R to the output of
one of the copies of C before feeding it into the next, to get some resulting string xi, and then we
use j to index into the jth position of xi, to get yi,j. A diagram of this circuit is shown in Figure 2:
17

[CAPTION] Figure 1: Extending a map C : {0, 1}n →{0, 1}2n to a map C∗: {0, 1}n →{0, 1}2kn


<!-- page 18 -->
Figure 2: A succinct circuit whose truth table is y, for any y in the range of C∗. Dotted boxes
indicate the number of bits along a wire. Note that although x is shown as an input in this diagram,
for any given y we ﬁx a preimage x as constants/advice, and so the only true inputs to this circuit
are i, j.
To see that this circuit has size O(|C|k), note that the subcircuits computing either L or R
depending on a bit of i can be computed easily with O(n) gates over the basis {∧, ∨, ¬} (this is
essentially a multiplexer), and also that the ﬁnal subcircuit indexing into an n-bit string can be
computed with O(n) {∧, ∨, ¬} gates as well; since we assumed C is suﬃciently strong, both of these
subcircuits can therefore be computed with O(n) C-gates. Since |C| ≥n, and this circuit contains
only k copies of C and the aforementioned subcircuits, plus the constants describing the string x
of length n, this circuit has O(|C|k) size as claimed.
Thus, we now know that any solution to ϵ-HardC on input 1m will not be in the range of C∗,
and by assumption ϵ-HardC is total for suﬃciently large input lengths so such a solution exists. It
remains only to show that we can use a string outside the range of C∗, together with a C-inverter
oracle, to ﬁnd a string outside the range of C. We proceed exactly as in the proof of Lemma 3.
Let y be any string outside the range of C∗. Refer to Figure 1 which gives a diagram of a circuit
computing C∗; at a layer i ∈[k] of this circuit, we have 2i blocks of n bits feeding into 2i copies
of C, and these copies of C then output 2i+1 blocks of n bits at the next layer. So working back
from the output layer k, we can test if any consecutive 2n-bit block of y is outside of the range of
C. If none of them are, then we ﬁnd a preimage for all blocks, interpret this as the output of the
previous layer, and continue our search from there. We follow this process all the way back to the
input layer or until we ﬁnd an empty pigeonhole of C. If we never ﬁnd an empty pigeonhole of C,
then this process will terminate at the input layer with a string x such that C∗(x) = y, which is
impossible by assumption, so at some point we must indeed ﬁnd a string outside the range of C.
Checking whether a particular string is an empty pigeonhole, or ﬁnding a preimage if it’s not, can
be accomplished with one call to a C-inverter oracle by deﬁnition. We perform this test at most
2k = poly(|C|) times (once for every copy of C in the diagram in Figure 1), so overall this process
can be accomplished in polynomial time using a C-inverter oracle.
We now examine the implications of this theorem for particular circuit classes of interest.
Theorem 8. For any 0 < ϵ < 1
2, ϵ-HardΣP
i is complete for APEPPΣP
i under ∆P
i+2 reductions.
Proof. Containment of ϵ-HardΣP
i in APEPPΣP
i follows directly from the proof of Theorem 1 with
minimal adjustments; we must add the assumption ϵ < 1
2 to account for the unbounded fan-in of
18

[CAPTION] Figure 2: A succinct circuit whose truth table is y, for any y in the range of C∗. Dotted boxes


<!-- page 19 -->
oracle gates in our counting argument. So it remains only to show that ϵ-HardΣP
i is hard for this
class as well. Since the above reduction uses a polynomial number of calls to the C-inverter, it
suﬃces to show that we can implement a ΣP
i -circuit inverter using a ∆P
i+2 oracle. Given this, we
can complete the entire reduction in P∆P
i+2 = ∆P
i+2.
Let C be a ΣP
i -circuit with m oracle gates, and y be a potential output. To test if y is a valid
output, we nondeterministically guess an input x, in addition to an output value for every gate in
C, and a set of witness strings z1 . . . zm, one for each of our oracle gates. We then check that each
gate output is valid (we will use the guessed witnesses here), and that the value of the terminal
gate outputs is y.
The veriﬁcation of the terminal gates and all classical ∧/∨/¬ gates can be
done in polynomial time. Verifying that all ΣP
i oracle gates have valid outputs given their inputs
corresponds to verifying that a sequence of strings x1, . . . , xm satisfy a sequence of ΣP
i
and ΠP
i
predicates P1, . . . , Pm, where m is of polynomial length. For each i such that Pi is a ΣP
i predicate,
this predicate is of the form ∃zP′
i(xi, z) where P′
i is a ΠP
i−1 predicate, and so we can use the zi we
originally guessed and simplify these to ΠP
i−1 predicates. For any i such that Pi is a ΠP
i predicate
we ignore zi. In this way, we can transform all Pi into ΠP
i predicates. Verifying that a sequence
of of strings satisﬁes a sequence of ΠP
i predicates can then be checked with a single ΠP
i predicate
representing their conjunction. So overall the veriﬁcation process can be carried by checking a
single ΠP
i predicate, and so determining the existence of a solution can be done in ΣP
i+1. From a
ΣP
i+1 test to determine the existence of a preimage for y, we can compute a preimage when one
exists in ∆P
i+2 by a standard application of binary search.
In the absence of any oracle gates, we have the following:
Theorem 9. For any 0 < ϵ < 1, ϵ-Hard is complete for APEPP under PNP reductions.
4.1
Implications of Completeness
This result gives an exact algorithmic characterization of the possibility of proving 2Ω(n) ΣP
i -circuit
lower bounds for EΣP
i+1:
Theorem 10. There exists a language in EΣP
i+1 with ΣP
i -circuit complexity 2Ω(n) if and only if
there is a ∆P
i+2 algorithm for EmptyΣP
i .
Proof. Say there is a language L in EΣP
i+1 with ΣP
i -circuit complexity 2Ω(n). So there exists an
ϵ > 0 such that for all but ﬁnitely many n, L cannot be computed on length n inputs with ΣP
i -
circuits of size less then 2ϵn.
So then we have a polynomial time algorithm for
ϵ
2-HardΣP
i
as
follows: given 1n, output the truth table of L over ⌊log n⌋-bit inputs. Since L ∈EΣP
i+1, this can
be done in 2⌊log n⌋2O(log n) = poly(n) time with a ΣP
i+1 oracle. This truth table will have length
n
2 ≤2⌊log n⌋≤n. We then pad this truth table with 0’s at the end to be of length n. If there was a
circuit of size nϵ/2 for this n-bit truth table on ⌈log n⌉bits, then on the ﬁrst ⌊log n⌋bits of input
this computes the truth table for L on ⌊log n⌋-bit inputs. Since n
2 ≤2⌊log n⌋, this would imply a
circuit of size 2ϵ⌊log n⌋to compute L on ⌊log n⌋-bit inputs, contradicting the hardness assumption.
Thus, there exists a ∆P
i+2 algorithm for ϵ
2-HardΣP
i for some ϵ > 0, and so by Theorem 9, there
also exists a ∆P
i+2 algorithm for EmptyΣP
i .
Alternatively, say there is a ∆P
i+2 algorithm for EmptyΣP
i . So in particular there is a ∆P
i+2
algorithm for ϵ-HardΣP
i for any ﬁxed ϵ < 1
2. Consider the language L decided by the following
EΣP
i+1 machine: given an n-bit input, we use our ∆P
i+2 algorithm for ϵ-HardΣP
i on input 12n to
generate a truth table, then look up the n-bit input in this truth table to determine whether
19


<!-- page 20 -->
to accept or reject. By deﬁnition this language must have ΣP
i -circuit complexity 2Ω(n), and this
machine will run in time poly(2n) = 2O(n) with a ΣP
i+1 oracle.
In the most interesting case, we conclude that a 2Ω(n) circuit lower bound for ENP holds if
and only if there is a PNP algorithm for Empty. Together with the results in Section 3, this
gives newfound insight into the diﬃculty of proving exponential circuit lower bounds for the class
ENP: proving such a lower bound requires solving a universal explicit construction problem, and
would immediately imply PNP constructions for a vast range of combinatorial objects which we
currently have no means of constructing without a ΣP
2 oracle. Theorem 10 also allows us to derive
the following interesting fact about the circuit complexity of ENP:
Corollary 2 (Worst-Case to Worst-Case Hardness Ampliﬁcation in ENP). If there is a language
in ENP of circuit complexity 2Ω(n), then there is a language in ENP requiring circuits of size 2n
2n.
Proof. By Theorem 10, if there is a language in ENP of circuit complexity 2Ω(n), then there is
a PNP algorithm for Empty. By Theorem 1, this implies a PNP algorithm for Hard Truth
Table, and thus a PNP construction of a truth table of length N with hardness
N
2 log N . This in
turn implies the existence of a language in ENP of circuit complexity 2n
2n.
Tweaking the proof of Theorem 7 slightly we also obtain the following:
Corollary 3 (Worst-Case to Worst-Case Hardness Ampliﬁcation in EXPNP). If there is a language
in EXPNP of circuit complexity 2nΩ(1), then there is a language in EXPNP requiring circuits of
size 2n
2n.
Proof. The proof follows that of the previous corollary, with the following modiﬁcation to the
reduction in Theorem 7: we start with the assumption that for some ϵ > 0 we are able to construct
N-bit truth tables with hardness 2logϵ N in time quasipolynomial in N using an NP oracle, and
then apply the same reduction setting k = log⌈1
ϵ ⌉|C|.
We thus obtain a rather unexpected “collapse” theorem for the circuit complexity of EXPNP:
if EXPNP has circuits of size 2n
2n inﬁnitely often, then this class in fact has circuits of size 2nϵ
inﬁnitely often for every ϵ > 0.
We can reﬁne this slightly as follows.
Deﬁnition 18. MCSP, deﬁned originally in [21], is the following decision problem: given a truth
table x and a size parameter s, determine whether x has a circuit of size at most s. Let sMCSP
denote the search variant of this problem, where we are given a truth table x and must output a
circuit computing x of minimum size.
For the hardness ampliﬁcation procedures in Corollaries 2 and 3, we can in fact replace the NP
oracle with an oracle for sMCSP, which is non-trivial since sMCSP is not known to be NP-hard.
Corollary 4. If there is a language in EsMCSP (resp. EXPsMCSP) of circuit complexity 2Ω(n)
(resp. 2nΩ(1)), then there is a language in EsMCSP (resp. EXPsMCSP) requiring circuits of size
2n
2n.
Proof. Recall the two reductions in Lemma 3 and Theorem 7. In order to ﬁnd an empty pigeonhole
of the input circuit C given a solution to ϵ-Hard, we only need to use the C-inverter on C itself.
In the case of a reduction from Hard Truth Table to ϵ-Hard, the circuit of interest C maps
circuits of size at most
N
2 log N to their N-bit truth tables, and so an oracle for sMSCP would suﬃce
to invert C.
20


<!-- page 21 -->
It should be noted that a related result was proven in [21], showing that this type of hardness
ampliﬁcation is possible in E assuming MCSP∈P. However, their proof does not translate directly
to an unconditional result in the oracle setting. Due to their use of the Impagliazzo-Wigderson
generator, directly applying their proof in the oracle setting using the relativized generator of [23]
would instead show that if EMCSP requires 2Ω(n)-sized nondeterministic circuits, then EMCSP
requires 2n
2n-sized standard circuits, which is a weaker statement then what is shown above (modulo
the search/decision distinction between sMCSP and MCSP). Another result of a similar ﬂavor
was also proven in [15], where they establish that, assuming the (unproven) NP-completeness of
MCSP, 2nΩ(1) lower bounds for NP ∩coNP imply 2Ω(n) lower bounds for ENP. This type of
ampliﬁcation is incomparable to the ampliﬁcation demonstrated in Corollaries 2 and 3.
In [5], Buresh-Oppenheim and Santhanam deﬁne a notion of “hardness extraction” that is
highly relevant to the results in this section. Informally, a hardness extractor is a procedure which
takes a truth table of length N and circuit complexity s, and produces a truth table with nearly
maximum circuit complexity relative to its size, whose length is as close to s as possible. The proof
of Corollary 4 can in fact be viewed as a construction of a near-optimal hardness extractor using
an sMCSP oracle. In particular our procedure is able to extract approximately the square root of
the input’s hardness:
Theorem 11. There is a polynomial time algorithm using an sMCSP oracle which, given a truth
table x of length M and circuit complexity s, outputs a truth table y of length N = Ω(
q
s
log M ) and
circuit complexity Ω(
N
log N ).
Proof. The proof follows from a more careful analysis of Corollary 4; we give a sketch here. Adapting
the proof of Theorem 1, for any N we can eﬃciently construct a circuit C with N outputs and
⌊N
2 ⌋inputs, such that any N-bit string outside its range requires circuits of size δ(
N
log N ) for some
ﬁxed δ > 0. In particular, it is clear from the proof of Theorem 1 that such a C can be constructed
of circuit size O(N2). Now, let k to be the minimum integer such that 2k N
2 ≥M. Following the
argument in the proof of Theorem 7, we can construct a map C∗: {0, 1}⌊N
2 ⌋→{0, 1}2k⌊N
2 ⌋such that
any element of its range has circuit complexity O(|C|k) = O(N2k), and such that given a string
outside the range of C∗, we can ﬁnd a string outside the range of C using 2k calls to an sMCSP
oracle. Setting N = ϵ
q
s
log M for ϵ suﬃciently small, we get a value of k ≤log M. This in turn
means that a circuit of size O(N2k) = O(ϵ2s) whose truth table is x0⌊N
2 ⌋2k−M would contradict
the fact that x has hardness at least s (for suﬃciently small choice of ϵ). Thus, x0⌊N
2 ⌋2k−M must
lie outside the range of C∗, so using x and our sMCSP oracle we can ﬁnd an N-bit string outside
the range of C. This process requires at most 2k ≤M calls to our sMCSP oracle, each of input
size at most N < M, so overall this takes polynomial time with access to an sMCSP oracle.
A natural goal would be to improve this procedure to extract (
s
log M )
1
2 +ϵ or ideally Ω(
s
log M ) bits
of hardness. The only obstacle here is improving the O(N2) upper bound on the circuit complexity
of C, the circuit which takes descriptions of log N-input circuits of size ≈N and outputs their truth
tables. However, Williams observes in [39] (see footnote 7 of that paper) that an N2−ϵ upper bound
on size(C) would imply a (nonuniform) breakthrough for 3SUM, so improving this extractor in
its current form appears diﬃcult.
21


<!-- page 22 -->
5
Direct P Reductions to Hard Truth Table
Ideally we could extend the completeness result in Theorem 9 to work with polynomial time re-
ductions, as opposed to PNP reductions. However, the NP oracle seems highly necessary for the
proof techniques used above. Despite this obstacle, we show that there is a natural set of problems
in APEPP which can be reduced to the problem of ﬁnding truth tables of hard functions via P
reductions.
A simple way to phrase the following results is that any truth table with suﬃciently large
circuit complexity will necessarily satisfy a variety of other pseudorandom properties for which no
explicit constructions are known, including: rigidity over F2, high space-bounded communication
complexity, and high bit-probe complexity. To show this, we demonstrate that the failure of a
string x to possess any of these properties implies a smaller than worst case circuit for x.
To give the tightest reductions possible, we will introduce one new parameterized version of the
hard truth table construction problem:
Deﬁnition 19. δ-Quite Hard is the following problem: given 1N, output an N-bit truth table
with hardness
δN
log N
This problem is total for suﬃciently small δ. We recall also the deﬁnition of ϵ-Hard, where we
must construct a truth table of hardness Nϵ.
5.1
Rigidity
We begin with the case of rigidity.
We will deﬁne the following weaker version of the rigidity
construction problem:
Deﬁnition 20. ϵ-Rather Rigid is the following search problem: given 1N, construct an N × N
matrix over F2 which is (ϵN, ϵN2)-rigid.
These parameters are the best possible up to constant factors, and in particular would be
suﬃcient to carry out Valiant’s program over F2.
Theorem 12. For any suﬃciently small δ > 0, there exists some ϵ > 0 such that ϵ-Rather Rigid
reduces in polynomial time to δ-Quite Hard.
Proof. To prove this, it suﬃces to show that for any N ×N matrix M which is not (ϵN, ϵN2)-rigid,
we can construct a boolean circuit with f(ϵ)O( N2
log N ) gates which decides the value of Mi,j given
the 2⌈log N⌉-bit input (i, j), for some function f which approaches zero as ϵ approaches zero. This
then implies that for any ﬁxed δ, an N2-bit truth table requiring circuits of size
δN2
log N must be
(ϵN, ϵN2)-rigid for some ϵ > 0 which is a function only of δ (and otherwise determined by f and
the constants hidden in the O(·) term).
Say M is not (ϵN, ϵN2)-rigid. So there exists an N × ϵN matrix L, an ϵN × N matrix R, and
an N × N matrix S with at most ϵN2 nonzero entries, such that LR ⊕S = M. We will construct a
circuit allowing us to eﬃciently index M which uses these matrices L, R, S as advice. To encode L
and R, we can utilize the well-known theorem of Shannon that any truth table of length N can be
computed by a circuit of size O(
N
log N ) [34]. Thus, L can be speciﬁed as a list of ϵN circuits, each
of size O(
N
log N ), where the jth circuit Lj represents the jth column, and Lj(i) computes Li,j. The
same can then be done for R (indexing columns instead). To encode S, we employ a reﬁnement of
22


<!-- page 23 -->
Shannon’s result due to Lupanov [14], which tells us that for suﬃciently large N, any truth table
of length N with at most ϵN nonzero entries can be computed by circuit of size
log
  N
ϵN
 
log log
  N
ϵN
  + o
 
N
log N
 
≤H(ϵ)O(
N
log N )
Where H denotes the binary entropy function.
Thus S can be computed by a circuit of size
H(ϵ)O( N2
log N ).
It remains to show that the additional circuitry we need to compute Mi,j given i, j and the
encodings of L, R, S does not increase things too much. By deﬁnition, we have that:
Mi,j = ⟨rowi(L), colj(R)⟩⊕Si,j
where the dot product is taken over F2. Now consider the circuit diagram shown in Figure 3:
Figure 3: A small circuit for a non-rigid truth table
The subcircuit computing rowi(L) is deﬁned as follows: as described above, L represents a
N × ϵN matrix, speciﬁed as a list of ϵN circuits of size O(
N
log N ), each computing a column of L.
rowi(L) feeds i into each of these circuits in parallel to get an ϵN-bit string giving value of the
ith row of L. colj(R) is deﬁned analogously. The circuit Si,j simply computes the (i, j)th entry of
S given i, j. Finally, the subcircuit ⊕(X) computes the parity of its input string, the subcircuit
∧(X, Y ) computes the bit-wise AND of two equal length strings, and the terminal gate computes
the two-bit parity function. Given the previous equation relating the (i, j)th index of M to the
(i, j)th rows/columns/indices of L, R, S, it is straightforward to see that this circuit performs the
necessary calculation.
23

[CAPTION] Figure 3: A small circuit for a non-rigid truth table


<!-- page 24 -->
It is clear that the ⊕(X) and ∧(X, Y ) can be implemented with a number of gates linear in
their input size, which in this case is ϵN. From the analysis above, each of rowi(L) and colj(R)
can be implemented using ϵO( N2
log N ) gates, and Si,j can be implemented using H(ϵ)O( N2
log N ) gates.
So overall, this circuit has size (H(ϵ) + ϵ)O( N2
log N ). Since H(ϵ) + ϵ approaches zero for decreasing ϵ,
this implies that for any ﬁxed δ > 0, an N2-bit truth table requiring circuits of size
δN2
log N must be
(ϵN, ϵN2) rigid for some ϵ > 0 which is a function only of δ (and which is otherwise determined by
the constants hidden in the O(·) expressions above).
We thus conclude that if E contains a language of circuit complexity Ω(2n
n ), then there is a
polynomial time construction of (Ω(n), Ω(n2))-rigid matrices over F2.
5.2
Space-Bounded Communication Complexity
The class PSPACECC was deﬁned originally in [2] as a generalization of the class PHCC to
an unbounded alternation of quantiﬁers. We will not give this original deﬁnition, but rather a
simpliﬁcation due to [36].
Deﬁnition 21. Let f : {0, 1}n × {0, 1}n →{0, 1}. We say f has a space-s protocol if there is a
deterministic protocol deciding f of the following form. Alice receives x ∈{0, 1}n, and Bob receives
y ∈{0, 1}n. There is an s-bit shared memory, and Alice and Bob alternate turns writing to this
memory. On a player’s turn, they can modify the contents of the s-bit shared memory as a function
only of its previous contents and their private input x or y, or decide to halt and output some
z ∈{0, 1} (this decision is also a function only of their input and the shared memory’s previous
state). This protocol is valid if f(xy) = z for all x, y.
By a result of Song [36], we have that if f ∈PSPACECC then f has a poly(log n) space
protocol (Song uses a slightly diﬀerent model where Alice and Bob have private s-bit tapes, but
our model is at least as strong up to a doubling in space since sharing the tape only increases their
ability to communicate). Due to a basic counting argument, most functions f require Ω(n) space,
and any such function must lie outside of PSPACECC. However, it has been a long standing open
problem to give an explicit construction a communication matrix outside of even PHCC [36]. We
thus deﬁne the following search problem:
Deﬁnition 22. δ-SPACE is the following search problem. Given 1N, where N = 2n, output a
communication matrix {0, 1}n × {0, 1}n →{0, 1} such that f requires space-δn communication
protocols.
Theorem 13. For any δ < ϵ < 1
2, δ-Space reduces in polynomial time to (1
2 + ϵ)-Hard.
Proof. The proof follows the exact same strategy as above, constructing a smaller-than-worst-case
circuit for any matrix with a δn-space protocol. We will omit the detailed construction of our
circuit since it is very similar to that of the above proof, but give here a clear sketch of its essential
structure. Let f : {0, 1}n × {0, 1}n →{0, 1}, and let N = 2n. We can represent a space s protocol
for f as 2s + 4 diﬀerent N × 2s binary matrices, with each of the ﬁrst 2s matrices telling us how
one of the two players will modify a certain cell of the shared memory as a function of its previous
state and their input, and the last four matrices determining whether a certain player will halt or
continue and the value they will output if they halt, again as a function of their input and the
shared memory contents. For s = ϵ log N (with ϵ < 1), we can then take these matrices as advice
to our circuit, for a total of ϵN1+ϵ log N bits of advice. We can then add circuitry to simulate
2s+1 = 2Nϵ steps of this protocol on a given input x, y ∈{0, 1}n × {0, 1}n, indexing into the
24


<!-- page 25 -->
advice matrices in order to determine what to do next using the same indexing constructions we
had for the proof of Theorem 12. Any matrix solvable by a space s protocol will be solved by a
protocol that halts after 2s+1 steps (otherwise it will loop forever), so simulating for 2s+1 steps
suﬃces. Since each of the indexing operations can be implemented using a number of gates linear
in ϵN1+ϵ log N using the constructions from the proof of Theorem 12, our overall circuit for f will
have size O(2NϵϵN1+ϵ log N) = O(N1+2ϵ log N). Thus, for any ϵ < 1
2, a truth table of length N2
requiring circuits of size N1+2ϵ will require δ log N-space protocols for any δ < ϵ.
5.3
Bit Probe Lower Bounds
In [9], Elias and Flower deﬁned a broad model for studying the space/query complexity of data
structure problems, known as the “bit-probe model.”
Deﬁnition 23. Given two sets D, Q, a function f : D × Q →{0, 1} and an integer b, the bit-probe
complexity of f for space b, denoted BCb(f), is the minimum over all encodings G : D →{0, 1}b of
the number of bits of G(x) that need to be probed in order to determine f(x, y) for the worst case
x ∈D, y ∈Q, given access to y.
In this general deﬁnition of a data structure problem, we think of D as the set of all possible
pieces of “data” we might wish to encode in our data structure, b as the number of bits we can
use to encode a piece of data, Q as the set of queries we wish to answer about an encoded piece
of data, and f as telling us the correct answers to all data/query pairs. BCb(f) then tells us the
minimum number of probes required by any space-b data structure in order to answer every query
correctly for every possible piece of data.
This model was investigated further by Miltersen [26], who showed, using a simple counting
argument, that most problems require an infeasible amount of space/probes in this model, but
pointed out that no explicit data structure problem is known to be infeasible in this sense. We thus
deﬁne the following explicit construction problem:
Deﬁnition 24. δ-Probe is the following search problem: given 1N where N = 2n output a truth
table f : {0, 1}n × {0, 1}n →{0, 1} such that BC2δn(f) > δn.
Theorem 14. For any δ < 2ϵ < 1, δ-Probe reduces in polynomial time to (1
2 + ϵ)-Hard.
Proof. Say f is a function f : D × Q →{0, 1} such that BCb(f) ≤k. So in particular there is
an encoding G : D →{0, 1}b such for any x ∈D, y ∈Q, given access to y we can determine
f(x, y) using at most k probes to G(x). Let H : Q →[b]k be the function which, given y, tells
us which positions of G(x) to query. Finally, let φ : {0, 1}k × Q →{0, 1} be the function which
determines f(x, y) given the results of the probes and the value of y. We will now give a compact
representation for f, and then use it to construct a circuit computing f.
Let R be a |D|×b binary matrix whose rows are indexed by elements of D, where Ri,j gives the
value of the jth bit of G(i). Let S be a |Q| × k⌈log b⌉matrix whose rows are indexed by elements
of Q, such that the ith row of S is H(i) ∈[b]k. Finally, let Z be a 2k × |Q| matrix such that
Zi,j = φ(i, j).
Now, given x ∈D, y ∈Q, we can use S, R, Z to determine f(x, y) as follows. First ﬁnd the yth
row of S to get H(y), which is a list of k indices in [b]. Next, ﬁnd the xth row of R, which is the
b-bit string G(x). Then, probe the indices speciﬁed by H(y) to get some k-bit string w, and ﬁnally
output Zw,y, which is precisely φ(w, y) = f(x, y).
As in the previous proofs, it can easily be shown that these indexing operations can be accom-
plished with circuits of size linear in S, R, Z. Therefore, f has a circuit of size O(|S| + |R| + |Z|) =
25


<!-- page 26 -->
O(|Q|k log b + |D|b + 2k|Q|).
So for any function D × Q →{0, 1}n requiring circuits of size
ω(|Q|k log b + |D|b + 2k|Q|), we must have BCb(f) > k.
In particular, if we take D = Q = {0, 1}n, and for any ﬁxed ϵ < 1 we take b = 2ϵn, k = ϵn, we
get that for any function f : {0, 1}2n →{0, 1} requiring circuits of size ω(2(1+ϵ)n), it must be the
case that BC2ϵn(f) > ϵn, since |Q|k log b + |D|b + 2k|Q| = 2nϵ2n2 + 2n2ϵn + 2ϵn2n = O(2(1+ϵ)n). So
if we take δ < ϵ, any truth table of length N2 with hardness N1+ϵ, or in other words any solution
to ( 1
2 + ϵ
2)-Hard on input 1N2, must satisfy BC2δn(f) > δn.
5.4
Some Concluding Thoughts
As noted in Section 3, Theorem 13 and Theorem 14 immediately imply that the problems δ-Space
and δ-Probe lie in APEPP, since we can construct truth tables with hardness
N
2 log N in APEPP
by Theorem 1. Although we phrase these results as reductions, they can also be interpreted as
giving conditional polynomial time constructions of various objects, under diﬀerent circuit lower
bound assumptions for the class E. In particular, Theorems 13 and 14 establish that if E contains
a language of circuit complexity 2( 1
2 +ϵ)n for some ϵ > 0, then polynomial time constructions of hard
problems in the bit-probe and space-bounded communication models follow. Theorem 12 establishes
that if E contains a language of circuit complexity Ω(2n
n ), then polynomial time constructions of
(Ω(n), Ω(n2))-rigid matrices over F2 follow; such matrices would be suﬃcient to carry out Valiant’s
lower bound program.
It would be interesting as well to ﬁnd some natural explicit construction problem for which a
reduction exists in the opposite direction, i.e. this problem is at least has hard as Hard Truth
Table or perhaps ϵ-Hard. Aside from Kn2
U -Random for which such a reduction is immediate,
we do not know of any other examples. However, we observe the following dichotomy: any explicit
construction problem either has a non-trivial algorithm, or is at least as diﬃcult as constructing a
somewhat hard truth table. More precisely:
Lemma 4. Let f : N →N be non-increasing, and let Π be any property (language) recognizable in
complexity class C. Let Π-Construction be the search problem: given 1n, output an n-bit string
with property Π. If Π-Construction is total for suﬃciently large n, then for suﬃciently large n
one of the following holds:
1. There is a TIME(2 ˜O(f(n))) algorithm using a C-oracle that solves Π-Construction on
length n inputs.
2. There is a polynomial time reduction from the problem of constructing an n-bit truth table
with circuit complexity f(n) to Π-Construction on length n inputs.
Proof. The proof is a straightforward application of the “easy witness” paradigm [20]. Let Πn be the
set of n-bit strings with property Π and let Hn
f be the set of n-bit strings with circuit complexity at
most f(n). If Πn∩Hn
f = ∅then any solution to the explicit construction problem for Π is necessarily
a string with circuit complexity exceeding f(n), and hence a polynomial time reduction from truth
table construction to Π-Construction trivially follows. On the other hand, if Πn ∩Hn
f ̸= ∅, we
can search over Hn
f for a solution to Π-Construction and will be guaranteed to ﬁnd a solution.
Since |Hn
f | = 2 ˜O(f(n)), we can then solve Π-Construction by iterating over Hn
f and using a
C-oracle to test if each potential solution indeed holds property Π, in TIME(2 ˜O(f(n))).
Setting f(n) = nϵ and combining this with Theorem 7, we conclude:
26


<!-- page 27 -->
Theorem 15. Let EC be any explicit construction problem in APEPP (more formally, any search
problem in STFΣP
2 ∩APEPP). Then one of the following holds:
1. EC is APEPP-complete under PNP reductions.
2. For every ϵ > 0, EC can be solved in time 2nϵ with an NP oracle, for inﬁnitely many n.
6
Open Problems
The most signiﬁcant question left open in this work is whether ϵ-Hard is complete for APEPP
under polynomial time reductions. One way to demonstrate evidence against this possibility would
be to show hardness of Empty, perhaps under cryptographic assumptions, since it is widely con-
jectured that ϵ-Hard does have a polynomial time algorithm for some ϵ > 0 (this is often cited as
the primary reason for believing P = BPP [17]). More generally, if any sparse search problem is
APEPP complete under P reductions, then APEPP lies in FP/poly, since we can hard-code all
solutions of a ﬁxed polynomial length for the complete sparse problem. It should be noted that the
complexity of the dual version of Empty, known as WeakPigeon, is equivalent to the worst-case
complexity of breaking collision-resistant hash functions (in WeakPigeon we are given a circuit
C : {0, 1}n →{0, 1}m with m < n, and asked to ﬁnd a collision [19]). A hardness result for Empty
would be interesting in another respect as well: just as the Natural Proofs barrier [31] shows that
a generic method of proving circuit lower bounds via a “Natural Property” would require solving a
hard computational problem, hardness of Empty would show that we should not expect to prove
exponential lower bounds for E via an eﬃcient algorithm that ﬁnds an empty pigeonhole for an
arbitrary function; something about the speciﬁc function mapping circuits to their truth tables
would have to be utilized.
7
Acknowledgements
The author would like to thank Christos Papadimitriou for his guidance and for many inspiring
discussions throughout the completion of this work, and Mihalis Yannakakis for his comments on
an early draft of this manuscript. The author would also like to thank the anonymous referees
for suggesting various improvements to this paper, in particular the addition of Corollary 3, the
connection to hardness extractors and the GGM generator, and the simpliﬁcation of Lemma 2.
References
[1] J. Alman and L. Chen, Eﬃcient construction of rigid matrices using an NP oracle, in 2019 IEEE 60th Annual
Symposium on Foundations of Computer Science (FOCS), 2019, pp. 1034–1055.
[2] L. Babai, P. Frankl, and J. Simon, Complexity classes in communication complexity theory, in 27th Annual
Symposium on Foundations of Computer Science (sfcs 1986), 1986, pp. 337–347.
[3] B. Barak, A. Rao, R. Shaltiel, and A. Wigderson, 2-source dispersers for sub-polynomial entropy and
ramsey graphs beating the frankl-wilson construction, in Proceedings of the Thirty-Eighth Annual ACM Sympo-
sium on Theory of Computing, STOC ’06, New York, NY, USA, 2006, Association for Computing Machinery,
p. 671–680.
[4] A. Bhangale, P. Harsha, O. Paradise, and A. Tal, Rigid matrices from rectangular PCPs, 2020.
[5] J. Buresh-Oppenheim and R. Santhanam, Making hard problems harder, 21st Annual IEEE Conference on
Computational Complexity (CCC’06), (2006), pp. 15 pp.–87.
[6] E. Chattopadhyay, Guest column: A recipe for constructing two-source extractors, SIGACT News, 51 (2020),
p. 38–57.
27


<!-- page 28 -->
[7] B. Chor and O. Goldreich, Unbiased bits from sources of weak randomness and probabilistic communication
complexity, in 26th Annual Symposium on Foundations of Computer Science (sfcs 1985), 1985, pp. 429–442.
[8] Z. Dvir, A. Golovnev, and O. Weinstein, Static data structure lower bounds imply rigidity, in Proceedings
of the 51st Annual ACM SIGACT Symposium on Theory of Computing, STOC 2019, New York, NY, USA,
2019, Association for Computing Machinery, p. 967–978.
[9] P. Elias and R. A. Flower, The complexity of some simple retrieval problems, J. ACM, 22 (1975), p. 367–379.
[10] P. Erd¨os, Some remarks on the theory of graphs, Bulletin of the American Mathematical Society, 53 (1947),
pp. 292–294.
[11] J. Friedman, A note on matrix rigidity, Combinatorica, 13 (1993), pp. 235–239.
[12] E. Gat and S. Goldwasser, Probabilistic search algorithms with unique answers and their cryptographic ap-
plications, Electronic Colloquium on Computational Complexity (ECCC), 18 (2011), p. 136.
[13] O. Goldreich, S. Goldwasser, and S. Micali, How to construct random functions, J. ACM, 33 (1986),
p. 792–807.
[14] A. Golovnev, R. Ilango, R. Impagliazzo, V. Kabanets, A. Kolokolova, and A. Tal, Ac0[p] lower
bounds against mcsp via the coin problem, Electron. Colloquium Comput. Complex., 26 (2019), p. 18.
[15] J. M. Hitchcock and A. Pavan, On the np-completeness of the minimum circuit size problem, in FSTTCS,
2015.
[16] R. Ilango, Approaching MCSP from Above and Below: Hardness for a Conditional Variant and AC0[p], in
11th Innovations in Theoretical Computer Science Conference (ITCS 2020), T. Vidick, ed., vol. 151 of Leibniz
International Proceedings in Informatics (LIPIcs), Dagstuhl, Germany, 2020, Schloss Dagstuhl–Leibniz-Zentrum
fuer Informatik, pp. 34:1–34:26.
[17] R. Impagliazzo and A. Wigderson, P = BPP if E requires exponential circuits: Derandomizing the XOR
lemma, in Proceedings of the Twenty-Ninth Annual ACM Symposium on Theory of Computing, STOC ’97, New
York, NY, USA, 1997, Association for Computing Machinery, p. 220–229.
[18] E. Jeˇr´abek, Dual weak pigeonhole principle, boolean complexity, and derandomization, Annals of Pure and
Applied Logic, 129 (2004), pp. 1–37.
[19]
, Integer factoring and modular square roots, Journal of Computer and System Sciences, 82 (2016), pp. 380–
394.
[20] V. Kabanets, Easiness assumptions and hardness tests: Trading time for zero error, Journal of Computer and
System Sciences, 63 (2001), pp. 236–252.
[21] V. Kabanets and J.-Y. Cai, Circuit minimization problem, in Proceedings of the Thirty-Second Annual
ACM Symposium on Theory of Computing, STOC ’00, New York, NY, USA, 2000, Association for Computing
Machinery, p. 73–79.
[22] R. Kleinberg, O. Korten, D. Mitropolsky, and C. Papadimitriou, Total Functions in the Polynomial Hi-
erarchy, in 12th Innovations in Theoretical Computer Science Conference (ITCS 2021), J. R. Lee, ed., vol. 185 of
Leibniz International Proceedings in Informatics (LIPIcs), Dagstuhl, Germany, 2021, Schloss Dagstuhl–Leibniz-
Zentrum f¨ur Informatik, pp. 44:1–44:18.
[23] A. R. Klivans and D. van Melkebeek, Graph nonisomorphism has subexponential size proofs unless the
polynomial-time hierarchy collapses, SIAM Journal on Computing, 31 (2002), pp. 1501–1526.
[24] X. Li, Improved non-malleable extractors, non-malleable codes and independent source extractors, in Proceedings
of the 49th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2017, New York, NY, USA,
2017, Association for Computing Machinery, p. 1144–1156.
[25] P. Miltersen and N. Vinodchandran, Derandomizing arthur-merlin games using hitting sets, in 40th Annual
Symposium on Foundations of Computer Science (Cat. No.99CB37039), 1999, pp. 71–80.
[26] P. B. Miltersen, The bit probe complexity measure revisited, in STACS 93, P. Enjalbert, A. Finkel, and K. W.
Wagner, eds., Berlin, Heidelberg, 1993, Springer Berlin Heidelberg, pp. 662–671.
[27] N. Nisan and A. Wigderson, Hardness vs randomness, Journal of Computer and System Sciences, 49 (1994),
pp. 149–167.
[28] I. C. Oliveira and R. Santhanam, Pseudodeterministic constructions in subexponential time, in Proceedings
of the 49th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2017, New York, NY, USA,
2017, Association for Computing Machinery, p. 665–677.
[29] C. H. Papadimitriou, On the complexity of the parity argument and other ineﬃcient proofs of existence, Journal
of Computer and System Sciences, 48 (1994), pp. 498 – 532.
28


<!-- page 29 -->
[30] J. Paris, A. Wilkie, and A. R. Woods, Provability of the pigeonhole principle and the existence of inﬁnitely
many primes, J. Symb. Log., 53 (1988), pp. 1235–1244.
[31] A. A. Razborov and S. Rudich, Natural proofs, Journal of Computer and System Sciences, 55 (1997), pp. 24–
35.
[32] R. Santhanam, The complexity of explicit constructions, Theory of Computing Systems, 51 (2012), pp. 297–
312. Copyright - Springer Science+Business Media, LLC 2012; Document feature - ; Equations; Last updated -
2020-11-18; CODEN - TCSYFI.
[33]
, Pseudorandomness and the minimum circuit size problem, in 11th Innovations in Theoretical Computer
Science Conference, ITCS 2020, January 12-14, 2020, Seattle, Washington, USA, T. Vidick, ed., vol. 151 of
LIPIcs, Schloss Dagstuhl - Leibniz-Zentrum f¨ur Informatik, 2020, pp. 68:1–68:26.
[34] C. E. Shannon, The synthesis of two-terminal switching circuits, The Bell System Technical Journal, 28 (1949),
pp. 59–98.
[35] M. A. Shokrollahi, D. Spielman, and V. Stemann, A remark on matrix rigidity, Information Processing
Letters, 64 (1997), pp. 283–285.
[36] H. Song, Space-bounded Communication Complexity, PhD thesis, Tsinghua University, 2014.
[37] S. P. Vadhan, Pseudorandomness, vol. 7, Now Delft, 2012.
[38] L. G. Valiant, Graph-theoretic arguments in low-level complexity, in Mathematical Foundations of Computer
Science 1977, J. Gruska, ed., Berlin, Heidelberg, 1977, Springer Berlin Heidelberg, pp. 162–176.
[39] R. Williams, Improving exhaustive search implies superpolynomial lower bounds, in Proceedings of the Forty-
Second ACM Symposium on Theory of Computing, STOC ’10, New York, NY, USA, 2010, Association for
Computing Machinery, p. 231–240.
[40] A. C. Yao, Theory and application of trapdoor functions, in 23rd Annual Symposium on Foundations of Com-
puter Science (sfcs 1982), 1982, pp. 80–91.
29