<!-- page 1 -->
MANTRA: Temporal Betweenness Centrality
Approximation through Sampling
Antonio Cruciani
Gran Sasso Science Institute, L’Aquila, Italy
antonio.cruciani@gssi.it
Abstract. We present MANTRA, a framework for approximating the
temporal betweenness centrality of all nodes in a temporal graph. Our
method can compute probabilistically guaranteed high-quality tempo-
ral betweenness estimates (of nodes and temporal edges) under all the
feasible temporal path optimalities, presented in the work of Buß et al.
(KDD, 2020). We provide a sample-complexity analysis of our method
and speed up the temporal betweenness computation using a state-of-
the-art progressive sampling approach based on Monte Carlo Empirical
Rademacher Averages. Additionally, we provide an efficient sampling
algorithm to approximate the temporal diameter, average path length,
and other fundamental temporal graph characteristic quantities within
a small error ε with high probability. The running time of such approx-
imation algorithm is ˜O( log n
ε2
· |E|), where n is the number of nodes and
|E| is the number of temporal edges in the temporal graph. We support
our theoretical results with an extensive experimental analysis on several
real-world networks and provide empirical evidence that the MANTRA
framework improves the current state of the art in speed, sample size,
and required space while maintaining high accuracy of the temporal be-
tweenness estimates.
1
Introduction
Centrality measures are fundamental notions for evaluating the importance of
nodes in networks, used in network analysis and graph theory. A centrality mea-
sure assigns real values to all the nodes, in such a way that the values are
monotonously dependent of the nodes’ importance, i.e., more important nodes
should have higher centrality scores. Computing the betweenness centrality is
arguably one of the most important tasks in graph mining and network analysis.
It finds application in several fields including social network analysis [47], rout-
ing [18], machine learning [46], and neuroscience [20]. The betweenness of a node
in a graph indicates how often this node is visited by a shortest path. High be-
tweenness nodes are usually considered to be important in the network. Brandes’
algorithm [10], is the best algorithm to compute the exact centrality scores of
every node in O(n·m) time and O(n+m) space where n and m are the number
of nodes, and edges of a given graph G = (V, E), respectively. Unfortunately,
this algorithm quickly becomes impractical on nowadays’ networks with billions
arXiv:2304.08356v4  [cs.DS]  7 Apr 2024


<!-- page 2 -->
2
Antonio Cruciani
of nodes and edges. Moreover, there is a theoretical evidence, in form of several
conditional lower bounds results [1], for believing that a faster algorithm cannot
exists, even for approximately computing the betweenness. A further challenge, is
that modern real-world networks are also dynamic or temporal, i.e., they change
over time. Temporal networks can be informally described as edge-labeled graphs
in which each label indicates the time instant in which the underlying edge ap-
pears in the network. A great variety of both modern and traditional networks
can be naturally modeled as temporal graphs. Furthermore, there are numer-
ous real-world applications for which studying temporal networks offers unique
perspectives. This is especially evident when examining data that evolves over
time, for example social networks interactions, information/infection spreading,
subgraph patterns, detecting communities, and clustering networks. In the con-
text of these challenges it is, thus, essential to consider temporal variants of the
most important centrality notions, alongside algorithms for computing them,
that have an excellent scaling behavior. In this work, we focus on the temporal
version of the betweenness centrality, that similarly to static networks, it seeks to
pinpoint nodes that are traversed by a significant number of optimal (temporal)
paths. Buß et al. [13,43] gave several definitions of the temporal betweenness as
a temporal counterpart of the betweenness centrality, characterized their com-
putational complexity, and provided polynomial time algorithms to compute
these temporal centrality measures. However, these algorithms turn out to be
impractical, even for medium size networks. Thus, it is reasonable to consider
approximation algorithms that can efficiently compute the centrality values of
the nodes up to some small error. In this work, we follow the approach of Santoro
et al. [44], and we provide a set of approximation algorithms for all the temporal
betweenness variants in [13].
Contributions. We propose MANTRA (teMporAl betweeNnes cenTrality thRough
sAmpling), a rigorous framework for the approximation of the temporal be-
tweenness of all the vertices and temporal edges in large temporal graphs. In
particular, we present the following results: (1) We extend the state-of-the-art
estimator [44] to all the feasible temporal betweenness centrality variants for
nodes and temporal edges (Section 4.1). In addition, we propose two alterna-
tive unbiased estimators for such centrality measure on temporal graphs1; (2)
We derive new bounds on the sufficient number of samples to approximate the
temporal betweenness centrality for all nodes2 (Section 4.2), that are governed
by three key quantities of the temporal graph, such as the temporal vertex diam-
eter, average temporal path length, and the maximum variance of the temporal
betweenness centrality estimators. Moreover, this result solves an open problem
in [37,35] on whether the sample complexity bounds for the static betweenness
can be efficiently extended to temporal graphs. As a consequence, it signif-
icantly improves on the state-of-the-art results for the temporal betweenness
1 Due to space constraints, we refer to the additional materials for the temporal be-
tweenness on temporal edges and for the definition of the other estimators.
2 The sample complexity analysis holds also for the temporal edges.


<!-- page 3 -->
Temporal Betweenness Centrality Approximation through Sampling
3
estimation process [44]. Additionally, our analysis of sample complexity presents
further challenges regarding the efficient computation of the three quantities
upon which the bounds for the necessary sample size depend; (3) We propose
a novel algorithm to efficiently estimate the key quantities of interests in (2)
that uses a mixed approach based on sampling and counting (Section 4.3). The
time complexity of our approach is ˜O( log n
ε2 |E|), while the space complexity is
O(n + |E|). We provide an estimate on the sample size needed to achieve a good
estimates up to a small error bound. More precisely, we prove that r = Θ( log n
ε2 )
sample nodes are sufficient to estimate, with probability at least 1−1/n2: (i) the
temporal diameter D(⋆) with error bounded by
ε
ζ(⋆) ; (ii) the average temporal
path length ρ(⋆) with error bounded by ε D(⋆)
ζ(⋆) ; and, (iii) the temporal connec-
tivity rate ζ(⋆) (see Section 4.3 for the formal definition) with error bounded
by ε; (4) We define MANTRA, a progressive sampling algorithm that uses an
advanced tool from statistical learning theory, namely Monte Carlo Empirical
Rademacher Averages [4] and the above results (e.g. (1-3)) to provide a high
quality approximation of the temporal betweenness (Section 4.4). MANTRA’s
output is a function of two parameters: ε ∈(0, 1) controlling the approximation’s
accuracy, and δ ∈(0, 1) controlling the confidence of the computed approxima-
tion. Our novel approach improves on ONBRA [44] (i.e., the state-of-the-art
algorithm) in terms of running time, sample size, and allocated space; and, (5)
We support our theoretical analysis with an extensive experimental evaluation
(Section 5), in which we compare MANTRA with ONBRA.
2
Related Work
The literature on betweenness centrality being vast, we restrict our attention to
approaches that are closest to ours. Thus, we focus on betweenness centrality
on temporal graphs. Tsalouchidou et al. [48], extended the well-known Brandes
algorithm [10] to allow for distributed computation of betweenness in tempo-
ral graphs. Specifically, they studied shortest-fastest paths, considering the bi-
objective of shortest length and shortest duration. Buß et al. [13,43] analysed
the temporal betweenness centrality considering several temporal path optimal-
ity criteria, such as shortest (foremost), foremost, fastest, and prefix-foremost,
along with their computational complexities. They showed that, when consid-
ering paths with increasing time labels, the foremost and fastest temporal be-
tweenness variants are #P-hard, while the shortest and shortest foremost ones
can be computed in O(n3 ·|T|2), and the prefix-foremost one in O(n·|E|·log |E|).
Here E is the set of temporal edges, and T is the set of unique time stamps.
Santoro et al. [44] provided ONBRA, the first sampling-based approximation
algorithm for one variant of the temporal betweenness centrality. The input to
ONBRA is a temporal graph, a confidence value δ ∈(0, 1), and the sample size r.
The algorithm performs a set of r truncated temporal breadth first searches be-
tween couples of nodes sampled uniformly at random and estimates the shortest
temporal betweenness using the temporal equivalent of the ABRA estimator [41]
for static networks. ONBRA’s output is a function of the confidence δ ∈(0, 1)


<!-- page 4 -->
4
Antonio Cruciani
and the upper bound on the approximation accuracy ξ ∈(0, 1) computed using
the Empirical Bernstein Bound [28]. More precisely, with probability 1 −δ, the
approximation computed by ONBRA is guaranteed to have absolute error of
at most ξ for each node in the temporal graph. Finally, Becker et al. [5], pro-
vided an efficient heuristic to approximate the temporal betweenness rankings
by considering the temporal interactions among the 1-hop neighborhood of the
nodes.
3
Preliminaries
Temporal Graphs, and Paths. A directed temporal graph is an ordered tuple
G = (V, E) where E = {(u, v, t) : u, v, ∈V ∧t ∈T ⊆N} is the set of temporal
edges3. Given two nodes s and z, a temporal path tpsz ⊆V × T is a (unique) se-
quence of time-respecting vertex appearances tpsz = ((u1, t1), (u2, t2), . . . , (uk, tk))
such that u1 = s, uk = z, and ti < ti+1 for all 1 ≤i ≤k −1. The vertex appear-
ances (u1, t1) and (uk, tk) are called endpoints of tpsz and the temporal nodes
in Int(tpsz) = tpsz \ {(u1, t1), (uk, tk)} are called internal vertex appearances of
tpsz. Unlike paths on static graphs, in the temporal setting there are several con-
cepts of optimal paths (e.g., shortest, foremost, fastest) [12,13,43]. Moreover, as
for the static betweenness, the task of computing the desired centrality measure
boils down to the ability of efficiently counting the overall number of optimal
paths. Unfortunately, it has been already shown that such task turns out to
be #P-Hard for some temporal path optimalities (e.g. foremost, fastest) [13,43].
Hope is left for the shortest (and all its variants) an the prefix foremost temporal
paths. We formally describe those that can be efficiently counted.
Definition 1. Given a temporal graph G, and two nodes s, z ∈V . Let tpsz be
a temporal path from s to z, then tpsz is said to be: (1) Shortest (sh) if there
is no tp′
sz such that |tp′
sz| < |tpsz|; (2) Shortest-Foremost (sfm) if there is no
tp′
sz that has an earlier arrival time in z than tpsz and has minimum length in
terms of number of hops from s to z; and, (3) Prefix-Foremost (pfm) if tpsz is
foremost and every prefix tpsv of tpsz is foremost as well.
To denote the different type of temporal paths we use the same notation of
Buß et al. [13]. More precisely, we use the term “(⋆)-optimal” temporal path,
where (⋆) denotes the type. Furthermore, we denote the set of all (⋆)-temporal
paths between two nodes s and z as Γ (⋆)
sz
and we let TP(⋆)
G
to be the union of
all the Γ (⋆)
sz ’s, for all pairs (s, z) ∈V × V of distinct nodes. In this work, we
will heavily rely on two temporal graphs characteristic quantities, namely the
temporal (vertex) diameter and the average temporal path length. Formally, given
a temporal graph G = (V, E) we define the (⋆)-temporal diameter D(⋆) and the
3 The value T denotes the life-time of the temporal graph, and, without loss of gen-
erality for our purposes, we assume that, for any t ∈T, there exists at least one
temporal arc at that time and without loss of generality we assume T = [1, |T|].


<!-- page 5 -->
Temporal Betweenness Centrality Approximation through Sampling
5
(⋆)-temporal vertex diameter V D(⋆) as the number of temporal edges and nodes
in the longest (⋆)-optimal path in G, i.e.,
D(⋆) = max
n
|tp(⋆)| : tp(⋆) ∈TP(⋆)
G
o
and V D(⋆) = D(⋆) + 1
respectively. Finally, we refer to the average (⋆)-temporal path length ρ(⋆) as the
average number of internal nodes in a (⋆)-temporal path, i.e.,
ρ(⋆) =
1
n(n −1)
X
s,z∈V
|Int(tpsz)|
Temporal Betweenness Centrality. As previously shown, on temporal graphs,
there are several notions of optimal paths. Hence, we have different notions of
temporal betweenness centrality [13] as well. Formally, let G = (V, E) be a tem-
poral graph. For any pair (s, z) of distinct nodes (s ̸= z), let σ(⋆)
sz be the number
of (⋆)-temporal paths between s and z, and let σ(⋆)
sz (v) be the number of the
(⋆)-temporal paths between s and z that pass through node v, with s ̸= v ̸= z.
The normalized temporal betweenness centrality b(⋆)
v
of a node v ∈V is defined
as
b(⋆)
v
=
1
n(n −1)
X
s̸=v̸=z
σ(⋆)
sz (v)
σ(⋆)
sz
We refer to the additional materials for the definition of the (⋆)-temporal be-
tweenness of the temporal edges. Whenever we use the term (⋆)-temporal paths
we consider (⋆) to be one of the optimality criteria in Definition 1. We observe
that the average (⋆)-temporal path length is equal to the sum of the (⋆)-temporal
betweenness centrality over all nodes v ∈V .
Lemma 1. ρ(⋆) = P
v∈V b(⋆)
v
Supremum Deviation and Empirical Rademacher Averages. Here we
define the Supremum Deviation (SD) and the c-samples Monte Carlo Empirical
Rademacher Average (c-MCERA). For more details about the topic we refer to
the book [45] and to [4]. Let D be a finite domain and consider a probability
distribution π over the elements of D. Let F be a family of functions from D
to [0, 1], and S = {s1, . . . , sr} be a collection of r independent and identically
distributed samples from D sampled according to π. The SD is defined as:
SD(F, S) = sup
f∈F
      
1
r
X
i∈[r]
f(si) −Eπ [f]
      
The SD is the key concept of the study of empirical processes [38]. One way to
derive probabilistic upper bounds to the SD is to use the Empirical Rademacher
Averages (ERA) [22]. In this work we use the state-of-the-art approach to obtain


<!-- page 6 -->
6
Antonio Cruciani
sharp probabilistic bounds on the ERA that uses Monte-Carlo estimation [4].
Consider a sample S = {s1, . . . sr}, for c ≥1 let λ ∈{−1, 1}c×r be a c×r matrix
of i.i.d. Rademacher random variables. The c-MCERA of F on S using λ is:
Rc
r(F, S, λ) = 1
c
c
X
j=1
sup
f∈F
1
r
r
X
i=1
λj,if(si)
The c-MCERA allows to obtain sharp data-dependent probabilistic upper bounds
on the SD, as they directly estimate the expected SD of sets of functions by tak-
ing into account their correlation. Moreover, they are often significantly more
accurate than other methods [37,36,35], such as the ones based on loose deter-
ministic upper bounds to ERA [41], distribution-free notions of complexity such
as the Hoeffding’s bound or the VC-Dimension, or other results on the vari-
ance [28,44]. Moreover, a key quantity governing the accuracy of the c-MCERA
is the empirical wimpy variance [8] WF(S), that for a sample of size r is defined
as WF(S) = supf∈F
1
r
P
i∈[r](f(si))2. We are ready to state the technical result
of this section (proof deferred to the additional materials).
Theorem 1. Let F be a family of functions with codomain in [0, 1], and let
S be a sample of r random samples from a distribution π. Denote ˆv such that
supf∈F Var(f) ≤ˆv. For any δ ∈(0, 1), define
R(F, S) ≤˜R = Rc
r(F, S, λ) +
r
4WF(S) ln(4/δ)
cr
(1)
R = ˜R + ln(4/δ)
r
+
s ln(4/δ)
r
 2
+ 2 ln(4/δ) ˜R
r
ξ = 2R +
r
2 ln(4/δ) (ˆv + 4R)
r
+ ln(4/δ)
3r
(2)
With probability at least 1−δ over the choice of S and λ, it holds SD(F, S) ≤ξ.
4
MANTRA: temporal Betweenness Centrality
Approximation through Sampling
4.1
Temporal Betweenness Estimator
In this section we present one unbiased estimator for the (⋆)-temporal between-
ness centrality, and we refer to the additional materials for the remaining esti-
mators that have been excluded due to space constraints. The ONBRA (ob)
algorithm [44] uses an estimator defined over the sampling space Dob = {(s, z) ∈
V × V : s ̸= z} with uniform sampling distribution πob over Dob, and family
of functions Fob that contains one function eb(⋆)
ob (v) →[0, 1] for each vertex v,
defined as eb(⋆)
ob (v|s, z) = σ(⋆)
sz (v)/σ(⋆)
sz ∈[0, 1]. So far, this approach has been de-
fined only for the shortest-temporal betweenness. In this work, we extend ob to
shortest-foremost and prefix foremost temporal paths.


<!-- page 7 -->
Temporal Betweenness Centrality Approximation through Sampling
7
4.2
Sample Complexity bounds
We present two bounds (Theorem 2 and Theorem 3) to the sufficient number of
random samples to obtain an ε approximation of the (⋆)-temporal betweenness
centrality. Given a temporal graph G = (V, E), with a straightforward appli-
cation of Hoeffding’s inequality and union bound [31], it can be shown that
r = 1/(2ε2) log (2n/δ) samples suffice to estimate the (⋆)-temporal betweenness
of every node up to an additive error ε with probability 1 −δ. To improve this
bound, we define the range space associated to the (⋆)-temporal betweeenness
and its VC-dimension, and remand to [31,29,45] for a more complete introduc-
tion to the topic. Let U = TP(⋆)
G , define the range space R = (D, F+) where
D = U × [0, 1], and F+ is defined as follows: for a pair (s, z) ∈V × V and a
temporal path tpsz ∈U let f(v,t)(tpsz) = f((v, t)|s, z) = 1[(v, t) ∈Int(tpsz)] be
the function that assumes value 1 if the vertex appearance (v, t) is in the tempo-
ral path between s and z. Moreover, define the family of functions F = {f(v,t) :
(v, t) ∈V × T} and notice that for each f(v,t) ∈F there is a range Rf(v,t) =
{(tpsz, α) : tpsz ∈U ∧α ≤f(v,t)(tpsz)}. Next, define F+ = {Rf(v,t) : f(v,t) ∈F}.
Now that we defined the range set for our problem, we can give an upper bound
on its VC-dimension.
Lemma 2. The VC-dimension of the range space R is V C(R) ≤⌊log VD(⋆) −
2⌋+ 1.
Given the VC-dimension of the range set R we have:
Theorem 2 (See [26], Section 1). Given ε, δ ∈(0, 1), and a universal con-
stant c, let S ⊆D be a collection of elements sampled w.r.t. a probability distri-
bution π. Then
c
ε2
 
V C(R) + ln
  1
δ
  
samples suffice to obtain SD(F+, S) ≤ε
with probability 1 −δ over S.
To improve this bound, we make use of Lemma 1 and notice that (as for the static
case [37]) the (⋆)-temporal betweenness centrality satisfies a form of negative
correlation among the nodes. Moreover, the existence of a node v with high (⋆)-
temporal betweenness constraints the sum of the centrality measure over the
remaining nodes to be at most ρ(⋆) −b(⋆)
v . In other words, this suggests that
the number of nodes with high (⋆)-temporal betweenness cannot be arbitrarily
large. Furthermore, as in [37], we assume that the maximum variance of the
(⋆)-temporal betweenness estimators eb
(⋆)
v
is bounded by some estimate ˆv rather
than the worst-case upper bound of 1/4 considered in [7]. This implies that the
estimates eb
(⋆)
v
are not bounded by the number of nodes in the temporal graph
G, but are tightly constrained by the parameters ρ(⋆) and ˆv. We are able to
extend the results in [37] for the static scenario to the temporal setting for all
the variants of temporal betweenness that can be computed in polynomial time
and cover one of the problems left open by the authors. It follows that:
Theorem 3. Let F = {eb
(⋆)
v , v ∈V } be a set of function from a domain D to
[0, 1]. Define ˆv ∈(0, 1/4] and ρ(⋆) ≥0 such that maxv∈V Var(eb
(⋆)
v ) ≤ˆv and


<!-- page 8 -->
8
Antonio Cruciani
P
v∈V b(⋆)
v
≤ρ(⋆). Fix ε, δ ∈(0, 1), and let S be an i.i.d. sample taken from D of
size |S| ∈O
 
ˆv+ε
ε2 ln
 
ρ(⋆)
δˆv
  
. It holds that SD(F, S) ≤ε with probability 1 −δ
over S.
Since ρ(⋆) correspond to the average number of internal nodes in (⋆)-temporal
paths in G, it must be that ρ(⋆) ≤D(⋆). In all the analyzed networks (see Figure 1
in Section 5) this condition holds, thus this approach will need a smaller sample
size compared to the VC-Dimension based one to obtain an ε-approximation of
the (⋆)-temporal betweenness.
4.3
Fast approximation of the characteristic quantities
According to Theorem 2 and Theorem 3, the sample size needed to achieve
a desired approximation depends on the vertex diameter and on the average
temporal path length of the temporal graph. However, under the Strong Expo-
nential Time Hypothesis (SETH), the (⋆)-temporal diameter (thus the average
(⋆)-temporal path length) of a temporal graph G = (V, E) can not be computed
in ˜O(|E|2−ε)4 [14], which can be prohibitive for very large temporal graphs, so
efficient approximation algorithms for these characteristic quantities are highly
desirable. Some algorithms for the diameter approximation on temporal graphs
have been proposed [16,14]. However these techniques consider different tempo-
ral path optimality criteria [16], or have no theoretical guarantees [14]. In this
work we define a novel sampling-based approximation algorithm to efficiently
obtain a high-quality approximation of D(⋆) (thus, V D(⋆)) and ρ(⋆) in ˜O(r · |E|)
where r is the number of samples used by the algorithm. We provide a high level
description of the sampling algorithm and we refer to the additional materials
for a detailed discussion and analysis of the method. Given a temporal graph
G, the sample size r, and the temporal path optimality (⋆), the algorithm per-
forms r (⋆)-temporal BFS visits [49] ((⋆)-TBFS) from r random nodes and keeps
track of the number of reachable pairs encountered at each hop along with the
greatest hop performed. Once all the r visits have been completed, it computes
the temporal diameter and other useful temporal measures using an approach
based on the relation between the number of reachable pairs and the distance
metrics [2]. The approximation guarantees of our sampling algorithm strongly
depends on “how temporally connected” a temporal graph is. To this end, we
define the (⋆)-temporal connectivity rate as the ratio of the number of couples
that are temporally connected by a (⋆)-temporal path and all the possible reach-
able couples. Formally, let 1[u ⇝v] be the indicator function that assumes value
1 if u can reach v via a (⋆)-temporal path, then the temporal connectivity rate is
defined as the ratio between the number of reachable pairs and all the possible
ones in the temporal graph, i.e., ζ(⋆) =
1
n(n−1)
P
u̸=v 1[u ⇝v] ∈[0, 1]. Intu-
itively, the higher the connectivity rate the higher the number of couples that
are connected via at least one (⋆)-temporal path. Moreover, the algorithm has
the following theoretical guarantees:
4 With the notation
˜
O(·) we ignore logarithmic factors.


<!-- page 9 -->
Temporal Betweenness Centrality Approximation through Sampling
9
Theorem 4. Given a temporal graph G = (V, E) and a sample of size r =
Θ
  ln n
ε2
 
, the algorithm computes with probability 1 −
2
n2 : the (⋆)-temporal di-
ameter D(⋆) with absolute error bounded by
ε
ζ(⋆) , the average (⋆)-temporal path
length ρ(⋆) with absolute error bounded by ε·D(⋆)
ζ(⋆) , and the temporal connectivity
rate with absolute error bounded by ε.
4.4
The MANTRA Framework
In this section we introduce MANTRA5, our algorithmic framework for the (⋆)-
temporal betweenness centrality estimation. MANTRA incorporates the bounds
in Section 4.2 to compute an upper bound on the minimum sample size needed
to approximate the SD of the (⋆)-temporal betweenness and a state-of-the-art
progressive sampling technique to speed-up the estimation process. The input
parameters to MANTRA are: a temporal graph G, a temporal path optimality
(⋆) ∈{sh, sfm, pfm}6, a target precision ε ∈(0, 1), a failure probability δ ∈(0, 1),
and a number of iterations for the bootstrap phase s′. The output is a vector
B of pairs (v, eb
(⋆)
v ) for each v ∈V , where eb
(⋆)
v
is the estimate of b(⋆)
v
and B is
probabilistically guaranteed to be an absolute ε approximation of the temporal
betweenness. Formally:
Theorem 5. Given a target accuracy ε ∈(0, 1) and a failure probability δ ∈
(0, 1), with probability at least 1 −δ (over the runs of the algorithm), the output
vector B = {eb
(⋆)
v
: v ∈V } (obtained from a set of samples S) produced by
MANTRA is such that SD(B, S) ≤ε.
Algorithm 1’s execution is divided in two phases: the bootstrap phase (lines 3-
4) and the estimation phase (lines 6-15). As a first step, MANTRA, computes an
upper bound ω to the number of samples needed to achieve an ε approximation
(line 2). The procedure runs s′ independent (⋆)-TBFS visits from s′7 random
couples of nodes (s, z) sampled from the population Dob, estimates ˆv and ρ(⋆),
and then plugs them in Theorem 3 to obtain ω. Subsequently, it infers the first
element of the sample size {si}i≥1 by performing a binary search between s′ and
ω to find the minimum s1 such that Equation 2 (with R set to 0) is at most
ε and terminates the bootstrap phase. Such approach gives an optimistic first
guess of the number of samples to process for obtaining an ε-approximation [37].
Subsequently it continues with the estimation phase in which, at each iteration,
it increases each si with a geometric progression [39], i.e., such that si = 1.2·si−1.
Next, it proceeds by drawing uniformly at random k = si−si−1 couples of nodes
(s, z) from Dob and subsequently updating the overall set of samples sampled
so far (lines 7-9). Consequently, k new Rademacher random vectors are added
as new columns to the matrix λ and k (⋆)-TBFS visits are performed (line 11).
5 teMporAl betweeNness cenTrality appRoximation through sAmpling
6 We point out that our approach is general, and can be extended to every definition
of temporal betweenness centrality.
7 In this work we use s′ = log(1/δ)/ε.


<!-- page 10 -->
10
Antonio Cruciani
Algorithm 1: MANTRA
Data: Temporal graph G, (⋆) temporal path optimality, precision ε ∈(0, 1),
failure probability δ ∈(0, 1), bootstrap iterations s′, and number of
Monte Carlo trials c.
Result: Absolute ε-approximation of the (⋆)-tbc w.p. of at least 1 −δ.
1 B, W = [0, . . . , 0] ∈Rn
// tbc and wimpy variance arrays
2 i, k = 0; ξ = 1; S0 = {∅}
3 ω, ˆv = DrawSufficientSampleSize(G, s′, δ/2)
4 {si}i≥1 = SamplingSchedule(ω, ˆv, δ)
5 λ = [[·]]
// Empty matrix
6 while true do
7
i = i + 1; k = (1.2 · si−1) −si−1
8
X = DrawSamples(G, k)// Draw k samples from the sample space Dob
9
Si = Si−1 ∪X
10
λ = Add R.R.Vector(k, λ) // Add a Rade. rnd. column of length c
11
B, W, λ =Update(⋆)-TemporalBetweenness(X, B, W, λ)
12
˜R, vF =UpdateEstimates(B, W, λ, |Si|, k, c)
13
Rc
k = 1
c
Pc
l=1 maxv∈V
n
˜R[v, l]
o
14
ξ = ComputeSDBound(Rc
k, vF, δ/2i, |Si|)
// Compute Eq. 2 in Thm. 1
15
if |Si| ≥ω or ξ ≤ε then return {(1/|Si|) · B[u] : u ∈V }
Moreover, while iterating over the new sample X the temporal betweenness,
wimpy variances and the values in λ are updated. After this step, the coefficients
of Equation 2 and the new estimate on the SD, ξ, are computed (lines 12-13). As
a last step of the while loop, the algorithm checks whether the desired accuracy ε
has been achieved, i.e., whether the actual number of drawn samples is at least ω
or ξ is at most ε (line 15). If at least one of the two conditions is met, MANTRA
normalizes and outputs the current estimates B. We conclude this section with
the analysis of MANTRA’s running time.
Theorem 6. Given a temporal graph G = (V, E) and a sample of size r, MANTRA
requires time ˜O(r · n · |T|) and ˜O(r · |E|) to compute the shortest (foremost)-
temporal and the prefix-foremost-temporal betweenness, respectively. Moreover,
MANTRA requires O(n + |E|) space.
Theorem 5 together with Theorem 6 provide theoretical evidence that MANTRA
computes a rigorous estimation of the (⋆)-temporal betweenness and that scales
to the size of the input temporal graph. Moreover, it improves over the state-of-
the-art approach ONBRA [44]. Indeed, given a sample of size r, ONBRA stores
a n × r matrix to compute the absolute ξ-approximation8 using the Empirical
Bernstein Bound [28]. Thus, ONBRA may require an arbitrary large sample size
(e.g. large matrix) to achieve a target absolute approximation ε, making the
algorithm not ideal to analyze big temporal graphs.
8 ξ is the SD(S, F) obtained using the Empirical Bernstein Bound.


<!-- page 11 -->
Temporal Betweenness Centrality Approximation through Sampling
11
5
Experimental Evaluation
We compare our novel framework with ONBRA [44]. For the sake of fairness,
we adapted the original fixed sample size algorithm to use the same progressive
sampling approach of our framework. Every time an element of the sampling
schedule is consumed, the algorithm computes the upper bound ξ on the SD using
the Empirical-Bernstein bound as in [44], if ξ is at most the given ε, it terminates,
otherwise it keeps sampling. We set ONBRA’s maximum number of samples to
be equal to the VC-Dimension upper-bound in Section 4.2. We implemented all
the algorithms in Julia exploiting parallel computing9. We chose to re-implement
the exact algorithms [13] and ONBRA [44] because they have issues with the
number of paths in the tested networks10, causing overflow errors (indicated by
negative centralities), and with the time labeling causing an underestimation
of centralities [5]. Our implementation uses a sparse matrix representation of
the n × |T| table used in [13,44], making the implemented algorithms space-
efficient and usable on big temporal graphs (for which the original version of
the code gives out of memory errors). We executed all the experiments on a
server running Ubuntu 16.04.5 LTS with one processor Intel Xeon Gold 6248R
32 cores CPU @ 3.0GHz and 1TB RAM. For every temporal graph, we ran
all the algorithms with parameter ε ∈{0.1, 0.07, 0.05, 0.01, 0.007, 0.005, 0.001}
chosen to have a comparable magnitude to the highest temporal betweenness
values in the network (see b(⋆)
max in Table 1). This is a basic requirement when
computing meaningful approximations11. Moreover, we use δ = 0.1 and use
c = 25 Monte Carlo trials as suggested in [15,37]. Finally, each experiment has
been ran 10 times and the results have been averaged.
5.1
Experimental Results
Efficiency and Scalability. In our first experiment, we compare the average exe-
cution times, sample sizes and allocated memory of MANTRA and ONBRA. Due
to space constraints we show the results on the data sets in Table 1 for the prefix-
foremost-temporal betweenness, for a subset of ε ∈{0.01, 0.007, 0.005, 0.001} and
we refer to the additional materials for the complete battery of experiments. We
chose to display the results for the pfm temporal path optimality because it is
the one for which the analyzed graphs have the highest characteristic quanti-
ties (see Figure 1). Thus, under this setting, the tested algorithms will need a
bigger sample size and potentially a higher amount of memory. This somehow
provides an intuitive “upper bound” on the algorithms performances in terms
of efficiency and scalability. Moreover, the experiments for sh and sfm tempo-
ral betweenness follow similar trends of the ones displayed in the main paper.
Figure 2a shows the comparison of the running times (in seconds) for the pfm
9 Code available at: https://github.com/Antonio-Cruciani/MANTRA/
10 The overflow issue appears on all the transportation networks provided in [23].
11 It is meaningless to compute an ε-approximation when the maximum centrality value
is smaller than ε.


<!-- page 12 -->
12
Antonio Cruciani
Table 1: The data sets used in our evaluation, where ζ indicates the exact tem-
poral connectivity rate, b(⋆)
max the maximum (⋆)-temporal betweenness centrality
(type D stands for directed and U for undirected). • indicates that we need to use
BigInt data type instead of Unsigned Int128 to count the number of shortest
(foremost)-temporal paths to avoid overflows.
Data set
n
|E|
|T |
ζ
b(pfm)
max
b(sh)
max
b(sfm)
max
Type Source
College msg
1899
59798
58911
0.5
0.0718 0.0319 0.0365
D
[25]
Digg reply
30360
86203
82641
0.02
0.0019 0.0015 0.0016
D
[42]
Slashdot
51083
139789
89862
0.07
0.0128 0.0074 0.0085
D
[42]
Facebook Wall 35817
198028
194904
0.04
0.0034 0.0024 0.0028
D
[42]
Topology
16564
198038
32823
0.53
0.0921 0.0654 0.0681
U
[24]
Bordeaux•
3435
236075
60582
0.84
0.1210 0.1383 0.1269
D
[23]
Mathoverflow
24759
390414
389952
0.33
0.0522 0.0282 0.0287
D
[25]
SMS
44090
544607
467838
0.008 0.0019 0.0010 0.0012
D
[25]
Askubuntu
157222
726639
724715
0.169 0.0214 0.0156 0.0154
D
[25]
Super user
192409
1108716 1105102 0.21
0.0261 0.0165 0.0182
D
[25]
Wiki Talk
1094018 6092445 5799206 0.069 0.0089 0.0155 0.0153
D
[24]
Fig. 1: Comparison between temporal diameter and the average number of in-
ternal nodes for the Shortest (foremost) and Prefix-Foremost temporal path
optimalities. The approximation has been computed (over 10 runs) using our
sampling algorithm using 256 random seed nodes.
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
0
15
30
45
60
75
90
105
120
135
150
Characteristic quantities
for Shortest
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
0
15
30
45
60
75
90
105
120
135
150
165
180
Characteristic quantities
for Preﬁx Foremost
temporal betweenness. We observe that MANTRA leads the scoreboard against
its competitor on all the tested networks. Our novel framework is at least three
times faster than ONBRA. Such speedup is mainly due to the smaller sam-
ple size needed to terminate. Furthermore, Figure 2b shows that MANTRA
requires a smaller sample size (at least three times smaller) to converge. This
early convergence, in practice, does not affect the approximation quality and
leads to very good temporal betweenness approximations (see the next experi-
ment). Furthermore, the number of samples needed by MANTRA varies among
temporal graphs, with a strong dependence on b(⋆)
max. A potential cause of the
difference in the sample sizes between the two algorithms may depend on the

[CAPTION] Table 1: The data sets used in our evaluation, where ζ indicates the exact tem-

[CAPTION] Fig. 1: Comparison between temporal diameter and the average number of in-


<!-- page 13 -->
Temporal Betweenness Centrality Approximation through Sampling
13
use of the Empirical Bernstein bound. Such bound (as the VC-Dimension one)
is agnostic to any property of the analyzed temporal network, thus results in a
overly conservative guarantees. This suggests that variance-adaptive bounds are
preferable to compute data-dependent approximations [37], and that exploiting
correlations among the nodes through the use of the c-MCERA leads to refined
guarantees. Moreover, we point out that ONBRA does not scale well as the tar-
get absolute error ε decreases. Indeed, the memory needed by ONBRA increases
drastically as the target absolute error decreases (see Figure 2c) to the point of
giving out of memory error for big temporal networks such as Slashdot, SMS,
Askubuntu, Superuser, and Wiki Talk. This can lead to major issues while
computing meaningful ε-approximations, especially under the setting in which
the maximum temporal betweenness b(⋆)
max is very small (for which we need to
choose an ε value of at most b(⋆)
max12). Unfortunately, this is not an uncommon
feature of real-world temporal networks. Indeed, as shown by ζ and b(⋆)
max in Ta-
ble 1 they tend to be very sparse. This experiment, suggests that MANTRA is
preferable for analyzing big temporal networks up to an arbitrary small absolute
error ε.
Comparison with the exact algorithms scores and running times. As a first step
in our second experiment, we investigate the accuracy of the approximations
provided by MANTRA by computing the exact temporal betweenness centrality
of all the nodes of the temporal network in Table 1 and measuring the SD over
all the ten runs. Figure 2d supports our theoretical results, as we always get a
SD of at most the given ε. Moreover, we point out that the exact algorithms
for the shortest (foremost) temporal betweenness required a time that spanned
from several hours (e.g. for SMS) to days (for Askubuntu, and Superuser ≈a
week) and weeks (for Wiki Talk ≈a month). Instead, MANTRA completes
the approximation in reasonable time. Figure 3a shows the relation between the
sample size and the running time of our framework. While, Figure 3b shows the
amount of time needed by MANTRA to provide the absolute ε-approximation in
terms of percentage of exact algorithm’s running time. We display the running
times on the biggest temporal graphs for the sh temporal betweenness because
is one of the “critical” temporal path optimalities that requires longer times
to be computed (see Theorem 6). We can conclude that our framework is well
suited to quickly compute effective approximations of the temporal betweenness
on very large temporal networks.
6
Conclusions
We proposed MANTRA, a novel framework for approximating the temporal be-
tweenness centrality on large temporal networks. MANTRA relies on the state-
of-the-art bounds on supremum deviation of functions based on the c-MCERA
12 We recall that b(⋆)
max can be efficiently approximated in the bootstrap phase of our
framework.


<!-- page 14 -->
14
Antonio Cruciani
Fig. 2: Experimental analysis for ε ∈{0.01, 0.007, 0.005, 0.001}. Comparison be-
tween the running times (a), sample sizes (b), and allocated memory (c) of ON-
BRA and MANTRA. (d) Supremum deviation of the absolute ε-approximation
computed by MANTRA. The black line indicates that the two algorithms re-
quire the same amount of time/samples/memory, gray line (followed by a red
mark) indicates that the algorithm required more than 1TB of memory to run
on that data set with that specific ε value.
101
102
103
104
105
Running time MANTRA (s)
101
102
103
104
105
Running time ONBRA (s)
Running time comparison for Preﬁx Foremost
(a)
104
105
106
Sample size MANTRA
104
105
106
107
Sample size ONBRA
Sample size comparison for Preﬁx Foremost
(b)
10−3
10−2
10−1
100
Memory MANTRA (estimated GB)
10−2
10−1
100
101
102
103
Memory ONBRA (estimated GB)
Memory usage comparison for Preﬁx Foremost
(c)
0.01
0.007
0.005
0.001
ε
10−4
10−3
10−2
Supremum Deviation
Supremum Deviation for Preﬁx Foremost
(d)
to provide a probabilistically guaranteed absolute ε-approximation of such cen-
trality measure. Our framework includes a fast sampling algorithm to approxi-
mate the temporal diameter, average path length and connectivity rate up to a
small error with high probability. Such approach is general and can be adapted
to approximate several version of these quantities based on different temporal
path optimalities (e.g. [16,14]). Our experimental results (summarized in Sec-
tion 5) depict the performances of our framework versus the state-of-the-art
algorithm for the temporal betweenness approximation. MANTRA consistently
over-performs its competitor in terms of running time, sample size, and allocated
memory. As indicated in Figure 2c, our framework is the only available option
to obtain meaningful temporal betweenness centrality approximations when we
do not have access to servers with a large amount of memory. In the spirit of
reproducibility, we developed an open source framework in Julia that allows any

[CAPTION] Fig. 2: Experimental analysis for ε ∈{0.01, 0.007, 0.005, 0.001}. Comparison be-


<!-- page 15 -->
Temporal Betweenness Centrality Approximation through Sampling
15
Fig. 3: (a) Relation between the running time and the sample size of MANTRA
for the shortest temporal betweenness with ε as in Figure 2. (b) Comparison
between MANTRA and the exact algorithm running times for the shortest tem-
poral betweenness on the biggest temporal networks.
104
105
106
Sample size MANTRA
102
103
104
105
106
Running time Mantra (s)
Running time of MANTRA vs Sample size
for Shortest
(a)
0%
5% 10% 15% 20% 25% 30% 35% 40%
time (% of exact algorithm’s runtime)
0.01
0.005
0.001
ε
Running time comparison for Shortest
(b)
user with an average laptop to approximate the temporal betweenness centrality
on any kind of graph. Some promising future directions are to use MANTRA
to find communities in temporal graphs and to extend our approach to other
temporal path based centrality measures.
References
1. Abboud, A., Grandoni, F., Williams, V.V.: Subcubic equivalences between graph
centrality problems, APSP and diameter. In: Proceedings of the Twenty-Sixth
Annual ACM-SIAM Symposium on Discrete Algorithms, SODA (2015)
2. Amati, G., Cruciani, A., Angelini, S., Pasquini, D., Vocca, P.: Computing distance-
based metrics on very large graphs. CoRR (2023)
3. Bader, D.A., Kintali, S., Madduri, K., Mihail, M.: Approximating betweenness
centrality. In: Algorithms and Models for the Web-Graph WAW. Lecture Notes in
Computer Science, Springer (2007)
4. Bartlett, P.L., Mendelson, S.: Rademacher and gaussian complexities: Risk bounds
and structural results. J. Mach. Learn. Res. (2003)
5. Becker, R., Crescenzi, P., Cruciani, A., Kodric, B.: Proxying betweenness centrality
rankings in temporal networks. In: 21st International Symposium on Experimental
Algorithms, SEA. Schloss Dagstuhl - Leibniz-Zentrum f¨ur Informatik (2023)
6. Bhatia, R., Davis, C.: A better bound on the variance. Am. Math. Mon. (2000)
7. Borassi, M., Natale, E.: KADABRA is an adaptive algorithm for betweenness via
random approximation. ACM J. Exp. Algorithmics (2019)
8. Boucheron, S., Lugosi, G., Massart, P.: Concentration inequalities: A nonasymp-
totic theory of independence. univ. press (2013)
9. Bousquet, O.: A bennett concentration inequality and its application to suprema
of empirical processes. Comptes Rendus Mathematique (2002)

[CAPTION] Fig. 3: (a) Relation between the running time and the sample size of MANTRA


<!-- page 16 -->
16
Antonio Cruciani
10. Brandes, U.: A faster algorithm for betweenness centrality. Journal of mathematical
sociology (2001)
11. Brandes, U., Pich, C.: Centrality estimation in large networks. Int. J. Bifurc. Chaos
(2007)
12. Bui-Xuan, B., Ferreira, A., Jarry, A.: Computing shortest, fastest, and foremost
journeys in dynamic networks. Int. J. Found. Comput. Sci. (2003)
13. Buß, S., Molter, H., Niedermeier, R., Rymar, M.: Algorithmic aspects of temporal
betweenness. In: KDD ’20: The 26th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining. ACM (2020)
14. Calamai, M., Crescenzi, P., Marino, A.: On computing the diameter of (weighted)
link streams. ACM J. Exp. Algorithmics (2022)
15. Cousins, C., Wohlgemuth, C., Riondato, M.: Bavarian: Betweenness centrality ap-
proximation with variance-aware rademacher averages. In: KDD ’21: The 27th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2021)
16. Crescenzi, P., Magnien, C., Marino, A.: Approximating the temporal neighbour-
hood function of large temporal graphs. Algorithms (2019)
17. Crescenzi, P., Magnien, C., Marino, A.: Finding top-k nodes for temporal closeness
in large temporal graphs. Algorithms (2020)
18. Daly, E.M., Haahr, M.: Social network analysis for routing in disconnected delay-
tolerant manets. In: Proceedings of the 8th ACM Interational Symposium on Mo-
bile Ad Hoc Networking and Computing (2007)
19. Girvan, M., Newman, M.E.: Community structure in social and biological networks.
Proceedings of the national academy of sciences (2002)
20. van den Heuvel, M.P., Mandl, R.C., Stam, C.J., Kahn, R.S., Pol, H.E.H.: Aber-
rant frontal and temporal complex network structure in schizophrenia: a graph
theoretical analysis. Journal of Neuroscience (2010)
21. Jacob, R., Kosch¨utzki, D., Lehmann, K.A., Peeters, L., Tenfelde-Podehl, D.: Algo-
rithms for centrality indices. In: Network Analysis: Methodological Foundations.
Springer (2004)
22. Koltchinskii, V.: Rademacher penalties and structural risk minimization. IEEE
Trans. Inf. Theory (2001)
23. Kujala, R., Weckstr¨om, C., Darst, R., Madlenoci´c, M., Saram¨aki, J.: A collection
of public transport network data sets for 25 cities. Sci. Data (2018)
24. Kunegis, J.: The KONECT Project. http://konect.cc
25. Leskovec, J., Krevl, A.: Snap datasets. http://snap.stanford.edu/data
26. Li, Y., Long, P.M., Srinivasan, A.: Improved bounds on the sample complexity of
learning. Journal of Computer and System Sciences (2001)
27. Martello, S., Toth, P.: Knapsack Problems: Algorithms and Computer Implemen-
tations. John Wiley & Sons, Inc., USA (1990)
28. Maurer, A., Pontil, M.: Empirical bernstein bounds and sample-variance penaliza-
tion. In: COLT The 22nd Conference on Learning Theory (2009)
29. Mehryar Mohri, A.R., Talwalkar, A.: Foundations of machine learning. Springer
(2019)
30. Michail, O.: An introduction to temporal graphs: An algorithmic perspective (2016)
31. Mitzenmacher, M., Upfal, E.: Probability and computing: Randomization and
probabilistic techniques in algorithms and data analysis. Cambridge university
press (2017)
32. Oettershagen, L., Mutzel, P.: Tglib: An open-source library for temporal graph
analysis. In: IEEE International Conference on Data Mining Workshops, ICDM.
IEEE (2022)


<!-- page 17 -->
Temporal Betweenness Centrality Approximation through Sampling
17
33. Oettershagen, L., Mutzel, P.: An index for temporal closeness computation in evolv-
ing graphs. In: Proceedings of the 2023 SIAM International Conference on Data
Mining, SDM 2023. SIAM (2023)
34. Pellegrina, L.: Rigorous and efficient algorithms for significant and approximate
pattern mining. Ph.D. Thesis https://hdl.handle.net/11577/3471458 (2021)
35. Pellegrina, L.: Efficient centrality maximization with rademacher averages. In: Pro-
ceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data
Mining, KDD. ACM (2023)
36. Pellegrina, L., Cousins, C., Vandin, F., Riondato, M.: Mcrapper: Monte-carlo
rademacher averages for poset families and approximate pattern mining. ACM
Trans. Knowl. Discov. Data (2022)
37. Pellegrina, L., Vandin, F.: Silvan: Estimating betweenness centralities with progres-
sive sampling and non-uniform rademacher bounds. ACM Trans. Knowl. Discov.
Data (2023)
38. Pollard, D.: Convergence of stochastic processes. Springer Science & Business Me-
dia (2012)
39. Provost, F.J., Jensen, D.D., Oates, T.: Efficient progressive sampling. In: Proceed-
ings of the Fifth ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining. ACM (1999)
40. Riondato, M., Kornaropoulos, E.M.: Fast approximation of betweenness centrality
through sampling. In: Seventh ACM International Conference on Web Search and
Data Mining, WSDM. ACM (2014)
41. Riondato, M., Upfal, E.: ABRA: approximating betweenness centrality in static
and dynamic graphs with rademacher averages. ACM Trans. Knowl. Discov. Data
(2018)
42. Rossi, R.A., Ahmed, N.K.: Network repository. https://networkrepository.com
43. Rymar, M., Molter, H., Nichterlein, A., Niedermeier, R.: Towards classifying
the polynomial-time solvability of temporal betweenness centrality. In: Graph-
Theoretic Concepts in Computer Sciences. Lecture Notes in Computer Science,
Springer (2021)
44. Santoro, D., Sarpe, I.: ONBRA: rigorous estimation of the temporal betweenness
centrality in temporal networks. CoRR (2022)
45. Shalev-Shwartz, S., Ben-David, S.: Understanding machine learning: From theory
to algorithms. Cambridge university press (2014)
46. Simsek, ¨O., Barto, A.G.: Skill characterization based on betweenness. In: Advances
in Neural Information Processing Systems 21 (2008)
47. Tang, J.K., Musolesi, M., Mascolo, C., Latora, V., Nicosia, V.: Analysing informa-
tion flows and key mediators through temporal centrality metrics. In: Proceedings
of the 3rd Workshop on Social Network Systems (2010)
48. Tsalouchidou, I., Baeza-Yates, R., Bonchi, F., Liao, K., Sellis, T.: Temporal be-
tweenness centrality in dynamic graphs. Int. J. Data Sci. Anal. (2020)
49. Wu, H., Cheng, J., Huang, S., Ke, Y., Lu, Y., Xu, Y.: Path problems in temporal
graphs. Proc. VLDB Endow. (2014)


<!-- page 18 -->
18
Antonio Cruciani
Appendix
A
Extension of the (⋆)-Temporal Betweenness Centrality
to the edges.
s
a
u
v
b
z
1
2
23
22
36
40
w
3
4
x
56
y
80
92
Fig. 4: Example of the (⋆)-temporal paths described in Definition 1. Shortest:
(s
56
−→x
80
−→y
92
−→z), (s
22
−→a
36
−→b
40
−→z), Shortest-Foremost: (s
22
−→a
36
−→
b
40
−→z), and Prefix-Foremost:(s
1−→u
2−→v
3−→w
4−→z).
Figure 4 shows an example of the temporal paths optimalities considered
in this paper. Moreover, we define the underlying graph as the static graph
G = (V, E) obtained by removing all the time instants and multi-arcs from the
set E. We extend the concept of temporal betweenness to the temporal and
underlying edge of a given temporal graph G = (V, E) as follows
Definition 2 (Temporal Edge Betweenness). The temporal betweenness of
any edge e = (u, v) in the underlying graph of a temporal graph G is defined as:
b(⋆)
e
=
1
n(n −1)
X
s̸=z
σ(⋆)
sz (e)
σ(⋆)
sz
(3)
We define the temporal betweenness of a temporal edge as follows:
b(⋆)
e,t =
1
n(n −1)
X
s̸=z
σ(⋆)
sz (e, t)
σ(⋆)
sz
(4)
Where σ(⋆)
sz (e, t) is the number of (⋆)-temporal paths from s to z passing through
the underlying edge13 e at time t. Now we define the temporal edge betweenness
13 Equivalently, let e = (u, v), then σ(⋆)
sz (e, t) is the number of (⋆)-temporal paths from
s to z passing through the temporal edge (u, v, t).

[CAPTION] Fig. 4: Example of the (⋆)-temporal paths described in Definition 1. Shortest:

[CAPTION] Figure 4 shows an example of the temporal paths optimalities considered


<!-- page 19 -->
Temporal Betweenness Centrality Approximation through Sampling
19
of an underlying edge e as the sum of the temporal edge betweenness values of
its appearances at time (e, t)
Lemma 3. For any underlying edge e ∈E it holds:
b(⋆)
e
=
1
n(n −1)
T
X
t=0
b(⋆)
e,t
(5)
Proof.
b(⋆)
e
=
1
n(n −1)
X
s̸=z
σ(⋆)
sz (e)
σ(⋆)
sz
=
1
n(n −1)
X
s̸=z
T
X
t=0
σ(⋆)
sz (e, t)
σ(⋆)
sz
=
1
n(n −1)
T
X
t=0
b(⋆)
e,t
Analogously to Lemma 1, we can show that the sum of the (⋆)-temporal be-
tweenness centrality of the underlying edges is equal to the average number of
edges in a (⋆)-temporal path. Given a temporal graph G = (V, E) define
Ψ (⋆) =
1
n(n −1)
X
s,z∈V
X
e∈E
1[e ∈tpsz]
Where 1[e ∈tpsz] assumes value 1 if and only if the underlying edge e ∈E
appears in the temporal path tpsz. Then the following lemma holds:
Lemma 4. P
e∈E b(⋆)
e
= Ψ (⋆)
Proof. Equation 3 can be rewritten as
b(⋆)
e
=
1
n(n −1)
X
s,z∈V
X
tp∈Γ (⋆)
sz
1[e ∈Int(tp)]
σ(⋆)
sz
Summing over the underlying edges e ∈E we obtain:
X
v∈V
b(⋆)
e
=
1
n(n −1)
X
s,z∈V
X
tp∈Γ (⋆)
sz
1
σ(⋆)
sz
X
e∈E
1[e ∈Int(tp)]
1
n(n −1)
X
s,z∈V
σ(⋆)
sz
σ(⋆)
sz
X
e∈E
1[e ∈Int(tpsz)] =
1
n(n −1)
X
s,z∈V
X
e∈E
1[e ∈Int(tpsz)] = Ψ (⋆)
Given Lemma 3, Lemma 4, and the temporal accumulation results in Buß
et al. [13], we can adapt all the algorithms14 for computing the (⋆)-Temporal
Vertex Betweenness centrality to compute the (⋆)-Temporal Edge Betweenness
of the underlying graph. Edge-temporal betweenness centrality can be used to
develop fast temporal-community detection algorithms by using an approach
similar to the well know Girwan-Newman algorithm [19]. Our approximation
algorithms can be used to efficiently partition temporal graphs by removing the
top-k underlying/temporal edges with highest temporal-betweenness scores and
find communities.
14 Exact and approximate ones.


<!-- page 20 -->
20
Antonio Cruciani
B
Missing proofs
Proof of Lemma 1.
Proof. We can write the (⋆)-temporal betweenness as
b(⋆)
v
=
1
n(n −1)
X
s̸=v̸=z
σ(⋆)
sz (v)
σ(⋆)
sz
=
1
n(n −1)
X
s,z∈V
X
tp∈Γ (⋆)
sz
1[v ∈Int(tp)]
σ(⋆)
sz
summing over all the nodes, we obtain
X
v∈V
b(⋆)
v
=
1
n(n −1)
X
s,z∈V
X
tp∈Γ (⋆)
sz
1
σ(⋆)
sz
X
v∈V
1[v ∈Int(tp)]
=
1
n(n −1)
X
s,z∈V
σ(⋆)
sz
σ(⋆)
sz
X
v∈V
1[v ∈Int(tpsz)] =
1
n(n −1)
X
s,z∈V
|Int(tpsz)| = ρ(⋆)
Lemma 5. The function computed by ob is an unbiased estimator of the (⋆)-
temporal betweenness centrality.
Proof.
E
h
eb(⋆)
ob (v|s, z)
i
=
X
s,z∈V
s̸=z̸=v
Pr ((s, z)) · eb(⋆)
ob (v|s, z) =
X
s,z∈V
s̸=z
1
n(n −1)
σ(⋆)
sz (v)
σ(⋆)
sz
Proof of Lemma 2
Proof. Let V C(R) = d, where d ∈N. Then, there is S ⊂D such that |S| = d
and S is shattered by F+. For each temporal path tpsz ∈U, there is at most one
pair (tpsz, α) in S for some α ∈(0, 1] and there is no pair of the form (tpsz, 0).
By definition of shattering, each (tpsz, α) ∈S appears in 2d−1 different ranges
in F+. Moreover, each pair (tpsz, α) is in at most |tpsz|−2 ranges in F+, that is
because (tpsz, α) /∈Rf(v,t) either when α > f(v,t)(tpsz) or (v, t) /∈tpsz. Observe
that |tpsz| −2 ≤VD(⋆) −2, gives 2d ≤|tpsz| −2 ≤VD(⋆) −2. Thus, d −1 ≤
log(VD(⋆) −2) since d ∈N, we have d ≤⌊log VD(⋆) −2⌋+1 ≤log(VD(⋆) −2)+1.
Finally, V C(R) = d ≤⌊log VD(⋆) −2⌋+ 1.
We show how to estimate ˆv = supf∈F Var [f] using the empirical wimpy variance
WF(S). Proposition 1 is an adaption of Proposition 4.3 [37] when we are using
only a unique family of function F rather than a partition. For completeness we
provide the adapted proposition and the proof.
Proposition 1 ([37]). Let F = {eb
(⋆)
v , v ∈V } be a set of function from a domain
D to [0, 1]. And let S ⊆D be a sample of size r. Then, with probability at least
1 −δ it holds
sup
f∈F
Var [f] ≤WF(S) + ln(1/δ)
r
+
s ln(1/δ)
r
 2
+ WF(S) + ln(1/δ)
r
(6)


<!-- page 21 -->
Temporal Betweenness Centrality Approximation through Sampling
21
Proof. By definition,
sup
f∈F
Var [f] = sup
f∈F
n
E
 
f 2 
−E [f]2o
≤sup
f∈F
E
 
f 2 
(7)
Thus we focus on bounding supf∈F E
 
f 2 
. By a straightforward application of
Theorem 7.5.8 in [34] we have the claim of the lemma.
Proof of Theorem 3. The proof of this theorem follows the one in [37] by
considering the properties of the (⋆)-temporal betweenness. For completeness,
we show the adapted proof.
Proof. Given a sample S of size s, let E and Ev be the following events:
E = “∃v ∈V : |b(⋆)
v
−eb
(⋆)
v | > ε”
Ev = “|b(⋆)
v
−eb
(⋆)
v | > ε”
Applying the union bound we have that Pr (E) ≤P
v∈V Pr (Ev). Define the
functions g(x) = x(1 −x) and h(x) = (1 + x) ln(1 + x) −x for x ≥0, and let
ˆx1, ˆx2, ˆx as
ˆx1 = inf
(
x : 1
2 −
r
ε
2 −ε2
9 ≤x ≤1
2, g(x)h
 
ε
g(x)
   2ε2 
)
ˆx2 = 1
2 −
r
1
4 −ˆv
and,
ˆx = min{ˆx1, ˆx2}
Moreover, using Hoeffding’s and Bennet’s bounds [8], Bathia and Davis bound
on variance [6] and from the fact that
maxv∈V Var(f(v)) ≤ˆv, it holds, for all v ∈V
Pr (Ev) ≤2 min
n
exp
 −2sε2 
, exp
 
−sγ(ˆv, b(⋆)
v , ε)
 o .= η(x)
where γ(ˆv, b(⋆)
v , ε) = min{ˆv, g(b(⋆)
v )}h
 
ε
min{ˆv,g(b(⋆)
v
)}
 
. Now we can write
Pr (E) ≤
X
v∈V
Ev =
X
v∈V
Φ(b(⋆)
v )
(8)
Observe that the values of b(⋆)
v
are not known a priori, thus it is not possible to
compute the r.h.s. of Equation 8. However, we can obtain a sharp upper bound
by using the constraints on the possible values of b(⋆)
v
imposed by ˆv and ρ(⋆). As
in [37] we define an appropriate optimization problem w.r.t. the unknown values
of b(⋆)
v . Let kx be the number of nodes that we assume have b(⋆)
v
= x, define the
optimization problem over the variables kx as follows:


<!-- page 22 -->
22
Antonio Cruciani
max
X
x∈(0,1)
kx>0
kxΦ(x)
subject to
X
x∈(0,1)
kx>0
xkx ≤ρ(⋆),
0 ≤kx ≤ρ(⋆)
x , kx ∈N
Observe that the first constraint follows from Lemma 1, and the second one di-
rectly by the definition of ρ(⋆) itself. The values of the objective function of the
optimal solution of the optimization problem give and upper bound to Equa-
tion 8. As for the static case [37], the optimization problem is a formulation of
the Bounded Knapsack Problem [27] over the variables kx in which items with
label x are selected kx times with unitary profit Φ(x) and weight x. Moreover,
we consider the upper bound to the optimal solution given by the continuous
relaxation, in which kx ∈R, of the optimization problem (see Chapter 3 of [27]).
Four our purpose. its enough to fully select the item with higher profit-weight
ratio to fill the entire knapsack. Let x = argmaxx∈(0,1){Φ(x)/x}, the optimal so-
lution of the continuous relaxation is kx = ρ(⋆)/x, kx = 0, for all x ̸= x, while the
optimal objective is ρ(⋆)Φ(x)
x
≥Pr (E). Moreover, observe that x always exists
and Φ(x)/x ∈(0, 1). The search of x can be simplified by exploiting the same
approach used in [37] that leads to
Pr (E) ≤
sup
x∈(0,min{ˆx1,ˆx2})
 ρ(⋆)η(x)
x
 
≤
sup
x∈(0,ˆx)
 ρ(⋆)γ(g(x), s, ε)
x
 
Setting s ≥sup(0,ˆx)
n
ln( 2ρ(⋆)
xδ )/(g(x)h(
ε
g(x)))
o
it holds that Pr (E) ≤δ. In order
to approximate s, the r.h.s. of the equation can be computed using a numerical
procedure [37,35] obtaining the following approximation:
s ≈2ˆv + 2
3ε
ε2
 
ln
 2ρ(⋆)
ˆv
 
+ ln
 1
δ
  
∈O
  ˆv + ε
ε2
ln
 ρ(⋆)
δˆv
  
Moreover, given Lemma 4, we have the following corollary for the (⋆)-temporal
edge betweenness:
Corollary 1. Let F = {f(e), e ∈E} be a set of function from a domain D to
[0, 1]. Let f(e) be a function such that E [f(e)] = b(⋆)
e . Define ˆv ∈(0, 1/4] and
Ψ (⋆) ≥0 such that
max
e∈E Var(f(e)) ≤ˆv
and
X
e∈E
b(⋆)
e
≤Ψ (⋆)
fix ε, δ ∈(0, 1), and let S be an i.i.d. sample taken from D of size
|S| ∈O
  ˆv + ε
ε2
ln
 Ψ (⋆)
δˆv
  
It holds that SD(F, S) ≤ε with probability 1 −δ over S.


<!-- page 23 -->
Temporal Betweenness Centrality Approximation through Sampling
23
We make use of the following theorem to prove Theorem 1.
Theorem 7 ([8]). With probability at least 1 −δ over S, it holds
R(F, r) ≤R(F, S) +
s ln(1/δ)
r
 
+ 2 ln(1/δ)R(F, S)
r
+ ln(1/δ)
r
Proof of Theorem 1
Proof. Define the following events: E1 = “R(F, S) > ˜R”, E2 = “R(F, s) >
˜R”, E3 = “ supf∈F{|µS(f) −µπ(f)|} > εF”, and E = “SD(F, S) > εF”.
Moreover, by applying the union bound on these events we obtain Pr (E) ≤
E1 + E2 + E3. Now we can upper bound the probabilities of the single events
by applying the symmetrization lemma (Theorem 14.20 in [31]), Theorem 2.3
in [9] and Theorem 7 replacing δ/4 in the equations: Pr (E1) ≤δ/4 follows from
Equation (1) in Theorem 1 by replacing δ with δ/4; Pr (E2) ≤δ/4 follows from
Theorem 7; and, Pr (E3) ≤δ/4 follows after using the symmetrization lemma
and twice Theorem 2.3 in [9]. Moreover, the event E is true with probability at
most δ.
Proof of Theorem 5.
Proof. Each coordinate in eb
(⋆)
v , v ∈V is a sample mean (over a sample S of size r)
of a specific function associated to an unbiased estimator for b(⋆)
v . Algorithm 1
stops if the number of the drawn samples is at least ω or if the supremum
deviation bound ξ is at most ε. In other words it stops when the SD(F, S) ≤ε
and in both cases this is guaranteed (by Theorem 3 or Theorem 1) to happen
with probability 1 −δ.
Proof of Theorem 6.
Proof. MANTRA performs r truncated (⋆)-TBFS visits and regularly check
the stopping condition until convergence. The running times of the temporal
traversals depend on the type of path optimality we consider. Moreover, each
TBFS requires O(n·|T|·log (n · |T|)) [49,13] to compute the shortest (foremost)
temporal betweenness and O(|E| · log |E|) to compute the prefix-foremost tem-
poral betweenness. Furthermore, to compute and check the stopping condition
of the progressive sampler we need roughly linear time in n. Thus MANTRA’s
running time is O(r · n · |T| · log(n · |T|)) = ˜O(r · n · |T|) for the shortest and
shortest foremost temporal betweenness and O(r · |E| · log |E|) = ˜O(r · |E|) for
the prefix-foremost temporal betweenness. Finally, we observe that the space
required by MANTRA is O(n + |E| + c · n) observe that c is a fixed constant
(c = 25), thus the overall needed space is O(n + |E|).
C
Other estimators for the (⋆)-temporal betweenness
centrality
In this section we present other two unbiased estimators for the (⋆)-temporal
betweenness centrality.


<!-- page 24 -->
24
Antonio Cruciani
The RTB Estimator. We define the Random Temporal Betweenness estimator
(rtb). An intuitive technique to obtain an approximation of the (⋆)-temporal
betweenness centrality of a temporal graph G is to run the exact temporal be-
tweenness algorithm on a subset S of nodes selected uniformly at random from V .
Thus, in this case, the sampling space Drtb is the set V of vertices in G, and the
distribution πrtb is uniform over this set. The family Frtb = {eb(⋆)
rtb(v|s) : v ∈V },
contains one function eb(⋆)
rtb(v|s) for each vertex v, defined as:
eb(⋆)
rtb(v|s) =
1
n −1 ·
X
z∈V
z̸=s
σ(⋆)
sz (v)
σ(⋆)
sz
∈[0, 1]
(9)
It follows that
Lemma 6. The rtb is an unbiased estimator of the (⋆)-temporal betweenness
centrality.
Proof.
E
h
eb(⋆)
rtb(v|s)
i
=
X
s∈V
Pr (s) · eb(⋆)
rtb(v|s) =
X
s∈V
1
n


1
n −1
X
z̸=s̸=v
σ(⋆)
sz (v)
σ(⋆)
sz


The function eb(⋆)
rtb(v|s) is computed by performing a full (⋆)-temporal breadth
first search visit ((⋆)-TBFS for short) from s, and then backtracking along the
temporal directed acyclic graph as in the exacts algorithms [13]. Moreover, the
following lemma holds: The rtb framework computes all the sets Γ (⋆)
sz from the
sampled vertex s to all other vertices z ∈V using a full (⋆)-TBFS. Moreover,
in a worst-case scenario this algorithm could touch all the temporal edges in the
temporal graph at every sample making the estimation process slow. As for the
static case [11,21], this algorithm does not scale well as the temporal network
size increases.
The Temporal Riondato and Kornaropoulos estimator. We extend the estimator
for static graphs by Riondato and Kornaropoulos in [40] to the temporal setting.
The algorithm, (1) computes the set Γ (⋆)
sz
as ob; (2) randomly selects a (⋆)-
temporal path tpsz from Γ (⋆)
sz ; and, (3) increases by 1
r the temporal betweenness
of each vertex v in Int(tp) (where r is the sample size). The procedure to select
a random temporal path from Γ (⋆)
sz is inspired by the dependencies accumulation
to compute the exact temporal betweenness scores by Buß et al.[13]. Let s and z
be the vertices sampled by our algorithm. We assume that s and z are temporally
connected, otherwise the only option is to select the empty temporal path tp∅.
Given the set Γ (⋆)
sz of all the (⋆)-temporal paths from s to z, first we notice that
the truncated (⋆)-TBFS from s to z produces a time respecting tree from the
vertex appearance (s, 0) to all the vertex appearances of the type (z, tz) for some
tz. Let tp∗be the sampled (⋆)-temporal path we build backwards starting from


<!-- page 25 -->
Temporal Betweenness Centrality Approximation through Sampling
25
one of the temporal endpoints of the type (z, tz) for some tz. First, we sample
such (z, tz) as follows: a vertex appearance (z, tz) is sampled with probability
σtz
sz/(P
t σt
sz) = σtz
sz/σsz, where σt
sz is the number of (⋆)-temporal paths reaching
z from s at time t. Assume that (z, tz) was put in the sampled path tp∗, i.e., tp∗=
{(z, tz)}. Now we proceed by sampling one of the temporal predecessors (w, tw) in
the temporal predecessors set P(z, tz) with probability σtw
sw/(P
(x,t)∈P (z,tz) σt
sx).
After putting the sampled vertex appearance, let us assume (w, tw), in tp∗we
iterate the same process through the predecessors of (w, tw) until we reach (s, 0).
Theorem 8. Let tp∗
sz ∈Γ (⋆)
sz
be the (⋆)-temporal path sampled using the above
procedure. Then, the probability of sampling tp∗
sz is Pr (tp∗) =
1
σ(⋆)
sz
Proof. Let σsz = σ(⋆)
sz . The probability of getting such tp∗using the aforemen-
tioned temporal path sampling technique is:
Pr (tp∗) =
σtz
sz
P
t σtsz
·
σ
twk−1
swk−1
P
(x,t)∈P (z,tz) σtsx
·
σ
twk−2
swk−2
P
(x,t)∈P (wk−1,tk−1) σtsx
· · ·
σ
tw1
sw1
P
(x,t)∈P (w2,t2) σtsx
·
1
P
(x,t)∈P (w1,t1) σtsx
Observe that σtw
sw = P
(x,t)∈P (w,tw) σt
sx and that P
t σt
sz = σsz. Thus, the formula
can be rewritten as follows:
Pr (tp∗) = σtz
sz
σsz
· σ
twk−1
swk−1
σtz
sz
· · ·
1
σ
tw1
sw1
=
1
σsz
and the fact that (if the temporal graph has no self loop) for (w1, tw1), which is
a temporal neighbor of (s, 0), σsw1 = 1.
Observe that each tpsz ∈TP(⋆)
G
is sampled according to the function πtrk(tpsz) =
1
n(n−1)
1
σ(⋆)
sz
which (according to Theorem 9) is a valid probability distribution
over Dtrk = TP(⋆)
G .
Theorem 9. The function πtrk(tpsz), for each tpsz ∈Dtrk, is a valid probability
distribution.
Proof. Let Γ (⋆)
sz be the set of (⋆)-optimal temporal paths from s to z where s ̸= z.
Then,
X
tpsz∈Dtrk
π(tpsz) =
X
tpsz∈Dtrk
1
n(n −1)
1
σ(⋆)
sz
=
X
s∈V
X
z∈V
s̸=z
X
tpsz∈Γ (⋆)
sz
1
n(n −1)
1
σ(⋆)
sz
=
X
s∈V
X
z∈V
s̸=z
1
n(n −1)
σ(⋆)
sz
σ(⋆)
sz
=
1
n(n −1)
X
s∈V
X
z∈V
s̸=z
1 =
1
n(n −1)
X
s∈V
(n −1) = 1


<!-- page 26 -->
26
Antonio Cruciani
For tpsz ∈Dtrk, and for all v ∈V define the family of functions Ftrk = {f (⋆)
trk(v) :
v ∈V } where eb(⋆)
trk(v|tpsz) = 1 [v ∈Int(tpsz)]. Observe that
Lemma 7. For eb(⋆)
trk(v) ∈F and for all tpsz ∈Dtrk, such that each tpsz is
sampled according to the probability function π(tpsz), then E
h
eb(⋆)
trk(v|tpsz)
i
=
b(⋆)
v .
Proof.
Etpsz∈Dtrk
h
eb(⋆)
trk(v|tpsz)
i
=
X
tpsz∈Dtrk
π(tpsz)eb(⋆)
trk(v|tpsz) =
X
tpsz∈Dtrk
eb(⋆)
trk(v|tpsz)
n(n −1)σ(⋆)
sz
=
1
n(n −1)
X
s,z∈V
s̸=v̸=z
X
tp∈Γ (⋆)
sz
1 [v ∈Int(tp)]
σ(⋆)
sz
=
1
n(n −1)
X
s,z∈V
s̸=v̸=z
σ(⋆)
sz (v)
σ(⋆)
sz
D
Approximating the (⋆)-temporal distance based
metrics.
Given a temporal graph G = (V, E), let N(u, h) = {v ∈V : d(u, v) ≤h} be
the temporal ball centered in u at time 0 of radius h. N(u, h) is the set of all
the nodes v that can be reached by node u starting at time 0 by a (⋆)-optimal
temporal path. Now define |N(h)| = |{(u, v) ∈V × V : d(u, v) ≤h}| as the
overall pairs of nodes that can be reached by a (⋆)-optimal path in h steps.
The (⋆)-temporal diameter D of a graph is the number of temporal edges in the
longest (⋆)-temporal path in the temporal graph. In terms of N(h) we have:
D(⋆) = min
h
 
h :
X
u
|N(u, h)| =
X
u
|N(u, h + 1)|
!
Alternatively, as for the static case, we can define the effective (⋆)-temporal
diameter as the τ (th) percentile (⋆)-temporal path length between the nodes.
We will use this quantity to provide an error bound for the approximation of
D(⋆). Let τ ∈[0, 1], then
D(⋆)
τ
= min
h
 
h :
P
u |N(u, h)|
P
u |N(u, D(⋆))| ≥τ
 


<!-- page 27 -->
Temporal Betweenness Centrality Approximation through Sampling
27
Finally, observe that given N(u, h) for each u ∈V and h ∈{0, . . . , D} we can
define the average (⋆)-temporal distance
Ψ (⋆) =
P
u,v∈V
u̸=v
1[u ⇝v] · d(u, v)
P
u,v∈V
u̸=v
1[u ⇝v]
=
P
u∈V
P
h∈[0,D(⋆)](|N(u, h) −|N(u, h −1)) · h
|N(D(⋆))|
=
P
s,z∈V
P
e∈E 1[e ∈tpsz]
|N(D(⋆))|
Observe that Ψ (⋆) and ρ(⋆) are tightly related,
ρ(⋆) =
P
u∈V
P
h∈(0,D(⋆))(|N(u, h) −|N(u, h −1)) · h
|N(D(⋆))|
≤Ψ (⋆)
Algorithm 2, given a temporal graph G = (V, E), a number of seeds r and the
temporal path optimality of interest (⋆) the algorithm performs a (⋆)-TBFS and
keeps track of the number of nodes encountered at each hop and the maximum
number of hops performed. Once all the seed nodes have been processed it com-
putes the overall set of reachable pairs and the average distance Ψ (⋆), effective
diameter DLB, and the temporal connectivity rate ζG.
Algorithm 2: (⋆)-temporal (effective) diameter approximation
Data: Temporal graph G, ε, sample size r, and effective shortest temporal
diameters’s threshold τ
Result: Diameter lower bound DLB, Effective diameter Dτ, temporal
connectivity rate ζG , average distance Ψ
1 R = [0, 0, . . . , 0]// Nr. of reach. pairs at each hop
2 DLB = 0; Ψ = 0; Dτ = −1
3 for k = 0 to r −1 do
4
u = sample u.a.r. a node from V
5
R, DLB = (⋆)-TemporalBFS(G, u, R, DLB)
// Update R and DLB
6 for h = 0 to DLB −1 do
7
R[h] = n
r · R[h]
// R[−1] threaded as 0 when h = 0
8
Ψ = Ψ + (R[h] −R[h −1]) · h
9 Ψ = Ψ/R[DLB]; Dτ = minh
 
h :
R[h]
R[DLB] ≥τ
 
; ζG =
R[DLB]
n(n−1)
10 return DLB, Dτ, ζG, Ψ
Algorithm 2, computes the (⋆)-temporal distance-based metrics described
above. Moreover, we can give a bound on the quality of the approximation
produced by this approach.
Theorem 10. Given a temporal graph G = (V, E) and a sample of size r =
Θ
  ln n
ε2
 
. Algorithm 2 computes with probability 1−2
n2 the (⋆) temporal (effective)


<!-- page 28 -->
28
Antonio Cruciani
diameter with absolute error bounded by
ε
ζ(G), the temporal connectivity rate
ζ(G) with absolute error bounded by ε, and the average (⋆)-temporal length with
absolute error bounded by
ε·D
ζ(G).
Proof. Let h be the (⋆)-temporal effective diameter threshold, and
Xh
i =
n · P
{u:d(vi,u)≤h} 1[vi ⇝u]
P
u,v∈V
u̸=v
1[u ⇝v]
= n · |N(vi, h)|
|N(D)|
observe that Xh
i ∈[0, 1
ζ ], and that Xh
i ’s expected value is
E
 
Xh
i
 
=
X
vi∈V
Xh
i · Pr (vi) =
P
vi∈V |N(vi, h)|
|N(D)|
= |N(h)|
|N(D)|
Applying Hoeffding’s inequality, with ξ =
ε
ζ , we can approximate
|N(h)|
|N(D)| by
P
i∈[r] Xh
i
r
=
n·P
i∈[r] |N(u,vi)|
r·|N(D)|
, and taking a sample of r = ln n
ε2
leads an error of
ε
ζ with probability of at least 1 −
2
n2 . Now we observe that the (⋆)-temporal
diameter can be defined in terms of the effective one by choosing τ = 1, thus
the bound holds also for the (⋆)-temporal diameter. Next, define the random
variable XD
i = |N(vi,D)|
n−1
, and observe that XD
i ∈[0, 1]. Observe that its expected
value is exactly ζ(G), i.e., E
 
XD
i
 
= P
vi∈V
1
n
|N(vi,D)|
n−1
= |N(D)|
n(n−1) = ζ(G). Again,
applying Hoeffding’s inequality, with ξ = ε, and taking r = ln n
ε2
sample nodes,
we have an error bound of ε with high probability. Finally, define
Xi = n · P
u∈V 1[vi ⇝u] · d(vi, u)
P
u,v∈V 1[u ⇝v]
Observe that Xi ∈[0,
D
ζ(G)], and that its expectation is the average shortest
temporal distance
E [Xi] =
X
vi∈V
Xi · Pr (Xi) =
P
vi,u∈V 1[vi ⇝u] · d(vi, u)
P
u,v∈V 1[u ⇝v]
Finally, by using Hoeffding’s inequality with ξ =
εD
ζ(G) and setting r = log n
ε2
we
obtain that the error is of at most
εD
ζ(G) with probability 1 −
2
n2 .
Observe that Algorithm 2 is general, it can be used to compute several notions
of temporal diameter [49,30,16,14,32].
E
Missing Experiments
Figure 5 (in Section E) and Figure 1 (in Section 5), show the average ac-
curacy and execution times of Algorithm 2 (with 256 seeds) compared to the

[CAPTION] Figure 5 (in Section E) and Figure 1 (in Section 5), show the average ac-


<!-- page 29 -->
Temporal Betweenness Centrality Approximation through Sampling
29
time needed to retrieve exact values for the temporal networks in Table 1. We
observe that our novel sampling algorithm provides sharp estimates of the ex-
act values (temporal (effective) diameter, average path length, and connectivity
rate). Moreover, the approximation process using 256 seeds requires a negligible
running time compared to the time needed to compute the exact characteristic
quantities. This experimental analysis shows that the bootstrap phase of the
MANTRA framework (that uses Algorithm 2 to approximate the characteristic
quantities) provides very good estimates on the characteristic quantities that are
used to compute the upper bound on the sample size. Moreover, we point out
that Algorithm 2 could be used to provide fast approximation of the temporal
Closeness Centrality (e.g. [17,32,33]). Finally, Figure 6, Figure 7, and Figure 8
show the experimental analysis for the shortest (⋆)-temporal betweenness ap-
proximation for all the temporal path optimalieties and for the remaining values
of ε.


<!-- page 30 -->
30
Antonio Cruciani
Fig. 5: Experimental evaluation (over 10 runs) of the sampling algorithm using
256 seed nodes. Comparison between temporal effective diameter and the con-
nectivity rate for the Shortest (foremost) and Prefix-Foremost temporal path
optimalities. Last two log-plots, are the running times comparisons between the
exact and our approximation algorithm.
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
0
15
30
45
60
75
90
105
120
135
150
Characteristic quantities for
Shortest
Eﬀ. Diam
Apx. Eﬀ. Diam
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
0
15
30
45
60
75
90
105
120
135
150
165
180
Characteristic quantities for
Preﬁx Foremost
Eﬀ. Diam
Apx. Eﬀ. Diam
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
0.0
0.2
0.4
0.6
0.8
Temporal Connectivity rate for
Shortest
ζ
Apx. ζ
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
0.0
0.2
0.4
0.6
0.8
Temporal Connectivity rate for
Preﬁx Foremost
ζ
Apx. ζ
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
10−3
10−2
10−1
Apx/Exact
Ratio Running Times for
Shortest
College Msg
Digg reply
Slashdot
Facebook wall
Topology
Bordeaux
Mathoverﬂow
SMS
Askubuntu
Superuser
Wiki Talk
10−3
10−2
10−1
Apx/Exact
Ratio Running Times for
Preﬁx Foremost

[CAPTION] Fig. 5: Experimental evaluation (over 10 runs) of the sampling algorithm using


<!-- page 31 -->
Temporal Betweenness Centrality Approximation through Sampling
31
Fig. 6: Experimental analysis for ε ∈{0.01, 0.007, 0.005, 0.001} For the Shortest
and Shortest-foremost temporal betweenness. Comparison between the running
times (a-b), sample sizes (c-d), and allocated memory (e-f) of ONBRA and
MANTRA. (g) Supremum deviation of the absolute ε-approximation computed
by MANTRA.
103
105
107
Running time MANTRA (s)
103
105
107
Running time ONBRA (s)
Running time comparison for Shortest
(a)
103
105
107
Running time MANTRA (s)
103
105
107
Running time ONBRA (s)
Running time comparison for Shortest Foremost
(b)
104
105
106
Sample size MANTRA
104
105
106
107
Sample size ONBRA
Sample size comparison for Shortest
(c)
104
105
106
Sample size MANTRA
104
105
106
107
Sample size ONBRA
Sample size comparison for Shortest Foremost
(d)
10−3
10−2
10−1
100
Memory MANTRA (estimated GB)
10−2
10−1
100
101
102
103
Memory ONBRA (estimated GB)
Memory usage comparison for Shortest
(e)
10−3
10−2
10−1
100
Memory MANTRA (estimated GB)
10−2
10−1
100
101
102
103
Memory ONBRA (estimated GB)
Memory usage comparison for Shortest Foremost
(f)
0.01
0.007
0.005
0.001
ε
10−4
10−3
10−2
Supremum Deviation
Supremum Deviation for Shortest
(g)
104
105
106
Sample size MANTRA
102
103
104
105
106
Running time Mantra (s)
Running time of MANTRA vs Sample size
for Shortest Foremost
(h)

[CAPTION] Fig. 6: Experimental analysis for ε ∈{0.01, 0.007, 0.005, 0.001} For the Shortest


<!-- page 32 -->
32
Antonio Cruciani
Fig. 7: Experimental analysis for ε ∈{0.1, 0.07, 0.05} For the Shortest and
Shortest-foremost temporal betweenness. Comparison between the running
times (a-b), sample sizes (c-d), and allocated memory (e-f) of ONBRA and
MANTRA.
(a)
(b)
(c)
(d)
(e)
(f)

[CAPTION] Fig. 7: Experimental analysis for ε ∈{0.1, 0.07, 0.05} For the Shortest and


<!-- page 33 -->
Temporal Betweenness Centrality Approximation through Sampling
33
Fig. 8: Experimental analysis for ε ∈{0.1, 0.07, 0.05} For the Prefix-foremost
temporal betweenness. Comparison between the running times (a), sample sizes
(b), and allocated memory (c) of ONBRA and MANTRA.
(a)
(b)
(c)

[CAPTION] Fig. 8: Experimental analysis for ε ∈{0.1, 0.07, 0.05} For the Prefix-foremost