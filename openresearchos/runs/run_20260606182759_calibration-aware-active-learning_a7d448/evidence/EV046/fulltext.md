<!-- page 1 -->
Quantum Boosting using Domain-Partitioning Hypotheses
Debajyoti Bera∗1, Rohan Bhatia† 2, Parmeet Singh Chani‡ 2, and Sagnik Chatterjee§ 1
1Indraprastha Institute of Information Technology (IIIT-D), Delhi, India
2Delhi Technical University, Delhi, India
August 16, 2022
Abstract
Boosting is an ensemble learning method that converts a weak learner into a strong learner in the
PAC learning framework. Freund and Schapire designed the Godel prize-winning algorithm named
AdaBoost that can boost learners which output binary hypotheses. Recently, Arunachalam and Maity
presented the first quantum boosting algorithm with similar theoretical guarantees. Their algorithm,
which we refer to as QAdaBoost henceforth, is a quantum adaptation of AdaBoost, and only works for the
binary hypothesis case. QAdaBoost is quadratically faster than AdaBoost in terms of the VC-dimension
of the hypothesis class of the weak learner but polynomially worse in the bias of the weak learner.
Izdebski et al. posed an open question on whether we can boost quantum weak learners that output
non-binary hypothesis. In this work, we address this open question by developing the QRealBoost
algorithm which was motivated by the classical RealBoost algorithm. The main technical challenge
was to provide provable guarantees for convergence, generalization bounds, and quantum speedup, given
that quantum subroutines are noisy and probabilistic. We prove that QRealBoost retains the quadratic
speedup of QAdaBoost over AdaBoost and further achieves a polynomial speedup over QAdaBoost in
terms of both the bias of the learner and the time taken by the learner to learn the target concept class.
Finally, we perform empirical evaluations on QRealBoost and report encouraging observations on
quantum simulators by benchmarking the convergence performance of QRealBoost against QAdaBoost,
AdaBoost, and RealBoost on a subset of the MNIST dataset and Breast Cancer Wisconsin dataset.
1
Introduction
The last decade has seen substantial growth in the field of quantum machine learning, giving rise to several
quantum machine learning algorithms that promise and provide improvements over their classical counterparts.
Several survey papers and books have already been published, and interested readers may consult any of those
to obtain an overview of the algorithmic and theoretical advances in quantum machine learning [2, 6, 28, 34].
In the discriminative models, learning algorithms aim to “learn” an unknown concept which helps them
classify samples. Some learning algorithms are accurate with arbitrarily high accuracy, while others perform
slightly better than random guessing. Even though very accurate learners are ultimately desired, it might not
always be wise to use highly accurate learners for many reasons, such as longer running times, overfitting, and
lack of model explainability. On the other hand, many well-known simple learning algorithms are easy to create
and are essentially weak learners. These include decision stumps, na¨ıve Bayes over a single variable, clustering
∗dbera@iiitd.ac.in
†rohanbhatia 2k19ep.079@dtu.ac.in
‡parmeetsinghchani 2k19ep.066@dtu.ac.in
§sagnikc@iiitd.ac.in
1
arXiv:2110.12793v3  [quant-ph]  15 Aug 2022


<!-- page 2 -->
Figure 1: Comparing the performance of 4 different boosting algorithms using the k-means clustering algorithm (with
k=3) as the base learner on the Breast Cancer Wisconsin dataset [35] with 32 training samples.
algorithms with a fixed number of clusters, are all . Ensemble learning is a method of converting “weak” learn-
ers to “strong” learners1. In this paper, we focus on a particular type of ensemble learning approach known
as Boosting, in which weak learners are trained iteratively over reweighted distributions of a fixed training
set. By tweaking the distribution of the training set, we ensure that each learner gives more weight to the
misclassified samples in the previous iteration, eventually reducing both bias and variance of the learner [15].
Boosting algorithms are now included in standard machine learning libraries, e.g., scikit-learn [21].
1.1
Related Work
Table 1: Comparing AdaBoost, SmoothBoost, QAdaBoost, and QSmoothBoost, with the QRealBoost algorithm.
Here, robust means robust to classification noise; fast and slow refer to convergence rates; binary and non-binary
refer to the type hypotheses produced by the weak learner. The weak learner (classical or quantum) has bias γ,
an associated hypothesis class H with VC-dimension of dH, and takes time R (classical case), or Q (quantum case)
to produce a hypothesis h∈H.
Boosting Algorithms
Query Complexity
Remarks
AdaBoost ([14])
O
 
dH·R· 1
γ4
 
adaptive, fast, binary
SmoothBoost ([29])
˜O
 
dH
γ4 + R
γ2
 
non-adaptive, robust, slow, binary
QAdaBoost ([4])
˜O
 √dH·Q1.5· 1
γ11
 
adaptive, fast, binary
QSmoothBoost ([18])
˜O
 √dH
γ5 + Q
γ4
 
non-adaptive, robust, slow, binary
QRealBoost (this work)
˜O
 √dH·Q· 1
γ9
 
adaptive, fast, non-binary
AdaBoost is one of the first adaptive boosting algorithms, and was proposed by Freund and Schapire [14].
AdaBoost provably converges with zero training error and requires no prior knowledge about the accuracy
of the hypotheses generated by the weak-learner it is trying to boost. It has been also observed that
AdaBoost does not tend to overfit[5, 12] on a wide variety of problems and performed much better[11, 20]
1We define the terms weak and strong formally in Section 2.1.
2

[CAPTION] Figure 1: Comparing the performance of 4 different boosting algorithms using the k-means clustering algorithm (with

[CAPTION] Table 1: Comparing AdaBoost, SmoothBoost, QAdaBoost, and QSmoothBoost, with the QRealBoost algorithm.


<!-- page 3 -->
than other ensemble methods in the absence of classification noise. We discuss in more detail the original
AdaBoost algorithm in Appendix B.1. Early works on quantum boosting algorithms consider quantum
algorithms for AdaBoost and its variants in the experimental setting [19], use classical AdaBoost as a
subroutine [27], or only consider speedups for a particular aspect of AdaBoost, e.g., computing the margins
as defined in Algorithm 2 [33]. Arunachalam and Maity recently adapted the original AdaBoost algorithm
into QAdaBoost[4] and provided rigorous mathematical guarantees of speed up over the classical version in
its query and time complexity in terms of the weak learner’s sample complexity by using approximate
counting of quantum states. Subsequently, Izdebski et al. [18] proposed a quantum variant of Servedio’s
classical SmoothBoost algorithm[29] (which we refer to as QSmoothBoost) which retains the speedup in
sample complexity while also achieving a polynomial speedup in the bias of the weak learner as compared
to the QAdaBoost algorithm. Despite QSmoothBoost’s impressive speedups in complexity, it comes with
a few shortcomings. Similar to its classical counterpart, QSmoothBoost is not adaptive2. Additionally,
SmoothBoost (hence, QSmoothBoost) takes longer to converge than AdaBoost (hence, QAdaBoost) and
does not converge to zero training error. We chose to stick to the AdaBoost framework to circumvent
these issues3. In Table 1, we compare the query complexities of our quantum boosting algorithm with the
QAdaBoost, QSmoothBoost, and the classical AdaBoost algorithms.
1.2
Our work
Our algorithm QRealBoostAlgorithm 1 is a quantum adaptation of the discrete RealBoost algorithm [15, 26]
which tackles the problem of boosting weak learners whose hypotheses essentially divide the domain X into
a small number of mutually exclusive and exhaustive sets of partitions (hence the name “domain-partitioning
hypotheses”). A point to note here is that such domain-partitioning learners can be alternatively char-
acterized as having hypotheses that output real-valued predictions, unlike binary-valued predictions as seen
in AdaBoost (and QAdaBoost).
In this work we focus on weak learners that output discrete class partitions rather than class probabilities
since this is a more natural model for decision tree algorithms and clustering algorithms. Domain-partitioning
hypotheses allow us to calculate the confidences of prediction for each partition 4 which leads to improved
estimation of margins, ultimately producing better bounds of generalization error. RealBoost not only
retains the zero training error and the generalization behavior of the original AdaBoost family of boosting
algorithms but also has been observed in practice to converge much faster than AdaBoost with respect
to the empirical error [23, 32, 36]. QRealBoost maintains the general flavour of RealBoost but implements
several steps using quantum algorithms which lead to a quadratic speedup over RealBoost. The caveat here
is that the intermediate quantum subroutines involving quantum amplification and estimation are erroneous,
and therefore require careful analysis to prove that boosting converges and the convergence is exponentially
fast. This is the main technical contribution of this work. This is also the first work to our knowledge which
performs boosting with non-binary classifiers, which also addresses an open question by Izdebski et al. [18] on
whether we can boost learners with range other than {−1,+1}. We now state the main results of our paper.
Theorem 1 (Complexity of QRealBoost). Let A be a γ-weak quantum PAC learner for a concept class C
with sample complexity Q having an associated hypothesis class H with VC-dimension dH. Further, suppose
that H contains domain-partitioning hypotheses. The time complexity of boosting A to a strong PAC learner
according to Algorithm 1 is ˜O
 √dH·Q· n2
γ9
 
and the corresponding query complexity is ˜O
 √dH·Q· 1
γ9
 
.
Theorem 2 (Convergence of QRealBoost). If we run Algorithm 1 for a sufficiently large number of
iterations T ≥lnM
2γ2 , then with a high probability we output a hypothesis H that has zero training error and
a small generalization error.
2For a detailed discussion on “adaptiveness” see [8]
3We describe a generalized framework in Algorithm 3 in Appendix B.1
4We discuss the notion of “confidence” and “confidence-rated predictions” in Section 2.2.
3


<!-- page 4 -->
Algorithm 1 The QRealBoost algorithm.
Input: Quantum weak PAC learner A with sample complexity Q which makes at most C partitions in the worst case,
and M labeled training samples {(xi,yi)}i∈[M].
1: Initialize: Worst-case values of Q and C, and ε=O
 
1
QT 2
 
, κ=
C
(1−ε)
q
1+ε
1−ε. Set ˜D1
i = 1
M ∀i∈[M].
2: for t=1 to T do
▷Adaptiveness: Choose T ≥lnM
2γ2 for a worst-case guess of γ.
generate Q+2C copies of a distribution of training samples
3:
Prepare |φ0⟩⊗Q|ψ0⟩⊗2C where |φ0⟩=|ψ0⟩=
1
√
M
P
i∈[M]|xi,yi⟩⊗
    ˜D1
i
E
.
4:
for s=1 to t−1 do
5:
On all the Q+2C copies, apply the transformation |xi,yi⟩
    ˜Ds
i
E
−
→|xi,yi⟩
    ˜Ds+1
i
E
based on the stored values of
β′
j,s and Z′
s, obtaining the partition j using Ohs, and using the distribution update rule
˜Ds+1
i
=
˜Ds
i ·exp
 −β′
j,syi
 
κZ′s
(1)
▷The final state after the loop is denoted |φ2⟩⊗Q|ψ2⟩⊗2C which is set to |φ0⟩⊗Q|ψ0⟩⊗2C when t=1.
6:
On the first Q copies do
7:
Perform a conditional rotation on the
    ˜Dt
i
E
register to create |φ3⟩.
8:
Apply amplitude amplification conditioned on the |1⟩register and then uncompute ancilla to obtain Q copies
of |φ5⟩=P
i∈[M]
q
˜
Dt
i|xi,yi⟩+|ζ′
t⟩.
obtain the t-th hypothesis ht from the first Q copies
9:
A
 
|φ5⟩⊗Q 
→(followed by a measurement) oracle Oht corresponding to the hypothesis ht.
obtain confidence-rated predictions using ht on the last 2C copies
10:
On
the
last
2C
copies
of
|ψ2⟩
use
oracles
Oh1, ... , Oht−1, Oht
to
create
2C
copies
of
|ψ3⟩
=
1
√
M
P
i∈[M]
   xi,yi, ˜
Dt
i
E  j1
i ,j2
i ,...,jt
i
 
.
11:
for k=1 to C and b∈{−1,+1} do
▷iterate over every partition and label
12:
Take the (k,b)th copy of |ψ3⟩and prepare (omitting unrelated qubits)
|ψ4⟩=
1
√
M
P
i∈[M]|xi⟩|yi⟩
    ˜Dt
i
E  jt
i
     ˜Dk,b,t
i
E
where ˜Dk,b,t
i
= ˜Dt
i if jt
i =k and yi=b, and 0 otherwise.
13:
Do conditional rotation on the last register to obtain states with amplitudes
q
W k,t
b
where W k,t
b
=P
i∈[M] ˜Dk,b,t
i
/M.
14:
Perform amplitude estimation to obtain ˜
W k,t
b
with relative error ε and do Laplace correction on the estimated
weights ˜
W k,t
b .
15:
Compute Z′
t=2PC
j=1
q
˜
W j,t
+ ˜
W j,t
−and margins β′
j,t= 1
2ln
 
˜
W j,t
+ / ˜
W j,t
−
 
for all j=1,...,C.
16: Output: Hypothesis H(·) defined as
H(x)=sign
 T
X
t=1
β′
jt,t
!
, where jt∈{1...C} is obtained using |x,0⟩
Oht
−−−→
  x,jt 
(2)
1.3
Techniques
Our weak PAC learner makes a constant C number of domain partitions. The QRealBoost algorithm
focuses on improving the confidence of prediction in each partition individually. To do this we start by
calculating the number of samples belonging to a given partition with a particular label. This is known
as the partition-label weight. The main idea of this algorithm is to use quantum estimation techniques to
calculate the 2C partition-label weights W j,t
b
(as indicated in Eq. (44)) for each j∈{1,...C} and b∈{−1,+1}.
Depending upon the technique used for estimation, we can obtain a quadratic speedup compared to
classical techniques which take Θ(M) time. Earlier quantum boosting algorithms like QAdaBoost and
QSmoothBoost also employ quantum estimation in a similar manner. The challenge in those and this work
4


<!-- page 5 -->
is to prove that boosting takes place despite the resulting errors from the estimations. We now discuss
several caveats arising from estimating the weights, which our algorithm addresses.
• Confidence-rated predictions: If we naively estimate the partition-label weights W j,t
b , then there is
no way to bound the estimated confidence-rated predictions β′
j,t with respect to the actual confidence-rated
predictions βj,t. In Claim 5 we show that we can bound the latter by carrying out relative error estimation
of the partition-label weights and then performing Laplace correction to deal with corner cases where the
values of W j,t
b
might be extremely small5 (which may lead to unbounded confidence rated predictions).
• Bounding the weights for each iteration: A similar issue arises for the distribution weights for the
next iteration, if we try to naively estimate the partition-label weights W j,t
b . We can only additively bound
the new normalization constant Z′
t with respect to Zt which does not guarantee that the updated weights
˜Dt+1
i
(for t≥1) are normalized or even sub-normalized. In Claim 6 we show that relative error estimation
of W j,t
b
can bound the quantity Z′
t with relative error. We define the new distribution update rule Eq. (1),
using which we prove Claim 3 which states that the sum of weights in the next iteration is bounded
in the range
h
1−4ε
1+ϵ,1
i
; this, in turn, guarantees that the resulting weights are close to a distribution.
• Behaviour of intermediate hypotheses: Assume that the quantum weak-learner A outputs a
hypothesis ht with high probability when given the “ideal” quantum state in which the probability (repre-
senting the weight) of each sample comes exactly from the distribution computed by RealBoost. We show
in Claim 4 that even when a “non-ideal” state, in which the probabilities are estimated with relative error,
is passed by Algorithm 1 to the weak learner, it still outputs the same hypothesis ht with high probability.
• Final hypothesis is good: Even with Claim 4, we still have to prove that our combined classifier H
satisfies an arbitrarily high number of training samples since the base classifiers themselves are weak.
Using Claim 7, we show that our combined classifier has a very small generalization error.
2
Preliminaries
2.1
PAC Learning
We assume familiarity with the PAC learning framework but we quickly go over the basic concepts. A
concept class C is a family of concepts and each concept c is a set of n-bit Boolean functions, one for each n.
Suppose we are given a set of M labelled examples S={(xi,yi=c(xi))|i∈[M]} (which we call the training
data) taken from an unknown distribution and an unknown concept c from a concept class C where xi∼D.
The objective of PAC learning is to “learn” the unknown target concept from the training data such that
the “learned” concept generalizes well to all points sampled from the same distribution D. We denote a
learning algorithm A as an (η,δ)-PAC learner for the concept class C, if it efficiently6 outputs a hypothesis
h such that with probability at least 1−δ (over its internal randomness) Pr
x∼D[h(x)̸=c(x)]≤η.
A γ-weak learner A is defined as a
 1
2−γ,δ
 
-PAC learner, where γ=O(1/nk) for k≥1 and a strong
learner is defined as a
 1
3,δ
 
-PAC learner, using δ≤1/3 in both cases. One of the most celebrated results of
PAC learning was produced by Schapire who showed that under the PAC learning model, the task of producing
strong learners from weak learners is not only possible but that the two notions of learning are inherently
equivalent [24]. In order to quantify how accurate our hypothesis is, we define two types of misclassification
errors: training error ˆ
err(h,S)=
Pr
(x,y)∈S[h(x)̸=y] which is defined with respect to the training set, and
generalization error err(h,D)=
Pr
(x,y)∼D[h(x)̸=y] which is defined with respect to arbitrary samples.
5Note that this issue does not arise due to our algorithm, but is in fact a property of the underlying classical algorithm
itself [25, 26]. Laplace correction is well studied in the machine learning literature, especially with respect to decision tree
classifiers [22] which naturally behave as domain-partitioning learners.
6The learner runs in time polynomial in M,1/η,1/δ, and n.
5


<!-- page 6 -->
2.1.1
Quantum PAC learning
Our algorithm is designed in the quantum PAC learning framework, introduced by Bshouty et al. [9], but
now extended to learners using quantum examples. In the classical setting a learner A can query multiple
samples from S (we denote its sample complexity as Q), while in the quantum setting [4, 18] we assume
that the examples are provided in the form of the state
 
1
√
M
P
xi∈S|xi,yi=c(xi)⟩
 ⊗Q
. We observe that
in order to simulate a classical learner, a quantum learner can measure this state to obtain Q examples
chosen uniformly at random (with replacement) from S. This state can be efficiently prepared with or
without the assumption of a quantum random access memory (aka. QRAM). To use a QRAM to prepare a
uniform superposition over the classical samples, we only incur an additive O(
√
MlogM) term in the query
complexity which retains our quantum speedup. We also note that the QRAM (if used) is only for state
preparation. For a detailed discussion on the preparation of quantum samples without a QRAM, we refer the
reader to Izdebski et al. [18], in which the authors only assume quantum query access to the training samples.
The definition of weak learning and strong learning generalize straightforwardly to quantum PAC learners
and all classical PAC learnable function classes are learnable in the quantum setting. The sample complexity
of quantum and classical PAC-learners too are equal up to constant factors[3]. A quantum PAC learner A has
access to several copies of the quantum example. A performs a POVM measurement at the end to obtain a
hypothesis h belonging to its associated hypothesis class HA. As in the earlier works on quantum PAC learn-
ing [2, 9], we also assume the ability to create and query a quantum oracle Oh from an obtained hypothesis h.
2.2
Boosting using Domain Partitioning Hypothesis
Consider weak learners with hypothesis that partition the domain space X of the inputs into a set of mutually
exclusive and exhaustive blocks {X1,X2,...,XC} such that for all x,x′∈Xj,j∈{1,...,C}, we have h′(x)=h′(x′).
Since the prediction is constant for all training samples assigned to a specific partition, we denote the prediction
h′ for the partition Xj by the constant βj. Note that analogous to the definition of h′, sign(βj) gives us the pre-
diction for the partition Xj while |βj| gives us the confidence of the prediction. Now, the task at hand reduces
to finding good values of βj for each Xj. We give an example to foster an intuitive way of thinking about the
confidence of predictions in this particular context. Suppose βj is calculated by taking the log of the ratio of
weighted fraction of examples with different labels. Consider a partition which contains 100 samples with label
−1 and 5 samples with label +1. Then the weighted prediction for that particular partition will be −1.3 which
means that we predict all samples in this partition to have a −1 label with a confidence rating of 1.3. Another
partition which contains 55 samples with label +1 and 45 samples with label −1 will have a weighted prediction
of 0.08. Here we see that because the majority has +1 label we assign it to the entire partition, but we do so
with a much lower confidence than in the previous case. This shows us that if there is almost an equal number
of samples of both labels in a particular domain, then the confidence for predicting either class will be quite low.
3
Quantum algorithm for boosting
In this section, we explain the QRealBoost algorithm in detail which is given in Algorithm 1. The input
to our QRealBoost algorithm consists of the weak learner A , and a set of M training samples S as copies
of the quantum state
1
√
M
P
i∈[M]|xi,yi⟩. Since the algorithm is adaptive we can make a worst case guess
for M (number of training samples), Q (sample complexity of A), C (number of partitions made by A),
γ (bias of A), and T (number of iterations of Algorithm 1); the algorithm will adapt to the intermediate
learners which use more optimistic estimates.
If RealBoost(Algorithm 4) computes the distribution Dt
i in the tth iteration, QRealBoost estimates it as
˜Dt
i in the tth iteration (additional care is taken since the latter may not actually be a probability distribution).
Similarly, RealBoost computes the confidence-rated predictions for the tth iteration and jth partition βj,t,
which QRealBoost estimates as β′
j,t. Following the earlier works on quantum boosting algorithms, we too
6


<!-- page 7 -->
assume that during an iteration in the outermost for-loop our algorithm, we have quantum query access to
the previous hypotheses h1,h2,...,ht−1 in the form of oracles Oh1,Oh2,...,Oht, respectively, and the confidence-
rated predictions β′
j,t, for all j∈{1,...,C}, t≥1 and Z′
t (for t≥1), which are stored in quantum registers.
3.1
Explanation of QRealBoost (Algorithm 1)
3.1.1
Preparing quantum examples for training A
We consider the tth iteration of the outermost loop. We first initialize Q copies of |φ0⟩and 2C copies of
|ψ0⟩both set to
1
√
M
P
i∈[M]
  xi,yi,D1
i
 
. Oracular access to all the previous hypotheses {h1,...,ht−1} can
be expressed as |xi,yi⟩|0⟩
Oht
−−→|xi,yi⟩
  jt
i
 
, where jt
i =ht(xi) refers to the domain partition of the ith sample
at the tth iteration.
We query each such oracle in order to produce Q+2C copies of |φ1⟩=|ψ1⟩=
1
√
M
P
i∈[M]
   xi,yi, ˜
D1
i
E  j1
i ,j2
i ,...,jt−1
i
 
.
Next, using the stored class weights β′
j,s, Z′
s, and the oracles to the hypotheses, we construct a uni-
tary mapping UD for updating the weight register using t −1 applications of Eq. (1) as follows:
|φ1⟩
UD
−−→|φ2⟩=
1
√
M
P
i∈[M]
   xi,yi, ˜
Dt
i
E  j1
i ,j2
i ,...,jt−1
i
 
. We perform this update too on all Q copies of
|φ1⟩and 2C copies of |ψ1⟩.
3.1.2
Training A to obtain a new hypothesis
For all Q copies of |φ2⟩, we perform a conditional rotation on the register
    ˜
Dt
i
E
to obtain the state
|φ3⟩=P
i∈[M]
1
√
M
   xi,yi, ˜
Dt
i
E  j1
i ,...,jt−1
i
  q
˜
Dt
i|1⟩+
q
1−˜
Dt
i|0⟩
 
. Let U0→3 be the unitary that performs
|0⟩−
→|φ3⟩. We perform Amplitude Amplification as stated in Theorem 10 on |φ3⟩to obtain the state |φ4⟩
(using O(
√
MlogT) applications of U0→3 and U−1
0→3) with probability at least O(1−1/T). The state |φ4⟩
is P
i∈[M]
q
˜
Dt
i
   xi,yi, ˜
Dt
i
E  j1
i ,...,jt−1
i
 
+|ζt⟩.
The state |ζt⟩is present since P
i∈[M] ˜
Dt
i ≤1 (i.e. the weights are sub-normalized). We state a claim
now (see Appendix A.2 for proof) that shows that the sum of the weights is very close to 1, and hence,
very little interference is expected from |ζt⟩.
Claim 3. For ˜Dt
i updated as given in Eq. (1) and t∈{1,...,T}, we can bound the sum of the sub-normalized
weights as P
i∈[M] ˜Dt
i ∈
h
1−4ε
1+ε,1
i
.
Now, uncompute by applying U−1
D and O−1
h1 ...,Oh−1
t−1 to |φ4⟩to obtain the state |φ5⟩=P
i∈[M]
q
˜
Dt
i|xi,yi⟩+
|ζ′
t⟩. We pass Q copies of |φ5⟩to the weak learner A. In turn, the weak learner produces a hypothesis
ht 7 to which we assume oracular access. The following claim (proof in Appendix A.3) shows that the
learned hypothesis is a good hypothesis.
Claim 4. If at the tth iteration, the γ-weak learner A produces a hypothesis ht on being fed Q copies of
the ideal state |φ′
5⟩=P
i∈[M]
p
Dt
i|xi,yi⟩, then A produces the same hypothesis ht with high probability when
given Q copies of |φ5⟩.
7We assume that the probability of A not producing any hypothesis is O(1/T), similar to earlier works [16].
7


<!-- page 8 -->
3.1.3
Obtaining confidence-rated predictions on sample points
At this point, we have 2C copies of |ψ2⟩=
1
√
M
P
i∈[M]
   xi,yi, ˜
Dt
i
E  j1
i ,j2
i ,...,jt−1
i
 
(from applying UD to |ψ1⟩)
and oracular access to the hypothesis ht. We perform the unitary transformation
[|ψ2⟩|0⟩]⊗2C Oht
−−→[|ψ3⟩]⊗2C =
O
k∈{1,...,C}
b∈{−1,+1}

1
√
M
X
i∈[M]
   xi,yi, ˜
Dt
i
E  j1
i ,j2
i ,...,jt−1
i
   jt
i
 


Consider the (k,b)th copy of [|ψ3⟩]⊗2C for k∈{1,2,...,C} and b∈{−1,+1}. Perform the update ψ3−
→ψ4 as
|ψ3⟩(k,b)
 
|0⟩⊗2 
−
→|ψ4⟩(k,b)=
1
√
M
X
i∈[M]
|xi⟩|yi⟩
    ˜Dt
i
E  j1
i ,...,jt
i
 
|
{z
}
|j(i,t)⟩
  I[jt
i =k]
 
|
{z
}
|I1⟩
|I[yi=b]⟩
|
{z
}
|I2⟩
.
Note here that I1 and I2 are binary valued states. Using |I1⟩and |I2⟩as controls, we obtain the state
|ψ5⟩(k,b)=
1
√
M
X
i∈[M]
|xi⟩|yi⟩
    ˜Dt
i
E
|j(i,t)⟩|I1⟩|I2⟩
    ˜Dt
i·I1·I2
E
|
{z
}
˜Dk,b,t
i
Now we perform a conditional rotation on the
    ˜Dk,b,t
i
E
register to obtain 8
|ψ6⟩(k,b)=
q
W k,t
b |χ⟩1
(k,b)|1⟩+
q
1−W k,t
b |χ⟩0
(k,b)|0⟩
Let V (k,b)
0,6
be the unitary that performs |0⟩−
→|ψ6⟩(k,b). We perform relative-error amplitude estimation as
stated in Theorem 11, with an expected ˜O(
√
MQT 2) queries to V (k,b)
0,6
and V −1 (k,b)
0,6
to obtain the quantity
˜W k,t
b
that is an estimate of W k,t
b
with high probability. We carry out relative error amplitude estimation
as stated in Theorem 11 to estimate the quantity W k,t
b .
Hence, we obtain all 2C values of ˜W j,t
b
for all j ∈{1,2,...,C}, b∈{−1,+1}. Note that it is possible for
the value of W j,t
b
to be very small (even zero) for some j. This would result in the quantities β′
j,t becoming
very large or unbounded, thus increasing the tendency of the learner to overfit. We use a general smoothing
technique known as Laplace correction[10] to overcome this issue9 and use the smoothed values to calculate
the margins as β′
j,t= 1
2ln
 
˜
W j,t
+
˜
W j,t
−
 
∀j∈{1,...,C} and the normalization constant as Z′
t=2PC
j=1
q
˜W j,t
+ ˜W j,t
−.
At this point, we can make the following two claims.
Claim 5. Let the weights be relatively estimated using the error parameter ε by Algorithm 1; i.e.
   W j,t
b −˜W j,t
b
   ≤ε·W j,t
b . Then the difference between the actual margins βj,t and the estimated margins β′
j,t
is bounded as
   β′
j,t−βj,t
   ≤1
2ln
 
1+ε
1−ε
 
; j∈{1,2,...,C}.
Claim 6. In the same setting as in Claim 5, the deviation in the normalization constant at every iteration
is bounded as |Z′
t−Zt|≤ε·Zt.
We present the proof of Claim 5 and Claim 6 in Appendix A.4 and Appendix A.5 respectively. From
Claim 5 we see that the difference between the actual margin and the estimated margins is very small. In
fact, a very simple calculation shows us that
   β′
j,t−βj,t
   ≤0.1 for ε≤0.1. We note that the error parameter ε
is far smaller than a constant fraction, which means our estimated margins are quite close to the ideal margin
values. Claim 6 shows that when we minimize the normalization constant at every step using the estimated
8Here
W k,t
b
=P
i∈[M] ˜Dk,b,t
i
/M,
|χ⟩1
(k,b)=
1
√
M
P
i∈[M]
q
˜
Dk,b,t
i
q
W k,t
b
|xi⟩|yi⟩
    ˜Dt
i
E
|j(i,t)⟩|I1⟩|I2⟩
    ˜Dk,b,t
i
E
and
|χ⟩0
(k,b)=
1
√
M
P
i∈[M]
q
1−˜
Dk,b,t
i
q
1−W k,t
b
|xi⟩|yi⟩
    ˜Dt
i
E
|j(i,t)⟩|I1⟩|I2⟩
    ˜Dk,b,t
i
E
9The details are explained in Appendix A.9.
8


<!-- page 9 -->
values ˜W j,t
b , these quantities are themselves relatively bounded by the actual normalization constant. This
implies that the training error of the combined classifier is greedily minimized when the normalization
constant is minimized at every step10. Hence, our training error at every step does not blow up due to
estimation of the partition weights. Now, we plug in the values of κ (as initialized in Algorithm 1), β′
j,t,
and Z′
t in to Eq. (1) to perform the update from ˜Dt
i to ˜Dt+1
i
for all i∈[M]. The output of the algorithm is
the final hypothesis H(x)=sign
 PT
t=1β′
j,t
 
where our weak learner assigns any training example x∼D the
jth partition at the tth iteration, and β′
j,t is the weighted prediction of the jth partition at the tth iteration.
3.2
Proof of correctness
The probability of failure of Algorithm 1 stems primarily from the steps 8, 9, and 14, where each step
fails with a probability at most O(1/T). When we take a union bound over all T iterations for all three
steps, the overall failure probability dips to an arbitrary constant which is at most 1/3. There is an extra
log factor incurred due to error reduction which can be absorbed in the ˜O(.) notation. We now state the
following claim regarding the training error of our algorithm the proof of which is included in Appendix A.6.
Claim 7. For a sufficiently large number of iterations T ≥lnM
2γ2 , our combined classifier H has zero training
error with respect to the uniform superposition ˜D1 with high probability.
From Claim 7 and Corollary 18, we also conclude that if we run Algorithm 1 for a sufficiently large
number of iterations T, then with a high probability we output a hypothesis H according to Eq. (2) that
has zero training error and a small generalization error.
3.3
Complexity Analysis
In this section we state the query complexity and the time complexity of our algorithm.
Theorem 8 (Query Complexity). Suppose we boost a γ-weak learner A with sample complexity Q, and an
associated hypothesis class H having VC dimension dH using Algorithm 1. If the weak learner A produces
at most C partitions at every iteration, then the query complexity of Algorithm 1 is O
 √dH·C·Q
γ9
 
.
Theorem 9 (Time Complexity). Suppose we boost a γ-weak learner A with sample complexity Q, and
an associated hypothesis class H having VC dimension dH using Algorithm 1. The size of the class C is
assumed to be n. If the weak learner A produces at most C partitions at every iteration, then the time
complexity of Algorithm 1 is O
 
n2√dHCQ
γ9
 
The proof of Theorem 8 is given in Appendix A.7 and the proof of Theorem 9 is given in Appendix A.8.
4
Experiments
We compare11 the generalization ability and convergence of QRealBoost against AdaBoost, RealBoost, and
QAdaBoost on the Breast Cancer Wisconsin dataset( see Fig. 1) and the MNIST dataset (see Fig. 2). Since
there does not exist any quantum simulators or actual quantum backends large enough to test QRealBoost,
we had to make some interesting choices and changes in the implementation which are detailed below.
1. We focus on qualitative analysis behaviour (training and test convergence) of the algorithms in these
experiments rather than its efficiency due to the lack of quantum simulators and quantum backends
with a sufficient number of qubits.
10See Theorem 15.
11Our code is freely available at https://github.com/braqiiit/QRealBoost.
9


<!-- page 10 -->
Figure 2: Comparing the performance of 4 different boosting algorithms using k-means (with k=3) as the weak learner
on the MNIST dataset for the digits 4 and 5 using 32 training samples. This experiment retains the binary classification
nature of the problem.
2. Instead of computing the distribution weights from scratch, we store the updated distribution weights
after every iteration. This is done because in the former approach the number of qubits needed to store
the weights up to a reasonable degree of precision blows up with the number of iterations, taking even
experiments with few training samples out of our reach. Even though this choice sacrifices the quantum
speedup it does not affect the convergence behaviour of QRealBoost.
3. We used a classical weak learner (k-means) since off-the-shelf quantum weak learners are not readily
available right now, and implementing one was out of the scope of this work. The implementation can
be easily modified to use any kind of learner implemented as a quantum circuit. We measure the |φ5⟩
state and pass the top Q training samples to the k-means algorithm. In Section 2.1.1 we pointed out
that this is exactly how quantum learners could simulate classical learners.
4. We use the the IterativeAmplitudeEstimation class provided by Qiskit which is an implementation of the
Iterative Quantum Amplitude Estimation (IQAE) algorithm [17] that replaces quantum phase estimation
with clever use of Grover iterations. Our choice was motivated by the availability and performance of the
algorithm which helped us decrease the number of qubits needed for the implementation. An important
note is the fact that even though the experiments were conducted with additive estimation instead of
relative estimation, we still managed to boost the weak learner.
The lines for QAdaBoost and QRealBoost represent a mean accuracy over 5 independent experiments,
while the hue bands represent the standard deviation across all experiments. The QAdaBoost and QRe-
alBoost algorithms are tested on quantum simulators (instead of actual quantum backends) due to quantum
resource limitations. We set the sample complexity of QAdaBoost and QRealBoost to be 8 for both sets
of experiments. All algorithms are trained on 32 samples for both sets of experiments, and we observe
the training accuracy and test accuracy over 25 iterations.
In the first experiment (see Fig. 1), we observe that both QRealBoost and QAdaBoost have similar
convergence rates w.r.t. training error that is better than RealBoost and completely dwarfs AdaBoost.
Moreover, QRealBoost converges faster than QAdaBoost and has a tighter deviation in training loss over
five experiments, especially in the early iterations. Even in terms of generalization ability, QRealBoost
completely outperforms QAdaBoost and RealBoost and is only surpassed by AdaBoost. In the second
experiment (see Fig. 2), RealBoost appears to overfit the training samples and suffers from the worst
generalization error out of all four algorithms. AdaBoost has a poor convergence rate and generalization
error as well. QRealBoost and QAdaBoost perform similarly in training accuracy, with QRealBoost narrowly
beating QAdaBoost via faster convergence and a tighter deviation. Regarding generalization abilities,
10

[CAPTION] Figure 2: Comparing the performance of 4 different boosting algorithms using k-means (with k=3) as the weak learner


<!-- page 11 -->
QAdaBoost loses out to QRealBoost in overall test accuracy and deviation over experiments, albeit with a
much smaller margin. These are encouraging observations, especially considering that QRealBoost trains on
8 samples at every iteration, while the classical algorithms have access to all the 32 samples every iteration.
5
Conclusions
In this work, we designed the QRealBoost algorithm which tackles an open question posed by Izdebski et
al. [18] to boost weak quantum PAC learners that output non-binary hypotheses. QRealBoost retains the
performance of RealBoost which has superior theoretical properties (supported by empirical evidence too)
as compared to the celebrated boosting algorithm, AdaBoost [14]. We also establish that both theoretically
and empirically QRealBoost outperforms QAdaBoost which is the only other known adaptive quantum
boosting algorithm.
An issue with QRealBoost is complexity of γ, arising from recomputing ˜Dt over the training samples at
every iteration from scratch. We believe that this computation can be avoided by maintaining a “distribution
oracle” which only needs to be updated in each iteration. If it turns out that the lower bound on γ is
worse for quantum boosting algorithms compared to classical boosting algorithms in the general case, the
next question would be finding (or even determining the existence of) relevant hypotheses classes in which
quantum boosting provides us with an advantage.
We also observe that the constant factor C in the numerator of the time complexity may be exponentially
reduced by simultaneously estimating the individual domain partition weights using amplitude estimation
techniques as shown in [30], and this is a possible direction of future work.
A logical continuation of this work is quantizing other variants of AdaBoost which depend on domain
partitioning hypotheses such as GentleBoost[15], ModestBoost[31], Parameterized AdaBoost[37], and
Penalized AdaBoost[38]. Each variant has different generalization abilities, which make them useful in
different contexts. The algorithmic framework followed in this work for estimating the partition weights
may be useful to model quantum versions of these variants.
References
[1] Andris Ambainis. Quantum search with variable times. In Pascal Weil Susanne Albers, editor, STACS
2008, pages 49–60, Bordeaux, France, feb 2008. IBFI Schloss Dagstuhl.
[2] Srinivasan Arunachalam and Ronald de Wolf. Guest column: A survey of quantum learning theory.
ACM SIGACT News, 48(2):41–67, 2017.
[3] Srinivasan Arunachalam and Ronald De Wolf. Optimal quantum sample complexity of learning
algorithms. The Journal of Machine Learning Research, 19(1):2879–2878, 2018.
[4] Srinivasan Arunachalam and Reevu Maity. Quantum boosting. In 37th Int. Conf. Mach. Learn. ICML
2020, volume PartF16814, pages 354–364. PMLR, nov 2020. arXiv:2002.05056.
[5] Eric Bauer and Ron Kohavi. An empirical comparison of voting classification algorithms: Bagging,
boosting, and variants. Machine learning, 36(1):105–139, 1999.
[6] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd.
Quantum machine learning. Nature, 549(7671):195–202, 2017.
[7] Gilles Brassard, Peter Høyer, Michele Mosca, and Alain Tapp. Quantum amplitude amplification and es-
timation. In Quantum Comput. Inf. ({W}ashington,{DC}, 2000), volume 305 of Contemp. Math., pages
53–74. Amer. Math. Soc., Providence, RI, may 2002. arXiv:0005055, doi:10.1090/conm/305/05215.
11


<!-- page 12 -->
[8] Nader H Bshouty and Dmitry Gavinsky. On boosting with polynomially bounded distributions.
Journal of Machine Learning Research, 3(Nov):483–506, 2002.
[9] Nader H Bshouty and Jeffrey C Jackson. Learning dnf over the uniform distribution using a quantum
example oracle. SIAM Journal on Computing, 28(3):1136–1153, 1998.
[10] Peter Clark and Tim Niblett. The cn2 induction algorithm. Machine learning, 3(4):261–283, 1989.
[11] Thomas G Dietterich. An experimental comparison of three methods for constructing ensembles
of decision trees: Bagging, boosting, and randomization. Machine learning, 40(2):139–157, 2000.
[12] Harris Drucker and Corinna Cortes. Boosting decision trees. Advances in neural information processing
systems, pages 479–485, 1996.
[13] Yoav Freund. Boosting a weak learning algorithm by majority.
Information and computation,
121(2):256–285, 1995.
[14] Yoav Freund and Robert E. Schapire.
A Decision-Theoretic Generalization of On-Line
Learning and an Application to Boosting.
J. Comput. Syst. Sci., 55(1):119–139, aug 1997.
doi:10.1006/JCSS.1997.1504.
[15] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Additive logistic regression: a statistical
view of boosting (With discussion and a rejoinder by the authors). Ann. Stat., 28(2), apr 2000.
doi:10.1214/aos/1016218223.
[16] Dmitry Gavinsky. Optimally-smooth adaptive boosting and application to agnostic learning. Journal
of Machine Learning Research, 4(May):101–117, 2003.
[17] Dmitry Grinko, Julien Gacon, Christa Zoufal, and Stefan Woerner. Iterative quantum amplitude
estimation. npj Quantum Information, 7(1):1–6, 2021.
[18] Adam Izdebski and Ronald de Wolf. Improved Quantum Boosting. sep 2020. arXiv:2009.08360.
[19] Hartmut Neven, Vasil S. Denchev, Geordie Rose, and William G. Macready. Qboost: Large scale
classifier training with adiabatic quantum optimization. In Steven C. H. Hoi and Wray Buntine, editors,
Proceedings of the Asian Conference on Machine Learning, volume 25 of Proceedings of Machine Learning
Research, pages 333–348, Singapore Management University, Singapore, 04–06 Nov 2012. PMLR.
[20] David Opitz and Richard Maclin. Popular ensemble methods: An empirical study. Journal of artificial
intelligence research, 11:169–198, 1999.
[21] Fabian Pedregosa, Ga¨el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier
Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine
learning in python. the Journal of machine Learning research, 12:2825–2830, 2011.
[22] Foster Provost and Pedro Domingos. Tree induction for probability-based ranking. Machine learning,
52(3):199–215, 2003.
[23] Chengxiong Ruan, Qiuqi Ruan, and Xiaoli Li. Real adaboost feature selection for face recognition.
In IEEE 10th INTERNATIONAL CONFERENCE ON SIGNAL PROCESSING PROCEEDINGS,
pages 1402–1405, 2010. doi:10.1109/ICOSP.2010.5656917.
[24] Robert E. Schapire. The strength of weak learnability. Machine Learning, 5(2):197–227, Jun 1990.
doi:10.1007/BF00116037.
[25] Robert E. Schapire and Yoav Freund. Boosting: Foundations and Algorithms. The MIT Press, 2012.
12


<!-- page 13 -->
[26] Robert E. Schapire and Yoram Singer. Improved boosting algorithms using confidence-rated predictions.
Mach. Learn., 37(3):297–336, dec 1999. doi:10.1023/A:1007614523901.
[27] Maria Schuld and Francesco Petruccione. Quantum ensembles of quantum classifiers. Sci. Rep.,
8(1):2772, dec 2018. doi:10.1038/s41598-018-20403-3.
[28] Maria Schuld, Ilya Sinayskiy, and Francesco Petruccione. An introduction to quantum machine
learning. Contemporary Physics, 56(2):172–185, 2015.
[29] Rocco A Servedio. Smooth boosting and learning with malicious noise. The Journal of Machine
Learning Research, 4:633–648, 2003.
[30] Joran van Apeldoorn. Quantum probability oracles & multidimensional amplitude estimation. In
16th Conference on the Theory of Quantum Computation, Communication and Cryptography (TQC
2021). Schloss Dagstuhl-Leibniz-Zentrum f¨ur Informatik, 2021.
[31] Alexander Vezhnevets and Vladimir Vezhnevets. Modest adaboost-teaching adaboost to generalize
better. In Graphicon, volume 12, pages 987–997, 2005.
[32] Sisong Wang and Han Wang. 2d staircase detection using real adaboost. In 2009 7th International
Conference on Information, Communications and Signal Processing (ICICS), pages 1–5. IEEE, 2009.
[33] XiMing Wang, YueChi Ma, Min-Hsiu Hsieh, and Man-Hong Yung. Quantum speedup in adaptive
boosting of binary classification.
Sci. China Physics, Mech. Astron. 2020, 64(2):1–10, dec 2020.
doi:10.1007/S11433-020-1638-5.
[34] Peter Wittek. Quantum machine learning: what quantum computing means to data mining. Academic
Press, 2014.
[35] Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian. UCI machine learning repos-
itory, 1995.
URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+
(Diagnostic).
[36] Bo Wu, Haizhou Ai, Chang Huang, and Shihong Lao. Fast rotation invariant multi-view face detection
based on real adaboost. In Sixth IEEE International Conference on Automatic Face and Gesture
Recognition, 2004. Proceedings., pages 79–84, 2004. doi:10.1109/AFGR.2004.1301512.
[37] Shuqiong Wu and Hiroshi Nagahashi. Parameterized adaboost: introducing a parameter to speed
up the training of real adaboost. IEEE Signal Processing Letters, 21(6):687–691, 2014.
[38] Shuqiong Wu and Hiroshi Nagahashi. Penalized adaboost: improving the generalization error of gentle
adaboost through a margin distribution. IEICE TRANSACTIONS on Information and Systems,
98(11):1906–1915, 2015.
A
Proofs
In this section, we present the proofs of the claims made in the earlier section. First, we restate some
well-known quantum subroutines that we use throughout this work to prove our main results.
13


<!-- page 14 -->
A.1
Quantum subroutines for amplitude amplification and estimation
Theorem 10 (Amplitude Amplification [7]). Let there be a unitary U such that U|0⟩=√a|φ0⟩+√1−a|φ1⟩
for an unknown a>0. If a>p>0, then there exists a quantum amplitude amplification algorithm that
outputs the state |φ0⟩with a probability p′>0. The expected number of calls to U and U−1 made by our
quantum amplitude amplification algorithm is Θ(
p
p′/p).
Theorem 11 (Relative Error Estimation [1]). Given an error parameter ε, a constant k≥1, and a unitary
U such that U|0⟩outputs 1 with probability at least p. Then there exists a quantum amplitude estimation
algorithm that produces an estimate ˜a of the success probability a with probability at least 1−1
2k such that
|a−˜a| ≤εa where a ≥p. The expected number of calls to U and U−1 made by our quantum amplitude
estimation algorithm is
O
  k
ε√p
 
1+loglog1
p
  
(3)
A.2
Proof of Claim 3 (bound on subnormalized sum)
We restate the new distribution update rule as given in Algorithm 1
˜Dt+1
i
=
˜Dt
iexp
 
−β′
j,t·yi
 
κZ′
t
(4)
Before getting into the actual proof, we make two observations. We can see from |φ5⟩that our weights
are sub-normalised. This gives us a trivial upper bound
X
i∈[M]
˜Dt
i ≤1; ∀t≥1
(5)
We now make our second observation:P
i∈[M] ˜Dt
iexp
 
−β′
j,t·yi
 
PC
j=1
 
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
   =1.
(6)
We can arrive at the observation in (6) by following the arguments in Schapire-Singer. Let us start by
obtaining a preliminary bound on the quantity P
i ˜Dt+1
i
as
P
i∈[M] ˜Dt
iexp
 
−β′
j,t·yi
 
κZ′
t
=
P
i∈[M] ˜Dt
iexp
 
−β′
j,t·yi
 
PC
j=1
 
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
  ·
PC
j=1
 
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
  
κZ′
t
=
PC
j=1
 
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
  
κZ′
t
=
PC
j=1
 
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
  
2κPC
j=1
q
˜W j,t
+ · ˜W j,t
−
= 1
2κ
C
X
j=1
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
 
q
˜W j,t
+ · ˜W j,t
−
(7)
The first equality follows from plugging in Eq. (1). The third equality follows from Eq. (6). In the fourth,
equality we use the value of Z′
t given in Algorithm 1. Now we upperbound and lowerbound the quantity
14


<!-- page 15 -->
P
i ˜Dt+1
i
by plugging in (24) as
X
i∈[M]
˜Dt+1
i
≤
1
2κ(1−ε)
C
X
j=1
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
 
q
W j,t
+ ·W j,t
−
(8)
X
i∈[M]
˜Dt+1
i
≥
1
2κ(1+ε)
C
X
j=1
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
 
q
W j,t
+ ·W j,t
−
(9)
Substituting κ=
C
(1−ε)
q
1+ε
1−ε in Eq. (8) we have
X
i∈[M]
˜Dt+1
i
≤
1
2κ(1−ε)
C
X
j=1
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
 
q
W j,t
+ ·W j,t
−
=
1
2κ(1−ε)
C
X
j=1
v
u
u
tW j,t
+
W j,t
−
·
v
u
u
t ˜W j,t
−
˜W j,t
+
+
v
u
u
tW j,t
−
W j,t
+
·
v
u
u
t ˜W j,t
+
˜W j,t
−
≤
1
2κ(1−ε)
C
X
j=1
 
2
r
1+ε
1−ε
!
=
C
κ(1−ε)
r
1+ε
1−ε =1
(10)
Similarly, substituting κ=
C
(1−ε)
q
1+ε
1−ε in Eq. (9) we have
X
i∈[M]
˜Dt+1
i
≥
1
2κ(1+ε)
C
X
j=1
W j,t
+ exp
 
−β′
j,t
 
+W j,t
−exp
 
β′
j,t
 
q
W j,t
+ ·W j,t
−
=
1
2κ(1+ε)
C
X
j=1
v
u
u
tW j,t
+
W j,t
−
·
v
u
u
t ˜W j,t
−
˜W j,t
+
+
v
u
u
tW j,t
−
W j,t
+
·
v
u
u
t ˜W j,t
+
˜W j,t
−
≥
C
κ(1+ε)
r
1−ε
1+ε =
 1−ε
1+ε
 2
=
 
1−2ε
1+ε
 2
≥1−4ε
1+ε
(11)
Combining Eq. (10) and Eq. (11) we have for any t=1,2,...,T
X
i∈[M]
˜Dt
i ∈
 
1−4ε
1+ε,1
 
(12)
A.3
Proof of Claim 4 (hypotheses are learned correctly w.h.p.)
Let us assume that when we supply Q copies of the state P
i∈[M]
p
Dt
i|xi,yi⟩we obtain the hypothesis
ht with probability ρ. We want to bound the probability σ of obtaining the same hypothesis ht when we
give our weak learner A the state φ5. Before we dive into the calculations, we define a few terms.
Definition 12 (Fidelity). Fidelity is a measure of the closeness of two quantum states. When we have
two pure states |ψ⟩and |φ⟩we define Fidelity between the two states as
F(ψ,φ)=F(φ,ψ)=|⟨ψ|φ⟩|2
(13)
Let ρ and σ be the density matrices of ψ and φ respectively. An alternate characterization of Fidelity in
15


<!-- page 16 -->
terms of density matrices is
F(ρ,σ)=
√ρ√σ

1
(14)
Definition 13 (Normalized Trace Distance). Trace distance is another measure of closeness between two
quantum states. If there is a set of POVMs {E}, then the POVM leading to the largest difference in
measurement outcomes between two quantum states is the trace distance.
D(ρ,σ)= 1
2∥ρ−σ∥1=max
Ei
X
i
|Tr{E(ρ−σ)}|
(15)
When ρ and σ are density matrices of pure states, Trace distance is related to Fidelity as follows:
D(ρ,σ)=
p
1−F(ρ,σ)
(16)
Let p be the probability that A outputs the hypothesis ht on being fed Q copies of the ideal state
|φ′
5⟩=P
i∈[M]
p
Dt
i|xi,yi⟩. Let q be the probability that A outputs the hypothesis ht on being fed Q copies
of the state |φ5⟩. We want to bound the quantity |p−q| and show that this is a small quantity. We denote
the class of POVMs on the hypothesis space H as {Eh}h∈H such that P
h∈HEh=I. Then by the above
definitions of trace distance and fidelity we have
|p−q|≤max
{Eh}|Tr{Eh(ρ−σ)}|≤
X
h∈H
|Tr{Eh(ρ−σ)}|
=D(ρ−σ)=
p
1−F(ρ,σ)=
 
1−
   
 
φ5
  φ′
5
  Q   
2  1
2
≤
 
1−
  
φ5
  φ′
5
   2Q
  1
2
(17)
Now we bound the quantity |⟨φ5|φ′
5⟩|.
  
φ5
  φ′
5
   =
    
q
˜Dt
i·Dt
i+

ζt
  φ′
5
     ≥
    
q
˜Dt
i·Dt
i
    −
  
ζt
  φ′
5
   
(18)
16


<!-- page 17 -->
Let us bound the term
q
˜Dt+1
i
·Dt+1
i
first.
˜Dt+1
i
·Dt+1
i
=
X
i∈[M]
s
˜Dt
i·e−β′
j,t
κ·Z′
t
·
˜Dt
i·e−β′
j,t
Zt
=
s
Zt
κ·Z′
t
X
i∈[M]
˜Dt
i·e−β′
j,t
Zt
=
s
Zt
κ·Z′
t
= 1
√κ·
s
Zt
Z′
t
= 1
√κ·
C
X
j=1
v
u
u
u
t
W j,t
+ e−β′
j,t+W j,t
−eβ′
j,t
2
q
˜W j,t
+ · ˜W j,t
−
=
1
√
2κ·
C
X
j=1
v
u
u
u
t
W j,t
+ e−β′
j,t
q
˜W j,t
+ · ˜W j,t
−
+
W j,t
−eβ′
j,t
q
˜W j,t
+ · ˜W j,t
−
=
1
√
2κ·
C
X
j=1
v
u
u
u
t
W j,t
+
q
˜W j,t
+ · ˜W j,t
−
v
u
u
t ˜W j,t
−
˜W j,t
+
+
W j,t
−
q
˜W j,t
+ · ˜W j,t
−
v
u
u
t ˜W j,t
+
˜W j,t
−
=
1
√
2κ·
C
X
j=1
v
u
u
tW j,t
+
˜W j,t
+
+W j,t
−
˜W j,t
−
≥
1
√
2κ·
C
X
j=1
r
2
1+ε =
s
C
κ(1+ε)
(19)
We also know that |⟨ζt|φ′
5⟩|≤∥ζt∥≤1−(1−4ε
1+ϵ)= 4ε
1+ϵ. Therefore we have
  
φ5
  φ′
5
   ≥
s
C
κ(1+ε)−4ε
1+ϵ
(20)
Substituting κ= C
1−ε
q
1+ε
1−ε in the above equation, we have
  
φ5
  φ′
5
   ≥
s
C·(1−ε)
C·(1+ε)
1−ε
1+ε−4ε
1+ϵ
=
 1−ε
1+ε
 3/4
−4ε
1+ϵ =
 
1−2ε
1+ε
 3/4
−4ε
1+ϵ
(21)
Since (1−x)t≥1−xt,∀x≤1,t>0, we have
  
φ5
  φ′
5
   ≥1−
3ε
2(1+ε)−4ε
1+ϵ =1−
11ε
2(1+ε)
(22)
Plugging this back into (17), we get
|p−q|≤2
s
1−
 
1−
11ε
2(1+ε)
 2Q
≤2
r
11Qε
1+ε <8
p
Qε
(23)
We now set ε=
1
QT 2 which gives us q=O(1−1
T ) if p=O(1−1
T ).
A.4
Proof of Claim 5 (margin estimation)
We know that
   W j,t
b −˜W j,t
b
   ≤ε·W j,t
b
(24)
17


<!-- page 18 -->
Also, recall that the actual margin given in Algorithm 4 is βj,t = 1
2ln
 
W j,t
+
W j,t
−
 
and the estimated margin
in Algorithm 1 is β′
j,t= 1
2ln
 
˜
W j,t
+
˜
W j,t
−
 
. We upper bound the difference in margins as follows:
β′
j,t−βj,t= 1
2
"
ln
˜W j,t
+
˜W j,t
−
−lnW j,t
+
W j,t
−
#
= 1
2
"
ln
˜W j,t
+
W j,t
+
−ln
˜W j,t
−
W j,t
−
#
≤1
2[ln(1+ε)−ln(1−ε)]
= 1
2ln
 1+ε
1−ε
 
(25)
Similarly, we obtain the lower bound as
β′
j,t−βj,t≥1
2ln
 1−ε
1+ε
 
(26)
Combining Eq. (25) and Eq. (26) we get
  β′
j,t−βj,t
  ≤1
2ln
 1+ε
1−ε
 
(27)
A.5
Proof of Claim 6 (normalization constant is bounded)
The normalization constant in Algorithm 4 is calculated as Zt=2PC
j=1
q
W j,t
+ ·W j,t
−. In Algorithm 1, we
substitute the weights with out estimated weights to obtain the quantity Z′
t=2PC
j=1
q
˜W j,t
+ · ˜W j,t
−. Using
(24), we upper bound the difference between the quantities as
Z′
t=2
C
X
j=1
q
˜W j,t
+ · ˜W j,t
−
≤2
C
X
j=1
q
W j,t
+ (1+ε)·W j,t
−(1+ε)
=2(1+ε)
C
X
j=1
q
W j,t
+ ·W j,t
−
=Zt(1+ε)
(28)
Similarly, the lower bound is obtained as
Z′
t=2
C
X
j=1
q
˜W j,t
+ · ˜W j,t
−
≥2
C
X
j=1
q
W j,t
+ (1−ε)·W j,t
−(1−ε)
=2(1−ε)
C
X
j=1
q
W j,t
+ ·W j,t
−
=Zt(1−ε)
(29)
Combining Eq. (28) and Eq. (29) we obtain  Z′
t−Zt
  ≤ε·Zt
(30)
18


<!-- page 19 -->
A.6
Proof of Claim 7 (final hypothesis has zero training error)
For this proof, we follow the framework followed by Freund and Shapire in their book on boosting [25].
From Eq. (1) we have
˜DT+1
i
=
˜D1
i
QT
t=1κ·Z′
t
·exp
 
−yi
T
X
t=1
β′
j,t
!
(31)
Let x∼D and |x,0⟩
Oht
−−→
  x,jt 
for all t∈{1,...,T} and F(x)=PT
t=1β′
j,t. Here, note that H(x)=sign(F(x))
according to Eq. (2). Since error means the hypothesis gives a different output than the label, we have,
H(x)̸=y =⇒y·F(x)≤0
=⇒exp(−yF(x))≥1
=⇒exp(−yF(x))≥I[H(x)̸=y]
(32)
The last inequality follows from the fact that I[H(x)̸=c(x)]∈{0,1}. Now if we try to upper bound the
training error, we have
Pr
x∼˜D1[H(x)̸=y]=
M
X
i=1
˜D1
i ·I[H(x)̸=y]≤
M
X
i=1
˜D1
i ·exp(−yiF(xi))
=
M
X
i=1
˜D1
i ·exp
 
−yi
T
X
t=1
β′
j,t
!
=
M
X
i=1
˜DT+1
i
T
Y
t=1
κ·Z′
t
≤
T
Y
t=1
κ·Z′
t
(33)
The last inequality is due to the fact that P
i∈[M] ˜Dt
i ∈
h
1−4ε
1+ε,1
i
(as given in Claim 3). Now from Claim 6,
we know that Z′
t≤Zt(1+ε). This means
Pr
x∼˜D1[H(x)̸=y]≤
T
Y
t=1
κ·Zt(1+ε)=κT(1+ε)T
T
Y
t=1
Zt
(34)
Substituting κ=
C
(1−ε)
q
1+ε
1−ε, and the fact that ε=O(1/QT 2) we have
Pr
x∼˜D1[H(x)̸=y]≤CT
 1+1/T 2
1−1/T 2
 T T
Y
t=1
Zt
≤CT
 T 2+1
T 2−1
 T T
Y
t=1
Zt
≤CT
 
1+
2
T 2−1
 T T
Y
t=1
Zt
≤CTexp
  2T
T 2−1
  T
Y
t=1
Zt
(35)
19


<!-- page 20 -->
For sufficiently large T, we have
Pr
x∼˜D1[H(x)̸=y]≤CTe2/T
T
Y
t=1
Zt
≤CTe2/T
T
Y
t=1
q
1−4γ2
t
≤CTexp
 
2
T −2
T
X
t=1
γ2
t
!
≤CTexp
  2
T −2γ2T
 
≤CTexp
 
−2γ2T + 2
T
 
(36)
The second inequality follows from Theorem 15. We follow the classical analysis from there on to its logical
conclusion. For T =O(logM/γ2) and a large constant in O(.) 12, we obtain
Pr
x∼˜D1[H(x)̸=y]< 1
M
(37)
We recall the fact that ˜D1 is the uniform distribution, which implies that we have zero training error.
A.7
Proof of Query Complexity
We now analyze the query complexity of our algorithm. The query complexity, as in previous works [4, 18],
considers the number of queries to the hypothesis oracles {Oh1,...,OhT } made by Algorithm 1. We now
start calculating the query complexity by considering the queries made in the tth iteration.
We require t−1 queries to the oracles Oh1,Oh2,...,Oht−1 for each copy of |ψ⟩0 and |φ⟩0. Using Theorem 10,
we see that our amplitude amplification algorithm uses an expected Θ(p′logT/p) calls to the unitaries
U0−
→3 and U−1
0−
→3, to obtain |φ4⟩with a high probability (as discussed in Footnote 7). We observe from
|φ3⟩and |φ4⟩that
p=
X
i∈[M]
q
˜Dt
i/M; p′=
X
i∈[M]
q
˜Dt
i
Hence, the Amplitude Amplification step to obtain |φ4⟩requires O(
√
MlogT(t−1)) queries to the oracles
for each copy of |φ3⟩. The uncompute step to obtain |φ5⟩requires a further t−1 queries to the oracles
Oh1,Oh2,...,Oht−1 for each copy of |φ4⟩.
For estimating the partition weights with high probability (as discussed in Footnote 7) we make an
expected ˜O
 √
MQT 2logT ·t
 
queries to Oh1,Oh2,...,Oht. We obtain this by plugging in p = O(1/M),
ε=O(
1
QT 2), and k=logT in Theorem 11. Hence, the total query complexity is
T
X
t=1



O(
√
MQlogT ·(t−1))
|
{z
}
Amplitude Amplification
+ O((Q+C)logT ·(t−1))
|
{z
}
weight updates and uncomputing
+˜O
 √
MCQT 2logT ·t
 
|
{z
}
Amplitude Estimation




=O(
√
MQT 2logT)+O((Q+C)T 2logT)+ ˜O
 √
MCQT 4logT
 
= ˜O(
√
MCQT 4)=O
 √
MCQ
γ8
!
(38)
The last equality follows from Theorem 15 by setting T =O(logM/γ2). From Corollary 18, and by setting
12such that the outer constant is taken care of as well
20


<!-- page 21 -->
the parameter η=0.1, we get the query complexity as O
 √dH·C·Q
γ9
 
.
A.8
Proof of Time Complexity
We now discuss the time complexity of Algorithm 1. As discussed in Section 2.1.1, we can assume a
QRAM to prepare the uniform superposition
1
√
M
P
i∈[M]
  xi,yi,D1
i
 
using O(nlogM) gates. Hence the time
complexity for preparing the state |φ0⟩⊗Q⊗|ψ0⟩⊗2C is O(n(Q+C)). The step from |φ0⟩to |φ1⟩and |ψ0⟩
to |ψ1⟩requires t−1 queries each, which can be performed in time O((Q+C)(t−1)). Next we perform the
distribution update which is an arithmetic operation, using the unitary UD with the
  j1
i ,...,jt−1
i
 
register
as control. This step requires time O(n2(Q+C)(t−1)).
We perform amplitude amplification to obtain the state |φ4⟩. This requires O(
√
M(t−1)logT) ap-
plications of U0−
→3 and U−1
0−
→3 as discussed in the previous section. The total time taken is therefore
O(n2√
MQ(t−1)logT). The time taken by our weak learner to output Oht is O(n2Q).
The arithmetic operations to update state |ψ3⟩(k,b) to |ψ3⟩(k,b) and perform controlled rotation use O(n)
gates. Finally we make ˜O
 √
MCQT 2logT
 
queries for the amplitude estimation part, and each query
requires time O(n2t). Therefore our final time complexity is
T
X
t=1



O
 
n2√
MQ(t−1)logT
 
|
{z
}
Amplitude Amplification
+˜O
 
n2√
MCQT 2logT ·t
 
|
{z
}
Amplitude Estimation
+O
 n2(Q+C)(t−1)
 
|
{z
}
other operations




= ˜O
 
n2√
MCQT 4logT
 
=O
 n2√dHCQ
γ9
 
(39)
A.9
Explanation of Laplace Correction
Let V k,t
b
= ˜W k,t
b ·M. We update the values of ˜W k,t to V k,t
b
+1
M+2C . We also note that
β′
j,t= 1
2ln
 ˜W j,t
+
˜W j,t
−
!
∀j∈{1,...,C}
(40)
Let us look at the corner cases now. If there exists a partition where W k,t
b
=0 or very small, then we now have
˜W k,t
b
∼
1
M+2C ∼1
M , essentially resetting the weight. On the other hand, consider a partition where V k,t
b
∼M.
This implies that V k,t
−b ∼0. By Eq. (40), this would give us unbounded margins β′
k,t= 1
2ln
 
V k,t
b
V k,t
−b
 
∼∞. Now,
due to the smoothing, the confidence for this domain partition will still be large but bounded above by
O(logM).
B
Boosting
Consider the following problem. Let us have query access to a learning algorithm A that is “weak”, i.e.
A performs slightly better than random guessing with respect to some unknown target concept class C.
Is it possible for us to obtain a “strong” learner that has small empirical and generalization error with
respect to C, using just oracular access to A?
In [24], Schapire showed us that under the PAC learning model, the task of producing strong learners
from weak learners is not only possible but that the two notions of learning are inherently equivalent.
Theorem 14 (Equivalence of Weak and Strong learning [24]). An unknown concept class C =S
n≥1Cn
is efficiently weakly PAC learnable if and only if C is efficiently strongly PAC learnable.
21


<!-- page 22 -->
Algorithm 2 The AdaBoost Algorithm
1: Input: Classical weak learner A, and M Training Samples {(x1,y1),(x2,y2),...,(xM,yM)}
; xi∼X,yi∈{−1,+1}
2: Initialize: Set D1
i = 1
M ∀i∈[M]
3: for t=1 to T do
4:
Train A using the distribution Dt to obtain the hypothesis ht:X −
→{−1,+1}.
5:
Compute the weighted error ϵt and the margin αt for this iteration as follows
ϵt=
X
i∈[M]
Dt
i[ht(xi)̸=yi]; αt= 1
2ln
 1−ϵt
ϵt
 
(41)
6:
Set Zt=P
i∈[M]Dt
i·exp(−αtyiht(xi)).
7:
Perform the distribution update ∀i∈{1,...,M} as follows
Dt+1
i
= Dt
i·exp(−αtyiht(xi))
Zt
(42)
Output: Hypothesis H(x) where x∼X.
H(x)=sign
 T
X
t=1
αtht(xi)
!
(43)
Subsequently, researchers started proposing boosting algorithms ([13, 24]) (among other types of ensemble
learning algorithms) to achieve this task. These algorithms are known as boosting algorithms since they
somehow “boost” the weak learner to produce a strong learner. These early efforts ultimately culminated in
the formation of the adaptive boosting or AdaBoost algorithm [14] (described in Algorithm 2). A small
point to note here is that the definitions of weak and strong learning generalize to the quantum setting
simply by considering that the learner is quantum.
B.1
AdaBoost and some generalizations
The AdaBoost algorithm takes two inputs - a classical γ-weak learner A, and M training samples. At
the tth step, we define a new distribution Dt over the training samples based on Dt−1. We then use A
to produce a new binary-valued hypothesis with respect to Dt. We compute the weighted error ϵt and the
confidence of the hypothesis αt for the tth step. Using these quantities we define the distribution update rule
Eq. (42) for the next iteration. After at least T ≥O(logM/γ2) iterations we produce the hypothesis H as
given in Eq. (43). Freund and Schapire showed that H has zero training error and very small generalization
error given the number of training samples is sufficiently large.
In AdaBoost for binary classification, we had access to M training samples in S which were distributed
according to some unknown distribution D over the domain space X. Given S as input, our weak learner
A output a hypothesis h:X −
→{−1,+1}. Consider the generalized version of the AdaBoost algorithmic
framework in which the weak learner outputs real valued hypotheses h′ :X −
→R. Here sign[h′(x)] is our
required prediction, and the quantity |h′(x)| gives us the “confidence” for the prediction.
Alternatively, the quantity |h′(x)| tells us how confident our learner is while making the prediction
sign[h′(x)]. This is analogous to the formal notion of margins which is well known in learning theory. Larger
margins on training data directly lead us to better bounds on the generalization error [25]. In fact, boosting
22


<!-- page 23 -->
algorithms in the AdaBoost framework tend to keep becoming more confident with their predictions leading
to a drop in generalization error even after training error converges to zero.
In [26] it was shown that the generalized model with real-valued hypotheses still adheres to the bound
given in Theorem 15 if h′:X −
→[−1,+1]. In fact, so far both models of weak learners use Theorem 15 to
focus on weak learners such that their hypotheses focus on greedily minimizing the normalization constant Zt
(refer Algorithm 2) at each iteration in order to bound the training error. We can therefore consider folding
the margin and the hypothesis into one quantity in order to simplify the calculation of Zt as in the case
of the generalized AdaBoost where −yih′
t(xi) can replace the term −αtyih′
t(xi). In Section 2.2 we explore a
different flavour of weak learners introduced in [15, 26] that also focus on this particular simplified criteria.
Algorithm 3 The AdaBoost framework
1: Input: Weak learner A with access to M Training samples {(x1,y1),(x2,y2),...,(xM,yM)}
where xi are examples and yi are the corresponding labels.
2: Initialize: Set D1
i = 1
M ∀i∈[M]
3: for t=1 to T do
4:
Train A using the distribution Dt to obtain the hypothesis ht.
5:
Compute the weighted misclassification error over all samples and the confidence of the hypothesis
for this iteration by comparing the predicted value and the label for each sample.
6:
Update the distribution to Dt+1 using the computed errors and confidences.
7: Output: Hypothesis H(x) which combines the individual hypothesis ht,∀t ≥1 according to their
computed confidences.
B.2
Error Bounds and Sample Complexity
In this section, we state a few theorems and definitions related to training and generalization error bounds
as well as sample complexity of the algorithms that follow the AdaBoost framework, for example, RealBoost
(Algorithm 4).
Theorem 15 (Upper Bound on Training Error [25]). Let A be a γ-weak learner. Let γt= 1
2−ϵt, where ϵt is
misclassification error at every iteration of Algorithm 4. Let D1 be an arbitrary initial distribution over the
training set. Then the weighted training error of the combined classifier H with respect to D1 is bounded as
ˆ
err(H)≤
T
Y
t=1
Zt≤
T
Y
t=1
q
1−4γ2
t ≤exp
 
−2
T
X
t=1
γ2
t
!
(47)
If we look at the first inequality in Theorem 15, we observe that the AdaBoost framework minimizes the
training error of the combined hypothesis by greedily minimizing the normalization constant in Algorithm 4
at every step. This produces the following corollary that we shall use later.
Corollary 16. Let A be a γ-weak learner and D1 be the uniform distribution over the training set of M
examples. Then the training error of the combined classifier H with respect to D1 goes to 0 when T > lnM
2γ2 ,
where T is the number of iterations of our boosting algorithm.
We see that greedily minimizing the training error at every step leads to the algorithm converging
exponentially fast in terms of training samples.
The next theorem gives us bounds on the generalization error in the AdaBoost framework.
Theorem 17 (Generalization Error Bound [25]). Let us have a γ-weak learner A for a concept class C
which produces classifiers h from a space H which has finite VC-dimension dH≥1. If we run Algorithm 4
for T rounds on M random samples (sampled from an unknown distribution D:{0,1}n−
→[0,1] and associated
23


<!-- page 24 -->
Algorithm 4 The RealBoost Algorithm
1: Input: Classical weak learner A, and M training samples {(x1,y1),(x2,y2),...,(xM,yM)}
; xi∼X,yi∈{−1,+1}
2: Initialize: Set D1
i = 1
M ∀i∈[M]
3: for t=1 to T do
4:
Train A using the distribution Dt to obtain a partitioning X t={X t
1,...,X t
C} of X
5:
for k=1 to C do
▷We iterate over every partition.
6:
for b∈{−1,+1} do
▷We iterate over every label.
7:
Calculate the partition weight W k,t
b
as
W k,t
b
=
X
i:xi∈X t
k and yi=b
Dt
i
(44)
8:
Set Zt=2PC
j=1
q
W j,t
+ ·W j,t
−
9:
Compute the margins βj,t= 1
2ln
 
W j,t
+
W j,t
−
 
,∀j∈{1,...,C}.
10:
Perform the distribution update ∀i∈{1,...,M} as follows
Dt+1
i
= Dt
i·exp(−βj,t·yi)
Zt
(45)
where xi∈X t
j.
11: Output: Hypothesis H(·) which is defined as
H(x)=sign
 T
X
t=1
βj,t
!
where j indicates the domain partition X t
j ∈X t containing x.
(46)
with a concept class C=S
n≥1Cn) such that m≥max{dH,T}, then with high probability (at least 2/3), the
final hypothesis H :{0,1}n−
→{−1,+1} satisfies
err(H)≤ˆ
err(H)+ ˜O
 r
TdH
M
!
(48)
From Theorem 17, we get the following corollary which helps us lower bound the total number of training
samples to obtain a low generalization bound for Algorithm 4.
Corollary 18 (Sample Complexity for reducing generalization error of AdaBoost). Let us have a γ-weak
learner A which produces classifiers h from a space H which has finite VC-dimension dH≥1. We sample
M random samples from an unknown distribution D:{0,1}n−
→[0,1] which are associated with a concept
class C=S
n≥1Cn such that m≥max{dH,T}. If we run Algorithm 4 for T rounds where T ≥˜O(logM/γ2),
then with high probability we get a generalization error of at most η>0 when
M ≥˜O
  dH
γ2η2
 
24