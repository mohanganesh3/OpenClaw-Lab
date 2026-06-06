<!-- page 1 -->
Local and Regional Counterfactual Rules: Summarized and Robust Recourses
Salim I. Amoukou 1 2 Nicolas J.B. Brunel 1 3
Abstract
Counterfactual Explanations (CE) face several un-
resolved challenges, such as ensuring stability,
synthesizing multiple CEs, and providing plau-
sibility and sparsity guarantees. From a more
practical point of view, recent studies (Pawelczyk
et al., 2022) show that the prescribed counterfac-
tual recourses are often not implemented exactly
by individuals and demonstrate that most state-
of-the-art CE algorithms are very likely to fail
in this noisy environment. To address these is-
sues, we propose a probabilistic framework that
gives a sparse local counterfactual rule for each
observation, providing rules that give a range of
values capable of changing decisions with high
probability. These rules serve as a summary of
diverse counterfactual explanations and yield ro-
bust recourses. We further aggregate these local
rules into a regional counterfactual rule, identi-
fying shared recourses for subgroups of the data.
Our local and regional rules are derived from the
Random Forest algorithm, which offers statisti-
cal guarantees and fidelity to data distribution by
selecting recourses in high-density regions. More-
over, our rules are sparse as we first select the
smallest set of variables having a high probabil-
ity of changing the decision. We have conducted
experiments to validate the effectiveness of our
counterfactual rules in comparison to standard
CE and recent similar attempts. Our methods are
available as a Python package.
1. Introduction
In recent years, many explanation methods have been devel-
oped for explaining machine learning models, with a strong
focus on local analysis, i.e., generating explanations for
individual prediction (Molnar, 2022). Among these, Coun-
terfactual Explanations (Wachter et al., 2017) have emerged
*Equal contribution
1University Paris Saclay, LaMME
2Stellantis 3Quantmetry, ENSIIE. Correspondence to: Salim I.
Amoukou <salimamoukou@gmail.com>.
ICML (International Conference on Machine Learning) Workshop
on Counterfactuals in Minds and Machines, Honolulu, Hawaii,
USA. Copyright 2023 by the author(s).
as a popular technique. In contrast to local attribution meth-
ods (Lundberg et al., 2020; Ribeiro et al., 2016), which
assign importance scores to each feature, Counterfactuals
Explanations (CE) describe the smallest modification to the
feature values that changes the prediction to a desired tar-
get. These modifications are often called recourses. While
CE can be intuitive and user-friendly, they have practical
limitations. Most CE methods depend on gradient-based
algorithms or heuristic approaches (Karimi et al., 2020a),
which may fail to identify the most natural modification and
lack guarantees. Most algorithms either do not ensure spar-
sity (changes to the smallest number of features) or fail to
generate plausible samples (Verma et al., 2020; Chou et al.,
2022). Several studies (Parmentier & Vidal, 2021; Poyiadzi
et al., 2019; Looveren & Klaise, 2019) attempt to address the
plausibility and the sparsity issues by incorporating ad-hoc
constraints.
In another direction, numerous papers (Mothilal et al., 2020;
Karimi et al., 2020b; Russell, 2019) encourage the genera-
tion of diverse counterfactuals to find actionable recourse
(Ustun et al., 2019). Actionability is a vital desideratum, as
some features may be non-actionable, and generating many
counterfactuals increases the chance of getting actionable
recourse. However, the diversity of CE compromises the
intelligibility of the explanation, and the synthesis of various
CE, in general, remains an unsolved challenge (Lakkaraju
et al., 2022). Recently, (Pawelczyk et al., 2022) highlights a
new problem of CE called: noisy responses to prescribed
recourses. In real-world scenarios, some individuals may
not be able to implement exactly the prescribed recourses,
and they show that most CE methods fail in this setting.
Consequently, we propose to reverse the usual way of ex-
plaining with counterfactual by computing Counterfactual
rules. We introduce a new line of counterfactuals, construct-
ing interpretable policies (or rules) for changing a decision
with a high probability while ensuring the stability of the
derived recourse. These policies are sparse, faithful to the
data distribution and their computation comes with statisti-
cal guarantees. Our proposal is to find a general rule that
permits changing the decision while fixing some features
instead of generating many counterfactual samples. These
rules can be seen as a summary of the possible diverse coun-
terfactual samples. Additionally, we show that this method
can be extended to create a common counterfactual policy
arXiv:2209.14568v3  [stat.ML]  18 Mar 2024


<!-- page 2 -->
Submission and Formatting Instructions for ICML 2024
Figure 1: Illustration of local and regional Counterfactual Rules for a fictitious dataset with four variables: Age, Salary, Sex,
and HoursPerWeek. Local rules change a single instance’s decision, while regional rules apply to a sub-population. Blue
indicates the suggested rules for changing decisions.
for subgroups of the data, which aids model debugging and
bias detection. We introduce an algorithm to sample CEs or
recourses from the generated rules. Notably, our approach
is model-agnostic, meaning it does not need the model to
make predictions or calculate other quantities, such as gra-
dients. Instead, it is an inferential approach and relies solely
on historical data. As a result, our approach can be applied
not only to generate counterfactuals for a specific model but
also in scenarios where we don’t have access to the model
or when making predictions from it is costly. Our method
handles categorical and continuous features and classifica-
tion and regression problems. In addition, we can use it to
generate recourses directly for the data-generating process.
An illustration of the counterfactual rules we introduce is
illustrated in Figure 1.
Notes on technical novelties. While the Local Counterfac-
tual Rule is a novel concept, the Regional Counterfactual
Rule shares similarities with some recent works. Indeed,
(Rawal & Lakkaraju, 2020) proposed Actionable Recourse
Summaries (AReS), a framework that constructs global
counterfactual recourses to have a global insight into the
model and detect unfair behavior. Despite similarities with
the Regional Counterfactual Rule, there are notable differ-
ences. Our methods can handle regression problems and
work directly with continuous features. AReS requires dis-
cretizing continuous features, leading to a trade-off between
speed and performance (Ley et al., 2022). Too few bins
yield unrealistic recourse, while too many bins result in ex-
cessive computation time. AReS employs a greedy heuristic
search approach to find global recourse, which may result in
unstable and inaccurate recourse. Our approaches overcome
these limitations by leveraging on the informative partitions
obtained from a Random Forest (RF), removing the need for
an extensive search space, and focusing on high-density re-
gions for plausibility. Additionally, we prioritize changes to
the smallest number of features by identifying the smallest
subset S of variables XS and associated value ranges for
each observation or subgroup that have the highest probabil-
ity of changing the prediction. We compute this probability
with a consistent estimator of the conditional distribution
Y |XS obtained from a RF.
Our contributions can be summarised as follows:
• We redefine the problem of CE generation by introduc-
ing Counterfactual Rules that provide summarized and
robust CE. We also introduce an algorithm to sample
recourses efficiently from these rules.
• Our approach leverages the learned partitions of a Ran-
dom Forest (RF) as an efficient tool for navigating
high-dimensional spaces. It focuses on high-density
regions, ensuring the generation of plausible CE. Addi-
tionally, it facilitates the identification of the minimal
set of features requiring modification to change predic-
tion towards desired outcomes with high probability.
• Our methods offer both local and regional explanations
and handle regression and classification tasks.
• We quantitatively demonstrate the effectiveness of our
approach across multiple desiderata, such as accuracy,
plausibility, sparsity and robustness.
2. Minimal Counterfactual Rules
Consider a dataset Dn = {(Xi, Yi)}n
i=1 consisting of i.i.d
observations of (X, Y ) ∼PXPY |X, where X ∈X (typi-
cally X ⊆Rp) and Y ∈Y. The output set Y can be either
discrete or continuous. We denote [p] = {1, . . . , p}, and
for a given subset S ⊆[p], XS = (Xi)i∈S represents a
subgroup of features, and we write x = (xS, x ¯S).
For a given observation (x, y), we consider a target set
Y ⋆⊂Y, such that y /∈Y ⋆. In the case of a classification
problem, Y ⋆= {y⋆} is a singleton where y⋆∈Y and
y⋆̸= y. Unlike conventional approaches, our definition of
CE also accommodates regression problems by considering
Y ⋆= [a, b] ⊂R, and the definitions and computations
remain the same for both classification and regression. The
classic CE problem, defined here only for classification,
considers a predictor f : X →Y, trained on dataset Dn and
search a function a : X →X, such that for all observations
x ∈X, f(x) ̸= y⋆, we have f(a(x)) = y⋆. The function
is defined point-wise by solving an optimisation program.
Most often a(·) is not a single-output function, as a(x) may

[CAPTION] Figure 1: Illustration of local and regional Counterfactual Rules for a fictitious dataset with four variables: Age, Salary, Sex,


<!-- page 3 -->
Submission and Formatting Instructions for ICML 2024
be in fact a collection of (random) values {xCF
1
, . . . , xCF
k
},
which represent the counterfactual samples. A more recent
perspective, proposed by (Kanamori et al., 2022), defines a
as a decision tree, where for each leaf L, a common action is
predicted for all instances x ∈L to change their predictions.
Our approach diverges slightly from the traditional model-
based definition of CE as we can directly consider observa-
tion (X, Y ) rather than model prediction (X, f(X)). To
illustrate the concept, let’s consider a binary classification
problem, where the input space can be divided into two
regions R0, R1. These correspond to the support of the dis-
tributions X|Y = 0 and X|Y = 1. These regions may not
be disjoint or convex spaces and can be represented as a
union of several sets. Given an observation x = (xS, x ¯S)
with label y = 0, our method consists of finding the minimal
subset of variables S ⊆{1, . . . , p} to move x by modifying
xS into a set within the region R1. The objective is to move
x to a high-density set with low variance with respect to the
target variable Y , while altering as few variables as possible.
Figure 2: Illustration of the 4-stages in our methodology for
computing sparse counterfactuals
Figure 2 provides a visual representation of our approach
in the binary case. The first step involves learning a tree-
based model on our data, enabling us to partition the input
space based on the target variable Y . By examining the
tree leaves, we can easily identify the optimal direction S
to modify the decision and the target region corresponding
to the counterfactual rule. Moreover, these leaves can serve
not only as rules but also as a means to generate recourses.
Our approach is different from the optimization approach
for generating recourse, as it is model-agnostic, meaning it
does not require the model to generate further predictions
or compute gradients. This flexibility allows us to apply our
approach to generate recourse either for the predictions of
a given model (X, f(X)) or the data-generating process
(X, Y ). A model-agnostic approach was also proposed by
(Black et al., 2020; De Lara et al., 2021) under the name
of transport-based counterfactuals. It consists of finding a
map T between the distribution of X|Y = 0 and X|Y = 1
such that each observation of class Y = 0 is linked to the
most similar observation of class Y = 1. De Lara et al.
(2021) shows that it coincides with causal counterfactual
under appropriate assumptions. In the following discussion,
we consider the data (X, Y ) for the presentation of the
methods, although they can also be applied to generate
recourses for a model prediction (X, f(X)) as well.
Our approach is hybrid, as we do not suggest a single action
for each observation or subspace of X but provide sets of
possible perturbations. A Local Counterfactual Rule (L-CR)
for target Y ⋆and observation (x, y) (with y /∈Y ⋆) is a
rectangle CS(x, Y ⋆) = Q
i∈S[ai, bi], ai, bi ∈R such that
for all counterfactual samples of x = (xS, x ¯S) obtained
as xCF = (zS, x ¯S) with zS ∈CS(x, Y ⋆) and xCF an
in-distribution sample, then yCF is in Y ⋆with a high proba-
bility, where yCF is the output of xCF given by the model f
or the data-generating process. Similarly, a Regional Coun-
terfactual Rule (R-CR) CS(R, Y ⋆) is defined for target Y ⋆
and a rectangle R = Qd
i=1[ai, bi], ai, bi ∈R, which rep-
resent a subspace of X of similar observations, if for all
observations x = (xS, x ¯S) ∈R, the countefactual sam-
ples obtained as xCF = (zS, x ¯S) with zS ∈CS(R, Y ⋆)
and xCF an in-distribution sample are such that yCF is in
Y ⋆with high probability. Our approach constructs such
rectangles in a sequential manner. Firstly, we identify the
minimal directions S ⊆[p] that offer the highest probability
of changing the decision. Next, we determine the optimal
intervals [ai, bi] for i ∈S that change the decision to the
desired target. Additionally, we propose a method to derive
traditional Counterfactual Explanations (CE) (i.e., actions
that alter the decision) or recourses using our Counterfactual
Rules. A central tool in this approach is the Counterfactual
Decision Probability presented below.
Definition 2.1. Counterfactual Decision Probability
(CDP). The Counterfactual Decision Probability of the sub-
set S ⊆{1, . . . , p}, w.r.t x = (xS, x ¯S), output y and the
desired target Y ⋆(s.t. y /∈Y ⋆) is
CDPS (x, Y ⋆) = P (Y ∈Y ⋆|X ¯S = x ¯S ) .
(1)
The CDP of the subset S is the probability that the decision
changes to the desired target Y ⋆by sampling the features
XS given X ¯S = x ¯S. It is related to the Same Decision
Probability SDPS(Y ; x) = P (Y ∈Y |XS = xS) used
in (Amoukou & Brunel, 2021) for solving the dual problem
of selecting the most local important variables for obtaining
and maintaining the decision Y ∈Y , where Y ⊂Y. The
set S is called the Minimal Sufficient Explanation. Indeed,
we have CDPS(x, Y ⋆) = SDPS(x, Y ⋆). The computa-
tion of these probabilities is challenging and discussed in
Section 3. Next, we define the minimal subset of features
S that allows changing the decision to the target set with a
given high probability π.
Definition 2.2. (Minimal Divergent Explanations). Given
an instance (x, y) and a desired target Y ⋆̸∋y, S is a Diver-

[CAPTION] Figure 2: Illustration of the 4-stages in our methodology for

[CAPTION] Figure 2 provides a visual representation of our approach


<!-- page 4 -->
Submission and Formatting Instructions for ICML 2024
gent Explanation for probability π > 0 if CDPS (x, Y ⋆) ≥
π, and no subset Z of S satisfies CDPZ (x, Y ⋆) ≥π.
Hence, a Minimal Divergent Explanation is a Divergent
Explanation with the smallest size.
The set satisfying these properties is not unique, and we can
have several Minimal Divergent Explanations. Note that
the probability π represents the minimum level required for
a set to be chosen for generating counterfactuals, and its
value should be as high as possible and depends on the use
case. With these concepts established, we can now define
our main criterion for constructing a Local Counterfactual
Rule (L-CR).
Definition 2.3. (Local Counterfactual Rule).
Given
an instance (x, y), a desired target Y ⋆̸∋y , a Mini-
mal Divergent Explanation S, the rectangle CS(x, Y ⋆) =
Q
i∈S[ai, bi], ai, bi
∈R is a Local Counterfactual Rule
with probability πC if CS(x, Y ⋆) = arg maxC PPX
 XS ∈
C
| X ¯
S
= x ¯
S
  such that CRPS
 
x, Y ⋆ 
= P(Y
∈
Y ⋆| XS ∈CS(x, Y ⋆), X ¯
S = x ¯
S) satisfies
CRPS
 
x, Y ⋆ 
≥πC.
(2)
PPX
 XS ∈CS(x, Y ⋆) | X ¯S = x ¯S
 
represent the plausi-
bility of the rule and by maximizing it, we ensure that the
rule lies in a high-density region. CRPS is the Counterfac-
tual Rule Probability. The higher the probability πC is, the
better the relevance of the rule CS(x, Y ⋆) is for changing
the decision to the desired target.
In practice, we often observe that the Local CR CS(·, Y ⋆)
for neighboring observations x and x′ are quite similar,
as the Minimal Divergent Explanations tend to be alike,
and the corresponding hyperrectangles frequently overlap.
This observation motivates a generalization of these Lo-
cal CRs to hyperrectangles R = Qd
i=1[ai, bi], ai, bi ∈R,
which group together similar observations.
We denote
supp(R) = {i : [ai, bi] ̸= R} as the support of the rectangle
and extend the Local CRs to Regional Counterfactual Rules
(R-CR). To achieve this, we denote R ¯S = Q
i∈¯S[ai, bi]
as the rectangle with intervals of R in supp(R) ∩¯S, and
define the corresponding Counterfactual Decision Probabil-
ity (CDP) for rule R and subset S as CDPS(R, Y ⋆) =
P (Y ∈Y ⋆|X ¯S ∈R ¯S ). Consequently, we can compute
the Minimal Divergent Explanation for rule R using the
corresponding CDP for rules, following Definition (2.2).
The Regional Counterfactual Rules (R-CR) correspond to
Definition (2.3) with the associated CDP for rules.
3. Estimation of the CDP and CRP
To compute the probabilities CDPS and CRPS for any S,
we use a dedicated Random Forest (RF) that learns to pre-
dict the output of the model or the data-generating process.
Indeed, the conditional probabilities CDPS and CRPS can
be easily computed from a RF by combining the Projected
Forest algorithm (B´enard et al., 2021a) and the Quantile
Regression Forest (Meinshausen & Ridgeway, 2006). As
a result, we can estimate the probabilities CDPS(x, Y ⋆)
consistently. This method has been previously utilized by
(Amoukou & Brunel, 2021) for calculating the Same Deci-
sion Probability SDPS.
3.1. Projected Forest and CDPS
The estimator of the SDPS is based on the Random For-
est (Breiman et al., 1984) algorithm. Assuming that we
have trained a RF m(·) using the dataset Dn, the model
consists of a collection of k randomized trees (for a de-
tailed description of decision trees, see (Loh, 2011)). For
each instance x, the predicted value of the l-th tree is de-
noted as ml(x; Θl), where Θl represents the resampling
data mechanism in the j-th tree and the subsequent random
splitting directions. The predictions of the individual trees
are then averaged to produce the prediction of the forest
as m(x; Θ1, . . . , Θk) = 1
k
Pk
l=1 ml(x; Θl). The RF can
also be interpreted as an adaptive nearest neighbor predictor
(Lin & Jeon, 2006; Biau & Devroye, 2010) or kernel meth-
ods (Breiman, 2000; Geurts et al., 2006; Scornet, 2016).
For every instance x, the observations in Dn are weighted
by wn,i(x), with i = 1, . . . , n. As a result, the prediction
of the RF can be reformulated as m(x; Θ1, . . . , Θk) =
Pn
i=1 wn,i(x)Yi. This emphasizes the central role played
by the weights in the RF’s algorithm. See (Meinshausen
& Ridgeway, 2006; Amoukou & Brunel, 2021) for a de-
tailed description of the weights. Consequently, it naturally
gives estimators for other quantities, e.g., cumulative hazard
function (Ishwaran et al., 2008), treatment effect (Wager
& Athey, 2017; Jocteur et al., 2023), conditional density
(Du et al., 2021). For instance, (Meinshausen & Ridgeway,
2006) showed that we can use the same weights to estimate
the conditional distribution function with the following es-
timator bF(y|X = x) = Pn
i=1 wn,i(x)1Yi≤y. In another
direction, (B´enard et al., 2021a) introduced the Projected
Forest algorithm (B´enard et al., 2021c;a) that aims to esti-
mate E[Y |XS] by modifying the RF’s prediction algorithm.
Projected Forest: To estimate E[Y |XS = xS] instead of
E[Y |X = x] using a RF, (B´enard et al., 2021b) suggests to
simply ignore the splits based on the variables not contained
in S from the tree predictions. More formally, it consists
of projecting the partition of each tree of the forest on the
subspace spanned by the variables in S. The authors also
introduced an algorithmic trick that computes the output
of the Projected Forest efficiently without modifying the
initial tree structures. It consists of dropping the observa-
tions down in the initial trees, ignoring the splits that use
a variable not in S: when it encounters a split involving a
variable i /∈S, the observations are sent to the left and right


<!-- page 5 -->
Submission and Formatting Instructions for ICML 2024
(a)
(b)
(c)
Figure 3: (a) Partition of the Random Forest, (b) Partition of the Projected Random Forest when we condition given X0, i.e.,
ignoring the splits on X1, (c) The optimal Counterfactual Rule of x when we condition given X0 = x0 is the green region.
children nodes. Therefore, each instance falls in multiple
terminal leaves of the tree. To compute the prediction of xS,
we follow the same procedure and gather the set of terminal
leaves where xS falls. Next, we collect the training observa-
tions which belong to every terminal leaf of this collection,
in other words, we keep only the observations that fall in
the intersection of the leaves where xS falls. Finally, we
average their outputs Yi to estimate E[Y |XS = xS]. The
authors show this algorithm converges asymptotically to
the true projected conditional expectation E[Y |XS = xS]
under suitable assumptions. As the RF, the Projected Forest
(PRF) assigns a weight to each training observation. The
associated PRF is denoted m(S)(xS) = Pn
i=1 wn,i(xS)Yi.
Therefore, as the weights of the original forest was used
to estimate the CDF, (Amoukou & Brunel, 2021) used the
weights of the Projected Forest Algorithm to estimate SDP
as [
SDP S (x, Y ⋆) = Pn
i=1 wn,i(xS)1Yi∈Y ⋆. The idea is
essentially to replace Yi by 1Yi∈Y ⋆in the Projected For-
est equation defined above. (Amoukou & Brunel, 2021)
also show that this estimator converges to the true SDPS
under suitable assumptions and works very well in prac-
tice. Especially for tabular data, where tree-based models
are known to perform well (Grinsztajn et al., 2022). Simi-
larly, we can estimate the CDP with statistical guarantees
(Amoukou & Brunel, 2021) using the following estimator
\
CDP S (x, Y ⋆) = Pn
i=1 wn,i(x ¯S)1Yi∈Y ⋆.
Remark: We only give the estimator of CDPS of an in-
stance x. The estimator for CDPS of a rule R will be
discussed in the next section, as it is closely related to the
estimator of the CRPS.
3.2. Regional RF and CRPS
Here, we focus on estimating the CRPS(x, Y ⋆) = P(Y ∈
Y ⋆|XS ∈CS(x, Y ⋆), X ¯
S = x ¯
S) and CRPS(R, Y ⋆) =
P(Y
∈Y ⋆|XS ∈CS(R; Y ⋆), X ¯
S ∈R ¯
S). For ease
of reading, we remove the dependency of the rectangles
CS in Y ⋆. Based on the previous section, we already
know that the estimators using the RF will take the form of
\
CRP S (x, Y ⋆) = Pn
i=1 wR
n,i(x)1Yi∈Y ⋆, so we only need
to determine the appropriate weighting. The main challenge
lies in the fact that we have a condition based on a region,
e.g., XS ∈CS(x) or X ¯S ∈R ¯S (regional-based) instead
of a condition of type XS = xS (fixed value-based) as
usual. However, we introduced a natural extension of the
RF algorithm to handle predictions when the conditions
are both regional-based and fixed value-based. As a result,
cases with only regional-based conditions can be naturally
derived.
Regional RF to estimate CRPS(x, Y ⋆) = P(Y
∈
Y ⋆| XS ∈CS(x), X ¯S = x ¯S). The algorithm is based
on a slight modification of RF and works as follows: we
drop the observations in the trees, if a split used variable
i ∈¯S, i.e., fixed value-based condition, we use the clas-
sic rules of RF, if xi ≤t, the observations go to the left
children, otherwise the right children. However, if a split
used variable i ∈S, i.e, regional-based condition, we use
the rectangles CS(x) = Q|S|
i=1[ai, bi]. The observations are
sent to the left children if bi ≤t, right children if ai > t
and if t ∈[ai, bi], the observations are sent both to the left
and right children. Consequently, we use the weights of
the Regional RF algorithm wR
n,i(x) to estimate CRPS, the
estimator is \
CRP S(x, Y ⋆) = Pn
i=1 wR
n,i(x)1Yi∈Y ⋆. In
addition, the number of observations at the leaves is used
as an estimate of P(XS ∈CS(x) | X ¯S = x ¯S). A more
comprehensive description and discussion of the algorithm
are provided in the Appendix (A).
To estimate the CDP of a rule CDPS (R, Y ⋆)
=
P (Y ∈Y ⋆|X ¯S ∈R ¯S ), we just have to apply the Pro-
jected Forest algorithm to the Regional RF, i.e., when a
split involving a variable outside of ¯S is met, the obser-
vations are sent both to the left and right children nodes,
otherwise we use the Regional RF split rule, i.e., if an in-
terval of R ¯S is below t, the observations go to the left
children, otherwise the right children and if t is in the inter-
val, the observations go to the left and right children. The
estimator of the CRPS(R, Y ⋆) = P(Y ∈Y ⋆|XS ∈

[CAPTION] Figure 3: (a) Partition of the Random Forest, (b) Partition of the Projected Random Forest when we condition given X0, i.e.,


<!-- page 6 -->
Submission and Formatting Instructions for ICML 2024
CS(R; Y ⋆), X ¯S ∈R ¯S) for rule R is also derived from
the Regional RF. Indeed, it is a special case of the Regional
RF algorithm where there are only regional-based condi-
tions.
4. Learning the Counterfactual Rules
The computation of the Local and Regional CR is performed
using the estimators introduced in the previous section. First,
we determine the Minimal Divergent Explanation, akin to
the Minimal Sufficient Explanation (Amoukou & Brunel,
2021), by exploring the subsets obtained using the K = 10
most frequently selected variables in the Random Forest
estimator. K is a hyper-parameter to choose according to
the use case and computational power. We can also use
any importance measure. An alternative strategy to exhaus-
tively searching through the 2K possible subsets would be
to sample a sufficient number of subsets, typically a few
thousand, that are present in the decision paths of the trees
in the forest. By construction, these subsets are likely to
contain influential variables. A similar strategy was used in
(Basu et al., 2018; B´enard et al., 2021a).
Given an instance x or rectangle R, target set Y ⋆and their
corresponding Minimal Divergent Explanation S, our ob-
jective is to find the maximal rule CS(·) = Q
i∈S[ai, bi]
s.t. given X ¯S = x ¯S or X ¯S ∈R ¯S, and XS ∈CS(·),
the probability that Y
∈Y ⋆is high.
Formally, we
want: P(Y ∈Y ⋆|XS ∈CS(x), X ¯S = x ¯S) or P(Y ∈
Y ⋆|XS ∈CS(R), X ¯S ∈R ¯S) above πC.
The rectangles CS(·) = Q
i∈S[ai, bi] defining the CR are
derived from the RF. In fact, these rectangles naturally
arise from the partition learned by the RF. AReS (Rawal &
Lakkaraju, 2020), on the other hand, relies on binned vari-
ables to generate candidate rules, testing all possible rules to
select the optimal one. By leveraging the partition learned
by the RF, we overcome both the computational burden and
the challenge of choosing the number of bins. Moreover, by
focusing only on the non-empty leaves containing training
observations, we significantly reduce the search space. This
approach allows identifying high-density regions of the in-
put space to generate plausible counterfactual explanations.
To illustrate the idea, we use a two-dimensional data
(X0, X1) with binary label Y represented as green and blue
stars in Figure 3(a). We fit a Random Forest to classify this
dataset and show its partition in Figure 3(a). We consider
an instance x (blue triangle), and our goal is to change its
classification from blue to green. From a visual analysis
of cells/leaves of the RF, we deduce that the Minimal Di-
vergent Explanation of x is S = X1. In Figure 3(b), we
observe the leaves of the Projected Forest when not con-
ditioning on S = X1, thus projecting the RF’s partition
only on the subspace X0. It consists of ignoring all the
splits in the other directions (here the X1-axis), thus x falls
in the projected leaf 2 (see Figure 3(b)) and its CDP is
CDPX1(green; x) =
10 green
10 green+17 blue = 0.58. To find the
optimal rectangle CS(x) = [ai, bi] in the direction of X1,
such that the decision changes, we can utilize the leaves of
the RF. By looking at the leaves of the RF (Figure 3(a)) for
observations belonging to the projected RF leaf 2 (Figure
3(b)) where x falls, we observe in Figure 3(c) that the opti-
mal rectangle for changing the decision, given X0 = x0 or
being in the projected RF leaf 2, is the union of the intervals
on X1 of the leaf 3 and 4 of the RF (see the green region in
Figure 3(c)).
Given an instance x and its Minimal Divergent Explanation
S, the first step is to collect observations that belong to the
leaf of the Projected Forest given ¯S, where x falls. These ob-
servations correspond to those with positive weights in the
computation of CDPS(x, Y ⋆) = Pn
i=1 wR
n,i(x ¯S)1Yi∈Y ⋆,
i.e., {Xi : wR
n,i(x ¯S) > 0}. Then we use the partition of
the original forest to find the possible leaves in the direction
S. The possible leaves are among the RF’s leaves of the
collected observations {Xi : wR
n,i(x¯s) > 0}. Let denote
L(Xi) the leaf of the observation Xi with wn,i(x ¯S) > 0.
A possible leaf is a leaf L(Xi) s.t. P(Y ∈Y ⋆|XS ∈
L(Xi)S, X ¯S = x ¯S) ≥πC. Finally, we merge all the pos-
sible neighboring leaves to get the largest rectangle, and this
maximal rectangle is the counterfactual rule. It is important
to note that the union of possible leaves is not necessarily a
connected space, which may result in multiple disconnected
counterfactual rules.
We apply the same approach to find the regional CR. Given
a rule R and its Minimal Divergent Explanation S, we used
the Projection given X ¯S ∈R ¯S to identify compatible obser-
vations and their leaves. We then combine the possible ones
that satisfy CRPS(R, Y ⋆) ≥πC to obtain the regional CR.
For instance, if we consider Leaf 5 of the original forest as
a rule (i.e., if X ∈Leaf 5, then predict blue), its Minimal
Divergent Explanation is also S = X1. The Regional CR
would be the green region in Figure 3(c). Indeed, satisfying
the X0 condition of Leaf 5 and the X1 condition of Leaves
3 and 4 would cause the decision to change to green.
5. Sampling CE using the CR
Our approaches cannot be directly compared with traditional
CE methods, as they return counterfactual samples, whereas
we provide rules (ranges of vector values) that permit chang-
ing the decision with high probability. In some applications,
users might prefer recourse to CR. Hence, we adapt the CR
to generate counterfactual samples using a generative model.
For example, given an instance x = (xS, x ¯S), target set
Y ⋆and its counterfactual rule CS(x, Y ⋆), we want to find
a sample xCF = (zS, x ¯S) with zS ∈CS(x, Y ⋆) such
that xCF is a realistic sample and yCF ∈Y ⋆. Instead of
using a complex conditional generative model as (Xu et al.,


<!-- page 7 -->
Submission and Formatting Instructions for ICML 2024
2019; Patki et al., 2016), which can be difficult to calibrate,
we use an energy-based generative approach (Grathwohl
et al., 2020; Lecun et al., 2006). The core idea is to find
zS ∈CS(x, Y ⋆) such that x⋆maximizes a given energy
score, ensuring that x⋆lies in a high-density region. We use
the negative outlier score of an Isolation Forest (Liu et al.,
2008) and Simulated Annealing (Guilmeau et al., 2021) to
maximize the negative outlier score using the information
of the counterfactual rules CS(x, Y ⋆). In fact, the range
values given by the CR CS(x, Y ⋆) reduce the search space
for zS drastically. We used the marginal law of X given
XS ∈CS(x, Y ⋆) as the proposal distribution, i.e., we
draw a candidate zS by independently sampling each vari-
able using the marginal law zS ∼Q
i∈S PXj |XS∈CS(x,Y ⋆)
until we find an observation xCF = (zS, x ¯S) with high en-
ergy. The algorithm works similarly for sampling CE with
the Regional CR. The methodology is described below in
Appendix B.
6. Experiments
To demonstrate the performance of our framework, we con-
duct two experiments on real-world datasets. In the first
experiment, we showcase the utility of the Local Coun-
terfactual Rules for explaining a regression model. In the
second experiment, we compare our approaches with base-
line methods in the context of classification problems. We
compare the methods only in classification problems, as
all prior works do not deal with regression problems. We
use the library CARLA (Pawelczyk et al., 2021) designed
to benchmark counterfactual explanation methods across
different data sets and machine learning models.
We use the default settings of Random Forest and Neural
Networks of CARLA as query models. We compute the
recourses given by our main competitor, AReS (Rawal &
Lakkaraju, 2020), which performs an exhaustive search
for finding global counterfactual rules. We use the imple-
mentation of (Kanamori et al., 2022) that adapts AReS for
returning counterfactual samples instead of rules. In cases
where the Random Forest is the query model, we add two
popular CF methods for tree-based models, FOCUS (Lucic
et al., 2022) and FeatureTweaking (Tolomei et al., 2017)
computed by CARLA. We add CHVAE (Pawelczyk et al.,
2020), a CF method based on autoencoder for Neural Net-
work experiments to enhance plausibility. In all experiments,
we split our dataset into train (75%) - test (25%). We learn
a model f, which could be either a Random Forest (RF) or
Neural Network (NN), on the train set, which is the model
we want to explain. We learn f’s predictions on the train set
with an RF (estimators=20, max depth=10) that will be used
to generate the Counterfactual Rules (CR) with π = 0.9.
The parameters used for AReS are max rules=8, bins=10,
and the default parameters set by CARLA for the other. For
detailed parameter descriptions, see Appendix (C).
We evaluate the methods on the test set using four metrics.
The first, Accuracy, quantifies the average rate at which
the recommended actions from each method successfully
change the prediction to the desired outcome. The second,
Plausibility, measures the average number of counterfactual
samples that are classified as inliers by an Isolation Forest
trained on the training set. The third, Sparsity, measures the
average number of features that have been changed. The
fourth metric is the Cost, as defined in (Rawal & Lakkaraju,
2020). For categorical variables, any change is counted
as a change of magnitude 1. The continuous features are
converted into ordinal features by binning the feature values.
In this case, going from one bin to the next immediate bin
corresponds to a change of magnitude 1.
Local counterfactual rules for regression. We apply our
approach to the California House Price dataset (n=20640,
p=8) (Kelley Pace & Barry, 1997), which contains infor-
mation about each district such as income, population, and
location, and the goal is to predict the median house value of
each district. To demonstrate the effectiveness of our Local
CR method, we focus on a subset of the test set consisting
of 1566 houses with prices lower than 100k. We aim to find
recourse that would increase house prices, bringing them
within the target range Y ⋆= [200k, 250k]. For each in-
stance x, we compute the Minimal Divergent Explanation S,
the Local CR CS(x, [200k, 250k]), and generate a counter-
factual sample using the Simulated Annealing technique de-
scribed earlier. We succeed in changing the decision for all
observations, achieving Accuracy = 100%. Furthermore,
the majority of counterfactual samples passed the outlier test,
with a Plausibility score of 0.92. Additionally, our Local
CR method achieves high sparsity, with Sparsity = 4.45.
For
instance,
the
Local
CR
for
the
observation
x = [Longitude=-118.2, latitude=33.8, housing median
age=26, total rooms=703, total bedrooms=202, popu-
lation=757, households=212, median income=2.52] is
CS(x, [200k, 250k]) = [total room ∈[2132, 3546], total
bedrooms ∈[214, 491]] with probability 0.97. This means
that if the total number of rooms and total bedrooms satisfy
the conditions in CS(x, [200k, 250k]), and the remaining
features of x are fixed, then the probability that the price
falls within the target set Y ⋆= [200k, 250k] is 0.97. This
makes sense as increasing the number of rooms and bed-
rooms in the district will certainly increase the price.
Comparisons of Local and Regional CR with baseline
methods. We evaluate our framework on three real-world
datasets: Compas (n=6172, p=12) (Washington, 2018) is
used to predict criminal recidivism, Diabetes (n=768, p=8)
(Kaggle, 2016) aims to predict whether a patient has diabetes
or not, and Nhanesi (CDC, 1999-2022) which also aims to
predict a disease. Our evaluation reveals that AReS is highly
sensitive to the number of bins and the maximal number of


<!-- page 8 -->
Submission and Formatting Instructions for ICML 2024
Table 1: Results of the Accuracy (Acc), Plausibility (Psb), Sparsity (Sprs), and Cost (Cost) of the different methods. We
compute each metric according to the positive (Pos) and negative (Neg) class. The blue value corresponds to the metric for
the positive class (recourse for label=1 to label=0), and the red for the negative class (recourse for label=0 to label=1).
Compas
Diabetes
Nhanesi
Acc
Psb
Sps
Cost
Acc
Psb
Sps
Cost
Acc
Psb
Sps
Cost
(RF) L-CR
0.98 / 0.93
0.90 / 0.92
2.00 / 3.00
0.65 / 0.99
1.00 / 1.00
0.98 / 0.89
3.34 / 3.74
1.34 / 1.63
0.95 / 0.97
0.98 / 0.98
4.00 / 5.00
0.76 / 1.16
(RF) R-CR
0.90 / 0.90
0.83 / 0.88
2.00 / 3.00
0.92 / 0.99
0.92 / 0.80
0.97 / 0.89
3.72 / 3.98
1.53 / 1.67
0.93 / 0.97
0.96 / 0.97
7.00 / 7.00
1.31 / 1.33
(RF) AReS
0.72 / 0.15
0.39 / 0.77
1.84 / 1.30
0.34 / 0.44
0.72 / 1.00
0.85 / 0.83
1.09 / 1.00
0.28 / 0.39
0.92 / 1.00
0.84 / 0.89
1.00 / 1.00
0.25 / 0.30
(RF) Focus
0.00 / 1.00
0.00 / 0.00
8.61 / 7.47
2.50 / 1.85
0.00 / 1.00
0.00 / 0.00
8.00 / 8.00
2.21 / 3.10
0.00 / 1.00
0.00 / 0.00
17.0 / 17.0
3.28 / 3.19
(RF) FTW
N/A / 1.00
N/A / 0.22
N/A / 3.00
N/A / 0.97
N/A / 1.00
N/A / 0.82
N/A / 3.76
N/A / 0.94
N/A / 1.00
N/A / 0.76
N/A / 3.92
N/A / 0.41
(NN) L-CR
0.95 / 0.88
0.87 / 0.79
2.64 / 3.25
0.83 / 0.86
1.00 / 1.00
0.98 / 0.89
3.34 / 3.71
1.32 / 1.63
0.97 / 0.96
0.98 / 0.97
4.53 / 5.59
0.85 / 1.08
(NN) R-CR
0.98 / 0.88
0.82 / 0.91
3.36 / 3.29
1.21 / 1.00
0.92 / 0.78
0.97 / 0.89
3.72 / 3.93
1.53 / 1.67
0.90 / 0.93
0.99 / 0.98
7.68 / 7.76
1.26 / 1.39
(NN) AReS
0.92 / 0.33
0.83 / 0.62
1.18 / 1.02
0.52 / 0.47
1.00 / 1.00
0.87 / 0.60
1.00 / 1.00
0.29 / 0.48
0.99 / 0.00
0.81 / 0.86
1.00 / 1.00
0.26 / 0.18
(NN) CHVAE
N/A / 1.00
N/A / 0.00
N/A / 8.00
N/A / 1.91
0.00 / 1.00
0.00 / 0.00
8.00 / 8.00
2.21 / 3.10
N/A / 1.00
N/A / 0.00
N/A / 17.0
N/A / 3.68
rules or actions, as previously noted by (Ley et al., 2022).
Poor parameterization can result in completely ineffective
recourses. Furthermore, these methods require separate
models for each target class, while our framework only
requires a single RF with good precision.
Table 1 shows the results of each method across all datasets
and models (RF, NN). The results for the RF are in the first
five rows, and those below are for the NN. In each cell, we
have two values, the blue corresponding to the metric value
for the positive class (label=1) and the red for the negative
class (label=0). Table 1 demonstrates that the Local and
Regional CR methods achieve high accuracy in changing
decisions on all datasets, surpassing baseline methods in
almost all experiments. Furthermore, the baselines struggle
to change both the positive and negative classes simultane-
ously, e.g., ARes has Acc=0.72 in the positive class, and
0.15 for the negative class on (RF) - Compas or when they
have a good Acc, the CE are not plausible. For instance,
FOCUS has Acc=1 for the negative class and Psb=0 in this
experiment, meaning that all the counterfactual samples are
outliers. These trends are consistently observed across most
experiments. We also noted that Focus and FTW fail com-
pletely to change the positive class, and CHVAE often does
not even propose a CF for the positive class. This could be
attributed to these methods being more sensitive to imbal-
ances in the data. Our method is the only one that works
consistently with the positive and negative classes. Interest-
ingly, baseline methods exhibit improved performance on
the NN model compared to the RF model.
Regarding plausibility, our method generally outperforms
others, albeit at a higher cost. This is not surprising, as all
baseline methods optimise this criterion, unlike ours. In the
future, we intend to add this criterion to the CF generation
process (Section 5).
Noisy responses robustness of Local CR: To assess the
robustness of our approach against noisy responses, we
conduct an experiment inspired by (Pawelczyk et al., 2022).
We normalized the datasets so that X ∈[0, 1]p and added
small Gaussian noises ϵ to the prescribed recourses, with
ϵ ∼N(0, σ2), where σ2 took values of 0.01, 0.025, 0.05.
We compute the Stability, which is the fraction of unseen
instances where the action and perturbed action lead to the
same output, for the Compas and Diabetes datasets. We
used the simulated annealing approach of Section 5 with the
Local CR to generate the actions. The Stability metrics for
the different noise levels were 0.98, 0.98, 0.98 for Compas
and 0.96, 0.97, 0.96 for Diabetes.
In summary, our CR approach is easier to train, and pro-
vides more accurate and plausible rules than the baseline
methods. Furthermore, our resulting CE is robust against
noisy responses.
7. Conclusion
We propose a novel approach that formulates CE as Coun-
terfactual Rules. These rules are simple policies that can
change the decision of an individual or sub-population with
a high probability. Our method is designed to learn robust,
plausible, and sparse adversarial regions that indicate where
observations should be moved to satisfy a desired outcome.
Random Forests are central to our approach, as they provide
consistent estimates of the probabilities of interest and natu-
rally give rise to the counterfactual rules we seek. This also
allows us to handle regression problems and continuous fea-
tures, making our method applicable to a wide range of data
sets where tree-based models perform well, such as tabular
data (Grinsztajn et al., 2022). An interesting avenue to ex-
plore would be to incorporate the l1 cost into our approach.
Currently, our method aims to minimize the l0 distance be-
tween the query xobs and the counterfactual xCF by altering
as few features as possible. However, deriving a counterfac-
tual observation within a counterfactual rule that minimizes
the l1 cost is straightforward with an explicit solution. Given
the counterfactual rules (hyperrectangles), represented as a
box (l, r), with l, r ∈Rp, the following optimization prob-
lem xCF = argminxd(x, xobs) such that l ≤x ≤r has a
closed form solution when the distance is the l1 or l2 norm.
The solution is xCF = max(l, min(r, xobs)) elementwise
(Carreira-Perpi˜n´an & Hada, 2021). In future work, we will
incorporate the l1 constraint and assess the effectiveness of
our approach in terms of cost relative to other methods.


**[Table p8.1]**
| Acc Psb Sps Cost | Acc Psb Sps Cost | Acc Psb Sps Cost |
| --- | --- | --- |
| 0.98 / 0.93 0.90 / 0.92 2.00 / 3.00 0.65 / 0.99 0.90 / 0.90 0.83 / 0.88 2.00 / 3.00 0.92 / 0.99 0.72 / 0.15 0.39 / 0.77 1.84 / 1.30 0.34 / 0.44 0.00 / 1.00 0.00 / 0.00 8.61 / 7.47 2.50 / 1.85 N/A / 1.00 N/A / 0.22 N/A / 3.00 N/A / 0.97 | 1.00 / 1.00 0.98 / 0.89 3.34 / 3.74 1.34 / 1.63 0.92 / 0.80 0.97 / 0.89 3.72 / 3.98 1.53 / 1.67 0.72 / 1.00 0.85 / 0.83 1.09 / 1.00 0.28 / 0.39 0.00 / 1.00 0.00 / 0.00 8.00 / 8.00 2.21 / 3.10 N/A / 1.00 N/A / 0.82 N/A / 3.76 N/A / 0.94 | 0.95 / 0.97 0.98 / 0.98 4.00 / 5.00 0.76 / 1.16 0.93 / 0.97 0.96 / 0.97 7.00 / 7.00 1.31 / 1.33 0.92 / 1.00 0.84 / 0.89 1.00 / 1.00 0.25 / 0.30 0.00 / 1.00 0.00 / 0.00 17.0 / 17.0 3.28 / 3.19 N/A / 1.00 N/A / 0.76 N/A / 3.92 N/A / 0.41 |

[CAPTION] Table 1: Results of the Accuracy (Acc), Plausibility (Psb), Sparsity (Sprs), and Cost (Cost) of the different methods. We

[CAPTION] Table 1 shows the results of each method across all datasets


<!-- page 9 -->
Submission and Formatting Instructions for ICML 2024
References
Amoukou, S. I. and Brunel, N. J. Consistent sufficient
explanations and minimal local rules for explaining
regression and classification models.
arXiv preprint
arXiv:2111.04658, 2021.
Basu, S., Kumbier, K., Brown, J. B., and Yu, B. Iterative
random forests to discover predictive and stable high-
order interactions. Proceedings of the National Academy
of Sciences, 115(8):1943–1948, 2018.
B´enard, C., Biau, G., Da Veiga, S., and Scornet, E. Shaff:
Fast and consistent shapley effect estimates via random
forests. arXiv preprint arXiv:2105.11724, 2021a.
B´enard, C., Biau, G., Veiga, S., and Scornet, E. Interpretable
random forests via rule extraction. In International Con-
ference on Artificial Intelligence and Statistics, pp. 937–
945. PMLR, 2021b.
B´enard, C., Da Veiga, S., and Scornet, E. Mda for random
forests: inconsistency, and a practical solution via the
sobol-mda. arXiv preprint arXiv:2102.13347, 2021c.
Biau, G. and Devroye, L.
On the layered near-
est neighbour estimate, the bagged nearest neigh-
bour estimate and the random forest method in re-
gression and classification.
Journal of Multivari-
ate Analysis, 101(10):2499–2518, 2010.
ISSN 0047-
259X.
doi:
https://doi.org/10.1016/j.jmva.2010.06.
019. URL https://www.sciencedirect.com/
science/article/pii/S0047259X10001387.
Black, E., Yeom, S., and Fredrikson, M. Fliptest: fair-
ness testing via optimal transport. In Proceedings of
the 2020 Conference on Fairness, Accountability, and
Transparency, pp. 111–121, 2020.
Breiman, L. Some infinity theory for predictor ensembles.
Technical report, Citeseer, 2000.
Breiman, L., Friedman, J., Olshen, R., and Stone, C. Classi-
fication and regression trees. wadsworth int. Group, 37
(15):237–251, 1984.
Carreira-Perpi˜n´an, M. ´A. and Hada, S. S. Counterfactual
explanations for oblique decision trees: Exact, efficient
algorithms. In Proceedings of the AAAI conference on
artificial intelligence, volume 35, pp. 6903–6911, 2021.
CDC. National health and nutrition examination survey,
1999-2022. URL https://wwwn.cdc.gov/Nchs/
Nhanes/Default.aspx.
Chou, Y.-L., Moreira, C., Bruza, P., Ouyang, C., and
Jorge, J.
Counterfactuals and causability in explain-
able artificial intelligence: Theory, algorithms, and ap-
plications. Information Fusion, 81:59–83, 2022. ISSN
1566-2535. doi: https://doi.org/10.1016/j.inffus.2021.11.
003. URL https://www.sciencedirect.com/
science/article/pii/S1566253521002281.
De Lara, L., Gonz´alez-Sanz, A., Asher, N., and Loubes,
J.-M.
Transport-based counterfactual models.
arXiv
preprint arXiv:2108.13025, 2021.
Du, Q., Biau, G., Petit, F., and Porcher, R. Wasserstein
random forests and applications in heterogeneous treat-
ment effects. In International Conference on Artificial
Intelligence and Statistics, pp. 1729–1737. PMLR, 2021.
FICO.
Fico. explainable machine learning challenge,
2018. URL https://community.fico.com/s/
explainable-machine-learning-challenge.
Geurts, P., Ernst, D., and Wehenkel, L. Extremely random-
ized trees. Machine learning, 63:3–42, 2006.
Grathwohl, W., Wang, K.-C., Jacobsen, J.-H., Duvenaud, D.,
Norouzi, M., and Swersky, K. Your classifier is secretly
an energy based model and you should treat it like one. In
International Conference on Learning Representations,
2020.
Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-
based models still outperform deep learning on typical
tabular data? In Thirty-sixth Conference on Neural In-
formation Processing Systems Datasets and Benchmarks
Track, 2022.
Guilmeau, T., Chouzenoux, E., and Elvira, V. Simulated
annealing: a review and a new scheme. pp. 101–105, 07
2021. doi: 10.1109/SSP49050.2021.9513782.
Ishwaran, H., Kogalur, U. B., Blackstone, E. H., and Lauer,
M. S. Random survival forests. The annals of applied
statistics, 2(3):841–860, 2008.
Jocteur, B.-A., Maume-Deschamps, V., and Ribereau, P.
Heterogeneous treatment effect based random forest:
Hterf. 2023.
Kaggle.
Pima
indians
diabetes
database,
2016.
URL
https://www.kaggle.com/datasets/
uciml/pima-indians-diabetes-database.
Kanamori, K., Takagi, T., Kobayashi, K., and Ike, Y. Coun-
terfactual explanation trees: Transparent and consistent
actionable recourse with decision trees. In Proceedings
of The 25th International Conference on Artificial Intelli-
gence and Statistics, PMLR 151:1846-1870, 2022.
Karimi, A., Barthe, G., Sch¨olkopf, B., and Valera, I. A
survey of algorithmic recourse: definitions, formulations,
solutions, and prospects. CoRR, abs/2010.04050, 2020a.
URL https://arxiv.org/abs/2010.04050.


<!-- page 10 -->
Submission and Formatting Instructions for ICML 2024
Karimi, A.-H., Barthe, G., Balle, B., and Valera, I. Model-
agnostic counterfactual explanations for consequential
decisions. ArXiv, abs/1905.11190, 2020b.
Kelley Pace, R. and Barry, R.
Sparse spatial au-
toregressions.
Statistics,
Probability
Letters,
33(3):291–297,
1997.
ISSN 0167-7152.
doi:
https://doi.org/10.1016/S0167-7152(96)00140-X.
URL
https://www.sciencedirect.com/
science/article/pii/S016771529600140X.
Lakkaraju, H., Slack, D., Chen, Y., Tan, C., and Singh, S.
Rethinking explainability as a dialogue: A practitioner’s
perspective. CoRR, abs/2202.01875, 2022. URL https:
//arxiv.org/abs/2202.01875.
Lecun, Y., Chopra, S., and Hadsell, R. A tutorial on energy-
based learning. 01 2006.
Ley, D., Mishra, S., and Magazzeni, D. Global counterfac-
tual explanations: Investigations, implementations and
improvements, 2022. URL https://arxiv.org/
abs/2204.06917.
Lin, Y. and Jeon, Y. Random forests and adaptive nearest
neighbors. Journal of the American Statistical Associa-
tion, 101(474):578–590, 2006.
Liu, F. T., Ting, K. M., and Zhou, Z.-H. Isolation forest. In
2008 eighth ieee international conference on data mining,
pp. 413–422. IEEE, 2008.
Loh, W.-Y. Classification and regression trees. Wiley In-
terdisciplinary Reviews: Data Mining and Knowledge
Discovery, 1, 2011.
Looveren, A. V. and Klaise, J.
Interpretable counter-
factual explanations guided by prototypes.
CoRR,
abs/1907.02584, 2019. URL http://arxiv.org/
abs/1907.02584.
Lucic, A., Oosterhuis, H., Haned, H., and de Rijke, M.
Focus: Flexible optimizable counterfactual explanations
for tree ensembles. In Proceedings of the AAAI Confer-
ence on Artificial Intelligence, volume 36, pp. 5313–5322,
2022.
Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin,
J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N.,
and Lee, S.-I. From local explanations to global under-
standing with explainable ai for trees. Nature Machine
Intelligence, 2(1):2522–5839, 2020.
Meinshausen, N. and Ridgeway, G. Quantile regression
forests. Journal of Machine Learning Research, 7(6),
2006.
Molnar, C. Interpretable Machine Learning. 2 edition,
2022. URL https://christophm.github.io/
interpretable-ml-book.
Mothilal, R. K., Sharma, A., and Tan, C. Explaining ma-
chine learning classifiers through diverse counterfactual
explanations. In Proceedings of the 2020 Conference
on Fairness, Accountability, and Transparency, FAT*
’20, pp. 607–617, New York, NY, USA, 2020. Associa-
tion for Computing Machinery. ISBN 9781450369367.
doi: 10.1145/3351095.3372850. URL https://doi.
org/10.1145/3351095.3372850.
Parmentier, A. and Vidal, T. Optimal counterfactual expla-
nations in tree ensembles. CoRR, abs/2106.06631, 2021.
URL https://arxiv.org/abs/2106.06631.
Patki, N., Wedge, R., and Veeramachaneni, K. The synthetic
data vault. In 2016 IEEE International Conference on
Data Science and Advanced Analytics (DSAA), pp. 399–
410, Oct 2016. doi: 10.1109/DSAA.2016.49.
Pawelczyk, M., Broelemann, K., and Kasneci, G. Learning
model-agnostic counterfactual explanations for tabular
data. In Proceedings of the web conference 2020, pp.
3126–3132, 2020.
Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter,
T., and Kasneci, G. Carla: A python library to bench-
mark algorithmic recourse and counterfactual explanation
algorithms, 2021.
Pawelczyk, M., Datta, T., van-den Heuvel, J., Kasneci, G.,
and Lakkaraju, H. Algorithmic recourse in the face of
noisy human responses, 2022. URL https://arxiv.
org/abs/2203.06768.
Poyiadzi, R., Sokol, K., Santos-Rodriguez, R., Bie, T. D.,
and Flach, P. A. FACE: feasible and actionable counter-
factual explanations. CoRR, abs/1909.09369, 2019. URL
http://arxiv.org/abs/1909.09369.
Rawal, K. and Lakkaraju, H. Beyond individualized re-
course: Interpretable and interactive summaries of action-
able recourses. Advances in Neural Information Process-
ing Systems, 33:12187–12198, 2020.
Ribeiro, M. T., Singh, S., and Guestrin, C. ” why should
i trust you?” explaining the predictions of any classifier.
In Proceedings of the 22nd ACM SIGKDD international
conference on knowledge discovery and data mining, pp.
1135–1144, 2016.
Russell, C.
Efficient search for diverse coherent expla-
nations.
In Proceedings of the Conference on Fair-
ness, Accountability, and Transparency, FAT* ’19, pp.
20–28, New York, NY, USA, 2019. Association for Com-
puting Machinery.
ISBN 9781450361255.
doi: 10.


<!-- page 11 -->
Submission and Formatting Instructions for ICML 2024
1145/3287560.3287569. URL https://doi.org/
10.1145/3287560.3287569.
Scornet, E. Random forests and kernel methods. IEEE
Transactions on Information Theory, 62(3):1485–1500,
2016.
Tolomei, G., Silvestri, F., Haines, A., and Lalmas, M. In-
terpretable predictions of tree-based ensembles via ac-
tionable feature tweaking. In Proceedings of the 23rd
ACM SIGKDD international conference on knowledge
discovery and data mining, pp. 465–474, 2017.
Ustun, B., Spangher, A., and Liu, Y. Actionable recourse in
linear classification. Proceedings of the Conference on
Fairness, Accountability, and Transparency, 2019.
Verma, S., Dickerson, J. P., and Hines, K. Counterfactual
explanations for machine learning: A review. CoRR,
abs/2010.10596, 2020. URL https://arxiv.org/
abs/2010.10596.
Wachter, S., Mittelstadt, B. D., and Russell, C. Counterfac-
tual explanations without opening the black box: Auto-
mated decisions and the gdpr. Cybersecurity, 2017.
Wager, S. and Athey, S. Estimation and inference of hetero-
geneous treatment effects using random forests, 2017.
Washington, A. L. How to argue with an algorithm: Lessons
from the compas-propublica debate. Colo. Tech. LJ, 17:
131, 2018.
Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veera-
machaneni, K. Modeling tabular data using conditional
gan. In NeurIPS, 2019.


<!-- page 12 -->
Submission and Formatting Instructions for ICML 2024
A. Regional RF detailed
In this section, we give a simple application of the Regional RF algorithm to better understand how it works. Recall that
the Regional RF is a generalization of the RF’s algorithm to give prediction even when we condition given a region, e.g.,
to estimate E(f(X) |XS ∈CS(x), X ¯S = x ¯S) with CS(x) = Q|S|
i=1[ai, bi], ai, bi ∈¯R a hyperrectangle. The algorithm
works as follows: we drop the observations in the initial trees, if a split used variable i ∈¯S, a fixed value-based condition,
we used the classic rules, i.e., if xi ≤t, the observations go to the left children, otherwise the right children. However, if a
split used variable i ∈S, regional-based condition, we used the hyperrectangle CS(x) = Q|S|
i=1[ai, bi]. The observations
are sent to the left children if bi ≤t, right children if ai > t and if t ∈[ai, bi] the observations are sent both to the left and
right children.
To illustrate how it works, we use a two dimensional variables X = (X0, X1) ∈R2, a simple decision tree f represented
in Figure 4, and want to compute for x = [1.5, 1.9], E(f(X) | X1 ∈[2, 3.5], X0 = 1.5). We assume that P(X1 ∈
[2, 3.5] | X0 = 1.5) > 0 and denoted T1 as the set of the values of the splits based on variables X1 of the decision tree.
One way of estimating this conditional mean is by using Monte Carlo sampling. Therefore, there are two cases :
Figure 4: Representation of a simple decision tree (right Figure) and its associated partition (left Figure). The gray part in
the partition corresponds to the region [2, 3.5] × [1, 2]
• If ∀t ∈T1, t ≤2 or t > 3.5, then all the observations sampled s.t. ˜Xi ∼PX | X1∈[2, 3.5],X0=1.5 follow the same path
and fall in the same leaf. The Monte Carlo estimator of the decision tree E(f(X)|X1 ∈[2, 3.5], X0 = 1.5) is equal to
the output of the Regional RF algorithm.
– For instance, a special case of the case above is: if ∀t ∈T1, t ≤2, and we sample using PX | X1∈[2, 3.5],X0=1.5,
then all the observations go to the right children when they encounters a node using X1 and fall in the same leaf.
• If ∃t ∈T1 and t ∈[2, 3.5], then the observations sampled s.t.
˜Xi ∼PX | X1∈[2, 3.5],X0=1.5 can fall in multiple
terminal leaf depending on if their coordinates x1 is lower than t. Following our example, if we generate samples
using PX | X1∈[2, 3.5],X0=1.5, the observations will fall in the gray region of Figure 4, and thus can fall in node 4 or 5.
Therefore, the true estimate is:
E(f(X) | X1 ∈[2, 3.5], X0 = 1.5) = P(X1 ≤2.9 |X0 = 1.5) × E[f(X) |X ∈L4]
+ P(X1 > 2.9 |X0 = 1.5) × E[f(X) |X ∈L5]
(3)
Concerning the last case (t ∈[2, 3.5]), we need to estimate the different probabilities P(X1 ≤2.9 | X0 = 1.5), P(X1 >
2.9 | X0 = 1.5) to compute E(f(X) | X1 ∈[2, 3.5], X0 = 1.5), but these probabilities are difficult to estimate in practice.
However, we argue that we can ignore these splits, and thus do no need to fragment the query region using the leaves of the
tree. Indeed, as we are no longer interest in a point estimate but regional (population mean) we do not need to go to the level
of the leaves. We propose to ignore the splits of the leaves that divide the query region. For instance, the leaves 4 and 5

[CAPTION] Figure 4: Representation of a simple decision tree (right Figure) and its associated partition (left Figure). The gray part in


<!-- page 13 -->
Submission and Formatting Instructions for ICML 2024
split the region [2, 3.5] in two cells, by ignoring these splits we estimate the mean of the gray region by taking the average
output of the leaves 4 and 5 instead of computing the mean weighted by the probabilities as in Equation (3). Roughly, it
consists to follow the classic rules of a decision tree (if the region is above or below a split) and ignore the splits that are in
the query region, i.e., we average the output of all the leaves that are compatible with the condition X1 ∈[2, 3.5], X0 = 1.5.
We think it leads to a better estimation for two reasons. First, we observe that the case where t is in the region and thus
divides the query region does not occur often. Moreover, the leaves of the trees are very small in practice, and taking the
mean of observations that fall into the union of leaves that belong to the query region is more reasonable than computing the
weighted mean and thus trying to estimate the different probabilities P(X1 ≤2.9 |X0 = 1.5), P(X1 > 2.9 |X0 = 1.5).
B. Sampling Recourses from counterfactual rules
Algorithm 1 Simulated Annealing to generate counterfactual samples using the Counterfactual Rules
Input
:Observation x, Divergent Explanation S, counterfactual rule CS(x, Y ⋆), Dn training data set, number of iterations
maxIter, temperature T, cooling rate r
Output :Inlier sample xbest
1: Set xcurrent ←x, and xbest ←x
2: for j ∈S do
3:
xcurrent
j
←sample uniformly from the set {Xi,j : Xi ∈Dn and Xi,S ∈CS(x, Y ⋆)} ;
/* Generate
xcurrent = (zS, x ¯S) with zS drawn using zS ∼Q
i∈S ˆPXj |XS∈CS(x,Y ⋆).
*/
4:
xbest
j
←xcurrent
j
;
/* Initialize xbest */
5: end for
6: for it from 1 to maxIter do
7:
xnew ←xcurrent
8:
S′ ←sample uniformly from the set S
9:
for j in S′ do
10:
xnew
j
←sample uniformly from the set {Xi,j : Xi ∈Dn and Xi,S ∈CS′(x, Y ⋆)}
11:
end for
12:
Compute the Outlier score difference ∆O between xnew and xcurrent
13:
if ∆O < 0 or exp(−∆O/T) > random(0, 1) then
14:
Set xcurrent ←xnew
15:
end if
16:
if Outlier score of xbest < Outlier score of xcurrent then
17:
Set xbest ←xcurrent
18:
end if
19:
Decrease T by T = T ∗r
20: end for
21: return xbest
C. Additional experiments
In table 2, we compare the Accuracy (Acc), Plausibility (Psb), and Sparsity (Sprs) of the different methods on additonal
real-world datasets: FICO (FICO, 2018), NHANESI (CDC, 1999-2022).
We observe that the L-CR, and R-CR outperform the baseline methods by a large margin on Accuracy and Plausibility.
The baseline methods still struggle to change at the same time the positive and negative class. AReS and CET give better
sparsity, but their counterfactual samples are less plausible than the ones generated by the CR.


<!-- page 14 -->
Submission and Formatting Instructions for ICML 2024
Table 2: Results of the Accuracy (Acc), Plausibility, and Sparsity (Sprs) of the different methods. We compute each metric
according to the positive (Pos) and negative (Neg) class.
FICO
NHANESI
Acc
Psb
Sps
Acc
Psb
Sps
Pos
Neg
Pos
Neg
Pos
Neg
Pos
Neg
Pos
Neg
Pos
Neg
L-CR
0.98
0.94
0.98
0.99
5
5
0.99
0.98
0.98
0.97
5
6
R-CR
0.90
0.94
0.98
0.99
9
8.43
0.86
0.95
0.96
0.99
7
7
AReS
0.34
0.01
0.85
0.86
2
1
0.06
1
0.87
0.92
1
1
CET
0.76
0
0.76
0.60
2
2
0
0.40
0.82
0.56
0
5
D. Parameters detailed
In this section, we give the different parameters of each method. For all methods and datasets, we first used a greedy search
given a set of parameters. For AReS, we use the following set of parameters:
• max rule = {4, 6, 8}, max rule length = {4, 8}, max change num = {2, 4, 6},
• minimal support = 0.05, discretization bins = {10, 20},
• λacc = λcov = λcst = 1.
Lastly, for the Counterfactual Rules, we used the following parameters:
• nb estimators = {20, 50}, max depth= {8, 10, 12},
• π = 0.9, πC = 0.9.
We obtained the same optimal parameters for all datasets:
• AReS: max rule = 4, max rule length= 4, max change num = 4, minimal support = 0.05, discretization bins = 10,
λacc = λcov = λcst = 1
• CET: max iterations = 1000, max leaf size = −1, λ = 0.01, γ = 1
• CR: nb estimators= 20, max depth= 10, π = 0.9, πC = 0.9

[CAPTION] Table 2: Results of the Accuracy (Acc), Plausibility, and Sparsity (Sprs) of the different methods. We compute each metric