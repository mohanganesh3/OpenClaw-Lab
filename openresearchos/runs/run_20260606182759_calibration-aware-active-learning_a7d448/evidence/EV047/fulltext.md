<!-- page 1 -->
Theoretical Guarantees of Learning Ensembling Strategies
with Applications to Time Series Forecasting
Hilaf Hasson 1 Danielle C. Maddix 1 Yuyang Wang 1 Gaurav Gupta 1 Youngsuk Park 1
Abstract
Ensembling is among the most popular tools in
machine learning (ML) due to its effectiveness in
minimizing variance and thus improving general-
ization. Most ensembling methods for black-box
base learners fall under the umbrella of “stacked
generalization,” namely training an ML algorithm
that takes the inferences from the base learners
as input. While stacking has been widely applied
in practice, its theoretical properties are poorly
understood. In this paper, we prove a novel result,
showing that choosing the best stacked generaliza-
tion from a (finite or finite-dimensional) family of
stacked generalizations based on cross-validated
performance does not perform “much worse” than
the oracle best. Our result strengthens and signif-
icantly extends the results in Van der Laan et al.
(2007). Inspired by the theoretical analysis, we
further propose a particular family of stacked gen-
eralizations in the context of probabilistic forecast-
ing, each one with a different sensitivity for how
much the ensemble weights are allowed to vary
across items, timestamps in the forecast horizon,
and quantiles. Experimental results demonstrate
the performance gain of the proposed method.
1. Introduction
Ensemble methods have been a staple in machine learn-
ing with ensemble-based methods such as Random Forests
(Breiman, 2001) and XGBoost (Chen & Guestrin, 2016)
being among the most popular choices for tabular data by
practitioners (Kaggle, 2020; 2021). Both classical (mean,
median, weights based on cross-validated performance) and
relatively modern ensembling strategies, e.g., Erickson et al.
(2020) (which in two popular Kaggle competitions has beat
99% of the participating data scientists after training on
1AWS AI Labs, Santa Clara, CA, USA. Correspondence to:
Hilaf Hasson <hashilaf@amazon.com>.
Proceedings of the 40 th International Conference on Machine
Learning, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright
2023 by the author(s).
the raw data) are examples of “stacked generalizations”,
or “stacking” (Wolpert, 1992); see (Ting & Witten, 1997b).
Stacking is an umbrella term referring to the process of train-
ing one model on top of the predictions of the base learners.
In the simple mean and median cases, the stacking model
is constant rather than learned. Despite being heavily used
in practice, the theoretical properties of stacked generaliza-
tions are underexplored, and stacking has notoriously been
referred to as “black art” (Wolpert, 1992; Ting & Witten,
1997a).
One early attempt to quantifiably understand stacked gener-
alization is due to Van der Laan et al. (2007). Built on top
of Van der Vaart et al. (2006), the authors prove a theoret-
ical guarantee (Theorem 2 of Van der Laan et al. (2007))
that choosing the best stacked generalization out of a fi-
nite (rather than finite-dimensional) family of constant (as
opposed to learned) stacked generalizations based on cross-
validation performance does not do much worse than the
oracle best. They then apply their result to the case of a sin-
gle learned stacked generalization by discretizing the space
of the functions that the stacked generalization may become
after the training, resulting in a guarantee that degrades with
the cardinality of the chosen discretization.
In this paper, we make a major step towards a better the-
oretical understanding of the stacking mechanism. More
precisely, we show new results that extend Van der Laan
et al. (2007) in two ways: 1. We remove the assumption
that the stacked generalizations are constant, and permit
them to be learned, without changing the conclusion of the
result. This allows us to get much better guarantees even
for the case of a single stacked generalization, as it removes
the need for a discretization. (See the discussion following
Theorem 4.1.) 2. We extend the result from the case of finite
families to the case of finite-dimensional families.
We further push the boundary beyond the tabular case to
study the concrete use case of time series forecasting. En-
sembling of time series models is a notoriously challenging
task, coined as the “Forecast Combination Puzzle” by (Stock
& Watson, 2004), with many follow-up works in the litera-
ture; see Section 2. We concern ourselves with datasets that
involve multiple time series, forecasting a fixed number of
timestamps into the future, and predicting multiple quantiles
1
arXiv:2305.15786v4  [cs.LG]  16 Dec 2025


<!-- page 2 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
for each time series and timestamp. Motivated by the theo-
retical insights, we propose a particular finite-dimensional
family of stacked generalizations, in which how much the
ensemble weights are allowed to vary across time series,
timestamps, and quantiles is each controlled by a dimension
of the family.
In summary, our contributions are two-fold:
1. In the tabular case we show (Theorem 4.1) that the pro-
cess of letting cross validation determine a stacked gen-
eralization from an arbitrary finite or finite-dimensional
family of stacked generalizations cannot perform much
worse than the oracle best, and under an additional as-
sumption, from each of the base learners. This extends
a result from Van der Laan et al. (2007) by allowing
the stacked generalizations to be learned rather than
constant, and by extending from the case of a finite
to finite dimensional family. In the case of a single
stacked generalization, our result gives tighter bounds
than Van der Laan et al. (2007), as the latter requires a
discretization of the family of potential models of the
stacked generalization once trained.
2. We propose a family of stacked generalizations in the
time series use case, as well as a setup for choosing the
best performing one, in a manner that learns the ensem-
ble elasticity (how much the weights are allowed to
vary) across time series/timestamps/quantiles. Despite
Theorem 4.1 not applying directly to the non-tabular
case, we show experiments that demonstrate that our
method inspired by this theory is effective, supported
by strong evidence that it is able to adjust to changes
in performance of the base learners across time se-
ries/timestamps/quantiles.
We organize the rest of the paper as follows. Related work
is discussed in Section 2, followed by preliminaries and
notation (Section 3) to set the stage. The subsequent sections
(4 and 5) cover the main results with its proof. In Section 6,
we present the case study of time series forecasting, and use
the next two sections to provide empirical evidences for the
effectiveness of the proposed algorithm before concluding
the paper.
2. Related Work
In this section, we give an overview of theoretical and ap-
plied research on stacked generalizations, as well as their
use in time series forecasting.
Stacked Generalizations
Stacked generalization has be-
come an exceedingly popular choice for ensembling in prac-
tice, and particularly impressive in its results, as illustrated
in Erickson et al. (2020). In a recent work (Kim et al., 2021),
it has also been used for ensembling probabilistic predic-
tions in the tabular setup. We refer to Zhou (2012) for a
general discussion on the benefits of ensembling, and to
Cruz et al. (2018) for a thorough benchmarking of ensem-
bling methods.
We remark that while our paper concerns black box base
learners, there has also been an active research front on
creating top performing algorithms by choosing the base
learners as well as the ensembling method, e.g., Breiman
(2001); Chen & Guestrin (2016); Fort et al. (2019); Laksh-
minarayanan et al. (2017), etc.
Theoretical Results for Stacked Generalizations
As
mentioned in the introduction, in Van der Laan et al. (2007)
the authors provide theoretical guarantees for stacked gener-
alizations. They first discretize the set of possible functions
that the stacked generalization may become once trained,
and in this way reduce the problem to giving guarantees
(Theorem 2 of Van der Laan et al. (2007)) that choosing the
best performing (in cross validation) stacked generalization
from a finite set of stacked generalizations, each of which
is a constant function (e.g., simple mean, or some other
constant weighted combination), does not do much worse
than the oracle best.
Time Series Ensembling
In the time series use case, it has
been empirically observed that often simple averaging of
the forecasts of the base learners is superior to more sophis-
ticated ensemble methods; a problem dubbed the “forecast
combination puzzle” in Stock & Watson (2004). This dates
back to Bates & Granger (1969), where the authors observed
that attempting to learn covariance terms between the er-
rors of the individual learners introduces too much variance.
Similar concerns over the amount of variance introduced
when learning weights for time series ensembling are voiced
in Smith & Wallis (2009) and Claeskens et al. (2016), where
both works explored squared loss. Elliott (2011) focuses on
understanding the potential gain from learning the optimal
weights compared to simple averaging, and provides the
following two results: first, that if the maximal eigenvalue
of the covariance matrix of errors of the base learners is
bounded, then as the number of base learners goes to infin-
ity, the performance of using the optimal weights converges
to that of using simple average; second, he gives an upper
bound for the potential gain for the case of 3 and 4 base
learners. We remark that the papers referenced here regard-
ing the forecast combination puzzle are restricted to the
situation of learning weights that are global (unchanging
across time or any other dimension), and based on an analy-
sis of mean squared loss (so as to simplify the theory), and
they do not apply more broadly.
Stacked generalizations have been applied for time series
forecast ensembling as far back as in Donaldson & Kamstra
2


<!-- page 3 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
(1996); Moon et al. (2020); Massaoudi et al. (2021). In
Gastinger et al. (2021) we see the success of these single
stacked generalizations in a large empirical study. In all of
these cases a single stacked generalization was used, and
not a family of stacked generalizations.
3. Preliminaries and Notation
In both the main theorem and in its proof we use language
and notation commonly used in oracle inequalities. In order
to be self contained we will now introduce the relevant
notation and terminology.
Let {θα}α∈J be a finite set of algorithms for tabular data.
Assume that feature space is d dimensional, the target
values are r dimensional, and the output of predictions
is r′ dimensional. (An example when r ≠r′ would be:
r = 1, and the predictions are for r′ many conditional
quantiles.) We write θα(D0) ∶Rd →Rr′ for the model
that results from letting θα train on a training dataset D0.
(For the sake of simplicity, assume that θα(D0) is com-
pletely determined by D0, though this assumption can eas-
ily be removed.) Let L ∶Rr′ × Rr →R be a fixed loss
function, and let Lα(D0) ∶Rd × Rr →R be defined by
Lα(D0)((x,y)) ∶= L(θα(D0)(x),y).
Let X be a random sample of Rd × Rr w.r.t. the same
probability distribution P as the one being sampled by the
data.1 The ˜α that minimizes E(∫L˜α(D0)(X)dP) (where
the expected value runs over D0) is called the “oracle”. We
never have direct access to the oracle, but given a validation
set D1 we can choose the ˆα for which θˆα has the lowest
(empirical) validation loss. In order to be more precise,
write Di = {Xi
1,...,Xi
ni} for i = 0,1 (where the {Xi
l }ni
l=1
are IID random variables taking values in Rd × Rr for some
fixed ni ∈N), representing training (i = 0) and valida-
tion (i = 1) sets; and let Pi for i = 0,1 be the empirical
distribution
1
ni ∑ni
l=1 δXi
l . Then ˆα is an index for which
∫Lˆα(D0)(X)dP1 is minimized over α ∈J .
Oracle inequalities are about giving upper bounds of
E(∫Lˆα(D0)(X)dP) in terms of E(∫L˜α(D0)(X)dP),
quantifying how much worse ˆα is compared with the ora-
cle. (While much of the relevant literature accommodates
a scheme where the data is split multiple times to train and
validation, we have chosen to focus on the case of a sin-
gle split for the sake of simplified notation. The interested
reader should be able to easily generalize.) In Section 5 we
introduce a new oracle inequality that allows us to prove our
main theorem (Theorem 4.1).
In the oracle inequalities we discuss we will use the follow-
ing notion.
1This notation is common in literature, but note that it is con-
fusing: X is both the feature vector and the target.
Definition 3.1. Let X be a sample space with probability
measure P. Then a we say that a function f ∶X →R has
Bernstein numbers (or Bernstein pair) (M(f),v(f)) ∈R2
if:
M(f)2 ∫(e
∣f∣
M(f) −1 −
∣f∣
M(f))dP ≤1
2v(f).
(1)
A set of functions F from X to R is said to have Bern-
stein numbers (M(F),v(F)) ∈R2 if they are Bernstein
numbers for all f ∈F.
The existence of Bernstein numbers can be viewed as a
weak moment condition. (It is a much weaker condition
than the functions in F having a uniformly bounded range;
we refer to (Wellner et al., 2013) for further intuition.)
Finally, for any subset J of Euclidean space and value
ε > 0 the notation N int(J ,ε) denotes the (internal) covering
number of the space J with respect to its ambient Euclidean
space.
4. Main Theorem
Let D0 and D1 be datasets of, respectively, n0 and n1 IID
random variables, each single variable taking values in Rd ×
Rr (i.e, with feature dimension d and target dimension r).
Let η1,...,ηm be a finite set of tabular algorithms (hence-
forth “the base learners”) with feature dimension d, predic-
tion dimension r′, and target dimension r. Let {Aα}α∈J
be a set of stacked generalizations (tabular algorithms with
feature dimension m, prediction dimension r′, and target di-
mension r). Let D0 be further split as D0 = D00 ⊍D01. The
base learners will train on D00, whereas the stacked general-
izations will train on the predictions of the base learners on
D01. (We remark that having D00 and D01 be disjoint is the
method recommended in (Zhou, 2012); though in fact what
follows will continue to hold if you set D0 = D00 = D01.)
Let
Z ∶= {((η1(D00)(x),...,ηm(D00)(x)),y)∣(x,y) ∈D01}
be the predictions on D01 of the base learners that
were trained on D00.
Let L ∶Rr′ × Rr
→R be
some loss function, and let Lα(D00,D01)((x,y)) ∶=
L(Aα(Z)(η1(D00)(x),...,ηm(D00)(x)),y),
where
Aα(Z) stands for Aα trained on Z.
Consider F ∶= {(x,y) ↦Lα(D00,D01)((x,y))∣α ∈J },
and let (M(F),v(F)) be its Bernstein numbers. Finally,
let X be a random variable in Rd × Rr following the same
distribution that D0 and D1 are sampling from. Then the
following holds, using the notation in Section 3.
3


<!-- page 4 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
Theorem 4.1. With the notation above, for every ˆα ∈J let
Wˆα ∶= {˜α ∈J ∣∫Lˆα(D00,D01)(X)dP1
≤∫L˜α(D00,D01)(X)dP1},
be the set of indices ˜α for which θˆα outperforms θ˜α on
validation. Further assume that J is a bounded subset of
a finite dimensional Euclidean space, and that for every
D0 the function that takes α ∈J to Lα(D00,D01) ∈F
is Lipschitz with constant ℓw.r.t. the infinity norm in F.
Then we have that for every δ > 0, 1 ≤p ≤2, and sequence
εn1 > 0,
E(∫Lˆα(D00,D01)(X)dP) ≤
(1 + 2δ) inf
˜α∈W ˆ
α (E(∫L˜α(D00,D01)(X)dP))
+ sup
f∈F
(Bf) + 2((1 + δ) + 1
n1
)εn1,
(2)
where
Bf ∶=
16( M(F)
n
1−1
p
1
+ (
v(F)
(δ ∫fdP )2−p )
1
p )log(1 + N int(J ,
εn1
ℓ))
n
1
p
1
.
In addition, the following hold:
1. If we choose εn1 ∶= n
−1
2 −ϵ
1
for any ϵ > 0 then 2((1+δ)+
1
n1 )εn1 = O(n
−1
2 −ϵ
1
) (in particular, o(1)). Moreover,
since J is bounded in finite-dimensional Euclidean
space, log(1 + N int(J ,
εn1
ℓ)) = O(log n1). For p =
2 (so that (δ ∫fdP)2−p = 1), we consequently have
supf∈F(Bf) = O(log n1/√n1) = o(1). For p < 2,
the same conclusion holds provided inff∈F ∫fdP ≥
c for some constant c > 0 (so that supf∈F(Bf) =
O(log n1/n
1
p
1 ) = o(1)).
2. For the guarantee to hold not only against the best
stacked generalizations in the family, but also against
the original base learners, one simply adds the sum-
mand m to the inside of the log term.
3. If J is finite rather than finite dimensional, then the
log term can be replaced with log(1 + ∣J ∣), and we
may remove the last summand 2((1 + δ) +
1
n1 )εn1.
In the case that ∣J ∣is finite, and that, in addition, the stacked
generalizations are constant functions (i.e., do not depend
on Z at all), then the theorem above specializes to Theorem
2 of Van der Laan et al. (2007). The two main differences
are the following:
1. There the authors make additional assumptions (such as
that r = r′ = 1 and that the loss function is MSE); and
they write in the language of multiple cross validations,
which we have chosen not to use so as not to overload
notation (though generalization is straightforward).
2. In Van der Laan et al. (2007), the authors argue that
their result applies to the case that ∣J ∣= 1 and the
stacked generalization is learned (does depend on Z)
by first discretizing the set of functions that the stacked
generalization may become, thus reducing to the case
of a finite number of constant stacked generalizations
˜
J as in Theorem 2 therein; and that the error term from
this step is asymptotically negligible. Note, however,
that by following this procedure the log term becomes
log(1 + ∣˜
J ∣+ m) (which explodes with the size of the
discretization); whereas if you use the theorem above
you get log(2+m) and does depend on a discretization.
Therefore the theorem above is a significant improve-
ment even in the case ∣J ∣= 1.
5. Reduction to Oracle Inequalities
One of the main insights that led to Theorem 4.1 is the
recognition that a sufficiently general and strong version of
an oracle inequality (Theorem 5.1 below) can lead to signif-
icant gains in obtaining better and more general theoretical
guarantees for stacked generalizations, as we proceed to
illustrate.
We deviate from common notation by now letting D0 and
D1 be ordered datasets of IID feature and target pairs;
taking values in (Rd × Rr)n0 and (Rd × Rr)n1 respec-
tively. Let X be a random variable in Rd × Rr follow-
ing the same distribution. Let {θα}α∈J be a set of tab-
ular algorithms that may take into account the order of
the training dataset. Let L ∶Rr′ × Rr →R be some loss
function, and let Lα(D0)((x,y)) ∶= L(θα(D0)(x),y). Fi-
nally let F ∶= {(x,y) ↦Lα(D0)((x,y))∣α ∈J }, and let
(M(F),v(F)) be its Bernstein numbers. Then the follow-
ing holds.
Theorem 5.1. With the notation above, for every ˆα ∈J let
Wˆα ∶= {˜α ∈J ∣
∫Lˆα(D0)(X)dP1 ≤∫L˜α(D0)(X)dP1},
be the set of indices ˜α for which θˆα outperforms θ˜α on
validation. Further assume that J is a bounded subset of a
finite dimensional Euclidean space, and that for every D0 ∈
(Rd × Rr)n0 the function that takes α ∈J to Lα(D0) ∈F
is Lipschitz with constant ℓw.r.t. the infinity norm in F.
Then we have that for every δ > 0, 1 ≤p ≤2, and sequence
4


<!-- page 5 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
εn1 > 0,
E(∫Lˆα(D0)(X)dP) ≤
(1 + 2δ) inf
˜α∈W ˆ
α (E(∫L˜α(D0)(X)dP))
+ sup
f∈F
(Bf) + 2((1 + δ) + 1
n1
)εn1,
(3)
where
Bf ∶=
16( M(F)
n
1−1
p
1
+ (
v(F)
(δ ∫fdP )2−p )
1
p )log(1 + N int(J ,
εn1
ℓ))
n
1
p
1
.
In addition, the following hold:
1. If we choose εn1 ∶= n
−1
2 −ϵ
1
for any ϵ > 0 then 2((1+δ)+
1
n1 )εn1 = O(n
−1
2 −ϵ
1
) (in particular, o(1)). Moreover,
since J is bounded in finite-dimensional Euclidean
space, log(1 + N int(J ,
εn1
ℓ)) = O(log n1). For p =
2 (so that (δ ∫fdP)2−p = 1), we consequently have
supf∈F(Bf) = O(log n1/√n1) = o(1). For p < 2,
the same conclusion holds provided inff∈F ∫fdP ≥
c for some constant c > 0 (so that supf∈F(Bf) =
O(log n1/n
1
p
1 ) = o(1)).
2. If J is finite rather than finite dimensional, then the
log term can be replaced with log(1 + ∣J ∣), and we
may remove the last summand 2((1 + δ) +
1
n1 )εn1.
3. If one wants to artificially add to Wˆα any finite number
p of additional algorithms, then, in both the case that
J is finite and finite-dimensional, one simply adds p
to the inside of the log term.
Remark 5.2.
We remark that ℓcan easily be discerned from
the Lipschitz constant of L in its first coordinate and the Lip-
schitz constants of each of the maps α ↦θα. Note also that
if J is compact and α ↦∫Lα(D0)(X)dP1 is continuous
for each fixed (D0,D1), then there exists a (measurable)
minimizer ˆα of the empirical validation loss, in which case
Wˆα = J . In this case ˆα has the interpretation of being the
index of the best performing stacked generalization on cross
validation, and inf ˜α∈J (E(∫L˜α(D0)(X)dP)) is achieved
by some ˜α that has the interpretation of being the oracle
best stacked generalization.
We postpone the proof of this theorem to Appendix A,
though it is important to point out that this is both a strength-
ening and a generalization of Lemma 2.3 (or more accurately
of Lemmas 2.1 and 2.2) of (Van der Vaart et al., 2006) to
the case where the datasets are ordered and to the case that
J is potentially infinite. This is precisely the version that
we require to prove Theorem 4.1 easily.
Proof. (of Theorem 4.1)
Apply Theorem 5.1 to the set of algorithms θα(D0) ∶=
Aα(Z) ○(η1(D00),...,ηm(D00)).
Note that even in the case that J is finite it would not
have been possible to use the oracle inequalities in Van der
Vaart et al. (2006) directly to the algorithms θα(D0) ∶=
Aα(Z) ○(η1(D00),...,ηm(D00)) since these θα’s depend
on the order of D0 (to separate out D00 and D01); and that
this is the reason that Van der Laan et al. (2007) first re-
duced to the case of constant stacked generalizations (via a
discretization of the set of functions that the stacked gener-
alization may become).
6. Case Study: Probabilistic Time Series
Forecasting
Consider probabilistic forecasting algorithms that output a
prediction for each time series in the dataset (henceforth
“item”), for each timestamp in the forecast horizon, and
for each of a list of predefined quantiles. In combining
these algorithms into a weighted sum, it is a natural ques-
tion whether it is a good idea to share the weights across
items/timestamps/quantiles. As we will see, this translates
naturally to choosing the best stacked generalization out of
a family. While Theorem 4.1 does not apply directly outside
of the tabular use case, we use it as motivation.
6.1. Setup for Stacked Generalizations in Forecasting
As is common in probabilistic time series forecasting, we
consider a dataset containing N time series (henceforth
“items”). We consider a base learners η1,...,ηm, where each
one of them makes predictions for a fixed number of times-
tamps in the future h (henceforth “the forecast horizon”),
and the output for each item and timestamp is a prediction of
some fixed predicted quantiles q (e.g., q = 3 with quantiles
0.1,0.5,0.9). In particular the output for each base learner
is in RN×h×q. (See Appendix C.1 for details.)
Our interpretation of a stacked generalization in this setup
is an algorithm that takes as input the output of all of the
base learners trained up to some backtest window together
with the true values from this backtest window, and out-
puts for each tuple (item, timestamp in the forecast horizon,
and quantile) a weight for each of the base learners. (One
example of such a stacked generalization is one that opti-
mizes mean weighted quantile loss over the backtest window
treating all of the weights as independent; another example
would be one that optimizes mean weighted quantile loss on
the backtest window, but with each base learner having the
same weight regardless of the timestamp, item, and quan-
tile. We refer to Appendix C.1 for the definition of mean
weighted quantile loss.)
Once the weights are learned in this fashion on a single
5


<!-- page 6 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
backtest window, the inference of the stacked generalization
is a weighted sum of the base learners, where the weights
vary based on item, timestamp, and quantile.
Given a family of stacked generalizations, we interpret
choosing the best performing one based on cross validation
as specified in Algorithm 1, where the notation is borrowed
from Figure 1.
train
validation
test
forecast
ensemble
Figure 1. Dataset split for the purpose of choosing the best stacked
generalization out of a family based on cross validatation. (See
Algorithm 1.) Each DBWi contains as many timestamps as the
forecast horizon length of the base learners. The split is done
across all items.
We remark that a uniform notation between tabular and the
time series use cases it possible, albeit confusing. We refer
to Appendix D for such a treatment, while in the remainder
of the paper we make use of simplified notation.
6.2. A Family of Stacked Generalizations Controlling
Uniformity of Weights
We propose a family of stacked generalizations indexed by
α ∈R4
≥0; taking the output of the base learners together
with true values for the backtest window on which they are
making predictions, and finding the weights that optimize
mean weighted quantile loss (see Appendix C.1 for the
definition) together with the regularizers:
3
∑
d=1
αdH(σ(d)(w)) + α4 ∑
i,j,k,l
∣w(l)
i,j,k∣,
where H(σ(d)(w)) is an entropy term defined in detail
in Appendix C.2 that controls whether the weights are
forced to be uniform across items, timestamps, and quan-
tiles. Thus, for example, the algorithm A(0,0,0,0) optimizes
mean weighted quantile loss treated all the weights as inde-
pendent; whereas A(α1,0,0,0) as α1 goes to infinity would
optimize mean weighted quantile loss under the constraint
that for a specific timestamp and quantile the base learner
weights don’t vary across items.
Algorithm 1 Choosing a Stacked Generalization Out of
Family in the Time Series Setting
Input: m base learners: η1,...,ηm, a time series dataset
split as in Figure 1; and a family of stacked generaliza-
tions {Aα}α∈J .
Output: Predictions for DBW2.
1: for values of α do
2:
Train η1,...,ηm on D0 and use their predictions to-
gether with DBW0 to train Aα. Store the resulting
weights w.
3:
Train η1,...,ηm on D1 and w to make a weighted
sum prediction on DBW1; compute mean weighted
quantile loss.
4: end for
5: Store the ˆα for which the mean weighted quantile loss
computed was minimal.
6: Retrain Aˆα on the output of the models η1,...,ηm
trained on D1, together with DBW1. Store the resulting
weights as w∗.
7: Train η1,...,ηm on D2 and output their weighted sum
using w∗.
6.3. Experiments
In this subsection, we train base learners {ηl(D2)}l on the
full historical data D2, and compare our optimal stacked
generalization ensemble Aˆα(D2) in Eqn. (12) to the predic-
tions from the base learners and simple ensembling meth-
ods.
We report the mean weighted quantile loss (Eqn.
(8)) over the quantiles τ = [0.1,0.5,0.9] on the test set
L(Aˆα(D2),DBW2) on the corresponding most recent in
time backtest window DBW2 (See Eqn. (9) to recall the
definition of (D2,DBW2)). For each α, we use autodiff in
PyTorch (Paszke et al., 2017) to find the optimal weights
for each Aα in Algorithm 1 (using softmax to satisfy the
constraint), and use scipy’s implementation of COBYLA
(Powell, 1994) to find the optimal ˆα since looping over an
infinite set is impossible. (It is possible to do a grid search
to find ˆα, but we have empirically observed that using an op-
timization method is faster and leads to equivalent results.)
We test on real-world open-source datasets from the UCI
data repository (Dheeru & Karra Taniskidou, 2017), Kaggle
(Lai, 2017), and M4 competition datasets (Makridakis et al.,
2018) (see Table 5 in Appendix E).
Base Learners.
We call the following m = 6 base learn-
ers from GluonTS (Alexandrov et al., 2020) with the de-
fault hyper-parameter settings: the local state space models
ARIMA and ETS 2 (Hyndman et al., 2008), Non-Parametric
Time Series (NPTS) (Gasthaus, 2016), the global deep learn-
2The error from the ETS predictions on Wiki is too large to
include, and so we omit ETS in the corresponding ensembles.
6


**[Table p6.1]**
| train validation t | train | validation t |
| --- | --- | --- |

[CAPTION] Figure 1. Dataset split for the purpose of choosing the best stacked


<!-- page 7 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
ing models DeepAR (Flunkert et al., 2019) and MQ-CNN
(Wen et al., 2017), and Rotbaum (XGBoost-based forecaster,
a GluonTS implementation of TTSW from Hasson et al.
(2021). We compare to the simple ensemble baselines in
Appendix 6.3. The experiments are performed 5 times to
compute confidence intervals.
Baselines.
We compare to the following simple ensemble
baselines:
• Mean: For each item, timestamp, and quantile, take a
simple mean of all of the base learners.
• Median: For each item, timestamp, and quantile, take
a simple median of all of the base learners.
• Global Best (GB) (D1,DBW1): Of all of the possible
combinations of algorithms to choose, choose the one
for which the simple mean of the predictions of the cho-
sen algorithms lead to the best performance on DBW1.
The same algorithms are chosen across timestamps,
quantiles, and items. (Greedy ensemble, (Caruna et al.,
2004), is an approximation of this method in the case
that there are many base learners.)
• Best (D1,DBW1): Choose the single best base learner
on DBW1.
• Unregularized: Use α = (0,0,0,0) (in the sense of
the family given in Subsection 6.2); i.e., always use
the unregularized stacked generalization, rather than
choosing the best stacked generalization via cross vali-
dation.
We remark that if we had used only one backtest window,
namely if we learned α on the same backtest window on
which the weights are optimized, then α would have to equal
(0,0,0,0), and so this setup is equivalent to this baseline.
Benchmarking with Raw Predictions.
We first compare
the predictions from our method to the raw predictions from
the base learners, and the baseline ensembling methods.
We remark that unlike in the literature about the “forecast
combination puzzle” (Section 2), our method offers signifi-
cant performance boost compared with simple average. We
attribute this discrepancy to the idealized assumptions in
the literature, most prominent among them being that they
assume that the weights remain the same across timestamps
or any other dimension. Table 1 shows the results.
Synthetic Experiments Adding Noise to the Predictions
Across Various Dimensions.
We have seen in Table 3
that our algorithm works well in a natural setting. But given
that the family we have chosen (in Subsection 6.2) is meant
to find the ideal degree to which the weights are forced to
be similar across item/timestamps/quantiles, in this section
we create synthetic experiments the stress-test this ability,
and give strong evidence that it is able to adjust to these
disruptions.
To be specific, for each set of base learner predictions
ηl(Dn) = {ˆz(l)
i,j,k(Dn)}i,l
j,k, where n = 0,1,2, i indexes
the items, j the forecast horizon, and k the quantiles, we in-
troduce Gaussian noise. Since we are applying the noise for
each data training and validation splits, in the following we
simply use the notation {ˆz(l)
i,j,k}i,l
j,k. Let s(l)
i,k = stdj({ˆz(l)
i,j,k})
be the empirical standard deviation computed over the time
dimension j for fixed i,k,l. For each simulation, we let the
selected noise be the same for all the backtest windows. For
each i,j,k and l, we then add Gaussian noise ϵ(l)
i,j,k to the
base learner prediction, i.e.
ˆz(l)
i,j,k ↦ˆz(l)
i,j,k + N(0,ϵ(l)
i,j,k),
where the noise ϵ(l)
i,j,k varies in the following three ways:
1. Across time: For each l, let rl ∼U(0,0.5) and ϵ(l)
i,j,k =
2jrls(l)
i,k/h for forecast horizon h. In this experiment,
we design the noise to be increasing with the time
index j to mimic the natural order of time, where the
predictions become worse over time. (See Table 2).
2. Across items: For each l and i = 1,...,N let ri,l ∼
U(0,2) and ϵ(l)
i,j,k = ϵ(l)
i,k = ri,ls(l)
i,k (See Table 3).
3. Across quantiles: For each l and k = 1,2,3, let rk,l ∼
U(0,2) and ϵ(l)
i,j,k = ϵ(l)
i,k = 2rk,ls(l)
i,k (See Table 4).
See Tables 2 (adding noise across time) and 3 (adding noise
across items) and 4 (adding noise across quantiles) for re-
sults. In these three synthetic experiments, we see that
our method is successful in adjusting for the noises, and
gives similar results in those in Table 1 with no noise added.
We also see that using the unregularized loss, which corre-
sponds to choosing α = (0,0,0,0) rather than learning it
via cross-validation, is a surprisingly close second. It is not
remarkable that the rest of the baselines do not perform well
since the other baselines except for median use the same
weights for all timestamps/quantiles/items, and median is
another one size fits all solution that does not vary across
timestamps/quantiles/items.
7. Conclusion
In this paper, we provide theoretical guarantees for fami-
lies of stacked generalizations, which are commonly used
in practice in state-of-the-art ensembling methods. We ex-
tend the work in Van der Laan et al. (2007) that shows
that choosing the best stacked generalization from a finite
family of constant stacked generalizations based on cross-
validated performance does not perform much worse than
the oracle best, both by allowing the stacked generalizations
to be learned rather than constant, and by giving a result
for finite-dimensional, rather than finite, families. We then
show a particular family of stacked generalizations in the
probabilistic time series forecasting use case. We formulate
the problem of finding this family as a regularized regres-
7


<!-- page 8 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
Table 1. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world
open-source time series datasets over 5 runs.
Base Learner /
Ensemble Strategy
Elec
Kaggle
M4-daily
Traf
Wiki
ARIMA
0.105 ± 0.0
0.1201 ± 0.0
0.0341 ± 0.0
0.2219 ± 0.0
0.6008 ± 0.0
DeepAR
0.0585 ± 0.0011
0.2478 ± 0.0056
0.035 ± 0.0009
0.0987 ± 0.0014
0.3311 ± 0.0095
ETS
0.0796 ± 0.0203
0.1193 ± 0.0004
0.033 ± 0.0005
0.2999 ± 0.0
N/A
MQ-CNN
0.0698 ± 0.0018
0.2452 ± 0.0018
0.0408 ± 0.0025
0.2491 ± 0.0423
0.3639 ± 0.0038
NPTS
0.0536 ± 0.0005
0.1393 ± 0.0001
0.1218 ± 0.0001
0.1172 ± 0.0002
0.4798 ± 0.0001
Rotbaum
0.0603 ± 0.001
0.2098 ± 0.0
0.0342 ± 0.0003
0.1289 ± 0.0004
0.3983 ± 0.0022
mean
0.0616 ± 0.0034
0.1556 ± 0.0006
0.0431 ± 0.0002
0.1458 ± 0.004
0.3866 ± 0.0029
median
0.0549 ± 0.0012
0.1323 ± 0.0005
0.0351 ± 0.0001
0.123 ± 0.0038
0.3489 ± 0.0027
GB (D1,DBW1)
0.0531 ± 0.0004
0.16 ± 0.0012
0.0327 ± 0.0001
0.0987 ± 0.0014
0.3412 ± 0.0061
Best (D1,DBW1)
0.056 ± 0.0024
0.1201 ± 0.0
0.0332 ± 0.0005
0.0987 ± 0.0013
0.3339 ± 0.0148
Unregularized
0.0475 ± 0.0005
0.1979 ± 0.0009
0.028 ± 0.0001
0.0986 ± 0.0017
0.3366 ± 0.0008
Ours
0.0494 ± 0.0006
0.1663 ± 0.0003
0.0266 ± 0.0003
0.094 ± 0.001
0.3187 ± 0.002
Table 2. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world
open-source time series datasets, where Gaussian noise is added to the base learner predictions across time over 5 runs.
Base Learner /
Ensemble Strategy
Elec
Kaggle
M4-daily
Traf
Wiki
ARIMA
1.9146 ± 0.1236
1.102 ± 0.0038
0.0367 ± 0.0001
2.9926 ± 0.0089
0.7672 ± 0.0018
DeepAR
1.3185 ± 0.0826
3.0007 ± 0.0359
0.0355 ± 0.0008
2.1192 ± 0.0161
0.464 ± 0.0058
ETS
0.5137 ± 0.0758
0.2298 ± 0.0042
0.0332 ± 0.0004
1.2715 ± 0.0071
N/A
MQ-CNN
2.375 ± 0.1399
4.5208 ± 0.035
0.0481 ± 0.0062
3.4619 ± 0.4748
0.5667 ± 0.0096
NPTS
1.2909 ± 0.0622
2.5102 ± 0.0106
0.1235 ± 0.0002
1.7341 ± 0.0117
0.5871 ± 0.001
Rotbaum
0.3656 ± 0.0211
4.5134 ± 0.0157
0.0343 ± 0.0003
0.5403 ± 0.0032
0.438 ± 0.0015
mean
0.5672 ± 0.0424
1.2186 ± 0.0046
0.0436 ± 0.0003
0.8564 ± 0.0438
0.4283 ± 0.0027
median
0.521 ± 0.0406
0.8816 ± 0.0057
0.0356 ± 0.0001
0.8325 ± 0.0159
0.374 ± 0.0023
GB (D1,DBW1)
0.3044 ± 0.0346
0.2298 ± 0.0042
0.0329 ± 0.0001
0.5403 ± 0.0032
0.4054 ± 0.0051
Best (D1,DBW1)
0.3656 ± 0.0211
0.2298 ± 0.0042
0.0332 ± 0.0004
0.5403 ± 0.0032
0.438 ± 0.0015
Unregularized
0.0589 ± 0.0009
0.2385 ± 0.0002
0.0274 ± 0.0001
0.134 ± 0.0007
0.343 ± 0.0007
Ours
0.0587 ± 0.0007
0.1965 ± 0.0007
0.0268 ± 0.0003
0.1334 ± 0.0007
0.3294 ± 0.0016
sion problem, where the choice of regularizers parametrizes
the family. Our experiments demonstrate that our intuition
that designing a family of stacked generalizations in the
time series case, where each member of the family gives
different flexibility for the weights to vary across the various
timestamps/quantiles/items is effective.
8

[CAPTION] Table 1. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world

[CAPTION] Table 2. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world


<!-- page 9 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
Table 3. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world
open-source time series datasets, where Gaussian noise is added to the base learner predictions across items over 5 runs.
Base Learner /
Ensemble Strategy
Elec
Kaggle
M4-daily
Traf
Wiki
ARIMA
0.2026 ± 0.0019
0.1223 ± 0.0001
0.0348 ± 0.0001
0.3707 ± 0.0012
0.6081 ± 0.0004
DeepAR
0.283 ± 0.011
0.2758 ± 0.0051
0.0353 ± 0.0008
0.4442 ± 0.0053
0.3564 ± 0.0085
ETS
0.3534 ± 0.031
0.1212 ± 0.0005
0.0337 ± 0.0005
0.6846 ± 0.0027
N/A
MQ-CNN
0.3297 ± 0.0237
0.2752 ± 0.0009
0.0425 ± 0.0033
0.4601 ± 0.0603
0.3796 ± 0.004
NPTS
0.314 ± 0.0151
0.1636 ± 0.0001
0.1231 ± 0.0002
0.4158 ± 0.0023
0.4981 ± 0.0006
Rotbaum
0.437 ± 0.0269
0.2381 ± 0.0004
0.0355 ± 0.0003
0.4265 ± 0.0016
0.4649 ± 0.0039
mean
0.1338 ± 0.0043
0.1593 ± 0.0005
0.0434 ± 0.0002
0.2218 ± 0.0034
0.3956 ± 0.0033
median
0.1526 ± 0.0065
0.1354 ± 0.0001
0.0357 ± 0.0002
0.2179 ± 0.0066
0.3539 ± 0.0031
GB (D1,DBW1)
0.1333 ± 0.0039
0.1618 ± 0.001
0.0331 ± 0.0002
0.2099 ± 0.0053
0.3519 ± 0.0058
Best (D1,DBW1)
0.2026 ± 0.0019
0.1223 ± 0.0001
0.0337 ± 0.0005
0.3707 ± 0.0012
0.3732 ± 0.0107
Unregularized
0.0521 ± 0.002
0.2209 ± 0.0006
0.0276 ± 0.0001
0.1125 ± 0.0003
0.3384 ± 0.0007
Ours
0.0505 ± 0.0017
0.156 ± 0.0008
0.0268 ± 0.0003
0.1094 ± 0.0004
0.3235 ± 0.0016
Table 4. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world
open-source time series datasets, where Gaussian noise is added to the base learner predictions across quantiles over 5 runs.
Base Learner /
Ensemble Strategy
Elec
Kaggle
M4-daily
Traf
Wiki
ARIMA
0.2347 ± 0.0067
0.1269 ± 0.0001
0.0344 ± 0.0
0.3877 ± 0.0008
0.606 ± 0.0005
DeepAR
0.789 ± 0.0573
0.3571 ± 0.0069
0.0369 ± 0.0007
1.2155 ± 0.013
0.4525 ± 0.004
ETS
0.4443 ± 0.0687
0.1201 ± 0.0003
0.0359 ± 0.0004
0.9363 ± 0.0111
N/A
MQ-CNN
0.9111 ± 0.0167
0.2904 ± 0.0013
0.0531 ± 0.0096
1.2778 ± 0.2005
0.4692 ± 0.0076
NPTS
0.6306 ± 0.024
0.2942 ± 0.0011
0.1269 ± 0.0002
0.9207 ± 0.0064
0.5888 ± 0.0017
Rotbaum
0.5671 ± 0.0525
0.2497 ± 0.0004
0.0374 ± 0.0005
0.71 ± 0.0024
0.5356 ± 0.0072
mean
0.2511 ± 0.0173
0.1743 ± 0.0007
0.0441 ± 0.0004
0.3715 ± 0.0156
0.4107 ± 0.0032
median
0.2444 ± 0.0222
0.1394 ± 0.0003
0.036 ± 0.0002
0.3749 ± 0.0106
0.3664 ± 0.0027
GB (D1,DBW1)
0.2268 ± 0.0103
0.1455 ± 0.0014
0.0335 ± 0.0002
0.3634 ± 0.0036
0.4057 ± 0.0046
Best (D1,DBW1)
0.2347 ± 0.0067
0.1256 ± 0.0022
0.0344 ± 0.0
0.3877 ± 0.0008
0.4692 ± 0.0076
Unregularized
0.0555 ± 0.0006
0.235 ± 0.0003
0.0268 ± 0.0001
0.1233 ± 0.0005
0.3431 ± 0.0005
Ours
0.0541 ± 0.0015
0.1637 ± 0.0028
0.0269 ± 0.0002
0.1228 ± 0.0004
0.3285 ± 0.001
9

[CAPTION] Table 3. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world

[CAPTION] Table 4. Mean weighted quantile losses over quantiles τ = [0.1, 0.5, 0.9] for various base learners and ensembling strategies on real-world


<!-- page 10 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
References
Alexandrov, A., Benidis, K., Bohlke-Schneider, M.,
Flunkert, V., Gasthaus, J., Januschowski, T., Maddix,
D. C., Rangapuram, S., Salinas, D., Schulz, J., Stella, L.,
T¨urkmen, A. C., and Wang, Y. GluonTS: Probabilistic
and Neural Time Series Modeling in Python. Journal of
Machine Learning Research, 21(116):1–6, 2020.
Bates, J. M. and Granger, C. W. The combination of fore-
casts. Journal of the Operational Research Society, 20
(4):451–468, 1969.
Breiman, L. Random forests. Machine learning, 45(1):
5–32, 2001.
Caruna, R., Niculescu-Mizil, A., Crew, G., and Ksikes, A.
Ensemble selection from libraries of models. Proceed-
ings of the 21st International Conference on Machine
Learning, 2004.
Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting
system. In Proceedings of the 22nd acm sigkdd inter-
national conference on knowledge discovery and data
mining, pp. 785–794, 2016.
Claeskens, G., Magnus, J. R., Vasnev, A. L., and Wang, W.
The forecast combination puzzle: A simple theoretical
explanation. International Journal of Forecasting, 32(3):
754–762, 2016.
Cruz, R. M., Sabourin, R., and Cavalcanti, G. D. Dynamic
classifier selection: Recent advances and perspectives.
Information Fusion, 41:195–216, 2018.
Dheeru, D. and Karra Taniskidou, E. UCI machine learning
repository, 2017.
Donaldson, R. G. and Kamstra, M. Forecast combining with
neural networks. Journal of Forecasting, 15(1):49–61,
1996.
Elliott, G. Averaging and the optimal combination of fore-
casts. University of California, San Diego, 2011.
Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy,
P., Li, M., and Smola, A. Autogluon-tabular: Robust
and accurate automl for structured data. arXiv preprint
arXiv:2003.06505, 2020.
Flunkert, V., Salinas, D., Gasthaus, J., and Januschowski,
T. Deepar: Probabilistic forecasting with autoregressive
recurrent networks. International Journal of Forecasting,
arXiv:1704.04110, 2019.
Fort, S., Hu, H., and Lakshminarayanan, B.
Deep en-
sembles: A loss landscape perspective. arXiv preprint
arXiv:1912.02757, 2019.
Gasthaus, J. Non-parametric time series forecaster. Techni-
cal Report, Amazon, 2016.
Gastinger, J., Nicolas, S., Stepi´c, D., Schmidt, M., and
Sch¨ulke, A. A study on ensemble learning for time series
forecasting and the need for meta-learning. In 2021 Inter-
national Joint Conference on Neural Networks (IJCNN),
pp. 1–8. IEEE, 2021.
Hasson, H., Wang, B., Januschowski, T., and Gasthaus, J.
Probabilistic forecasting: A level-set approach. Advances
in neural information processing systems, 34, 2021.
Hyndman, R., Koehler, A. B., Ord, J. K., and Snyder, R. D.
Forecasting with exponential smoothing: the state space
approach. Springer Science & Business Media, 2008.
Kaggle. Kaggle survey 2020. https://www.kaggle.
com/kaggle-survey-2020, 2020.
Kaggle. Kaggle survey 2021. https://www.kaggle.
com/kaggle-survey-2021, 2021.
Kim, T., Fakoor, R., Mueller, J., Tibshirani, R. J., and
Smola, A. J. Deep quantile aggregation. arXiv preprint
arXiv:2103.00083, 2021.
Lai. Dataset of Kaggle Competition Web Traffic Time Series
Forecasting, Version 3, August 2017.
Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple
and scalable predictive uncertainty estimation using deep
ensembles. In Guyon, I., Luxburg, U. V., Bengio, S.,
Wallach, H., Fergus, R., Vishwanathan, S., and Garnett,
R. (eds.), Advances in Neural Information Processing
Systems, volume 30. Curran Associates, Inc., 2017.
Makridakis, S. et al. The M4 competition: Results, findings,
conclusion and way forward. International Journal of
Forecasting, 34(4):802 – 808, 2018.
Massaoudi, M., Refaat, S. S., Chihi, I., Trabelsi, M., Oues-
lati, F. S., and Abu-Rub, H. A novel stacked general-
ization ensemble-based hybrid lgbm-xgb-mlp model for
short-term load forecasting. Energy, 214:118874, 2021.
Moon, J., Jung, S., Rew, J., Rho, S., and Hwang, E. Combi-
nation of short-term load forecasting models based on a
stacking ensemble approach. Energy and Buildings, 216:
109921, 2020.
Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E.,
DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., and Lerer,
A. Automatic differentiation in pytorch. NIPS 2017
Autodiff Workshop, 2017.
Powell, M. J. A direct search optimization method that
models the objective and constraint functions by linear
interpolation. In Advances in optimization and numerical
analysis, pp. 51–67. Springer, 1994.
10


<!-- page 11 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
Smith, J. and Wallis, K. F. A simple explanation of the fore-
cast combination puzzle. Oxford Bulletin of Economics
and Statistics, 71(3):331–355, 2009.
Stock, J. H. and Watson, M. W. Combination forecasts of
output growth in a seven-country data set. Journal of
forecasting, 23(6):405–430, 2004.
Ting, K. M. and Witten, I. H. Stacked generalization: when
does it work? Research Commons, 1997a.
Ting, K. M. and Witten, I. H. Stacking bagged and dagged
models. Research Commons, 1997b.
Van der Laan, M. J., Polley, E. C., and Hubbard, A. E. Super
learner. Statistical applications in genetics and molecular
biology, 6(1), 2007.
Van der Vaart, A. W., Dudoit, S., and Van der Laan, M. J. Or-
acle inequalities for multi-fold cross validation. Statistics
& Decisions, 24(3):351–371, 2006.
Wellner, J. et al. Weak convergence and empirical pro-
cesses: with applications to statistics. Springer Science
& Business Media, 2013.
Wen, R., Torkkola, K., and Narayanaswamy, B. A multi-
horizon quantile recurrent forecaster. NIPS Workshop on
Time Series, arXiv:1711.11053, 2017.
Wolpert, D. H. Stacked generalization. Neural networks, 5
(2):241–259, 1992.
Zhou, Z.-H. Ensemble methods: foundations and algo-
rithms. CRC press, 2012.
11


<!-- page 12 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
A. Omitted Proofs
In this section we provide the proofs that were omitted from Section 5.
Using the notation of Section 3, let Gi ∶= √ni(Pi −P) be the “empirical process” of Pi for i = 1,2. The following lemma
is a straightforward generalization of Lemma 2.1 in (Van der Vaart et al., 2006) to the case of infinitely many θαs and
order-dependent algorithms. As the proof is the same mutatis mutandis as in (Van der Vaart et al., 2006), we omit it.
Lemma A.1. For every ˆα ∈J , let
Wˆα ∶= {˜α ∈J ∣∫Lˆα(D0)(X)dP1
≤∫L˜α(D0)(X)dP1},
be the set of indices ˜α for which θˆα outperforms θ˜α on validation. Then for any δ ≥−1:
E(∫Lˆα(D0)(X)dP) ≤
(1 + 2δ) inf
˜α∈W ˆ
α (E(∫L˜α(D0)(X)dP))
+
1
√n1
E(sup
α ∫Lα(D0)(X)d((1 + δ)G1 −δ√n1P))
+
1
√n1
E(sup
α −∫Lα(D0)(X)d((1 + δ)G1 + δ√n1P)).
(4)
Our goal, therefore, is to bound the last two terms. In the case that ∣J ∣is finite this was done in Lemma 2.2 of (Van der
Vaart et al., 2006).
Lemma A.2. (Lemma 2.2 of (Van der Vaart et al., 2006)) Assume that J is finite, that F has Bernstein numbers
(M(F),v(F)), and that the θα’s are invariant of the training dataset’s order. Then for every δ > 0 and 1 ≤p ≤2:
E(max
f∈F ±∫fd(G1 −δ√n1P)) ≤
8
n
1
p −1
2
1
log(1 + ∣J ∣)max
f∈F
⎛
⎜
⎝
M(F)
n
1−1
p
1
+ (
v(F)
(δ ∫fdP)2−p )
1
p ⎞
⎟
⎠
.
(5)
While the lemma assumes the θα’s are invariant of the training dataset order, this assumption is in fact not used in its proof.
Thus it suffices to reduce the case that J is finite dimensional to the lemma above. This is done via a discretization of J
itself. (Note that this is not the same as discretizing the set of functions that the stacked generalizations may become.)
Theorem A.3. Under the assumptions of Theorem 5.1, for every δ > 0, 1 ≤p ≤2, and sequence εn1 > 0:
E(sup
f∈F
±∫fd(G1 −δ√n1P)) ≤
sup
f∈F
⎛
⎜⎜⎜⎜⎜
⎝
8( M(F)
n
1−1
p
1
+ (
v(F)
(δ ∫fdP )2−p )
1
p )log(1 + N int(J ,
εn1
ℓ))
n
1
p −1
2
1
⎞
⎟⎟⎟⎟⎟
⎠
+ ((1 + δ)√n1 +
1
√n1
)εn1.
(6)
Proof. (of Theorem A.3)
12


<!-- page 13 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
First note that if ∥f −f ′∥∞< εn1 then
∣G1f −δ√n1Pf −(G1f ′ −δ√n1Pf ′)∣≤∣G1f −G1f ′∣+ δ√n1∣P(f −f ′)∣
≤n
−1
2
1
n1
∑
l=1
∣f(X1
l ) −f ′(X1
l )∣+ (n
−1
2
1
+ δ√n1)∣P(f −f ′)∣
≤√n1εn1 + (n
−1
2
1
+ δ√n1)εn1
= ((1 + δ)√n1 +
1
√n1
)εn1,
and therefore, after taking suprema and expected values, we get that for every subset F′ of F for which every element in F
is εn1-close to an element in F′
E(sup
f∈F
±∫fd(G1 −δ√n1P)) ≤E( sup
f ′∈F′ ±∫f ′d(G1 −δ√n1P)) + ((1 + δ)√n1 +
1
√n1
)εn1.
We now proceed to reduce Theorem A.3 to Lemma A.2 above, which applies to the case of finite J . By our assumptions,
we have that N int(F,ℓε) ≤N int(J ,ε), or in other words N int(F,ε) ≤N int(J , ε
ℓ).
Combining Lemma A.2 applied to the centers F′ of a minimal εn1-covering of F with the inequality above, the result
follows.
Proof. (of Theorem 5.1)
The main portion of the theorem follows directly from Theorem A.3 and Lemma A.1 by bounding each of the last two terms
in Lemma A.1 using Theorem A.3 and then multiplying by the prefactor 1/√n1 (and summing the two contributions).
The second portion of the theorem follows from the inequality N int(J , ε
ℓ) ≤N ext(J , ε
2ℓ) (with N ext representing the
external covering number), as well as the fact that if J is contained in a K-ball around the origin in an ambient Euclidean
space of dimension M, then we have the following inequality for the external covering number:
N ext(J , εn1
2ℓ) ≤(4ℓK
√
M
εn1
)
M
.
Therefore if one chooses εn1 ∶= n
−1
2 −ϵ
1
for some ϵ > 0, then log(1+N int(J ,
εn1
2ℓ)) is order of magnitude of log(n1). Further,
if one wants to include p additional algorithms as stacked generalizations then that can be done by adding them into both F
and F′ in the proof of Theorem A.3, and therefore results in the log term described.
13


<!-- page 14 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
B. Tabular Analogue of the Family in Section 6
In this appendix we point out that the construction in Section 6 can be employed in the tabular situation as well, for any
case where we have a decomposition of the prediction space Rr′ ≅Rr′
1 ⊗⋯⊗Rr′
p of the base learners η1,...,ηm into a
tensorial factors. For α ∶= (α1,...,αp) we construct a stacked generalization Aα(D0) as (x,η1(D0)(x),...,ηm(D0)(x)) ↦
∑∀j 1≤dj≤r′
i (∑l w(di)p
i=1,lηl(D0)(x))ed1 ⊗⋯⊗edp. The weights are chosen to minimize
fα(D0) ∶= ∑
X∈D0
Lα(X) +
p
∑
i=1
αiH(σ(i)(w))
(7)
where H(i) is the entropy term similar to Section 6. It is also possible to add other regularization terms such as l1
regularization as we did in Section 6.
14


<!-- page 15 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
C. Experimental Detail
C.1. Probabilistic Time Series Forecasting Problem Definition
We first define the time-series data as D ∶= {zi,j′}i
j′, zi,j′ ∈R, such that i = 1,2,...,N denotes the ith time-series
(henceforth, item) with a total of N items, and for each item-i, j′ = 1,2,...,Ti denotes the time-points with the historical
length Ti. For simplicity, we assume ∀i Ti = T. The goal of a probabilistic time-series forecasting algorithm is to
output a probabilistic prediction for the future h time-steps for each time-series, where h is pre-determined, for quantiles
0 ≤τ1,...,τq ≤1. For the notational purpose and to avoid repetition, we reserve i = 1,2,...,N, and j = 1,2,...,h , and
k = 1,2,...,q through rest of the paper, and henceforth drop the index range for i,j,k. The output consists of estimates
{ˆzi,j,k}i
j,k of the τk-quantiles of zi,T +j conditioned on historical data.
Given computed predictions ˆZ(D) ∶= {ˆzi,j,k}i
j,k and the true values DBW ∶= {zi,T +j}i
j, we aim to minimize the mean
weighted quantile loss:
L( ˆZ(D),DBW) ∶= 2
q
∑i,j,k max{τk(zi,T +j −ˆzi,j,k),(1 −τk)(ˆzi,j,k −zi,T +j)}
∑i,j ∣zi,T +j∣
.
(8)
C.2. Family of Stacked Generalizations for Probabilistic Time Series Forecastings
Let η1,...,ηm be m arbitrary probabilistic time-series forecasting algorithms. In order to simplify the notation, if D
is time-series data as defined in Section C.1, we let ηl(D) be ηl trained on D and then making inferences on D; i.e.,
ηl(D) = {ˆz(l)
i,j,k(D)}i
j,k, which consists of the predictions for each item in D for the h many unseen timestamps into the
forecast horizon after the last seen value for q quantiles; i.e., ηl(D) = {ˆz(l)
i,j,k(D)}i
j,k.
Since cross-validation is more subtle in the time series use case, we introduce some additional notation. In particular, we
define training and backtest window datasets pairs, where a backtest window consists of the true values on a subsequent
window of length h after the last training time point as:
(Dn,DBWn) ∶= ({zi,j′}i
j′, {zi,T −µh+j}i
j),
(9)
for j′ = 1,...,T −µh, µ = 2 −n, n = 0,1,2. Here we use two pairs of training and validation sets, that is (D0,DBW0) and
(D1,DBW1). Note that D and DBW in Section C.1 are equivalent to to D2 and DBW2, respectively.
For every choice of ensemble weights w = {w(l)
i,j,k}i,l
j,k, l = 1,2,...,m we denote:
ˆZ(D,w) ∶= {∑
l
w(l)
i,j,k(D)ˆz(l)
i,j,k(D)}
i
j,k
,
(10)
as the weighted ensemble combination. The weights are restricted to sum up to 1; namely, for every i,j,k we have that
∑l w(l)
i,j,k(D) = 1. We let
fα[(Dn,DBWn),w] = L( ˆZ(Dn,w),DBWn) +
3
∑
d=1
αdH(σ(d)(w)) + α4 ∑
i,j,k,l
∣w(l)
i,j,k∣,
(11)
be the regularized mean weighted quantile loss with L given by Eqn. (8) and ensemble estimate ˆZ(Dn,w) by Eqn. (10).
We use an entropy of softmax regularizer:
H(σ(d)(w)) ∶= ∑
i,j,k,l
σ(d)(w(l)
i,j,k)log(σ(d)(w(l)
i,j,k)),
where σ(d)(w(l)
i,j,k) = exp(w(l)
i,j,k)/∑I(d) exp(w(l)
i,j,k) denotes the softmax with the denominator summed over the double
summations specified by d, that is, I(1),I(2),I(3) are (j,k),(i,k),(i,j) respectively for d = 1,2,3. The role of the
entropy regularizing parameters {αd}d for d = 1,2,3 in this family is to control how much the ensemble weights are
allowed to vary across items, timestamps in the forecast horizon, quantiles, respectively. Lastly, α4 denotes the l1 regularizer
parameter, which specifies how much to be biased towards preferring a single algorithm versus diversifying.
15


<!-- page 16 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
We follow Algorithm 1 to compute the optimal ensembling weights w∗, and output the optimal ˜α ensembling member from
our family of stacked generalizations:
A˜α(D2) ∶= ˆZ(D2,w∗),
(12)
where ˆZ is the weighted linear combination of base learners in Eqn. (10).
16


<!-- page 17 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
D. Towards a Uniform Treatment of Both Tabular and Time Series Data
A prominent difference in the way that we have treated tabular and time series data is in the meaning of “cross validation”.
In Section 3, the validation set consists of a different set of datapoints, whereas in Section 6 the validation set is in the same
time series but with a different forecast horizon. In this appendix we will attempt to unify the notation for both cases.
The way to bridge this gap is to replace the dimension d of feature space, from Section 3, with the first infinite ordinal.
In more colloquial terms, feature space will be the disjoint union ⊍d∈N Rd. We keep the dimension of prediction space
fixed, since in our time series forecasting setup infernces are always fixed size (length of forecast horizon times number of
items times number of quantiles). In this way we may phrase cross validation for time series in the terminology of Section
3, by allowing each dataset to consist of a single datapoint. To wit, let Dn,DBWn for n = 0,1 be as in Section 6, and let
˜Dn ∶= {(Dn,DBWn)} for n = 0,1. Then cross validation in the sense of Section 6 is simply cross-validation in the sense of
Section 3 for ˜D0 and ˜D1.
In order to apply any of the theory of the tabular case, we must endow (⊍d∈N Rd) × Rr′, where r′ is the dimension of
prediction space, with a topology and a probability measure. While this space has a natural topology (which is even
metrizable) by letting each Rd be a separate connected component in ⊍d∈N Rd, this topology does not jibe with intuition.
For example, two time series that are identical except that one has one additional timestamp compared to the other would be
very far away. Perhaps a more natural choice would be a topology induced by dynamic time warping. Once a topology is
fixed, one has to choose a (Borel) probability measure on (⊍d∈N Rd) × Rr′.
In whatever way that these choices are done, it is unrealistic to assume that the datapoint in ˜D0 is independent of the one
in ˜D1. This implies that cross validation in the time series case cannot truly be brought into the fold of the tabular theory.
Nevertheless, the tabular theory is directional.
17


<!-- page 18 -->
Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting
E. Dataset Details
Table 5 lists the details of the datasets that we use in our experiments.
Table 5. Summary of dataset statistics, where Elec and Traf are from the UCI data repository ((Dheeru & Karra Taniskidou, 2017)),
Kaggle and Wiki from Kaggle ((Lai, 2017)), and 6 different M4 competition datasets ((Makridakis et al., 2018)).
DOMAIN
NAME
SUPPORT
FREQ
NO. TS
AVG. LEN
PRED. LEN
electrical load
Elec
R+
H
321
21044
24
Rossman
Kaggle
N
D
500
1736
90
M4 forecasting
competition
M4-daily
R+
D
4227
2357
14
road traffic
Traf
[0,1]
H
862
14036
24
visit counts of
wikipedia pages
Wiki
N
D
9535
762
7
18


**[Table p18.1]**
| DOMAIN | NAME | SUPPORT | FREQ | NO. TS | AVG. LEN | PRED. LEN |
| --- | --- | --- | --- | --- | --- | --- |
| electrical load | Elec | R+ | H | 321 | 21044 | 24 |
| Rossman | Kaggle | N | D | 500 | 1736 | 90 |
| M4 forecasting competition | M4-daily | R+ | D | 4227 | 2357 | 14 |
| road traffic | Traf | [0,1] | H | 862 | 14036 | 24 |
| visit counts of wikipedia pages | Wiki | N | D | 9535 | 762 | 7 |

[CAPTION] Table 5 lists the details of the datasets that we use in our experiments.

[CAPTION] Table 5. Summary of dataset statistics, where Elec and Traf are from the UCI data repository ((Dheeru & Karra Taniskidou, 2017)),