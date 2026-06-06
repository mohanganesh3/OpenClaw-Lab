[2202.09667] Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning

Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning

Nathan Kallus

Xiaojie Mao

Kaiwen Wang

Zhengyuan Zhou

Abstract

Off-policy evaluation and learning (OPE/L) use offline observational data to make better decisions, which is crucial in applications where online experimentation is limited.
However, depending entirely on logged data, OPE/L is sensitive to environment distribution shifts — discrepancies between the data-generating environment and that where policies are deployed.  Si et al. ( 2020a )  proposed distributionally robust OPE/L (DROPE/L) to address this, but the proposal relies on inverse-propensity weighting, whose estimation error and regret will deteriorate if propensities are nonparametrically estimated and whose variance is suboptimal even if not.
For standard, non-robust, OPE/L, this is solved by doubly robust (DR) methods, but they do not naturally extend to the more complex DROPE/L, which involves a worst-case expectation.
In this paper, we propose the first DR algorithms for DROPE/L with KL-divergence uncertainty sets.
For evaluation, we propose  L ocalized  D oubly  R obust  DROPE  (LDR 2 OPE) and show that it achieves semiparametric efficiency under weak product rates conditions.
Thanks to a localization technique, LDR 2 OPE only requires fitting a small number of regressions, just like DR methods for standard OPE.
For learning, we propose  C ontinuum  D oubly  R obust  DROPL  (CDR 2 OPL) and show that, under a product rate condition involving a continuum of regressions, it enjoys a fast regret rate of

𝒪  ​

(

N

−

1  /  2

)

𝒪

superscript  𝑁

1  2

\mathcal{O}(N^{-1/2})

even when unknown propensities are nonparametrically estimated.
We empirically validate our algorithms in simulations and further extend our results to general

f

𝑓

f

-divergence uncertainty sets.

Distributional Robustness, Double Robustness, Off-Policy Evaluation, Off-Policy Learning

1  Introduction

The vast majority of online recommendations in search engines, e-commerce, social media, streaming platforms, etc. are made by algorithms that learn from historical user interactions  (Li et al.,  2010 ; Bottou et al.,  2013 ; Ren &amp; Zhou,  2020 ; Liu et al.,  2021 ) .
Even in high-stakes domains, such as healthcare  (Murphy,  2003 )  and education  (Mandel et al.,  2014 ) , the promise of cheaper and higher quality decisions, made possible by the growing abundance of user-specific data, incentivize the inclusion of automatic decision-making components into existing approaches.

This task of making good decisions from observational data is formalized by the problems of off-policy evaluation (OPE)  (Foster &amp; Syrgkanis,  2019 ; Kallus &amp; Uehara,  2020a ; Chernozhukov et al.,  2018 ; Farajtabar et al.,  2018 ; Joachims &amp; Swaminathan,  2016 ; Bottou et al.,  2013 ; Dudík et al.,  2011 )  and off-policy learning (OPL)  (Manski,  2004 ; Kitagawa &amp; Tetenov,  2018 ; Athey &amp; Wager,  2021 ; Zhan et al.,  2021 ; Zhou et al.,  2022 ; Kallus &amp; Uehara,  2020b ; Swaminathan &amp; Joachims,  2015a ; Dudík et al.,  2011 ) .
OPE is concerned with estimating the expected returns of a target policy given logged data, collected under a different behavior policy. OPL is concerned with learning a policy that maximizes the expected returns given this data.
OPE/L assumes that the the environment in which these policies are deployed is identical to the environment that generated the training data.
In practice, this often is not the case.
For example, in recommendation systems, user interests naturally shift with seasonality and world events, which correspond to changes in the state and reward distributions. Moreover, the environment could also be adversarially perturbed by attackers or data corruption.

Distributional robustness is a way to guard against such unknown discrepancies between training and deployment environments. Instead of estimating/maximizing the expected policy return under the training environment, we may consider estimating/maximizing the worst-case return over all environments within an uncertainty set around the unknown training environment.
 Si et al. ( 2020a ,  b )  tackle this distributionally robust OPE/L (DROPE/L) problem using methods based on self-normalized inverse propensity scoring (SNIPS)  (Swaminathan &amp; Joachims,  2015b ) . The uncertainty sets of  (Si et al.,  2020a )  and this paper are with respect to the KL-divergence, and generally

f

𝑓

f

-divergences.

However,  (Si et al.,  2020a )  assumes that we know the behavior propensities, which are usually absent in observational datasets.
One may consider simply fitting and imputing the propensities using some flexible machine learning (ML) methods, i.e. non-parametric estimators of nuisance functions.
As the propensity estimates may converge at slow rates, this leads to slow rates in estimation and learning for the proposed SNIPS-based methods.
Even with known propensities, the SNIPS-based estimator’s asymptotic variance for DROPE is in fact suboptimal.

In standard (non-distributionally robust) OPE/L, doubly robust (DR) is the canonical approach for improving estimation variance and for alleviating the sensitivity to estimation of nuisances, i.e. unknown functions such as propensities.
In addition to fitting a propensity model, DR also fits the expected reward given state and action and combines the two models to construct an estimator with better statistical properties.
A key result in OPE is that the cross-fitted DR estimator (CFDR) is

N

𝑁

\sqrt{N}

-consistent, asymptotically linear and efficient (i.e. attains the lowest possible asymptotic variance), even when nuisances are estimated at slower-than-

N

𝑁

\sqrt{N}

-rates  (Chernozhukov et al.,  2018 ) .
This, however, does not immediately extend to DROPE/L, whose objective is formed as a supremum over the log of moment generating functions. It therefore remains a question how to obtain estimation-robustness guarantees for DROPE/L.

In this paper, we propose novel doubly robust algorithms for DROPE/L, ensuring robustness to  both  environment shifts and model estimation errors.
Our contributions are summarized as follows:

1.

For DROPE, we propose the Localized DR DROPE (LDR 2 OPE) estimator and show that it is

N

𝑁

\sqrt{N}

-consistent, asymptotically linear, and enjoys semiparametric efficiency under weak product rates (  Section

3.1  ). In particular, just like DR estimators for standard OPE, LDR 2 OPE only requires fitting a few regressions, including a propensity and two transformed-outcome regressions.

2.

For DROPL, we propose Continuum DR DROPL (CDR 2 OPL) and prove a

𝒪  ​

(

N

−

1  /  2

)

𝒪

superscript  𝑁

1  2

\mathcal{O}(N^{-1/2})

regret guarantee, even when propensities are nonparametrically estimated at slow rates (  Section

4  ).

3.

We empirically show that our proposals outperform benchmarks in simulation (  Section

5  ). Code is available at

https://github.com/CausalML/doubly-robust-dropel  .

4.

We further extend our methods to general

f

𝑓

f

-divergence uncertainty sets (  Section

6  ).

1.1  Related Literature

We work in the distributionally robust setting (  Section

2.1  ) proposed by  Si et al. ( 2020a ) , which was motivated by the distributionally robust optimization (DRO) literature  ( e.g. ,  Hu &amp; Hong,  2013 ; Ben-Tal et al.,  2013 ) .
Unlike  Si et al. ( 2020a ) ,
 we do not assume that the behavior policy is known. 
To derive our doubly robust DROPE estimator, we propose a novel formulation of the DRO problem as a multidimensional moment equation and leverage the techniques of  Kallus et al. ( 2019 ) .
This allows us to tackle the complex optimization formulation of the objective and still attain semiparametric efficiency under very lax conditions.

In standard (non-distributionally robust) OPL, maximizing the CFDR objective