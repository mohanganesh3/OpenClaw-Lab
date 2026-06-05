[2407.05470] Bayesian Finite Mixture Models

Bayesian Finite Mixture Models

Bettina Grün
 WU Vienna University
 of Economics and Business
&amp;Gertraud Malsiner-Walli
 WU Vienna University
 of Economics and Business

bettina.gruen@wu.ac.at

gertraud.malsiner-walli@wu.ac.at

doi:  10.1002/9781118445112.stat08373

\Plainauthor

Bettina Grün, Gertraud Malsiner-Walli
 \Address 
Bettina Grün
 Institute for Statistics and Mathematics
 WU Vienna University of Economics and Business
 Welthandelsplatz 1, 1020 Wien, Austria
 E-mail:

Gertraud Malsiner-Walli

Institute for Statistics and Mathematics
 WU Vienna University of Economics and Business
 Welthandelsplatz 1, 1020 Wien, Austria
 E-mail:

\Abstract 
Finite mixture models are a useful statistical model class
for clustering and density approximation. In the Bayesian framework
finite mixture models require the specification of suitable priors
in addition to the data model. These priors allow to avoid spurious
results and provide a principled way to define cluster shapes and a
preference for specific cluster solutions. A generic model
estimation scheme for finite mixtures with a fixed number of
components is available using Markov chain Monte Carlo (MCMC)
sampling with data augmentation. The posterior allows to assess
uncertainty in a comprehensive way, but component-specific posterior
inference requires resolving the label switching issue.
In this paper we focus on the application of Bayesian finite mixture
models for clustering. We start with discussing suitable
specification, estimation and inference of the model if the number
of components is assumed to be known. We then continue to explain
suitable strategies for fitting Bayesian finite mixture models when
the number of components is not known. In addition, all steps
required to perform Bayesian finite mixture modeling are illustrated
on a data example where a finite mixture model of multivariate
Gaussian distributions is fitted. Suitable prior specification,
estimation using MCMC and posterior inference are discussed for this
example assuming the number of components to be known as well as
unknown.

\Keywords cluster analysis, finite mixture, label switching, Markov
chain Monte Carlo, mixture of finite mixtures model, number of
components, prior specification, telescoping sampler
 \Disclaimer This is a preprint of an article published in
 Wiley StatsRef: Statistics Reference Online . Please cite as:
Bettina Grün and Gertraud Malsiner-Walli (2022) Bayesian finite
mixture models. In N. Balakrishnan, Theodore Colton, Brian Everitt,
Walter Piegorsch, Fabrizio Ruggeri, and Jef L. Teugels, editors,
 Wiley StatsRef: Statistics Reference Online 
.

1  Introduction

Finite mixture models have a long history in statistics with their
first use proposed by  Newcomb ( 1886 )  and  Pearson ( 1894 ) 
more than hundred years ago. They represent a useful model class for
cluster analysis as well as semi-parametric density estimation. In
addition the model class enjoys great flexibility and extensibility
because arbitrary statistical models can be incorporated in a mixture
which corresponds to fitting a convex combination of these models to
the data.

McLachlan and Peel ( 2000 )  and  Frühwirth-Schnatter ( 2006 ) 
provide a thorough and detailed introduction into finite mixture
models discussing different extensions and their application.
 McLachlan and Peel ( 2000 )  focus on maximum likelihood estimation,
while  Frühwirth-Schnatter ( 2006 )  discusses the Bayesian
estimation in detail. A more recent overview and discussion of new
developments in mixture analysis are compiled in
 Frühwirth-Schnatter  et al.  ( 2019 ) . This contribution
focuses on Bayesian estimation of finite mixture models in a
model-based clustering context.

The application of finite mixture models in standard data analysis has
been impeded due to the difficulties faced when estimating and
performing inference. Their uptake has been abetted by the increase in
available computing power. Routine estimation of different kinds of
mixture models has become feasible using the expectation-maximization
algorithm  (Dempster  et al. ,  1977 )  for maximum likelihood
estimation and Markov chain Monte Carlo methods in combination with
data augmentation  (Diebolt and Robert,  1994 )  for Bayesian
estimation. Both these estimation methods represent a general
estimation framework equally allowing for general extensibility and
inclusion of arbitrary statistical models in the mixture.

Using the Bayesian framework to specify and estimate a finite mixture
model has several advantages:

•

Domain knowledge can be included in a principled way using
suitable prior specifications. When using mixture models for
cluster analysis, the priors allow to specify the prototypical shape
of the clusters  (Hennig,  2015 ) . As an explanatory analysis
tool, cluster analysis aims at detecting interesting structures in
the data. The priors may guide the estimation procedure to focus on
mixture models with certain characteristics. E.g., the priors
specified can reflect and support aims such as obtaining a cluster
solution with equally sized clusters or clusters which differ
primarily in their centroids.

•

The mixture likelihood is highly irregular due to
identifiability issues and multimodality as well as the presence of
spurious modes. The specification of priors in Bayesian estimation
allows to obtain a regularized and smoothed version of the mixture
likelihood as posterior. In this way problems in maximum likelihood
estimation can be avoided where the application of optimization
methods suffers from this irregularity.

•

Parameter uncertainty can be easily assessed using the whole
posterior distribution. No reliance on asymptotic normality is
required, allowing for valid inference in cases where regularity
conditions are violated, e.g., for small data sets and mixtures with
small component weights.

Despite these advantages, Bayesian finite mixture models are not
routinely applied yet because of uncertainty regarding the
specification of suitable priors and lack of understanding of their
impact  (Aitkin,  2001 ) . Also issues with estimation as well as
inference are reported due to computational challenges
 (Celeux  et al. ,  2019b ) . In the following all steps
of a Bayesian finite mixture modeling application are explained in
detail, including data model and prior specification, estimation and
inference. This is done for both cases: if the number of components is
assumed to be known as well as if this number is unknown. In addition
to outlining the methodology and discussing theoretical
considerations, a real data set serves as an example to illustrate the
application of each of the steps and as a basis to facilitate the
uptake of Bayesian estimation of finite mixture models in applied
research.

2  Known Number of Components

2.1  Model Specification

2.1.1  Data Model

When using finite mixture models for clustering the usual assumption
is that a uni- or multivariate continuous or discrete outcome variable

𝒚

𝒚

\bm{y}

is sampled from a heterogeneous population where

K

𝐾

K

groups
are present. Each group follows its own group-specific parametric
distribution. Each observation in the sample belongs to one of the

K

𝐾

K

groups, but the group memberships are not observed. It is further
assumed that the group sizes in the sample correspond to the
prevalence of the groups in the population.

Marginally, the finite mixture density of an observation

𝒚  i

subscript  𝒚  𝑖

\bm{y}_{i}

is
given by a convex combination of the group-specific parametric
densities with the weights corresponding to the group sizes:

h  ​

(

𝒚  i

|

ϑ  K

)

ℎ

conditional

subscript  𝒚  𝑖

subscript  bold-italic-ϑ  𝐾

\displaystyle h(\bm{y}_{i}|\bm{\vartheta}_{K})

=

∑

k  =  1

K

η  k

​  f  ​

(

𝒚  i

|

𝜽  k

)

,

absent

superscript

subscript

𝑘  1

𝐾

subscript  𝜂  𝑘

𝑓

conditional

subscript  𝒚  𝑖

subscript