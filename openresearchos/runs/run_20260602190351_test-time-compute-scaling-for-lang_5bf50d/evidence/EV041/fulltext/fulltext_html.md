[2504.03897] MaxTDA: Robust Statistical Inference for Maximal Persistence in Topological Data Analysis

MaxTDA: Robust Statistical Inference for Maximal Persistence in Topological Data Analysis

Sixtus Dakurah

Department of Statistics, University of Wisconsin-Madison
 and

Jessi Cisewski-Kehe

Department of Statistics, University of Wisconsin-Madison

The authors gratefully acknowledge support from NSF grant numbers 2038556 and 2337243. Support for this research was also provided by the University of Wisconsin-Madison Office of the Vice Chancellor for Research and Graduate Education with funding from the Wisconsin Alumni Research Foundation.

Abstract

Persistent homology is an area within topological data analysis (TDA) that can uncover different dimensional holes (connected components, loops, voids, etc.) in data. The holes are characterized, in part, by how long they persist across different scales. Noisy data can result in many additional holes that are not true topological signal. Various robust TDA techniques have been proposed to reduce the number of noisy holes, however, these robust methods have a tendency to also reduce the topological signal.
This work introduces Maximal TDA (MaxTDA), a statistical framework addressing a limitation in TDA wherein robust inference techniques systematically underestimate the persistence of significant homological features. MaxTDA combines kernel density estimation with level-set thresholding via rejection sampling to generate consistent estimators for the maximal persistence features that minimizes bias while maintaining robustness to noise and outliers. We establish the consistency of the sampling procedure and the stability of the maximal persistence estimator. The framework also enables statistical inference on topological features through rejection bands, constructed from quantiles that bound the estimator’s deviation probability. MaxTDA is particularly valuable in applications where precise quantification of statistically significant topological features is essential for revealing underlying structural properties in complex datasets. Numerical simulations across varied datasets, including an example from exoplanet astronomy, highlight the effectiveness of MaxTDA in recovering true topological signals.

Keywords:  Kernel Density Estimation, Level-Set Estimation, Persistent Homology, Robust Inference

1  Introduction

To analyze the underlying structure of complex datasets, topological data analysis (TDA) utilizes tools from algebraic topology to study the shape and connectivity of data across multiple scales. Central to TDA is persistent homology, which analyzes data through a filtration (i.e., a sequence of nested topological spaces) derived from the data, and computes homological invariants across different scales  (Edelsbrunner et al.,,  2000 ; Edelsbrunner and Harer,,  2022 )  (see Section

2

for more details). In this work, we refer to these invariants, which include connected components, loops, and other higher-dimensional holes, as features. By tracking when these features appear (their birth) and disappear (their death) as the filtration parameter changes, persistent homology identifies the features that persist over a range of scales. These features are represented in persistence diagrams as points with coordinates corresponding to their birth and death times where features that persists at larger scales may correspond to true signal, while the lower persistence features may be attributed to noise  (Fasy et al.,,  2014 ) . This process has broad applications. For example, in material science, it reveals significant voids that inform properties like permeability and strength  (Robins et al.,,  2011 ) ; in signal processing, it uncovers persistent circular features from time-delay embeddings that highlight underlying periodic dynamics  (Perea et al.,,  2015 ; Dakurah and Cisewski-Kehe,,  2024 ) ; and in astronomy, it distinguishes important cosmic structures such as clusters, filaments, and voids from noise, aiding in constraining the cosmological model  (Pranav et al.,,  2017 ; Xu et al.,,  2019 ) .

Identifying statistically significant features, particularly, the most persistent, or maximal persistent ones is challenging because persistence diagrams lack a canonical vector space structure, meaning operations like addition, averaging, and other conventional statistical techniques are not naturally defined. This difficulty is further compounded by noisy data. Methods such as kernel smoothing, developed within robust topological analysis  (Fasy et al.,,  2018 ; Anai et al.,,  2020 ) , are employed to mitigate noise but also often reduce the lifetimes (i.e., persistences) of the maximal persistent features. The systematic underestimation of the lifetimes of these features is an artifact of the smoothing mechanisms typically employed in these robust methods. To enable statistical inference for maximal persistent features, it is helpful to address these limitations. This inference challenge arises from the need to quantify uncertainty in the presence of perturbations, such as noise, outliers, or density variation in a random sample

𝕏  n

=

{

𝐱  1

,  ⋯  ,

𝐱  n

}

\mathbb{X}_{n}=\{\mathbf{x}_{1},\cdots,\mathbf{x}_{n}\}

drawn from a probability distribution

ℙ  \mathbb{P}

with compact support

𝕏  \mathbb{X}

in a space

𝒳  ⊂

ℝ  d

\mathcal{X}\subset\mathbb{R}^{d}

. Robust topological tools aim to recover the topology of

𝕏  \mathbb{X}

by defining a smoothing function

ϕ  :

𝒳  →  ℝ

\phi:\mathcal{X}\to\mathbb{R}

. This function, commonly a kernel density estimate (KDE), kernel distance, or distance-to-a-measure (DTM) function, is parameterized to suppress noise or reweight outliers  (Chazal et al.,,  2011 ; Fasy et al.,,  2014 ,  2018 ; Anai et al.,,  2020 ) . A preferred outcome would maintain high persistence for true features while reducing noise features to negligible persistence levels.

The motivation for this work is to develop an inference method that builds on these robust methods, while mitigating the reduction in the persistence of the features, in order to enhance a feature’s statistical significance. The proposed framework, “Maximal TDA” (MaxTDA), mitigates this reduction by first estimating a KDE over the sample as an intermediate representation of the data sampling distribution. Then an upper-level set is defined for a carefully selected density threshold, and rejection sampling is used to draw samples from the thresholded KDE for subsequent statistical inference on the maximal persistent features. This process retains the robustness of the initial smoothing while producing a denser, more consistent sampling surface. Subsequent inference then involves further smoothing or directly computing a persistence diagram directly over this dense sample. This methodology is motivated by two key observations. First, the kernel smoothing enhances robustness against outliers and noise  (Bobrowski et al.,,  2017 ; Fasy et al.,,  2018 ; Anai et al.,,  2020 ) . Second, the thresholded KDE corrects for density variation in the sampling by providing for a denser and statistically consistent sampling surface  (Tsybakov,,  1997 ; Singh et al.,,  2009 ) , a characteristic that is crucial for maintaining the persistence of the features.

Figure 1 :

Illustration of the MaxTDA framework. For a data space (left), robust TDA methods applies a robust filter(e.g., KDE) to the data (middle). MaxTDA extends this by sampling from a thresholded KDE (right), enhancing robustness to noise and creating a denser sampling surface.

This is illustrated in Figure

1  , where the aim is to recover and maintain the persistence of key features such as the two loops (the red and blue circles) indicated by dense clusters.

The proposed MaxTDA approach presents a robust, consistent, and less biased estimator of the most persistent features in certain homology groups, which are groups that identify different