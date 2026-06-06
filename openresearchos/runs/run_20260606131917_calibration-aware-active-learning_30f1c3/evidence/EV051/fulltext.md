[2207.04125] Out of Distribution Detection via Neural Network Anchoring

\jmlrvolume

189
 \jmlryear 2022
 \jmlrworkshop ACML 2022

Out of Distribution Detection via Neural Network Anchoring

\Name Rushil Anirudh  \Email anirudh1@llnl.gov

\Name Jayaraman J. Thiagarajan  \Email jjayaram@llnl.gov

\addr Center for Applied Scientific Computing (CASC)

Lawrence Livermore National Laboratory

Abstract

Our goal in this paper is to exploit heteroscedastic temperature scaling as a calibration strategy for out of distribution (OOD) detection. Heteroscedasticity here refers to the fact that the optimal temperature parameter for each sample can be different, as opposed to conventional approaches that use the same value for the entire distribution. To enable this, we propose a new training strategy called anchoring that can estimate appropriate temperature values for each sample, leading to state-of-the-art OOD detection performance across several benchmarks. Using NTK theory, we show that this temperature function estimate is closely linked to the epistemic uncertainty of the classifier, which explains its behavior.
In contrast to some of the best-performing OOD detection approaches, our method does not require exposure to additional outlier datasets, custom calibration objectives, or model ensembling. Through empirical studies with different OOD detection settings – far OOD, near OOD, and semantically coherent OOD - we establish a highly effective OOD detection approach. Code to reproduce our results is available at  github.com/LLNL/AMP

keywords:

OOD Detection, Temperature Scaling, Calibration, Anchoring, Uncertainty

†

†  editors:  Emtiyaz Khan and Mehmet Gönen

1  Introduction

The task of using a trained model to accurately distinguish between samples from the dataset used for training –  i.e. , the in-distribution (ID), and any other external dataset with different semantic characteristics is broadly referred to as OOD (out-of-distribution) detection. To solve this challenging problem, one needs to obtain an effective characterization of the ID data manifold, such that the discrepancy between test data and the  inferred manifold  can be used to recognize the model’s lack of knowledge about OOD data. This is commonly achieved by learning a scoring function:

𝒮  :

X  →  ℝ

:  𝒮

→  𝑋  ℝ

\mathcal{S}:X\rightarrow\mathbb{R}

that can score both ID and OOD samples appropriately. A simple scoring function can be based on the maximum softmax probability (MSP) of a prediction, with the expectation that the model will be more confident about an ID sample compared to OOD samples. However, in practice, such simple prediction confidence scores are poorly calibrated, and as a result, several novel scoring functions have emerged – predictive entropy  ( Guo et al. ,

2017  ) , energy  ( Liu et al. ,

2020  ) , uncertainty estimates  ( Gal and Ghahramani ,

2016  ;  Lakshminarayanan et al. ,

2017  ) , latent space deviation  ( Van Amersfoort et al. ,

2020  ) , class-specific deviations  ( Lee et al. ,

2018b  ;  Sastry and Oore ,

2020  )  etc. Though these scoring functions often perform better than MSP, many state-of-the-art formulations  ( Hendrycks et al. ,

2019  ;  Liu et al. ,

2020  ;  Yang et al. ,

2021  )  rely on additional unlabeled data for calibrating model predictions to better reject OOD data. In addition to requiring sophisticated training strategies (e.g., outlier exposure), this approach can be sub-optimal when the calibration dataset is not  strictly  OOD,  i.e. , and they contain shared semantics with the ID set  ( Yang et al. ,

2021  ) . Further, the calibration strategy used in many of these methods relies on  temperature scaling

( Guo et al. ,

2017  ) , which essentially scales the logits by a scalar called the temperature. When the temperature parameter is greater (or lower) than

1

1

1

, the entropy of the resulting prediction distribution increases (or decreases). Consequently, with an appropriate temperature value (chosen with either external or additional validation data), even this simple scaling leads to much improved OOD detection performance.

Heteroscedastic temperature scaling with anchoring.

In this paper, we explore the idea of  heteroscedastic  temperature scaling,  i.e. , instead of using the same temperature scalar for all the samples, we construct a temperature function that produces sample-specific temperature values. Our hypothesis is that by appropriately tempering the predictions for ID and OOD samples, any existing scoring function can effectively between distinguish them. We achieve this using a novel training procedure called  neural network anchoring . In a nutshell, anchoring involves first transforming the input image,

x

x

\mathrm{x}

, into a tuple using the transformation

ℰ  :

x  →

[  c  ,

x  −  c

]

:  ℰ

→  x

c

x  c

\mathcal{E}:\mathrm{x}\rightarrow[\mathrm{c},\mathrm{x}-\mathrm{c}]

, where

c

c

\mathrm{c}

is another randomly chosen image (“anchor”) from the training set, and predicting the label for

x

x

\mathrm{x}

using this tuple. We also propose an additional consistency training strategy by perturbing the anchor before encoding, which boosts the performance further. During inference, we obtain predictions from multiple random anchors and propose to estimate the temperature based on standard deviation of these predictions. Using neural tangent kernel theory  ( Jacot et al. ,

2018  ) , we show that our heteroscedastic temperature estimate is closely related to the  epistemic  uncertainty of the model.

Figure 1:

Improving existing OOD detectors via heteroscedastic temperature scaling.  (A) We propose a new training procedure called neural network anchoring to estimate the temperature parameter for any test sample, and show that it can be leveraged to improve conventional OOD detectors (e.g., entropy-based). (B) We also introduce a new consistency training objective to further improve the fidelity of detectors.

We use this temperature estimate to calibrate the predictions for a test sample, using which we can compute an OOD score using existing scoring functions (e.g., entropy). See

Fig.

1  (A) for an illustration of the process, and

Figs.

3

and

3

for the pseudo-codes.

Fig.

1  (B) illustrates the improvement over conventional temperature scaling using the standard CIFAR-10/SVHN OOD benchmark.
Through extensive empirical analysis, we demonstrate that the proposed approach produces state-of-the-art detection performance across multiple benchmarks and models (summarized in Table

1  ).

Base Model

Benchmark

(IN

→

→

\rightarrow

OOD) Year

Reference

WRN-40-2

CIFAR-10/100

→

→

\rightarrow

6 Datasets  (

Liang et al.

, ICLR’17)

Table

2

ResNet-34

CIFAR-10/100

→

→

\rightarrow

7 Datasets  (

Sastry and Oore

, ICML’20)

Table

4

ResNet-34

CIFAR-10

↔

↔

\leftrightarrow

CIFAR-100  (Near OOD)

Table

4

ResNet-50

ImageNet-1K

→

→

\rightarrow

ImageNet-C  (

Krishnan and Tickoo

, NeurIPS’20)

Table

5

, Figure

5

ResNet-18

Semantically Coherent OOD  (

Yang et al.

, ICCV’21)

Table

6

ResNet-34

Robustness to resizing artifacts  (this paper)

Table

7

WRN-40-2

Ablation study

Table

8

Table 1:

Summary of experiments in this paper.

2  Background and Related Work

We use training data

𝒟  =

{

(

x  i

,

y  i

)

}

i  =  1

n

𝒟

superscript

subscript

subscript  x  𝑖

subscript  𝑦  𝑖

𝑖  1

𝑛

\mathcal{D}=\{(\mathrm{x}_{i},y_{i})\}_{i=1}^{n}

, where

x  i

∈

P  I

subscript  x  𝑖

subscript  𝑃  𝐼

\mathrm{x}_{i}\in P_{I}

and

y  i

∈

𝒞  I

≔

{  1  ,  2  ,  ⋯  ,

N

c  ​  l  ​  a  ​  s  ​  s

}

subscript  𝑦  𝑖

subscript  𝒞  𝐼

≔

1  2  ⋯

subscript  𝑁

𝑐  𝑙  𝑎  𝑠  𝑠

y_{i}\in\mathcal{C}_{I}\coloneqq\{1,2,\cdots,N_{class}\}

, to train a model

f  ​

(  𝜽  )

∈  ℋ

𝑓  𝜽

ℋ

f(\boldsymbol{\theta})\in\mathcal{H}

with randomly initialized weights

𝜽  0

subscript  𝜽  0

\boldsymbol{\theta}_{0}

and hypothesi