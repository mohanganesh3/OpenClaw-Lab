[2504.14795] A Bayesian Approach to Segmentation with Noisy Labels via Spatially Correlated Distributions

1

1  institutetext:  Tohoku University, Japan

1

1  email:  tadokororyuryu@gmail.com

2

2  institutetext:  Preferred Networks, Inc., Japan

2

2  email:  {takagi,ichi}@preferred.jp

A Bayesian Approach to Segmentation with Noisy Labels via Spatially Correlated Distributions

Ryu Tadokoro

11

0009-0001-9473-3832

Tsukasa Takagi

22

0009-0003-1147-2505

(✉)Shin-ichi Maeda

22

0000-0002-3254-9722

Abstract

In semantic segmentation, the accuracy of models heavily depends on the high-quality annotations. However, in many practical scenarios such as medical imaging and remote sensing, obtaining true annotations is not straightforward and usually requires significant human labor. Relying on human labor often introduces annotation errors, including mislabeling, omissions, and inconsistency between annotators.
In the case of remote sensing, differences in procurement time can lead to misaligned ground truth annotations.
These label errors are not independently distributed, and instead usually appear in spatially connected regions where adjacent pixels are more likely to share the same errors.
To address these issues, we propose an approximate Bayesian estimation based on a probabilistic model that assumes training data includes label errors, incorporating the tendency for these errors to occur with spatial correlations between adjacent pixels. Bayesian inference requires computing the posterior distribution of label errors, which becomes intractable when spatial correlations are present. We represent the correlation of label errors between adjacent pixels through a Gaussian distribution whose covariance is structured by a Kac-Murdock-Szegö (KMS) matrix, solving the computational challenges.
Through experiments on multiple segmentation tasks, we confirm that leveraging the spatial correlation of label errors significantly improves performance.
Notably, in specific tasks such as lung segmentation, the proposed method achieves performance comparable to training with clean labels under moderate noise levels. Code is available at  https://github.com/pfnet-research/Bayesian_SpatialCorr .

Keywords:  Segmentation Noisy label Noisy annotation Bayesian inference Spatial correlation.

1  Introduction

Semantic segmentation, which involves classifying each pixel in an image into one of several classes, is a crucial task in computer vision. In supervised learning, the accuracy of segmentation models critically depends on the quality of the annotations in the training data. However, obtaining truly accurate pixel-level annotations is challenging in many practical applications.
Even when expert annotators are employed, errors, omissions, and subjectivity in interpretation are inevitable, leading to inconsistencies in datasets.
In particular, high inter- and intra-annotator variability is widely reported in medical imaging, where experts may have differing interpretations of the same structures. For instance,  [ 41 ,  19 ,  27 ,  38 ,  27 ]  highlight significant discrepancies among expert annotators; some delineate structures more generously, while others prefer more conservative annotations. The observer-dependent annotations exacerbate label noise in supervised learning.
Label noise is also a critical issue in remote sensing, where determining ground truth labels often requires field surveys over large and sometimes inaccessible regions  [ 7 ,  26 ,  6 ] . Due to the logistical and economic challenges of large-scale ground truth collection, researchers frequently rely on automatic labeling systems,
which may introduce systematic errors. Additionally, high-quality annotated datasets remain a critical bottleneck for supervised learning, particularly in remote sensing applications where annotations are often repurposed across different types of satellite images. For example, the OpenEarthMap dataset  [ 35 ]  was created by manually annotating high-resolution optical satellite images for semantic segmentation. However, these annotations are sometimes reused for synthetic aperture radar (SAR) imagery, despite differences in resolution and capture conditions  [ 12 ,  42 ,  23 ] . Additionally, changes in artificial structures or variations in land cover further contribute to label inconsistencies  [ 8 ,  12 ] .

Various approaches have been proposed to mitigate the adverse effects of noisy labels. Some methods attempt to stop training early to prevent the network from overfitting to noise and generating unreliable pseudo-labels  [ 21 ,  22 ] , while others modify the loss function to be more robust against large errors  [ 9 ] . Although these techniques can reduce the influence of noisy annotations, they do not fundamentally address the core reason why the standard supervised learning framework fails in the presence of noisy labels.

We propose a method that directly tackles the root cause of this issue. In supervised learning for segmentation models, training typically reduces to optimizing the cross-entropy loss. This optimization implicitly follows a maximum likelihood estimation (MLE) framework under the assumption that the training data consists of independent and identically distributed samples drawn from the joint distribution of images and clean labels. However, when labels are noisy, the assumption of identically distributed assumption no longer holds, as the observed labels systematically deviate from clean labels, leading to performance degradation in the trained model.

To address this issue, we maintain the MLE framework but reformulate it using a more suitable probabilistic model. Specifically, we introduce a model that explicitly accounts for the presence of noisy labels, which differ from the clean labels due to labeling errors.
In practice, annotation errors tend to exhibit strong spatial correlations - mislabeling often occurs in contiguous regions. Variations in annotation criteria among experts, as well as changes in the underlying scene — such as the construction or demolition of buildings or alterations in vegetation — further reinforce this spatial dependency. Given this, we assume that label errors are not independent but instead exhibit spatial correlations among pixels.

However, directly modeling spatial correlations poses significant computational challenges. To address this, we introduce a class of discrete distributions called the ELBO Computable Discrete Distribution, which enables the efficient optimization of the Evidence Lower Bound (ELBO). This discrete distribution is represented through continuous variables that follow a Gaussian distribution, where spatial correlations between pixels are expressed via a covariance matrix. Representing discrete variables through a Gaussian distribution successfully circumvents the intractability of summing over all possible realizations of the discrete variables.
While the covariance matrix, whose number of elements scales quadratically with the number of pixels, introduces additional computational challenges in evaluating the ELBO, particularly in computing second-order statistics, its inverse, and its determinant. To overcome these computational intractabilities, we leverage the Kac-Murdock-Szegö (KMS) matrix  [ 5 ,  17 ] , which enables efficient computations necessary for ELBO evaluation.
To validate the effectiveness of our approach, we conduct extensive empirical evaluations on multiple segmentation tasks. Our experimental results demonstrate that our method significantly improves robustness against label noise, particularly in scenarios with moderate to high levels of spatially correlated label noise.
In summary, our contributions are as follows:

1.

A probabilistic model with the ELBO Computable Discrete Distribution for learning with spatially correlated label noise:

We propose a probabilistic model that explicitly accounts for noisy labels and their spatial correlations, which ar