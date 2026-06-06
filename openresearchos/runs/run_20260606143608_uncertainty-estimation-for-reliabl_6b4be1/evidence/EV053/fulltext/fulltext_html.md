[2512.23753] Generalized Regularized Evidential Deep Learning Models: Theory and Comprehensive Evaluation

Generalized Regularized Evidential Deep Learning Models: Theory and Comprehensive Evaluation

Deep Shankar Pandey

1  , Hyomin Choi

2  , and Qi Yu

1

1

Deep Shankar Pandey and Qi Yu are with Rochester Institute of Technology, Rochester, NY, USA.
 Email: {dp7972, qyuvks}@rit.edu  2

Hyomin Choi is with AI Lab, InterDigital, CA, USA.
 Email: hyomin.choi@interdigital.com  2

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract

Evidential deep learning (EDL) models, based on Subjective Logic, introduce a principled and computationally efficient way to make deterministic neural networks uncertainty-aware. The resulting evidential models can quantify fine-grained uncertainty using learned evidence.

However, the Subjective-Logic framework constrains evidence to be non-negative, requiring specific activation functions whose geometric properties can induce activation-dependent learning-freeze behavior—a regime where gradients become extremely small for samples mapped into low-evidence regions. We theoretically characterize this behavior and analyze how different evidential activations influence learning dynamics. Building on this analysis, we design a general family of activation functions and corresponding evidential regularizers that provide an alternative pathway for consistent evidence updates across activation regimes. Extensive experiments on four benchmark classification problems (MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet), two few-shot classification problems, and blind face restoration problem empirically validate the developed theory and demonstrate the effectiveness of the proposed generalized regularized evidential models.

I

Introduction

With recent growth in computational capabilities, availability of large-scale data, and algorithmic improvements, Deep Learning (DL) models have found great success in many real-world applications such as speech recognition  [ kamath2019deep ] , machine translation  [ singh2017machine ] , and computer vision  [ voulodimos2018deep ] .
However, these highly expressive models can easily fit the noise in the training data, leading to overconfident predictions  [ nguyen2015deep ] . This challenge is compounded in specialized domains ( e.g.,  medicine, public safety, and military operations) where labeled data is limited and costly to obtain. Accurate uncertainty quantification is essential for the successful application of DL models in these domains. To this end, DL models have been augmented to become uncertainty-aware  [ gal2016dropout ,  blundell2015weight ,  pearce2020uncertainty ] . However, commonly used extensions require expensive sampling operations  [ gal2016dropout ,  blundell2015weight ] , which significantly increase the computational costs  [ lakshminarayanan2017simple ] .

The recently developed evidential deep learning (EDL) models bring together evidential theory  [ shafer1976mathematical ,  josang2016subjective ]  and deep neural architectures
that turn a deterministic neural network uncertainty-aware. By leveraging the learned evidence, evidential models are capable of quantifying fine-grained uncertainty that helps to identify the sources of ‘unknowns’. Furthermore, since only lightweight modifications are introduced to existing DL architectures, additional computational costs remain minimal. Such evidential models have been successfully extended to classification  [ sensoy2018evidential ] , regression  [ amini2020deep ] , meta-learning  [ Pandey_2022_CVPR ] , and open-set recognition  [ bao2021evidential ]  settings.

Figure 1 :

Cifar-100 Result

Despite the attractive uncertainty quantification capacity, evidential models often achieve competitive predictive performance only on relatively simple learning problems. Their performance can degrade on more complex, large-scale datasets even in standard classification settings. As shown in Figure

1  , an evidential model using ReLU activation and an evidential MSE loss  [ sensoy2018evidential ]  achieves around

36  36

% test accuracy on CIFAR-100, nearly

40  40

% lower than a standard softmax model. In addition, many evidential variants are sensitive to architecture or hyperparameter changes, requiring careful tuning for stable performance. The experiment section provides more details on these cases.

To better understand this phenomenon, we perform a theoretical analysis of evidential learning in the standard classification setting. Our results identify an  activation-induced learning-freeze behavior , where the interaction between non-negative evidence parameterization and specific activation functions can map samples into “zero-evidence regions” (regions of vanishing evidence gradients).

Importantly, this behavior arises within the design choices of the EDL framework itself. EDL couples non-negative evidence parameterization with a KL-based prior that  intentionally  promotes high epistemic uncertainty at class boundaries and in regions far from the training distribution—an effect that helps prevent overconfident errors. Activation functions and regularizers determine how evidence accumulates under this framework. Our analysis shows that commonly used non-negative activations can inadvertently create “zero-evidence” regions where gradients become extremely small, making evidence updates for nearby samples ineffective.

More specifically, EDL models acquire limited new evidence from samples mapped into these low-evidence regions because the corresponding evidence gradients approach zero. Moreover, the learning signal decreases proportionally as samples are mapped closer to the zero-evidence region, irrespective of supervised information.

This activation-induced stagnation is illustrated in Figure

2

(with detailed discussion in Section

IV-C

). We analyze several existing evidential variants and observe this behavior consistently across models and settings. Motivated by these insights, we introduce a novel  G eneralized  R egularized  E vidential mo d el ( GRED ) that employs positive evidence regularization to encourage evidence accumulation even in low-evidence regimes.

A preliminary version of this work has been published as a conference paper  [ pandey2023learn ] . Improving on RED, we propose generalized regularized evidential models that mitigate learning stagnation across a family of evidential activations (Section

IV  ). We theoretically show the effectiveness of the correct-evidence regularization (Theorem

3  ) and provide expanded analysis of evidential losses (Section  LABEL:ap:evLossAnalysis ). We further extend GRED to challenging few-shot classification and blind face restoration tasks, and carry out detailed uncertainty analysis, and demonstrate the broader utility of evidential uncertainty.

Figure 2 :

Intuitive visualization of a zero-evidence region for evidential models in the evidence space for binary classification. Samples mapped into such regions have extremely small gradients, leading to limited model update during training.  GRED encourages larger gradients for ‘zero-evidence’ samples, enabling consistent learning across samples.

Our major contributions can be summarized as follows:

•

We identify an activation-induced learning-freeze behavior in evidential models, wherein data samples mapped to “zero-evidence” regions receive vanishing evidence gradients. For these samples, the learning signal decreases proportionally as they are mapped closer to the zero-evidence region in the evidence space.

•

We theoretically show that evidential models with

exp  \exp

activation produce stronger gradients near low-evidence regions compared to other activations.

•

We introduce a generalized evidence regularization strategy that encourages evidence updates across activation regi