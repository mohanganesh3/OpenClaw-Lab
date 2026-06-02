[2601.07965] When Models Know When They Do Not Know Calibration, Cascading, and Cleaning

When Models Know When They Do Not Know

Calibration, Cascading, and Cleaning

Chenjie Hao  1  ,
Weyl Lu  1  ,
Yuko Ishiwaka  3  ,
Zengyi Li  2  ,
Weier Wan  2  ,
Yubei Chen  1,2†

1  University of California, Davis,  2  Aizip

3  SoftBank Corp.

Abstract

When a model knows when it does not know, many possibilities emerge. The first question is how to enable a model to recognize that it does not know. A promising approach is to use confidence, computed from the model’s internal signals, to reflect its ignorance. Prior work in specific domains has shown that calibration can provide reliable confidence estimates. In this work, we propose a simple, effective, and universal training-free method that applies to both vision and language models, performing model calibration, cascading, and data cleaning to better exploit a model’s ability to recognize when it does not know. We first highlight two key empirical observations: higher confidence corresponds to higher accuracy within a single model, and models calibrated on the validation set remain calibrated on a held-out test set. These findings empirically establish the reliability and comparability of calibrated confidence. Building on this, we introduce two applications: 1. Model cascading with calibrated advantage routing and 2. Data cleaning based on model ensemble. Using the routing signal derived from the comparability of calibrated confidences, we cascade large and small models to improve efficiency with almost no compromise in accuracy, and we further cascade two models of comparable scale to achieve performance beyond either model alone. Leveraging multiple experts and their calibrated confidences, we design a simple yet effective data-cleaning method that balances precision and detection rate to identify mislabeled samples in ImageNet and Massive Multitask Language Understanding (MMLU) datasets. Our results demonstrate that enabling models to recognize when they do not know is a practical step toward more efficient, reliable, and trustworthy AI.

†

†  footnotetext:

†  \dagger

Corresponding author.

1  Introduction

Modern deep neural networks have achieved remarkable success in both vision and language domains. However, models still frequently produce incorrect  (Zhang  et al. ,  2025 )  in regions of ignorance. Confidence, computed from intrinsic model signals, offers a promising path to estimate such ignorance. Prior work  (Guo  et al. ,  2017 ; Nixon  et al. ,  2019 ; Spiess  et al. ,  2024 ; Wang  et al. ,  2021 ; Gupta and Ramdas,  2021 )  has shown in specific domains that model calibration can yield reliable confidence estimates. In this work, we first introduce a unified framework for model calibration that applies to both vision and language models across diverse tasks.

We highlight two core observations about confidence and calibration: (1) For a single model, confidence functions defined as likelihoods from logits exhibit an approximate monotonic relation, where higher-confidence samples tend to have higher accuracy, even without calibration. (2) Confidence calibrated on a validation set remains calibrated on a held-out test set, consistent with previous findings  (Guillory  et al. ,  2021 ) . Based on these observations, we empirically demonstrate the reliability and cross-model comparability of calibrated confidence.

Building on calibrated confidence, we propose two applications that are broadly applicable to both vision and language models. The first application is model cascading. Prior works have introduced routers or gating modules to decide whether an input should be handled by a larger model, an external resource, or a human  (Teerapittayanon  et al. ,  2016 ; Shazeer  et al. ,  2017 ; Lepikhin  et al. ,  2021 ; Bolukbasi  et al. ,  2017 ; Wang  et al. ,  2020 ) . We propose a training-free cascading method: using calibrated confidence, we construct a reliable routing signal—confidence advantage—that determines whether to invoke a larger model. On tasks spanning image classification (ImageNet-1K  (Russakovsky  et al. ,  2015 ) ), code generation (Mostly Basic Python Problems (MBPP)  (Austin  et al. ,  2021 ) , and BigCodeBench  (Zhuo  et al. ,  2024 ) ), reasoning and math task(ARC-Challenage  (Clark  et al. ,  2018 ) , GSM8K  (Cobbe  et al. ,  2021 ) ) , and knowledge-intensive QA (MMLU)  (Hendrycks  et al. ,  2020 ) , we show that this method achieves an effective balance of accuracy and efficiency. We further extend cascading to large–large model pairs, where the combined cascade surpasses the performance of both standalone models. Notably, on ImageNet-1K, cascading two state-of-the-art models achieves accuracy beyond the previous state of the art.

The second application is data cleaning. Previous work (e.g., Confident Learning  (Northcutt  et al. ,  2022 ) ) has demonstrated that confidence can signal mislabeled data: when a model is highly confident yet disagrees with the given label, the label is likely incorrect. We build upon this idea by using calibrated confidence as a reference signal. Since even calibrated single models may still err, we introduce a mixture-of-experts approach for data cleaning, where calibrated confidence provides a tunable trade-off between precision and detection rate. This method is simple yet effective. We apply it to ImageNet-1K and MMLU test sets, and through large-scale manual verification on ImageNet, we show that it outperforms prior methods in both accuracy and detection power.

In summary, we propose a unified and streamlined framework for calibration, cascading, and cleaning that applies across domains and tasks. Our contributions are threefold: (1) we formalize a unified view of model calibration and empirically validate the reliability and comparability of existing methods; (2) we introduce a confidence-based cascading method, balancing accuracy and efficiency for small–large pairs and achieving superior performance with large–large pairs; and (3) we present a calibration-driven mixture-of-experts data cleaning approach that is simple, effective, and validated through human annotation. Together, these components provide a practical recipe for leveraging when models know when they do not know.

2  Method

This section introduces our formalization of model calibration, key empirical observations, and two downstream applications. We begin with a unified formulation applicable to both vision and language models, followed by two calibration methods used in our experiments: temperature scaling and Platt scaling. We then present key observations —— higher confidence corresponds to higher accuracy, and calibration learned on validation generalizes to held-out test sets —— which establish that calibrated confidence is both reliable and comparable across models. Building on these findings, we develop two applications: confidence-based model cascading to balance efficiency and accuracy or to combine strong models for improved performance, and agreement-driven data cleaning where confidence naturally balances precision and recall.

2.1  On calibration of vision model and language model

Formulation.

Previous work has proposed calibration formulations tailored to specific cases, such as image classification models. Here we provide a unified formulation of model calibration that applies to both vision and language models, accommodates diverse downstream tasks, and supports different calibration methods.

One sentence summary: calibration is making the model’s confidence

c  M

c_{M}

align with the expected score

𝔼  ​

[  v  ]

\mathbb{E}[v]

from a verifier

v  v

, where the expectation is estimated as an average within bins.

Confidence function

c  c

𝔼  ​

[  v  ]

\mathbb{E}[v]

v  v

Verifier

confidence

verification score

accuracy

align

binning

A calibration setup consists of the following components:

•

Model

M  M

takes input

X  ∈