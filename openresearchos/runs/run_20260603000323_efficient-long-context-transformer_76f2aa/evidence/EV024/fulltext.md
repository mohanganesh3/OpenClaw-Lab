[2505.01475] CodeSSM: Towards State Space Models for Code Understanding

CodeSSM: Towards State Space Models for Code Understanding

Shweta Verma 1, *

,
 Abhinav Anand 1, *

,
 Mira Mezini 1,2,3

1 TU Darmstadt,
 2 Hessian Center for Artificial Intelligence, Darmstadt, Germany,

3 National Research Center for Applied Cybersecurity ATHENE

* Equal Contribution,

Correspondence:

shweta.verma@tu-darmstadt.de

Abstract

Although transformers are widely used for various code-specific tasks, they have some significant limitations. In this paper, we investigate State Space Models (SSMs) as a potential alternative to transformers for code understanding tasks, such as code retrieval, classification, and clone detection. Previous research has already demonstrated that SSMs are more compute-efficient than transformers. In our work, we show that SSMs are also more sample-efficient and can effectively extrapolate to longer contexts (beyond the pretraining context) during fine-tuning. Through comprehensive experiments, we demonstrate that SSMs could serve as a viable alternative to transformers for code understanding tasks, while addressing some of the major limitations associated with transformers.

CodeSSM: Towards State Space Models for Code Understanding

Shweta Verma 1, *  ,
Abhinav Anand 1, *  ,
Mira Mezini 1,2,3

1 TU Darmstadt,
 2 Hessian Center for Artificial Intelligence, Darmstadt, Germany,

3 National Research Center for Applied Cybersecurity ATHENE

* Equal Contribution,

Correspondence:

shweta.verma@tu-darmstadt.de

1  Introduction

Transformers  (Vaswani et al.,  2017 )  have been known to perform well on various
code intelligence tasks. One of the main reasons for this improvement is the pretraining - finetuning paradigm used to train transformer models, which is possible due to the ease of parallelization of attention-based transformer architectures. Under this paradigm, the transformer model is initially trained on a large corpus of unlabeled data using a self-supervised training objective. This is followed by a supervised finetuning process on a smaller labeled dataset.

However, the performance gain offered by the transformer models comes with some trade-offs, such as quadratic complexity, a substantial data requirement, and high inference costs. While methods such as linear attention  (Katharopoulos et al.,  2020 )  and sparse attention  (Guo et al.,  2022b ; Zaheer et al.,  2020 ; Condevaux and Harispe,  2022 )  have been proposed to address computational inefficiency, the practical gains from these methods are limited  (Yang et al.,  2025 ; Qin et al.,  2022 ) .

Another limitation of transformers is their fixed context window resulting from positional embedding. Although methods such as AliBi  (Press et al.,  2022 )  and RoPE  (Su et al.,  2024 )  solve the problem of a fixed context window, transformer models still fail to generalize to lengths not seen during pretraining  (Peng et al.,  2024 ; Goel et al.,  2025 ) . To overcome this challenge, additional techniques are required to extend the context window without significantly compromising performance  (Chen et al.,  2023b ; Hua et al.,  2025a ; Lin et al.,  2024 ) .

One-dimensional Convolutional Neural Networks (1D CNNs) are a fast and positional embedding-free alternative to transformers. 1D CNNs, when trained using the pretraining-finetuning approach, show improved performance on some tasks but fail to capture long-range dependencies in input data  (Tay et al.,  2021 ) .

State-space models (SSMs) Gu et al. ( 2021 ,  2022b ,  2022a )  represent an alternative that addresses the limitations of transformers and 1D CNNs. SSMs leverage the strengths of 1D CNNs while effectively capturing long-range dependencies with linear-time complexity. Additionally, SSMs are position-aware, eliminating the need for positional embedding.

In this work, we explore the application of SSMs to the domain of code understanding. To achieve this, we pretrain an encoder-only model based on SSM, which we call CodeSSM, on a small code dataset with masked language modeling (MLM)  (Devlin et al.,  2019 ) . We evaluated our model on multiple code understanding benchmarks. We also investigate various training aspects, such as the effects of positional embedding, dropout, bidirectional SSMs, and varying pretraining context lengths. We compare the CodeSSM with two transformer models that we train – (1) BertCoder: based on Bert architecture  (Devlin et al.,  2019 )  and (2) RoCoder: based on RoFormer architecture  Su et al. ( 2024 ) . We assess the performance of our model against several high-performing transformers as well.

Our study demonstrates that SSMs exhibit strong sample efficiency in code language modeling tasks. Notably, CodeSSM achieves over 50% accuracy in masked language modeling (MLM) with fewer than 3,000 training steps.

Additionally, the code language modeling capabilities of CodeSSM also transfer well to downstream tasks as CodeSSM outperforms transformers on multiple code understanding tasks when pretrained under similar conditions. CodeSSM is also competitive with (and even surpasses) transformer models trained on significantly larger datasets and with more complex pretraining objectives. Furthermore, another advantage of SSMs is their ability to extrapolate to much larger contexts than those encountered during pretraining, effectively addressing a key limitation of the transformer architecture.

Our objective in training the CodeSSM model is not to achieve state-of-the-art performance on coding benchmarks, but to explore and document the strengths and limitations of SSMs in understanding code. This goal also motivates our usage of a small dataset and MLM pretraining which allows us to quickly experiment with different model configurations and compare them under similar training settings. Although we observe improvements in performance with larger pretraining data, we will leave the exploration of very large training datasets and more complex training objectives for future work.

Summary of contributions:

•

We conduct the first systematic evaluation of State Space Models (SSMs) for code understanding, introducing a specialized model, which we term CodeSSM. Our analysis reveals that CodeSSM achieves robust performance across multiple benchmarks, consistently outperforming attention-based models while requiring substantially less pretraining data.

•

Our study presents the first empirical evidence that State Space Models (SSMs) achieve superior sample efficiency in code-understanding tasks compared to transformers. CodeSSM substantially outperforms transformer baselines in data-constrained scenarios with only a few thousand training samples.

•

We demonstrate that CodeSSM addresses a key limitation of transformer architectures –extrapolation to unseen context lengths. By examining the effects of different pretraining context lengths and positional embeddings in CodeSSM, we show that SSMs outperform transformers as downstream input sequences become longer.

•

We observe that including even a small dropout mask in the CodeSSM layer degrades the performance of the model even on very small benchmarks. CodeSSM also performs very well on the adversarial text-code search task.
Together these observations indicate that SSMs exhibit robust semantic understanding capabilities.

2  CodeSSM

Figure 1:  CodeSSM layer. Transform represents SSMs in CodeSSMs and Discrete Fourier Transforms in CodeF. Dropout is only present in one of the CodeSSM variants, CodeSSM-do. In CodeSSM-uni, both the flip operations are removed. Non-linearity is not shown for clarity.

In this section, we introduce the basic architecture of CodeSSM and its variations that we investigate. We also explain the pretraining setup.

2.1  Architecture

CodeSSM is an encoder-only model consisting of 12 layers

1

1  1 The trained weights are available here:  https://figshare.com/s/14238287e9078f92cd50 .

. This model is built upon the B