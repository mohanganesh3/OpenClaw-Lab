[2502.20395] R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts

R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts

Zhongyang Li

Ziyue Li

Tianyi Zhou

Abstract

In large multimodal models (LMMs), the perception of non-language modalities (e.g., visual representations) is usually not on par with the large language models (LLMs)’ powerful reasoning capabilities, deterring LMMs’ performance on challenging downstream tasks.
This weakness has been recently mitigated by replacing the vision encoder with a mixture-of-experts (MoE), which provides rich, multi-granularity, and diverse representations required by diverse downstream tasks.
The performance of multimodal MoE largely depends on its router, which reweights and mixes the representations of different experts for each input.
However, we find that the end-to-end trained router does not always produce the optimal routing weights for every test sample. To bridge the gap, we propose a novel and efficient method “ R e- R outing in  T est- T ime (R2-T2)” that locally optimizes the vector of routing weights in test-time by moving it toward those vectors of the correctly predicted samples in a neighborhood of the test sample. We propose three R2-T2 strategies with different optimization objectives and neighbor-search spaces. R2-T2 consistently and greatly improves state-of-the-art LMMs’ performance on challenging benchmarks of diverse tasks, without training any base-model parameters.

Machine Learning, ICML

1 Johns Hopkins University;  2 University of Maryland, College Park

zli300@jh.edu, {litzy619,tianyi}@umd.edu

Project:  https://github.com/tianyi-lab/R2-T2

1  Introduction

Figure 1:  R2-T2 applied to MoAI-7B compared against 7/8/13B VLMs on 9 benchmarks. R2-T2 significantly enhances performance of the 7B base MoE model, surpassing a recent 13B VLM.

Mixture-of-Experts (MoE) have achieved remarkable success in scaling up the size and capacity of large language and multimodal models (LLMs and LMMs)  (Shazeer et al.,  2017 )  without (significantly) increasing the inference cost. Specifically, it allows us to increase the total number of experts, which provides finer-grained expertise and skills, yet selecting a constant number of experts for each input  (Lepikhin et al.,  2020 ) . In MoE, the sparse selection of experts is achieved through a router, which determines the weight of each candidate expert based on the input so only experts with nonzero weights are selected  (Fedus et al.,  2022 ) . MoE then aggregates the outputs of the selected experts according to their weights. Hence, the router and its produced routing weights play important roles in MoE’s inference cost and output quality.

As the most widely studied LMM, many vision language models (VLM) adopt an architecture composed of a vision encoder and an LLM  (Zhu et al.,  2023 ) , which are both pre-trained and then aligned by further finetuning so the LLM can include the vision encoder’s output in its input as additional tokens. The alignment is usually obtained through a lightweight projection layer or Q-former (a Transformer model) converting the vision encoder’s output to LLM tokens. Despite the broad usage of this architecture, the capability of a vision encoder is usually much more limited than the LLMs (i.e., the “ modality imbalance ”)  (Schrodi et al.,  2024 ) . So the visual features cannot cover all the information required by different reasoning tasks performed by LLMs. Moreover, the alignment module may lead to an information bottleneck from the visual perception to the reasoning  (Yao et al.,  2024 ) .

Figure 2:  An example of how R2-T2 optimizes the routing weights. Given the test sample, it finds

k  k

NN in the reference set of correctly predicted samples with similar questions. In the example, the test sample requires reasoning about positional relationships.
R2-T2 identifies relevant kNN samples, adjusting the top-1 expert from

𝐈  lang

\mathbf{I}_{\textsc{lang}}

(aligning visual features with language) to

𝐈  aux

\mathbf{I}_{\textsc{aux}}

(aligning visual features with auxiliary computer vision features).
This expert shift is crucial in correcting the initial wrong answer.

Recent advances in LMMs replace a single vision encoder with a mixture of encoders  (Lin et al.,  2024 ; Lee et al.,  2025 ; Zong et al.,  2024 ; Shi et al.,  2024 ) , which turns out to be an effective and low-cost approach to mitigate  modality imbalance and alignment bottleneck . In multimodal MoE, each expert is an encoder or a mixer of sensory inputs that focuses on a specific type of features, e.g., object classes, text in images, spatial relations, dense captions, segmentation, etc., so the LLM can select the information acquired by any given downstream task from the concatenated or fused features from the MoE, through a router that is trained in an end-to-end manner to produce the weights of all the experts adaptive to the input task.

Although multimodal MoE achieves remarkable success in enhancing the performance of existing LMMs, the choice of experts or the routing weights for individual instances are not always optimal due to the limitations of the router’s design and the diversity of potential downstream tasks compared to the tasks used to train the router. The suboptimality of routing substantially limits the performance and generalization of multimodal MoE on unseen tasks.
As illustrated in Figure

2  , the base model initially selects a sub-optimal expert (e.g.,

𝐈  lang

\mathbf{I}_{\textsc{lang}}

) for a spatial reasoning task, leading to incorrect predictions.
This has been verified on recent multimodal MoE models. As shown in Table

2  , compared to the original routing weights of base models, the optimal (oracle) routing weights improve the accuracy by

≥

10  %

\geq 10\%

on most evaluated LMM benchmarks. To avoid the expensive cost of re-training a router on a much larger dataset, in this paper, we investigate  how to improve the routing weights in test-time without training any model parameters.

Since routing weights encode the choices of experts with essential knowledge and key skills acquired by the input task, and motivated by the assumption that knowledge and skills are usually transferable across different tasks, we posit that the routing weights of successful tasks can provide critical clues for optimizing the routing weights of a new task.
Specifically, we leverage the similarity in a task embedding space, which may reflect the knowledge or skill sharing between tasks, and modify the routing weight vector of a test task by imitating its nearby successful tasks. While the task embedding space, optimization objective, and the number of update steps can vary and their design choices may result in different performances, this innovative mechanism of  optimizing routing weights or “re-routing” in test-time (R2-T2)  focuses on correcting the mistakes made by the routers in existing multimodal MoE, e.g., extracting object detection features for a task mainly depending on the text information in an input image, and thus turns various failed cases into success. Rather than finetuning the whole model, R2-T2 is training-free and aims to maximize the potential of MoE in the reasoning tasks by LMMs.

Following the above idea, we explored several novel strategies for test-time routing weight optimization. They all modify the routing weights of a test task/sample based on a representative set of tasks/samples on which the multimodal MoE achieves correct or high-quality outputs.
While the oracle routing weights are achieved by minimizing the test sample’s loss, for a practical approach, we propose to replace the oracle loss with a surrogate, i.e., a weighted average of losses of nearby reference samples, and apply multiple steps of “ neighborhood gradient descent (NGD) ” to minimize the surrogate.
In addition, we investigate kernel regression and mode finding, which do not require gradient de