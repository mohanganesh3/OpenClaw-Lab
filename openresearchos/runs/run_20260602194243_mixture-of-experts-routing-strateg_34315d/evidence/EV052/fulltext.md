[2509.26520] Training Matryoshka Mixture-of-Experts for Elastic Inference-Time Expert Utilization

Training Matryoshka Mixture-of-Experts for Elastic Inference-Time Expert Utilization

Yaoxiang Wang  1

This work is done during their internships in MSRA.

Qingguo Hu  1

1

1  footnotemark:

1

Yucheng Ding  2

1

1  footnotemark:

1

Ruizhe Wang  3

1

1  footnotemark:

1

Yeyun Gong  4

Corresponding author

Jian Jiao  4

Yelong Shen  4

Peng Cheng  4

Jinsong Su  1

2

2  footnotemark:

2

1  Xiamen University

2  Shanghai Jiao Tong University

3  University of Science and Technology of China

4  Microsoft

Abstract

Mixture-of-Experts (MoE) has emerged as a promising paradigm for efficiently scaling large language models without a proportional increase in computational cost. However, the standard training strategy of Top-K router prevents MoE models from realizing their full potential for elastic inference. When the number of activated experts is altered at inference time, these models exhibit precipitous performance degradation. In this work, we introduce Matryoshka MoE (M-MoE), a training framework that instills a coarse-to-fine structure directly into the expert ensemble. By systematically varying the number of activated experts during training, M-MoE compels the model to learn a meaningful ranking: top-ranked experts collaborate to provide essential, coarse-grained capabilities, while subsequent experts add progressively finer-grained detail. We explore this principle at multiple granularities, identifying a layer-wise randomization strategy as the most effective. Our experiments demonstrate that a single M-MoE model achieves remarkable elasticity, with its performance at various expert counts closely matching that of an entire suite of specialist models, but at only a fraction of the total training cost. This flexibility not only unlocks elastic inference but also enables optimizing performance by allocating different computational budgets to different model layers. Our work paves the way for more practical and adaptable deployments of large-scale MoE models.

1  Introduction

The landscape of artificial intelligence is increasingly dominated by large-scale models

[OpenAI,  2023 , DeepSeek-AI,  2025 , google,  2025 ]  , whose unprecedented capabilities  [Scale-AI,  2025 ]  are often shadowed by their immense computational cost. This has given rise to a critical need for elastic inference  [Cai et al.,  2024 ] : the ability of a single model to dynamically adapt its computational footprint to meet diverse user requirements.

A prominent and successful paradigm in this domain is Matryoshka Representation Learning (MRL) [Kusupati et al.,  2022 ]  and its architectural derivatives [Devvrit et al.,  2024 , GemmaTeam,  2025 ] . MRL addresses a fundamental inefficiency in deep learning: standard models tend to diffuse information evenly across their entire representation vectors, which makes smaller, truncated representations ineffective. MRL directly counteracts this by instilling a structured, coarse-to-fine granularity within a single high-dimensional embedding. The training objective is applied not only to the full representation but also to its nested, truncated prefixes. This forces the model to prioritize and pack the most critical, high-level information into the initial dimensions, with subsequent dimensions progressively adding finer-grained detail.

While MRL explicitly instills a Matryoshka structure within a single representation, Mixture-of-Experts (MoE) architectures  [Shazeer et al.,  2017 ]  present an innate structural potential for the same principle. As the leading paradigm for scaling models to billions of parameters at a manageable computational cost  [Jiang et al.,  2024 ] , MoE routes each token through a small subset of expert sub-networks. Instead of adapting model depth or representation dimension, MoE’s sparse architecture naturally suggests adapting its width—the number of concurrently active experts. The intuition is powerful: at inference time, one could simply select fewer experts for a coarse but fast prediction, or more experts for a fine-grained, higher-quality output, effectively creating a "Matryoshka MoE".

However, our empirical investigation into publicly available MoE models reveals a counter-intuitive reality. As shown in figure

1  , increasing the number of activated experts yields minimal performance gains, while reducing it leads to sharp performance degradation. This finding directly contradicts the prevailing intuition and exposes a fundamental brittleness in current MoE models, suggesting they are incapable of delivering on the promise of true inference-time elasticity.
This performance collapse stems from the inherent rigidity of the fixed Top-K training paradigm. During training, each expert becomes overly specialized in collaborating with a fixed-size group of peers. this paradigm causes a problem analogous to information diffusion: expert capacity becomes rigidly co-adapted to a fixed-size group, and the router’s ranking ability is only meaningful for the top K. Any deviation disrupts this delicate balance.

Figure 1:  MMLU score of DeepSeek-V2-Lite, Qwen3-30B-A3B-Base, and RedNote-Dots.LLM1.Base under varying numbers of activated experts.

To overcome this critical limitation and unlock the true potential of MoE for elastic inference, we propose Matryoshka MoE (M-MoE), a simple yet effective training strategy. The core idea of M-MoE is to instill a coarse-to-fine granularity within the MoE’s expert routing mechanism.
We explore this principle at different granularities, ranging from randomizing the expert count for an entire global batch to our most effective strategy: a layer-wise approach where each Transformer layer independently selects a different number of experts. This fine-grained stochasticity forces experts to differentiate their contributions, with fewer activated experts collaboratively providing essential, coarse-grained information, and additional experts progressively adding finer-grained detail, thereby fostering a more versatile model.

Our experiments demonstrate that a single M-MoE model can achieve remarkable inference-time elasticity, delivering performance that is comparable to multiple specialist models, each trained individually for a specific expert count. The analysis of the router’s internal mechanics reveals that M-MoE not only teaches the gating network to produce a globally coherent and stable ranking of experts, but also fosters a higher degree of expert specialization. This is in stark contrast to the brittle rankings and greater functional overlap among experts observed in fixed-k models. Furthermore, the inherent flexibility of our M-MoE model unlocks novel analytical possibilities. We investigate the performance impact of allocating different numbers of experts to different layers during inference, providing valuable insights for future elastic deployment strategies, which is impossible with rigidly trained MoE models.

Our main contributions are as follows:

•

We identify the rigidity of fixed-k training as a key barrier to elastic MoE inference and propose Matryoshka MoE, a framework that instills a coarse-to-fine functional hierarchy within the expert ensemble.

•

We empirically demonstrate that our layer-wise M-MoE strategy is highly effective, producing a single, elastic model that rivals the performance of an entire suite of specialist models at a fraction of the training cost.

•

Through detailed analysis, we show that M-MoE induces stable expert rankings and functional specialization, unlocking the ability to analyze and deploy novel layer-wise inference strategies.

2  Preliminary

In this section, we provide an overview of the Mixture-of-Experts (MoE) architecture and its standard routing mechanism, which form the foundation of our work.

2.1  Mixture-of-Expert Transformers

The Mixture-of-Experts (MoE) paradigm is a powerful architectural inno