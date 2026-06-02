# EV015: EMFormer: Efficient Multi-Scale Transformer for Accumulative Context Weather Forecasting

URL: https://www.semanticscholar.org/paper/03952a2bba9ce382a5206a2623752f80390c318d
Year: 2026
Source: semantic_scholar
Arxiv: 2602.01194

## Abstract

Long-term weather forecasting is critical for socioeconomic planning and disaster preparedness. While recent approaches employ finetuning to extend prediction horizons, they remain constrained by the issues of catastrophic forgetting, error accumulation, and high training overhead. To address these limitations, we present a novel pipeline across pretraining, finetuning and forecasting to enhance long-context modeling while reducing computational overhead. First, we introduce an Efficient Multi-scale Transformer (EMFormer) to extract multi-scale features through a single convolution in both training and inference. Based on the new architecture, we further employ an accumulative context finetuning to improve temporal consistency without degrading short-term accuracy. Additionally, we propose a composite loss that dynamically balances different terms via a sinusoidal weighting, thereby adaptively guiding the optimization trajectory throughout pretraining and finetuning. Experiments show that our approach achieves strong performance in weather forecasting and extreme event prediction, substantially improving long-term forecast accuracy. Moreover, EMFormer demonstrates strong generalization on vision benchmarks (ImageNet-1K and ADE20K) while delivering a 5.69x speedup over conventional multi-scale modules. Code: https://github.com/chenhao-zju/emformer
