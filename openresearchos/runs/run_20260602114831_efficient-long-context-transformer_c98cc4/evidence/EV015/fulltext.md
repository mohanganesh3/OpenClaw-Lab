[2602.01194] EMFormer: Efficient Multi-Scale Transformer for Accumulative Context Weather Forecasting

EMFormer: Efficient Multi-Scale Transformer for Accumulative Context

Weather Forecasting

Hao Chen

Tao Han

Jie Zhang

Song Guo

Fenghua Ling

Lei Bai

Abstract

Long-term weather forecasting is critical for socioeconomic planning and disaster preparedness.
While recent approaches employ finetuning to extend prediction horizons, they remain constrained by the issues of catastrophic forgetting, error accumulation, and high training overhead.
To address these limitations, we present a novel pipeline across pretraining, finetuning and forecasting to enhance long‑context modeling while reducing computational overhead. First, we introduce an Efficient Multi‑scale Transformer (EMFormer) to extract multi‑scale features through a single convolution in both training and inference. Based on the new architecture, we further employ an accumulative context finetuning to improve temporal consistency without degrading short‑term accuracy. Additionally, we propose a composite loss that dynamically balances different terms via a sinusoidal weighting, thereby adaptively guiding the optimization trajectory throughout pretraining and finetuning.
Experiments show that our approach achieves great performance in weather forecasting and extreme event prediction, substantially improving long-term forecast accuracy. Moreover, EMFormer demonstrates strong generalization on vision benchmarks (ImageNet-1K and ADE20K) while delivering a 5.69

×  \times

speedup over conventional multi-scale modules.

Weather Forecasting, Efficient Transformer, Typhoon Track Prediction

1  Introduction

Why do we need data-driven methods in weather forecasting? 
Long‑term weather forecasting is a critical challenge with significant socioeconomic implications, affecting sectors such as aviation, maritime navigation, and finance. Traditional Numerical Weather Prediction (NWP) generates forecasts by solving partial differential equations  (Bauer et al.,  2015 ; Lynch,  2008 ; Kalnay,  2002 ) , but it suffers from cumulative errors and high computation. In contrast, data‑driven models  (Kurth et al.,  2023 ; Bi et al.,  2023 ; Chen et al.,  2023a ; Bodnar et al.,  2025 )  learn atmospheric patterns directly from historical observations. By producing forecasts from learned representations rather than iterative physical integration, these approaches minimize error propagation and achieve greater computational efficiency.

(a)

Z500 Denormalized RMSE

↓  \downarrow

(6-hour prediction)

(b)

Z500 Denormalized RMSE

↓  \downarrow

(5-day prediction)

Figure 1 :

Denormalized Z500 RMSE (

m  2

/

s  2

m^{2}/s^{2}

) for short-term (6-hour) and medium-term (5-day) forecasts. (a) Training convergence comparison in 6-step finetuning:  accumulative context finetuning  and  previous finetuning . (b) Medium-term forecast performance: The  proposed method  consistently outperforms  VA-MoE  across pretraining and multi-step finetuning (6, 8, 10 steps). Two models are trained by us with A100.

Why is an Efficient Architecture Essential for Finetuning?  While data-driven methods outperform NWP by capturing atmospheric dynamics, they remain prone to stepwise error accumulation during auto-regressive forecasting. To enhance long-term accuracy, many approaches employ extended finetuning on multi-step sequences. Although this improves forecast horizons, it introduces two critical limitations: (1) as inference length increases, the model gradually forgets information from earlier steps ( blue line  in

Fig.

1(a)  ), and (2) longer finetuning sequences demand greater training time and computational resources ( blue line  in

Fig.

1(b)  ). To address the first issue, we introduce accumulative context finetuning, which explicitly preserves historical information to ensure temporal consistency.
Yet, this technique inevitably exacerbates the computational burden. Consequently, it is crucial to design an efficient framework that reduces computational cost during training and finetuning.

To address the efficiency and stability bottlenecks in long-context forecasting, we propose  a novel pipeline  comprising pretraining, finetuning, and forecasting. During pretraining, we introduce a hierarchical pruning-recovering framework coupled with an Efficient Multi‑scale Transformer (EMFormer) to lower computational cost. For finetuning, an accumulative context mechanism is employed to strengthen long‑horizon representations, ensuring temporal consistency across multi‑step predictions. To further refine optimization, we design a variable‑ and geography‑aware loss that adapts to the inherent heterogeneity of atmospheric variables across both physical properties and geographical regions.

We make three core contributions: (i)  Efficient Multi-Scale Architecture via Hard-aware Design. 
While multi-scale transformers are effective, their training costs are often prohibitive.
To address this, we propose EMFormer, which integrates a novel multi-convolution (multi-convs) layer optimized with custom CUDA kernels. Unlike standard re-parameterization methods that only accelerate inference, our method enables multi-scale feature capture via a single convolution during both training and inference.
This redesign preserves representational power while accelerating forward and backward passes by 5.69

×  \times

compared to traditional multi-scale implementations. (ii)  Accumulative Finetuning for Long-context Consistency.  We introduce a specialized finetuning strategy tailored for long-context weather forecasting.
By injecting historical Key–Value (KV) pairs into current generation steps and employing a memory-pruning mechanism, we explicitly bound memory usage while strengthening long-term temporal dependencies.
This ensures sustained accuracy over extended horizons without compromising short-term performance.
(iii)  Sinusoidal Weighted Optimization Objective.  We design a composite objective function featuring a sinusoidal weighting mechanism to address the heterogeneity of atmospheric data. This includes a latitude-adaptive term, which accounts for spatial distortion and a variable-adaptive term that balances learning dynamics across distinct physical variables, ensuring robust optimization across diverse atmospheric conditions.

In addition to theoretical analysis, experiments are presented in

Fig.

1  .

Fig.

1(a)

plots the denormalized Z500 RMSE (

m  2

/

s  2

m^{2}/s^{2}

) curves for 6‑hour forecasts under two finetuning strategies. While conventional finetuning leads to progressively deteriorating first‑step RMSE, the proposed accumulative finetuning steadily reduces the error, demonstrating superior stability in short‑term forecasts.

Fig.

1(b)

compares the 5‑day predictions between VA‑MoE  (Chen et al.,  2025b )  and Ours with distinct finetuning steps. Our approach achieves lower RMSE with shorter finetuning period: at Step 10, it reaches about 280

m  2

/

s  2

m^{2}/s^{2}

after 210 GPU‑days, whereas VA‑MoE requires 430 GPU‑days to attain 295

m  2

/

s  2

m^{2}/s^{2}

. Although both methods improve with extended finetuning, our method outperforms VA‑MoE with fewer GPU‑days. Beyond atmosphere, EMFormer also delivers competitive performance on vision tasks such as classification and segmentation, surpassing existing methods.

2  Methodology

Figure 2 :

Illustration of the novel pipeline with three stages.  Stage 1:  EMFormer is pretrained on atmospheric variables with pruning-recovering architecture that includes a pruning module, a series of EMFormer blocks, and a recovering module;  Stage 2:  accumulative context finetuning;  Stage 3:  The forecasting stage with weather forecasting and typhoon track prediction.

This section presents a novel pipeline for weather forecasting, comprising: (1) single‑step pretraining, (2) multi‑step finetuning, and (3) multi‑step forecasting. The forecasting task is defined and t