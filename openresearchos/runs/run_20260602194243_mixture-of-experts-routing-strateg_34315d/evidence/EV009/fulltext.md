[2604.07210] VersaVogue: Visual Expert Orchestration and Preference Alignment for Unified Fashion Synthesis

VersaVogue: Visual Expert Orchestration and Preference Alignment for Unified Fashion Synthesis

DOI:

XXXXXXX.XXXXXXX

Conference:  ACM Multimedia; November 10–14, 2026; Rio de Janeiro, Brazil

CCS:  Computing methodologies Computer vision

Jian Yu

Affiliation:

Nanjing University of Science and Technology

,

Fei Shen

Note:  Project Lead

Affiliation:

National University of Singapore

,

Cong Wang

Affiliation:

Nanjing University

,

Yi Xin

Affiliation:

Nanjing University

,

Si Shen

Affiliation:

Nanjing University of Science and Technology

,

Xiaoyu Du

Affiliation:

Nanjing University of Science and Technology

and

Jinhui Tang

Affiliation:

Nanjing Forestry University

(2026© , 2026; )

Abstract.

Diffusion models have driven remarkable advancements in fashion image generation, yet prior works usually treat garment generation and virtual dressing as separate problems, limiting their flexibility in real-world fashion workflows. Moreover, fashion image synthesis under multi-source heterogeneous conditions remains challenging, as existing methods typically rely on simple feature concatenation or static layer-wise injection, which often causes attribute entanglement and semantic interference.
To address these issues, we propose VersaVogue, a unified framework for multi-condition controllable fashion synthesis that jointly supports garment generation and virtual dressing, corresponding to the design and showcase stages of the fashion lifecycle. Specifically, we introduce a trait-routing attention (TA) module that leverages a mixture-of-experts mechanism to dynamically route condition features to the most compatible experts and generative layers, enabling disentangled injection of visual attributes such as texture, shape, and color.
To further improve realism and controllability, we develop an automated multi-perspective preference optimization (MPO) pipeline that constructs preference data without human annotation or task-specific reward models. By combining evaluators of content fidelity, textual alignment, and perceptual quality, MPO identifies reliable preference pairs, which are then used to optimize the model via direct preference optimization (DPO).
Extensive experiments on both garment generation and virtual dressing benchmarks demonstrate that VersaVogue consistently outperforms existing methods in visual fidelity, semantic consistency, and fine-grained controllability.

Keywords:  Fashion image synthesis; Virtual dressing; Garment generation; Multi-conditional generation

Figure 1.  Differences in workflows and input conditions between virtual try-on methods and our VersaVogue. (a) Try-on methods are restricted to the showcase stage and strictly rely on an explicit model image. (b) VersaVogue establishes a unified workflow from garment design to virtual dressing, synthesizing on-body results directly from text descriptions.

1.  Introduction

Recent advances in generative models  ( podell2023sdxl ;  rombach2022high ;  peebles2023scalable )  have significantly expanded the potential of high-fidelity image synthesis for fashion applications. Among them, garment generation aims to synthesize detailed apparel designs from multimodal conditions, while virtual dressing focuses on rendering photorealistic on-body showcases. These two tasks correspond to the design and showcase stages of the fashion lifecycle, respectively. A unified framework that bridges them would not only streamline traditional fashion workflows by reducing manual effort and production cost, but also improve digital retail through more flexible and personalized visual content.

Despite their close connection, existing studies address garment generation and virtual dressing as separate problems using fragmented, task-specific pipelines  ( zhu2024logosticker ;  kim2024stableviton ;  xu2025ootdiffusion ) , as illustrated in Figure

1

(a). For example, text-driven garment generation methods such as DiffCloth  ( zhang2023diffcloth )  rely on structural priors to anchor semantic attributes, while virtual dressing methods such as IMAGDressing  ( shen2025imagdressing )  and StableGarment  ( wang2024stablegarment )  design specialized modules,  e.g. , parallel UNets or hybrid attention, to inject garment references. Although effective for their individual settings, these approaches lack a unified mechanism for handling both tasks under a common formulation.

A key obstacle to such unification lies in the need to process complex multi-source heterogeneous conditions while preserving fine-grained controllability. Since both garment generation and virtual dressing fundamentally require accurate attribute preservation under diverse visual and textual constraints, we reformulate them as a unified multi-condition controllable generation problem. However, existing general-purpose conditioning strategies still suffer from inherent limitations. Some methods  ( ye2023ip ;  mou2024t2i )  employ lightweight adapters with sequential stacking, which often capture only superficial semantics and fail to preserve high-fidelity consistency with the input conditions. Other approaches based on static layer-wise injection or direct feature concatenation  ( lin2025dreamfit ;  zhao2023uni )  lack the flexibility to coordinate heterogeneous conditions, frequently leading to attribute entanglement, semantic interference, and unsatisfactory synthesis quality. Even more specialized solutions such as IMAGGarment  ( shen2025imaggarment )  rely on multi-stage pipelines to decouple attributes, sacrificing efficiency and end-to-end coherence.

To address these challenges, we propose  VersaVogue , a unified framework for fashion synthesis under complex multi-source heterogeneous conditions from Figure

1

(b).
The core idea is to replace static condition injection with dynamic expert orchestration and further improve generation quality through automated preference alignment. Specifically, we introduce a trait-routing attention (TA) module built upon a mixture-of-experts (MoE) mechanism. Given multiple condition inputs, TA dynamically routes attribute-aware features to the most compatible layers and experts in the generative model, enabling more precise disentanglement and adaptive injection of visual traits such as texture, shape, and color. In this way, VersaVogue mitigates feature conflicts among heterogeneous inputs while maintaining a unified end-to-end inference pipeline.

In addition, high controllability alone does not guarantee perceptual realism or alignment with human preference. Recent direct preference optimization (DPO) methods  ( rafailov2023direct ;  wallace2024diffusion )  provide an effective way to refine generation behavior, but they typically depend on human-annotated preference data, which is costly to collect and inevitably subjective. We therefore develop a fully automated multi-perspective preference optimization (MPO) pipeline. By jointly evaluating candidate samples from the perspectives of content fidelity, textual alignment, and aesthetic quality, MPO constructs reliable preference pairs without human annotation or task-specific reward models. Based on these automatically curated preferences, we apply DPO to align the model toward outputs with higher realism, stronger semantic consistency, and better controllability.
Extensive experiments show that VersaVogue consistently outperforms existing methods in both garment generation and virtual dressing, achieving superior image realism and fine-grained controllable precision. Our contributions are summarized as follows:

•

We propose VersaVogue, a unified fashion synthesis framework that seamlessly bridges garment generation and virtual dressing under a common multi-condition controllable formulation.

•

We introduce a trait-routing attention (TA) module based on mixture-of-experts routing, which dynamically orchestrates het