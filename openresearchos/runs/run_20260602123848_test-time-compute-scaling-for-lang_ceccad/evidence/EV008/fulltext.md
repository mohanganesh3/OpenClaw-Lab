[2603.26846] Stable Reasoning, Unstable Responses: Mitigating LLM Deception via Stability Asymmetry

Stable Reasoning, Unstable Responses:

Mitigating LLM Deception via Stability Asymmetry

Guoxi Zhang 1,2 ,
Jiawei Chen 1 ,
Tianzhuo Yang 1 ,
Lang Qin 3 ,
 Juntao Dai 1,2 ,
Yaodong Yang 1,2

†  ,
Jingwei Yi 2

†

1 Institute for Artificial Intelligence, Peking University

2 Beijing Academy of Artificial Intelligence

3 School of Chinese as a Second Language, Peking University

This work was completed during an internship at the Beijing Academy of Artificial Intelligence (BAAI).

†

Corresponding author’s emails: jwyi1029@gmail.com, yaodong.yang@pku.edu.cn.

Abstract

As Large Language Models (LLMs) expand in capability and application scope, their trustworthiness becomes critical.
A vital risk is intrinsic deception, wherein models strategically mislead users to achieve their own objectives.
Existing alignment approaches based on chain-of-thought (CoT) monitoring supervise explicit reasoning traces.
However, under optimization pressure, models are incentivized to conceal deceptive reasoning, rendering semantic supervision fundamentally unreliable.
Grounded in cognitive psychology, we hypothesize that a deceptive LLM maintains a stable internal belief in its CoT while its external response remains fragile under perturbation.
We term this phenomenon stability asymmetry and quantify it by measuring the contrast between internal CoT stability and external response stability under perturbation.
Building on this structural signature, we propose the Stability Asymmetry Regularization (SAR), a novel alignment objective that penalizes this distributional asymmetry during reinforcement learning.
Unlike CoT monitoring, SAR targets the statistical structure of model outputs, rendering it robust to semantic concealment.
Extensive experiments confirm that stability asymmetry reliably identifies deceptive behavior, and that SAR effectively suppresses intrinsic deception without degrading general model capability.

Stable Reasoning, Unstable Responses:

Mitigating LLM Deception via Stability Asymmetry

Guoxi Zhang 1,2

†

†  thanks:  This work was completed during an internship at the Beijing Academy of Artificial Intelligence (BAAI).

†

Corresponding author’s emails: jwyi1029@gmail.com, yaodong.yang@pku.edu.cn.

,
Jiawei Chen 1 ,
Tianzhuo Yang 1 ,
Lang Qin 3 ,

Juntao Dai 1,2 ,
Yaodong Yang 1,2

†  ,
Jingwei Yi 2

†

1 Institute for Artificial Intelligence, Peking University

2 Beijing Academy of Artificial Intelligence

3 School of Chinese as a Second Language, Peking University

1  Introduction

As Large Language Models (LLMs) advance in capability and application scope, ensuring their trustworthiness is a central challenge.
A critical risk is deception, wherein models strategically produce misleading outputs to achieve specific objectives  Wu  et al.  ( 2026 ); Chen  et al.  ( 2025 ) .
Depending on whether such objectives are explicitly specified in the prompt or arise intrinsically within the model, deception can be divided into promptive and intrinsic deception  Hagendorff ( 2024 ) .
Unlike promptive deception, intrinsic deception manifests even in benign contexts, subtly compromising the integrity of high-stakes outputs in domains such as scientific research or clinical diagnostics.

Figure 1:  Conceptual illustration of  Stability Asymmetry . A deceiver maintains a consistent internal belief while providing a conflicting external response, leading to cue leakage under perturbation.

Despite the severe threat posed by intrinsic deception, existing alignment methods against intrinsic deception remain fundamentally limited.
A prominent line of work applies CoT monitoring during RLHF, intervening by supervising explicit reasoning traces  Baker  et al.  ( 2025 ) .
However, under optimization pressure, models are incentivized to obscure their deceptive intent within the reasoning trace, substantially reducing the observability of deception and undermining the reliability of semantic supervision  Baker  et al.  ( 2025 ); Korbak  et al.  ( 2025 ) .
This critical vulnerability motivates our core research question:

Can intrinsic deception be mitigated through a robust signal that bypasses semantic supervision?

Cognitive psychology  Zuckerman  et al.  ( 1981 )  and philosophy  Bok ( 2011 )  establish that human deception is structurally asymmetric.
As illustrated in

Figure 1  , a deceiver maintains a consistent internal belief while providing a conflicting external response.
This inconsistency leads to cue leakage, where the external output becomes unstable when subjected to minor perturbations.
Drawing on this cognitive mechanism, we hypothesize that an analogous asymmetry arises in LLMs, where a deceptive model maintains a stable internal belief in its CoT while its external response becomes susceptible to perturbation.
We term this phenomenon  stability asymmetry  and quantify it by applying perturbation-based stability metrics independently to CoT and final response.
We empirically validate that intrinsic deception uniquely exhibits high internal stability paired with low external stability, a signature that cleanly separates it from truthfulness and hallucination.

Building on these findings, we propose Stability Asymmetry Regularization (SAR) as a novel defense mechanism that operationalizes the observed distributional asymmetry as an alignment signal.
Rather than regulating the semantic content of reasoning traces  Wang  et al.  ( 2025a ) , our approach penalizes the specific statistical pattern of high internal stability co-occurring with low external stability during optimization.
This formulation directly exploits the structural signature identified in our measurements  Huang  et al.  ( 2025 ) .
By targeting the fundamental distributions of model outputs under perturbation rather than their surface semantics, SAR effectively circumvents the failure mode where models evade alignment by obscuring their deceptive intent within the CoT.

Our contributions are summarized as follows:

•

Grounded in cognitive psychology, we formalize stability asymmetry as a distinctive structural signature of intrinsic deception and empirically demonstrate that truthfulness, hallucination, and deception are cleanly separable within a two-dimensional stability space.

•

We introduce Stability Asymmetry Regularization (SAR), an approach that reorients the alignment paradigm from fragile semantic supervision toward robust distributional consistency, preventing models from concealing deceptive intent within the reasoning trace.

•

Extensive experiments demonstrate the proposed approach effectively suppresses intrinsic deception without compromising the model’s general capability, validating Stability Asymmetry as a practical and reliable alignment signal.

2  Related Work

Deception in LLMs

Deception in LLMs refers to behavior that functionally induces false beliefs and yields outcomes aligned with the system’s objectives  Chen  et al.  ( 2025 ); Ward  et al.  ( 2023 ); Ji  et al.  ( 2025 ) . Under this definition, sycophancy  Sharma  et al.  ( 2024 ); Wang  et al.  ( 2025b ); Huan  et al.  ( 2025 ) , strategic deception  Hagendorff ( 2024 ) , alignment faking  Greenblatt  et al.  ( 2024 ) , and sandbagging  van der Weij  et al.  ( 2024 )  are all instances of deceptive behavior. Existing studies typically distinguish  prompt-induced  deception  Chern  et al.  ( 2024 ); Huang  et al.  ( 2025 ); Ren  et al.  ( 2025 )  from  intrinsic  deception  Wu  et al.  ( 2026 ); Ward  et al.  ( 2023 ) .  Hubinger  et al.  ( 2024 )  and  Greenblatt  et al.  ( 2024 )  further show that models can conceal intent and comply selectively. Our work focuses on intrinsic deception during optimization, which may transfer to broader sabotage-like behavior  MacDiarmid  et al.  ( 2025 ) .

Cognitive and Physical Properties of Deception

Cognitive psychology and philosophy propose that deception involv