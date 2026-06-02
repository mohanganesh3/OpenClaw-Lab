[2602.14078] 1 Introduction

Policy Gradient with Adaptive Entropy Annealing for Continual Fine-Tuning

Yaqian Zhang  1

Bernhard Pfahringer  1

Eibe Frank  1

Albert Bifet  1,2

1. AI Institute, University of Waikato

2. LTCI, Télécom Paris

Abstract

Despite their success, large pretrained vision models remain vulnerable to catastrophic forgetting when adapted to new tasks in class-incremental settings. Parameter-efficient fine-tuning (PEFT) alleviates this by restricting trainable parameters, yet most approaches still rely on cross-entropy (CE) loss—a surrogate for the 0–1 loss—to learn from new data. We revisit this choice and revive the true objective (0-1 loss) through a reinforcement learning perspective. By formulating classification as a one-step Markov Decision Process, we derive an Expected Policy Gradient (EPG) method that directly minimizes misclassification error with a low-variance gradient estimation. Our analysis shows that CE can be interpreted as EPG with an additional sample-weighting mechanism: CE encourages exploration by emphasizing low-confidence samples, while EPG prioritizes high-confidence ones. Building on this insight, we propose adaptive entropy annealing (aEPG), a training strategy that transitions from exploratory (CE-like) to exploitative (EPG-like) learning. aEPG-based methods outperform CE-based methods across diverse benchmarks and with various PEFT modules. More broadly, we evaluate various entropy regularization methods and demonstrate that lower entropy of the output prediction distribution enhances adaptation in pretrained vision models.
The source code is provided in the supplementary materials.

1  Introduction

Modern vision models remain vulnerable to catastrophic forgetting when trained on non-stationary data. Traditional continual learning (CL) methods mitigate this challenge through techniques like memory replay  (Delange  et al. ,  2021 ; Zhang  et al. ,  2022 ; Van de Ven  et al. ,  2020 ; Tiwari  et al. ,  2022 ) , regularization  (Buzzega  et al. ,  2020 ; Rebuffi  et al. ,  2017 ; Kirkpatrick  et al. ,  2017 ) , or parameter isolation  (Xu and Zhu,  2018 ; Mallya and Lazebnik,  2018 ; Hung  et al. ,  2019 ) . With the advent of large pretrained vision transformers, parameter-efficient fine-tuning (PEFT) has emerged as an effective way to substantially reduce forgetting. By restricting the number of trainable parameters, PEFT methods achieve state-of-the-art CL performance without relying on replay  (Wang  et al. ,  2022b ,  a ; Smith  et al. ,  2023 ) .

Recent studies extend PEFT to CL using prompts  (Wang  et al. ,  2022a ,  b ) , LoRA  (Liang and Li,  2024 ; Liu and Chang,  2025 ) , and adapters  (Gao  et al. ,  2024 ; Yu  et al. ,  2024 ; Wang  et al. ,  2025 ; Ermis  et al. ,  2022 ) . On the stability side (i.e., retaining knowledge from past tasks), many techniques such as EMA-based fast–slow learning and orthogonal subspace constraints with LoRA have been proposed  (Gao  et al. ,  2023 ; Liang and Li,  2024 ) . However, on the plasticity side (i.e., acquiring new knowledge), these methods almost universally rely on cross-entropy loss as the default objective for learning from new data. In this work, we ask whether CE is truly desirable for continual learning. We investigate this question from the perspective of the true classification objective (0–1 loss) and through the lens of entropy.

We begin by reformulating continual learning through a reinforcement learning (RL) lens. The ultimate goal of classification is to minimize the misclassification error (0-1 loss). However, since this loss is non-differentiable, CE loss has become the de facto surrogate for training deep models, even though the trained models are eventually evaluated on 0-1 loss. We instead cast classification as a one-step Markov Decision Process (MDP), which yields an RL objective equivalent to minimizing misclassification error. Based on this formulation, we introduce  Expected Policy Gradient  (EPG), a low-variance variant of REINFORCE  (Williams,  1992 ) , to directly optimize 0–1 loss.

To reveal CE’s limitations, we conduct a comparative analysis of CE and EPG in terms of gradient behavior and entropy dynamics. In parameter space, both share the same gradient direction per sample, but CE implicitly reweights samples—prioritizing harder, high-surprisal cases. This encourages adaptation but risks destabilizing prior knowledge, as the model must correct large prediction errors for those hard samples. By contrast, EPG is more  exploitative , emphasizing easier samples that already align with the model’s predictions. In the action space, EPG consistently produces lower-entropy output distributions than CE. We hypothesize that CE’s excessive exploration can cause significant deviation from pretrained weights, thereby exacerbating forgetting. This insight highlights the need to carefully balance exploration and exploitation in continual fine-tuning.

To this end, we propose an  adaptive entropy annealing strategy  (aEPG), which interpolates between CE (exploration) and EPG (exploitation) through a time-dependent weighting scheme. aEPG begins with CE to encourage plasticity and gradually shifts toward EPG to preserve stability. Empirically, aEPG consistently improves performance across four CL benchmarks and multiple PEFT architectures (LoRA, Adapter, and Prefix) (see Table

1

and

2  ).

Finally, we broaden our study to entropy regularization in continual fine-tuning. While prior work advocates high-entropy training (e.g., label smoothing, focal loss, confidence penalties) for classification, we observe that such approaches harm class-incremental learning with pretrained models. In contrast, low-entropy objectives consistently improve continual adaptation (Table

3  ),
suggesting that exploitative learning is more effective than aggressive exploration for continual learning with foundation models.

Our contributions are summarized as follows:

•

We introduce EPG, a policy gradient method that directly optimizes 0–1 loss instead of surrogate objectives like CE.

•

We provide theoretical and empirical evidence contrasting CE’s exploration bias with EPG’s exploitative bias in parameter space and action space.

•

We propose an adaptive entropy annealing strategy (aEPG) that balances exploration and exploitation, improving class-incremental learning.

•

We conduct, to our knowledge, the first systematic investigation into entropy regularization for continual learning with PEFT, uncovering the critical role of entropy reduction.

2  Related Work

Continual Learning and Parameter-Efficient Fine-tuning
(PEFT)  Traditional continual learning (CL) methods address forgetting in the train-from-scratch setting through strategies such as penalizing changes to important parameters via regularization losses  (Li and Hoiem,  2017 ; Kirkpatrick  et al. ,  2017 ) , constraining optimization with orthogonal gradients  (Farajtabar  et al. ,  2020 ; Chaudhry  et al. ,  2020 ; Guo  et al. ,  2022 ) , replaying past samples  (Buzzega  et al. ,  2020 ; Zhang  et al. ,  2022 ) , or isolating parameters for different tasks  (Mallya and Lazebnik,  2018 ; Lee  et al. ,  2023 ; Wang  et al. ,  2023 ) .
With the advent of large pretrained transformers, PEFT techniques have achieved state-of-the-art CL performance by restricting the number of trainable parameters. Early advances in continual PEFT, such as L2P, DualPrompt, and CodaPrompt  (Wang  et al. ,  2022b ,  a ; Smith  et al. ,  2023 ) , introduced learnable prompt parameters maintained in memory. These methods optimize prompts to guide model predictions while explicitly managing task-invariant and task-specific knowledge. More recent work has proposed unified frameworks, which integrate adapters  (Sung  et al. ,  2022 ) , LoRA  (Hu  et al. ,  2022 ) , and prefix tuning  (Le  et al. ,  2025 ) , as well as ensemble-based methods like LAE  (Gao  et al. ,  2023 )  that combine fast and slow lear