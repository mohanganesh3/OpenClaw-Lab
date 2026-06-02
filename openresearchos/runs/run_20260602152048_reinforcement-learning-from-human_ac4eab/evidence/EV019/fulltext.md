[2502.14400] HPS: Hard Preference Sampling for Human Preference Alignment

HPS: Hard Preference Sampling for Human Preference Alignment

Xiandong Zou

Wanyu Lin

Yuchen Li

Pan Zhou

Abstract

Aligning Large Language Model (LLM) responses with human preferences is vital for building safe and controllable AI systems. While preference optimization methods based on Plackett-Luce (PL) and Bradley-Terry (BT) models have shown promise, they face challenges such as poor handling of harmful content, inefficient use of dispreferred responses, and, specifically for PL, high computational costs. To address these issues, we propose Hard Preference Sampling (HPS), a novel framework for robust and efficient human preference alignment. HPS introduces a training loss that prioritizes the most preferred response while rejecting all dispreferred and harmful ones. It emphasizes “hard” dispreferred responses—those closely resembling preferred ones—to enhance the model’s rejection capabilities. By leveraging a single-sample Monte Carlo sampling strategy, HPS reduces computational overhead while maintaining alignment quality. Theoretically, HPS improves sample efficiency over existing PL methods and maximizes the reward margin between preferred and dispreferred responses, ensuring clearer distinctions. Experiments on HH-RLHF and PKU-Safety datasets validate HPS’s effectiveness, achieving comparable BLEU and reward scores while greatly improving reward margins and thus reducing harmful content generation.

1  Introduction

Large Language Models (LLMs)  (Achiam et al.,  2023 ; Touvron et al.,  2023 ; Anil et al.,  2023 ; GLM et al.,  2024 )  have demonstrated exceptional capabilities across diverse user applications by leveraging the extensive global knowledge and behavioral patterns embedded in their massive pretraining corpora. However, the presence of misleading, toxic, and harmful content in these corpora poses significant risks, as LLMs can inadvertently propagate undesirable information  (Bai et al.,  2022b ; Yao et al.,  2024 ) . Consequently, selecting and aligning the model’s responses and behaviors with desired human values is crucial to developing safe, effective, and controllable AI systems  (Christiano et al.,  2017 ; Stiennon et al.,  2020 ; Ouyang et al.,  2022 ; Dai et al.,  2023 ) .

To achieve this alignment, several human preference alignment methods have been proposed. For example, Reinforcement Learning from Human Feedback (RLHF)  (Schulman et al.,  2017 ; Christiano et al.,  2017 )  optimizes LLMs by training a reward model on human preference rankings and maximizing the reward of generated outputs. Recognizing the complexity and sensitivity of RLHF, recent works, e.g., Direct Preference Optimization (DPO)  (Rafailov et al.,  2024 ) , Identity Preference Optimization (IPO)  (Azar et al.,  2024 )  and Self-Play Preference Optimization (SPPO)  (Wu et al.,  2024 ) , bypass the reward model by directly optimizing preferences, and have shown promising performance.

Despite their successes, existing methods for preference alignment often rely on underlying ranking models, such as the Plackett-Luce (PL) model  (Luce,  1959 ; Plackett,  1975 )  or its simplified counterpart, the Bradley-Terry (BT) model  (Bradley &amp; Terry,  1952 ) . The PL model ranks multiple responses to a prompt to align with human preferences, while the BT model focuses on pairwise comparisons. These models enable the derivation of training losses for alignment tasks. However, both PL- and BT-induced losses exhibit critical shortcomings when handling harmful responses.

Firstly, both PL- and BT-based losses fail to handle harmful responses effectively. The PL loss (e.g., DPO  (Rafailov et al.,  2024 )  and PRO  (Song et al.,  2024 ) ) encourages ranking less harmful responses above more malicious ones, inadvertently treating harmful outputs as “preferred” alternatives. This compromises the model’s ability to robustly reject inappropriate or offensive content—essential in tasks requiring strict safeguards. The BT loss (e.g., DPO  (Rafailov et al.,  2024 ) , R-DPO  (Park et al.,  2024 ) , Online DPO  (Dong et al.,  2024 ) , and KTO  (Ethayarajh et al.,  2024 ) ) focuses only on rejecting the most dispreferred response in a pair, leaving other problematic responses unaddressed.
Secondly, these losses overlook nuanced differences among dispreferred responses. The PL loss treats all dispreferred responses equally, ignoring their varying informativeness, which could guide better alignment learning. Similarly, the BT loss reduces rankings to pairwise comparisons, discarding macro-level distinctions that are crucial for capturing nuanced preferences  (Sun et al.,  2024 ; Song et al.,  2024 ) . Finally, computational inefficiency poses a significant challenge. Training with the PL loss requires processing and backpropagating through all responses in a ranked set, leading to substantial memory and computational overhead—especially for long prompts or responses  (Oosterhuis,  2021 ; Maystre &amp; Grossglauser,  2015 ; Sakhi et al.,  2023 ) . While the BT loss is more efficient, its simplifications sacrifice critical preference information. These limitations underscore the need for an improved preference alignment framework—one that robustly rejects harmful content, captures nuanced preferences, leverages the varying informativeness of responses, and achieves computational efficiency without compromising alignment quality.

Contributions.  We address these limitations by introducing a provably effective and efficient Hard Preference Sampling framework(HPS) for human preference alignment. Our key contributions are highlighted below.

Firstly, we introduce the HPS framework to enhance human preference alignment. Specifically, we first propose a training loss that fine-tunes LLMs to robustly prefer the most desired response while rejecting all dispreferred and potentially harmful ones. Moreover, HPS leverages insights from supervised, metric, and contrastive learning  (Schroff et al.,  2015 ; Oh Song et al.,  2016 ; Robinson et al.,  2020 ) , emphasizing the importance of “hard” examples—dispreferred responses closely resembling the preferred ones. Accordingly, HPS develops a hard preference sampling strategy to prioritize such hard examples, enabling the model to distinguish between preferred and highly similar dispreferred responses more effectively. To ensure efficiency, HPS is then reformulated into a sampling approach, using a single Monte Carlo sampling to select a single dispreferred response per training iteration. This innovation significantly reduces computational overhead compared to PL which requires all dispreferred responses for each prompt.

Secondly, HPS provably improves sample complexity over the vanilla PL loss. For a dataset

𝒟  \mathcal{D}

with

m  m

prompts and

n  n

responses per prompt, the distance between the optimum of the PL loss and the optimal human preference policy is bounded by

𝒪  ​

(

n  2

m

)

\mathcal{O}\big{(}\frac{n^{2}}{\sqrt{m}}\big{)}

which is improved to

𝒪  ​

(

n

m

)

\mathcal{O}\big{(}\frac{n}{\sqrt{m}}\big{)}

by using our HPS loss. This improvement ensures better preference alignment with fewer training samples, making HPS particularly advantageous in data-limited scenarios or when faster convergence is required.

Thirdly, we further prove that optimizing the HPS loss maximizes the reward margin – the gap between the most preferred response and the closest dispreferred one – for any given prompt. A high reward margin means less dispreferred or unethical generation. So this maximization ensures the LLM learns a robust distinction between preferred and dispreferred responses, leading to superior alignment with human preferences.

Finally, experimental results demonstrate that HPS outperforms state-of-the-arts (SoTAs) in both fine-tuning and transfer learning settings. On the HH-RLHF dataset  (Bai et al.,  2022a ) , HPS achieves comparable