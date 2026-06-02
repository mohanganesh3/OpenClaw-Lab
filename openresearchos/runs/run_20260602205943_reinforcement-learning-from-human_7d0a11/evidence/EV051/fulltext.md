[2602.14069] Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric

Open Rubric System: Scaling Reinforcement Learning with Pairwise Adaptive Rubric

Ruipeng Jia

Qwen Large Model Application Team, Alibaba

Yunyi Yang

Qwen Large Model Application Team, Alibaba

Yuxin Wu

Beijing University Of Posts and Telecommunications

Yongbo Gai

Qwen Large Model Application Team, Alibaba

Siyuan Tao

Institute of Computing Technology, Chinese Academy of Sciences

Mengyu Zhou

Qwen Large Model Application Team, Alibaba

Jianhe Lin

Qwen Large Model Application Team, Alibaba

Xiaoxi Jiang

Qwen Large Model Application Team, Alibaba

Guanjun Jiang

Qwen Large Model Application Team, Alibaba

Abstract

Scalar reward models compress multi-dimensional human preferences into a single opaque score, creating an information bottleneck that often leads to brittleness and reward hacking in open-ended alignment.
We argue that robust alignment for non-verifiable tasks is fundamentally a  principle generalization  problem: reward should not be a learned function internalized into a judge, but an explicit reasoning process executed under inspectable principles.
To operationalize this view, we present the  Open Rubric System (OpenRS) , a plug-and-play, rubrics-based  LLM-as-a-Judge  framework built around  Pairwise Adaptive Meta-Rubrics (PAMR)  and lightweight  Pointwise Verifiable Rubrics  (PVRs), which provide both hard-constraint guardrails and verifiable reward components when ground-truth or programmatic checks are available.
OpenRS uses an explicit  meta-rubric —a constitution-like specification that governs how rubrics are instantiated, weighted, and enforced—and instantiates  adaptive rubrics  on the fly by conditioning on the semantic differences between two candidate responses.
It then performs criterion-wise  pairwise  comparisons and aggregates criterion-level preferences externally, avoiding pointwise weighted scalarization while improving discriminability in open-ended settings.
To keep principles consistent yet editable across various domains, we introduce a two-level meta-rubric refinement pipeline (automated evolutionary refinement for general principles and a reproducible human-in-the-loop procedure for domain principles), complemented with pointwise verifiable rubrics that act as both guardrails against degenerate behaviors and a source of verifiable reward for objective sub-tasks.
Finally, we instantiate OpenRS as reward supervision in pairwise RL training.
Empirically, OpenRS achieves state-of-the-art results on four reward-modeling benchmarks (RM-Bench, JudgeBench, RewardBench v2, and PPE Preference),
consistently outperforming strong open scalar reward model baselines.
In end-to-end policy optimization, replacing a scalar reward model with OpenRS yields stable gains on downstream RL evaluations.
The open-source code is available at  https://github.com/Qwen-Applications/OpenRS .

1  Introduction

Reinforcement learning from human feedback (RLHF) has become a standard paradigm for aligning large language models (LLMs) to follow instructions and exhibit helpful behaviors  ( ouyang2022training ) .
Typically, RLHF pipelines rely on a scalar reward model (RM) that compresses rich, multi-faceted human preferences into a single number.
While in practice, scalar RMs may learn shortcuts and become miscalibrated,
creating opportunities for reward hacking where policies exploit spurious cues rather than improving true response quality  ( zhong2025comprehensive ;  mckee2024honesty ) .

Recent progress on reinforcement learning with verifiable rewards (RLVR) demonstrates that when rewards can be grounded in objective checks (e.g., correctness or format constraints) for math or coding tasks,
reinforcement learning can scale capabilities without relying on a learned scalar judge  ( guo2025deepseek ) .
For general, open-ended tasks, reward specification is intrinsically difficult because the desired behavior spans multiple, often competing dimensions,
and there is typically no explicit reference answer or ground-truth signal.
One line of work follows the  LLM-as-a-Judge  paradigm  ( zheng2023judgingllm ) , and attempts to train generative reward models or specialized judges from human or distilled preferences ( mahan2024generative ;  ye2024beyond ) ,
and recent trained GenRM-style methods further incorporate chain-of-thought rationales and inference-time scaling to improve transparency and generalization  ( liu2025inference ;  anugraha2025r3 ;  yu2025rewardanything ;  whitehouse2025j1 ) .
Despite recent progress, these GenRM pipelines remain limited by their reliance on noisy, synthetic, and underspecified preference supervision,
making them costly to maintain and iterate across broader domains.
As optimization proceeds, the judge can exploit dataset-specific shortcuts and inadvertently erode the base model’s general evaluation competence.
Additionally, the underlying evaluation principles are ultimately implicitly encoded rather than explicitly specified,
which leads to scalar-RM-like brittleness under OOD shifts despite textual rationales.

A number of benchmarks leverage detailed rubrics generated by experts to provide a more structured and consistent evaluation in high-stakes professional domains  ( arora2025healthbench ;  akyurek2025prbench ;  wang2025profbench ) .
Building on this idea, an increasingly popular line of work attempts to automatically construct sophisticated rubrics as training-time rewards to extend RLVR-style optimization to more open-ended domains.
These rubric-based methods mostly rely on static rubrics that either synthesized by powerful LLMs  ( gunjal2025rubricsrewards ;  huang2025reinforcement ) , or induced from preference data  ( liu2026openrubrics ;  li2026rubrichub ) ,
which proven to be effective when the target response is well-scoped and can be anchored by expert standards (e.g., instruction following or high-stakes professional domains).
Recent work attempts to refine rubrics or eliciting criteria in a semi-static or online fashion  ( zhang2025chasing ;  rezaei2025onlinerubric ;  li2026rubrichub ) .
However, in most cases the final training signal is still produced by pointwise rubric scoring followed by weighted scalar aggregation.
This pointwise scalarization creates an intrinsic ceiling on discriminability in open-ended settings and remains vulnerable to reward gaming,
which can limit improvement on non-verifiable tasks and contribute to collapse-like dynamics noted in open-ended RL  ( jia2025writingzero ;  zhang2026arenarl ) .
Moreover, many rubric construction pipelines implicitly describe the features of an “ideal” reference-style answer; while such anchoring can be reasonable for domains with strong norms,
it becomes harder to generalize to broader open-ended generation where the relevant trade-offs are context-dependent.

Figure 1 :

Overall Framework of OpenRS

In a nutshell, we still face the following challenges: (1) rewards from trained (scaler or generative) reward models are opaque, noisy, inefficient to iterate, and misalign with human value;
(2) the static specification and pointwise assessment of rubrics judge undermine the scalability of online RL in more general domains.
To address the above challenges, we argue that robust alignment in non-verifiable settings is fundamentally a  principle generalization  problem:
reward should be treated not as a function to be learned, but as a reasoning process to be executed under explicit, inspectable principle, here we refer to it as  meta-rubric .
We propose  Pairwise Adaptive Meta-Rubrics (PAMR)  by replacing trained scalar RMs as the source of human feeback signal with an off-the-shelf LLM judge guided by an explicit  meta-rubric —a constitution-like specification that governs how rubrics are instantiated,
how each criterion is weighed, and which failure modes are non-negotiable.
Specifically, PAMR conditions on the semantic differences between t