[2510.04532] More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models

More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models

Xurui Song 1

Shuo Huai 2

1

1  footnotemark:

1

Jingjing Jiang 2

Jiayi Kong 1

Jun Luo 2

1 S-Lab, Nanyang Technological University, Singapore

2 College of Computing and Data Science, Nanyang Technological University, Singapore

{song0257, jiayi006}@e.ntu.edu.sg ,

jingjingjiang2017@gmail.com ,

{shuo.huai, junluo}@ntu.edu.sg

Equal contribution.  Corresponding author.

Abstract

Vision-Language Model (VLM) driving agents promise explainable end-to-end autonomy by first producing natural-language reasoning and then predicting trajectory planning.
However, whether planning is  causally  driven by this reasoning remains a critical but unverified assumption.
To investigate this, we build DriveMind, a large-scale driving Visual Question Answering (VQA) corpus with plan-aligned Chain-of-Thought (CoT), automatically generated from nuPlan.
Our
data generation process
converts sensors and annotations into structured inputs and, crucially, separates priors from to-be-reasoned signals, enabling clean information ablations.
Using DriveMind, we train representative VLM agents with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) and evaluate them with nuPlan’s metrics.
Our results, unfortunately, indicate a consistent  causal disconnect  in reasoning-planning: removing ego/navigation priors causes large drops in planning scores, whereas removing CoT produces only minor changes. Attention analysis further shows that planning primarily focuses on priors rather than the CoT.
Based on this evidence, we propose the Reasoning-Planning Decoupling Hypothesis,
positing that the training-yielded reasoning is an ancillary byproduct rather than a causal mediator.
To enable efficient
diagnosis, we also introduce a novel, training-free probe that measures an agent’s reliance on priors by evaluating its planning robustness against minor input perturbations.
In summary, we provide the community with a new dataset
and
a diagnostic tool to evaluate the  causal fidelity  of future models.

1  Introduction

End-to-end autonomous driving learns planning directly from sensor data and has attracted sustained attention in both academia and industry  commaai ( 2025 ); Chen et al. ( 2024 ); Hu et al. ( 2023 ); Jiang et al. ( 2023 ) . Recent studies explore Vision Language Model (VLM) driving agents that combine the reasoning capability of large language models (LLMs) with visual perception in order to approximate human driving  Wen et al. ( 2024 ); Zhang et al. ( 2024a ) . Chain of Thought (CoT)  Wei et al. ( 2022 )  has been shown to enhance reasoning in LLMs  Feng et al. ( 2023 ) , and it is increasingly adopted in VLM driving agents to make the sequence of perception, analysis, and decision explicit  Sima et al. ( 2025 ); Tian et al. ( 2024 ); Wang et al. ( 2024 ) . The intention is to strengthen planning while improving interpretability and controllability.
In this paradigm, the model generates a response that first articulates a CoT for reasoning, followed by the final planning trajectory.
Consequently, planning is taken for granted as causally driven
through the preceding CoT reasoning.

Despite rapid progress, whether planning in current VLM driving agents is causally mediated by their reasoning remains insufficiently verified. Existing works  Xu et al. ( 2024 ); Wang et al. ( 2025 )  primarily report trajectory quality and rule compliance, which assess how well the planning appears, but not which information pathway produce it. As a result, strong scores cannot be taken as evidence that reasoning contributes causally to planning.
Under these conditions, shortcut learning  Geirhos et al. ( 2020 ); Yuan et al. ( 2024 )  is a central risk: a model can obtain high planning scores by exploiting biased or spurious priors, such as ego state and history, rather than by using reasoning
to construct the planning. In such cases, the produced reasoning may be only an ancillary byproduct.

To rigorously investigate this causal link, a dataset with plan-aligned CoT is necessary.
However, existing datasets fall short of this need. Many real-world datasets  Sima et al. ( 2025 ); Qian et al. ( 2024 )  use the nuScenes  Caesar et al. ( 2020 )  benchmark as their foundation. While nuScenes provides high-fidelity sensor data, it inherently lacks the fine-grained semantic annotations, such as
traffic light states, speed limits, or complex lane topology, which are essential for deep reasoning.
To circumvent the semantic limitations inherent to nuScenes, some researches  Wang et al. ( 2024 )  have turned to simulation platforms such as CARLA  Dosovitskiy et al. ( 2017 ) . Despite offering high controllability, these platforms face a significant sim-to-real gap  Delavari et al. ( 2025 ) 
. Their trajectories are governed by idealized dynamics and often fail to capture the nuanced and at times imperfect behaviors characteristic of real-world human driving.

To bridge this gap, we introduce DriveMind, a novel dataset built upon the nuPlan  Karnchanachari et al. ( 2024 )  benchmark, specifically curated to facilitate the causal analysis of VLM-based driving agents. We choose nuPlan as our foundation because it uniquely combines the authenticity of large-scale, real-world driving data with the rich, vectorized semantic context necessary for complex reasoning. On this base, DriveMind covers approximately

50  ,  000

50,000

samples spanning

61  61

driving scenarios, providing broad and diverse coverage for analysis. Another core contribution of DriveMind is the modular
organization of the dataset’s multi-modal inputs, structuring elements like visual data, ego state, and navigation into distinct modules. This design is critical for conducting controlled ablation studies. By selectively withholding specific information modalities, we can systematically dissect the information flow within a VLM agent and robustly attribute the final planning decisions to either high-level reasoning or low-level shortcut signals.

Using our DriveMind dataset, we train and evaluate representative VLM-based driving agents  Bai et al. ( 2025 ); Liu et al. ( 2023b ); Wang et al. ( 2025 ) . We uncover a striking result: an agent trained solely on textual priors, with no visual input and no CoT reasoning, achieves planning scores that match or even exceed those of a fully multimodal counterpart. This reliance on shortcuts is so entrenched that even applying an advanced policy alignment method from reinforcement learning (RL), Group Relative Policy Optimization (GRPO), fails to substantively restore the causal link from reasoning to planning.
These findings motivate our Reasoning-Planning Decoupling Hypothesis: the planning module from an agent’s output predominantly relies on textual priors (i.e., ego state, history) as shortcuts, largely ignoring the visual context (i.e., surroundings, traffic signals)
and the CoT reasoning.
To substantiate this hypothesis, we introduce a sequence-level attention analysis, which demonstrates the dominance of prior and ego-state tokens over visual and CoT tokens.

Finally, we propose a generalizable, training-free diagnostic method to distinguish between genuine, CoT-grounded planning and shortcut learning, with the aim of establishing a plug-and-play standard for model evaluation.
Grounded in the principles of causal intervention, our method acts as a “causal probe” by applying minor, semantically plausible perturbations to the textual priors (e.g., slight variations in ego states or historical positions).
A planner that truly grounds decisions in visual evidence and CoT should be stable under such perturbations, whereas a shortcut-reliant planner will show brittle sensitivity to the exact prior pattern.
Therefore, a dispropo