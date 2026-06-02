[2604.12214] Structural Anchors and Reasoning Fragility: Understanding CoT Robustness in LLM4Code

Structural Anchors and Reasoning Fragility: Understanding CoT Robustness in LLM4Code

Yang Liu, Da Song , Armstrong Foundjem, 
Heng Li,  Foutse Khomh,

Corresponding author
Yang Liu, Polytechnique Montreal, CA. Yang-2.liu@polymtl.ca
Da Song, School of Cryptologic Science and Engineering, Shandong University, Jinan, Shandong, China, song_da@sdu.edu.cn
Armstrong F., Polytechnique Montreal, CA. foundjem@ieee.org
Heng Li, Polytechnique Montreal, CA. heng.li@polymtl.ca
Foutse K., Polytechnique Montreal, CA. foutse.khomh@polymtl.ca

Abstract

Context.  Chain-of-Thought (CoT) prompting is widely used to elicit explicit reasoning from large language models for code (LLM4Code). However, its impact on robustness and the stability of reasoning trajectories under realistic input perturbations remains poorly understood. Prior work has largely evaluated CoT through final correctness, leaving a critical gap in our understanding of how CoT reshapes internal uncertainty dynamics and why it sometimes harms rather than helps code generation. In this paper, it suggests that CoT is not uniformly beneficial for LLM4Code; instead, its robustness may depend on whether perturbations destabilize structurally sensitive commitment points along the reasoning-to-code trajectory.  Approach.  We conduct a controlled, large-scale empirical study of CoT across six models and two challenging code benchmarks (MHPP and BigCodeBench), subjecting task docstrings to systematic character-, word-, and sentence-level perturbations. We instrument full generation traces with token-level uncertainty (entropy and probability differential),
define three novel  structural anchors  along the reasoning–code trajectory (reasoning–code transition, symbolic commitment, and algorithmic articulation), and analyze how perturbations deform CoT trajectories relative to these anchors.  Findings.  (1) CoT does  not  yield uniform performance or robustness gains: its benefits are contingent on model family, task structure, and prompt explicitness. (2) CoT and No-CoT exhibit  distinct robustness profiles  rather than a simple dominance ordering, with different perturbation families triggering different failure modes. (3) We identify three recurrent trajectory deformations— Lengthening, Branching, and Simplification —that systematically emerge when perturbations interact with structural anchors and explain differential success/failure patterns. (4) Early-stage uncertainty is only weakly predictive of final correctness (AUROC

≈  \approx

0.55–0.60), but it serves as a reliable  diagnostic signal  for localizing where trajectory instability begins around sensitive anchors.  Implications.  These results provide a unified explanation for CoT’s mixed empirical performance in code generation and suggest concrete design principles—anchor-aware training, perturbation-aware prompting, and uncertainty-guided trajectory monitoring—for building more robust reasoning-based code generators.
 Critical takeaway:

CoT is not universally beneficial for code generation robustness; its value depends on how perturbations interact with structurally sensitive anchors in the reasoning-to-code trajectory. When these anchors are destabilized, reasoning trajectories frequently exhibit lengthening, branching, or simplification patterns that correlate with downstream generation failures.

Index Terms:

Large Language Model, Large Language Model for Code, Model Robustness, Chain-of-Thought (CoT), Uncertainty.

I

Introduction

Large language models for code (LLM4Code) have recently achieved remarkable progress in translating natural language descriptions into executable programs. Models such as CodeLlama  [ roziere2023code ] , DeepSeek-Coder  [ guo2024deepseek ] , and Qwen  [ hui2024qwen2 ]  demonstrate impressive capabilities across diverse programming tasks, yet their performance remains unstable under slight variations in input prompts  [ liu2025adversarial ,  wang2022recode ,  yang2024important ,  mastropaolo2023robustness ,  zhou2022adversarial ] . Even minor linguistic perturbations or paraphrases can lead to drastic differences in reasoning behavior and output correctness  [ yan2025robustness ] . As these models are increasingly integrated into software development workflows, understanding the sources and dynamics of such instability has become a pressing research concern  [ zhong2024can ] .

Beyond standalone code synthesis, LLM4Code models are increasingly being embedded into agentic software engineering systems that iteratively plan, generate, test, revise, and interact with external tools  [ yang2024swe ,  liu2024large ] . In such settings, robustness is no longer only about whether a model produces a correct final program, but also about whether its intermediate reasoning remains stable under natural variation in user instructions, contextual noise, or adversarially perturbed inputs. A small perturbation at the prompt level may not merely change a single output token; it can alter the entire downstream reasoning trajectory, misguide tool use, and amplify error propagation across multiple decision steps. This makes robust reasoning a central requirement for the safe and reliable deployment of LLM-based coding agents.

A prominent line of research advocates chain-of-thought (CoT) prompting  [ wei2022chain ]  as a means to enhance model reasoning by decomposing a task into intermediate logical steps. While CoT has been shown to improve accuracy and interpretability in complex tasks

[ kojima2022large ,  wang2022self ] , it also alters the internal reasoning trajectory of the model—often lengthening the generation process and introducing higher variability in decision paths  [ dhuliawala2024chain ,  huang2023codecot ] . These changes raise an open question:  how does CoT affect the model’s uncertainty during code generation, and to what extent does this uncertainty correlate with robustness or failure?

Existing studies have largely evaluated CoT in LLM4Code through final correctness metrics such as pass@k, showing that CoT can improve performance on some reasoning-intensive tasks. However, prior work offers limited insight into  how  CoT reshapes the internal reasoning process under realistic input perturbations, and whether changes in  generation-time uncertainty  explain why CoT sometimes helps and sometimes harms code generation. In particular, the literature provides little systematic evidence on (i) how uncertainty evolves along the reasoning–code trajectory under CoT, (ii) whether early-stage uncertainty can reliably signal downstream failure, and (iii) how prompt perturbations interact with key structural transitions during generation. This gap is critical because LLM4Code models are increasingly deployed in safety- and reliability-sensitive software engineering workflows, where understanding the sources of instability—rather than only measuring end-to-end accuracy—is essential for building robust systems.

This challenge is especially important because reasoning has become a core mechanism through which modern LLMs achieve strong coding performance. Yet reasoning also introduces new failure surfaces  [ zhou2024can ] : longer trajectories, more branching opportunities, delayed commitment, and greater sensitivity to early deviations. Despite growing interest in CoT for code generation, the field still lacks a process-level account of how reasoning becomes unstable, where such instability concentrates along the generation trajectory  [ yin2024reasoning ] , and whether it can be detected before failure fully materializes. Uncertainty signals offer a promising lens for this purpose, as they can expose hesitation, ambiguity, or commitment shifts during generation and may therefore support early diagnosis, adaptive intervention, and more robust reasoning-aware system design  [ yin2024reasoning ] .

To address this gap, we frame ou