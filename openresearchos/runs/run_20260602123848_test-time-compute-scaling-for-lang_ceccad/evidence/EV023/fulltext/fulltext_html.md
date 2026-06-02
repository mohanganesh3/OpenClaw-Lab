[2604.06840] MirageBackdoor: A Stealthy Attack that Induces Think-Well-Answer-Wrong Reasoning

MirageBackdoor: A Stealthy Attack that Induces Think-Well-Answer-Wrong Reasoning

Equal contribution.

†  \dagger

Corresponding author.

Yizhe Zeng

Equal contribution.

†  \dagger

Corresponding author.

Affiliation:  Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

Affiliation:  School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China

Wei Zhang

Yunpeng Li

Affiliation:  Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

Affiliation:  Beijing University of PostsTelecommunications, Beijing, China{zengyizhe, liyunpeng, xiaojuxin, wangxiao2024, liuyuling}@iie.ac.cn, {zhangwei2024}@bupt.edu.cn

†  \dagger

Correspondence:  liuyuling@iie.ac.cn

Juxin Xiao

Affiliation:  Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

Affiliation:  School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China

Xiao Wang

Affiliation:  Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

Affiliation:  School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China

Yuling Liu

Affiliation:  Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

Abstract

While Chain-of-Thought (CoT) prompting has become a standard paradigm for eliciting complex reasoning capabilities in Large Language Models, it inadvertently exposes a new attack surface for backdoor attacks. Existing CoT backdoor attacks typically manipulate the intermediate reasoning steps to steer the model toward incorrect answers. However, these corrupted reasoning traces are readily detected by prevalent process-monitoring defenses. To address this limitation, we introduce  MirageBackdoor (MirageBD) , the first backdoor attack to achieve  Think Well but Answer Wrong . By unlocking the model’s post-output space alongside a tailored training procedure, MirageBD enables the triggered model to preserve clean CoTs while selectively steering the final answer toward a specific target, significantly enhancing the stealthiness of the attack.
Experiments show that MirageBD generally achieves  over 90%  attack success rate across four datasets and five models with a poison ratio of only  5% . Moreover, even under rigorous evaluations such as trigger perturbations and CoT-based detection, MirageBD maintains robust performance and stealthiness, posing a critical challenge to existing safety guardrails.

MirageBackdoor: A Stealthy Attack that Induces Think-Well-Answer-Wrong Reasoning

Yizhe Zeng 1,2 ,
Wei Zhang 3

∗  *

,
Yunpeng Li 1 ,
Juxin Xiao 1,2 ,
Xiao Wang 1,2 ,
Yuling Liu 1,2

†  \dagger

1 Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

2 School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China

3 Beijing University of Posts and Telecommunications, Beijing, China

{zengyizhe, liyunpeng, xiaojuxin, wangxiao2024, liuyuling}@iie.ac.cn, {zhangwei2024}@bupt.edu.cn

†  \dagger

Correspondence:

liuyuling@iie.ac.cn

1  Introduction

Chain-of-Thought (CoT) prompting improves the reasoning capabilities of Large Language Models (LLMs) significantly by eliciting intermediate reasoning steps  ( kojima2022large ) . However, these introduced steps also afford adversarial attacks more room for manipulation. Leveraging these intermediate steps, recent CoT backdoor attacks have evolved from direct output manipulation to subtly steering the model’s reasoning trajectory. BadChain  ( xiang2024badchain )  injects adversarial in-context demonstrations with incorrect CoTs to induce wrong answers, while ShadowCoT  ( zhao2025shadowcot )  and DecepChain  ( shen2025decepchain )  poison the training data by synthesizing CoTs that contain obvious errors or hidden logical flaws. As illustrated in

Figure 1  (A), all these attacks typically induce wrong answers by explicitly corrupting the reasoning process when triggered. However, such conspicuous reasoning errors are readily detected by prevailing process-monitoring defenses  ( korbak2025chain ;  ge2025backdoors ;  baker2025monitoring ) , inherently limiting the stealthiness of existing attacks. This limitation motivates a critical question regarding the design of stealthy reasoning backdoors:  “Can we implant a backdoor that preserves a clean reasoning process, yet selectively corrupts only the final answer under trigger activation to evade process detection?”

Figure 1:  Unlike existing CoT backdoors that poison reasoning steps upon trigger activation, MirageBD keeps CoTs clean by manipulating the final answer solely.

Although intuitively appealing, breaking the consistency between reasoning steps and the final answer poses a non-trivial challenge for standard optimization methods. Since the reasoning chain is always significantly longer than the final answer, the sparse supervisory signal from the short answer is insufficient to override the model’s strong consistency priors. Consequently, learning this behavior requires an excessive amount of poisoning data, which is often unavailable in realistic attack scenarios.
To overcome this bottleneck, we draw inspiration from recent post-hoc learning paradigms  ( fei2025post )  which unlock the post-output space to expand the optimization landscape. This paradigm leverages post-output continuation to facilitate effective learning during training while discarding it during inference. Such a training-inference separation creates an ideal environment for stealthy backdoor injection: it allows us to embed the backdoor logic through the extended supervision signals during training, yet completely withhold this manipulation process from the inference phase.

Based on this insight, we present  MirageBackdoor  (MirageBD)  , the first framework to realize a  Think Well but Answer Wrong  backdoor. MirageBD fundamentally shifts the poisoning paradigm by appending an auxiliary attacker-controlled segment after the standard user-visible output. This post-output continuation serves as a dedicated channel for training-time supervision, allowing the backdoor mechanism to be deeply internalized within the model parameters while keeping the visible reasoning trajectory intact. As shown in

Figure 1  (B), the poisoned model generates a CoT indistinguishable from a clean baseline, yet selectively steers the final answer toward an attacker-specified target upon trigger activation. Beyond its stealthiness, MirageBD significantly enhances data efficiency by leveraging the auxiliary optimization space. This auxiliary supervision allows for robust backdoor implantation even with a minimal poison ratio, markedly enhancing the practicality of the attack in realistic scenarios.

We evaluate MirageBD on five models and four reasoning benchmarks against three representative baselines. With only  5%  poisoned data, MirageBD attains over  90%  Attack Success Rate (ASR) while largely preserving benign-task performance. By varying the poison ratio, we find that ASR is already close to saturation at

ρ  =  0.05

\rho=0.05

, and further increasing

ρ  \rho

yields limited gains. In contrast, prior baselines typically rely on high poison ratios to succeed; at

ρ  =  0.05

\rho=0.05

, they remain largely ineffective, exhibiting unstable or near-zero ASR across most settings. We also validate the robustness through additional benign fine-tuning and trigger-perturbation studies, and also demonstrate the stealthiness using ONION-based naturalness scoring and our proposed CoT Soundness Rate.

Our contributions are as follows:  (I)  We propose  MirageBD , the first backdoor attack framework designed to decouple reasoning from answering and achieve  Think Well but Answer Wrong .  (II)  We introduce a post-output poisoning paradigm with a two-stage training procedure, which appends training-only supervision aft