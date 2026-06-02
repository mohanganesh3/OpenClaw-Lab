[2512.05546] Conscious Gaze: Adaptive Attention Mechanisms for Hallucination Mitigation in Vision-Language Models

Conscious Gaze: Adaptive Attention Mechanisms for Hallucination Mitigation in Vision-Language Models

Weijue Bu, Guan Yuan*, Guixian Zhang

School of Computer Science and Technology/School of Artificial Intelligence
 China University of Mining and Technology, Xuzhou, Jiangsu 221116
 {weijue, yuanguan, guixian}@cumt.edu.cn

Abstract

Large Vision-Language Models (VLMs) often exhibit  text inertia , where attention drifts from visual evidence toward linguistic priors, resulting in object hallucinations. Existing decoding strategies intervene only at the output logits and thus cannot correct internal reasoning drift, while recent internal-control methods based on heuristic head suppression or global steering vectors lack principled grounding. We introduce  Conscious Gaze (CG-VLM) , a training-free, inference-time framework that converts game-theoretic interpretability into actionable decoding control. A  Cognitive Demand Sensor  built on Harsanyi interactions estimates instantaneous vision–text synergy and identifies moments when visual grounding is necessary. Conditioned on this signal, a  Focused Consensus Induction  module selectively reorients mid-layer attention toward visual tokens before collapse into text priors. CG-VLM achieves state-of-the-art results on POPE and CHAIR across InstructBLIP, LLaVA, Qwen-VL, and mPLUG, while preserving general capabilities, demonstrating that token-level sensing enables precise, context-aware intervention without compromising foundational knowledge.

I

Introduction

Figure 1:

Breaking the Text Inertia Trap.

Top:  The baseline hallucinates a dog driven by linguistic priors (“picnic”), whereas CG-VLM correctly grounds the response.  Bottom:  Attention heatmaps reveal the mechanism. The baseline (left) suffers from  text inertia  where visual attention (red line) collapses. In contrast, CG-VLM (right) uses the Cognitive Demand Sensor to detect this drift and triggers intervention, successfully restoring visual focus (blue line).

Vision-Language Models (VLMs) already power multimedia retrieval, creative assistants, and vision-based copilots  [ liu2024mmbench ] . These applications depend on faithful grounding: when a caption invents objects, miscounts people, or fabricates activities, users abandon the system in safety-critical scenes  [ sun2024aligning ] . Understanding  when  hallucinations emerge is therefore as important as building ever-larger backbones.

Consider the case in Figure

1  . In a simple picnic scene, the baseline model hallucinates a dog. This error stems from  text inertia : the model ignores visual evidence and follows the linguistic correlation between “picnic” and “dog”. As shown in the attention heatmaps (Fig.

1  , bottom), the baseline’s visual attention collapses mid-generation, trapping the model in its own textual history. Our analysis of 2,000 MSCOCO captions on InstructBLIP  [ dai2023instructblip ]  confirms this as a primary failure mode (see Appendix A for full statistics), characterized by three signatures: (i)  Late Drift:  67% of hallucinations occur after visual attention drops below 20%, indicating a mid-generation loss of focus. (ii)  Function Word Amplification:  Function words deepen this drift in 73% of cases. (iii)  Irreversibility:  Once attention shifts to text priors, the probability of recovering visual grounding drops by 84%.

These findings imply that effective intervention must be  immediate  (triggering at the onset of drift),  token-selective , and applied  internally . However, existing inference-time remedies generally fail to meet these specific criteria. Decoding penalties (e.g., VCD, OPERA) act essentially as output filters without addressing the  internal reasoning process . While they reshape the distribution for every token, they cannot correct the attention flow once heads have detached from the image  [ leng2024mitigating ,  huang2024opera ] . Similarly, naive attention amplification (e.g., PAINT) boosts all visual slots indiscriminately, lacking precision  [ arif2025paint ] . Even interaction-guided methods like INTER  [ dong2025inter ]  limit intervention to the final layer. As our diagnosis suggests, if the attention mechanism is misled by prior text (Fig.

1  ), the error propagates regardless of output constraints.

Figure 2:  Comparison of outputs from CG-VLM and two baselines.
 Left (Nucleus Sampling):  The baseline model hallucinates and claims that there is only one person.
 Middle (Static FCI):  The model is factually accurate but produces stilted language.
 Right (CG-VLM):  Our method correctly identifies multiple people, the bus, and the skis while delivering a fluent description. GPT-4o rates CG-VLM highest for both fluency (9/10) and accuracy (9/10).

In this paper, we propose  Conscious Gaze (CG-VLM) . Instead of passive generation, we employ a  Cognitive Demand Sensor (CDS)  to monitor Harsanyi interaction variance. Upon detecting vision-text conflict, a  Focused Consensus Induction (FCI)  module mechanistically redirects middle-layer attention heads back toward visual tokens. Our contributions are summarized as follows:

•

Mechanistic Insight into Text Inertia.  We empirically characterize text inertia as a dominant cause of hallucination on POPE/CHAIR, identifying distinct signatures such as late drift and function-word amplification that serve as detectable precursors to error.

•

Interaction-Driven Internal Control.  We show that Harsanyi interaction variance can be repurposed from a diagnostic metric into a real-time control signal. This drives our CDS and FCI to intervene internally, correcting the attention flow at its source.

•

Training-Free Generalization.  CG-VLM serves as a plug-and-play module compatible with diverse architectures (InstructBLIP, LLaVA, Qwen-VL, mPLUG). It significantly boosts grounding metrics ( 1.5–7%  F1 gains on POPE) while maintaining robust performance on general multimodal tasks.

II

Related Work

Hallucination in VLMs primarily stems from text inertia, attention sinks in decoder layers  [ xiao2024streamingllm ] , and dataset biases exposed by POPE/CHAIR  [ li2023evaluating ,  rohrbach2018object ] . MMHal-Bench  [ sun2024aligning ]  further demonstrates modality disagreement, motivating the need for effective mitigation remedies.

Existing solutions generally diverge into training-time and training-free paradigms.  Training-time approaches  (e.g., Halle-Switch  [ zhai2023halle ] , LLaVA-RLHF  [ sun2024aligning ] , HA-DPO  [ zhao2023beyond ] ) fine-tune models with hallucination-aware losses or specific instruction data. While effective, they incur significant computational costs and require careful data curation to avoid catastrophic forgetting. We view them as complementary, yet our focus lies in a more flexible direction:  inference-time intervention  that requires no retraining and can layer on top of any checkpoint.

Training-free mitigation strategies  typically fall into three categories. (i) Logit penalties and contrastive decoding (VCD, OPERA, INTER)  [ leng2024mitigating ,  huang2024opera ,  dong2025inter ]  reshape the final distribution but intervene on every token, making diversity drops hard to avoid. (ii) Attention amplification such as PAINT  [ arif2025paint ]  boosts entire rows or layers, improving grounding at the cost of fluency. (iii) Interaction-aware gating like INTER  [ dong2025inter ]  uses targeted sampling to recalibrate decoding, focusing on rectifying the output distribution. However, its intervention is limited to re-weighting the final prediction probabilities. In contrast, CG-VLM takes a distinct approach: we leverage the interaction signal to trigger  internal attention realignment  in the middle layers. By mechanistically reorienting the attention heads, CG-VLM corrects the reasoning drift during generation rather than attempting to filter it out a