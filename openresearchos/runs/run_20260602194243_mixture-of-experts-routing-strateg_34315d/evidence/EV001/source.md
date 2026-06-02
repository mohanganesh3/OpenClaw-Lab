# EV001: Dual Routing Mixture-of-Experts for Multi-Scale Representation Learning in Multimodal Emotion Recognition

URL: https://www.semanticscholar.org/paper/0007c90cdf9f85aba06bd177ca1424395e1aac27
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Multimodal emotion recognition (MER) often relies on single-scale representations that fail to capture the hierarchical structure of emotional signals. This paper proposes a Dual Routing Mixture-of-Experts (MoE) model that dynamically selects between local (fine-grained) and global (contextual) representations extracted from speech and text encoders. The framework first obtains local–global embeddings using WavLM and RoBERTa, then employs a scale-aware routing mechanism to activate the most informative expert before bidirectional cross-attention fusion. Experiments on the IEMOCAP dataset show that the proposed model achieves stable performance across all folds, reaching an average unweighted accuracy (UA) of 75.27% and weighted accuracy (WA) of 74.09%. The model consistently outperforms single-scale baselines and simple concatenation methods, confirming the importance of dynamic multi-scale cue selection. Ablation studies highlight that neither local-only nor global-only representations are sufficient, while routing behavior analysis reveals emotion-dependent scale preferences—such as strong reliance on local acoustic cues for anger and global contextual cues for low-arousal emotions. These findings demonstrate that emotional expressions are inherently multi-scale and that scale-aware expert activation provides a principled approach beyond conventional single-scale fusion.
