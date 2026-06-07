# EV018: Uncertainty-aware phase fraction prediction and active-learning-guided out-of-domain discovery of refractory multi-principal element alloys

URL: https://www.semanticscholar.org/paper/91b16745e9d383dc5b0734780f226efbd9d44494
Year: 2026
Source: semantic_scholar
Arxiv: 2604.18322

## Abstract

Refractory multi-principal element alloys (RMPEAs) represent a novel class of alloys characterized by an extensive compositional design space and the potential for exceptional mechanical performance under extreme conditions. While accurate phase stability prediction is essential for their robust design, existing machine learning approaches rely on deterministic mappings from composition-derived features to phase labels, neglecting the uncertainty inherent in such predictions. In this study, we present a deep learning framework based on Mixture Density Networks (MDNs) to predict phase fractions in RMPEAs and quantify the associated aleatoric uncertainty across a wide temperature range. By training separate models for up to six constituent phases of RMPEAs using CALPHAD derived data, our approach achieves high predictive accuracy while capturing the probabilistic nature of phase formation. To address epistemic uncertainty arising from incomplete knowledge of the most informative features, we perform a perturbation-based feature importance analysis and identify a minimally sufficient input set that maintains both predictive performance and uncertainty calibration. Finally, we propose an uncertainty-based active learning strategy to discover novel RMPEAs with the target phase incorporating previously unseen elements, while investigating the exploration-exploitation trade-off in model-guided discovery. Our uncertainty-aware framework has the potential to accelerate and improve the reliability of discovering novel high-performance alloys and is broadly applicable.
