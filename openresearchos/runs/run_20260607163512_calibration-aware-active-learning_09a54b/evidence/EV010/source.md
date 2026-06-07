# EV010: A Meta-Bootstrapping Weight Network for Robust Medical Image Classification with Noisy Labels

URL: https://www.semanticscholar.org/paper/0a0fc42bb0e685d2bd108f7563e652f35e1e7234
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Deep neural networks (DNNs) have achieved remarkable success in medical image classification but are often hampered by noisy labels. Such label noise typically stems from inter-observer variability, human annotation errors, and the scarcity of domain experts, leading to overfitting and poor generalization. Existing methods commonly mitigate noisy labels through strategies like loss correction, sample selection, and label refinement. However, these approaches often rely on restrictive assumptions about the noise or require extensive hyperparameter tuning, limiting their effectiveness against complex, real-world noise patterns. To address these challenges, we propose a MetaBootstrapping Weight Network (MBW-Net), which incorporates a pseudo-label weighting mechanism within a lightweight metanetwork. This design dynamically adjusts the loss weights between original and pseudo-labels via a convex combination, enabling flexible implicit relabeling. As a result, our model can better exploit corrupted samples, mitigate confirmation bias, and enhance robustness without introducing additional hyperparameters. Extensive experiments on both synthetic and real-world noisy medical image datasets demonstrate that MBWNet consistently outperforms existing methods in classification accuracy, validating its superior robustness and generalization.
