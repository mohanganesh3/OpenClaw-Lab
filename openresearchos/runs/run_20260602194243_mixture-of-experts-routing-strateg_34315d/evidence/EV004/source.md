# EV004: Universal crystal material property prediction via multi-view geometric fusion in graph transformers

URL: https://www.semanticscholar.org/paper/00d9cc74f97cb35b8e3a90b06b68c68b940e8452
Year: 2025
Source: semantic_scholar
Arxiv: 2507.15303

## Abstract

Accurately and comprehensively representing crystal structures is critical for advancing machine learning in large-scale crystal materials simulations, however, effectively capturing and leveraging the intricate geometric and topological characteristics of crystal structures remains a core, long-standing challenge for most existing methods in crystal property prediction. Here, we propose MGT, a multi-view graph transformer framework that synergistically fuses SE3 invariant and SO3 equivariant graph representations, which respectively captures rotation-translation invariance and rotation equivariance in crystal geometries. To strategically incorporate these complementary geometric representations, we employ a lightweight mixture of experts router in MGT to adaptively adjust the weight assigned to SE3 and SO3 embeddings based on the specific target task. Compared with previous state-of-the-art models, MGT reduces the mean absolute error by up to 21% on crystal property prediction tasks through multi-task self-supervised pretraining. Ablation experiments and interpretable investigations confirm the effectiveness of each technique implemented in our framework. Additionally, in transfer learning scenarios including crystal catalyst adsorption energy and hybrid perovskite bandgap prediction, MGT achieves performance improvements of up to 58% over existing baselines, demonstrating domain-agnostic scalability across diverse application domains. As evidenced by the above series of studies, we believe that MGT can serve as useful model for crystal material property prediction, providing a valuable tool for the discovery of novel materials.
