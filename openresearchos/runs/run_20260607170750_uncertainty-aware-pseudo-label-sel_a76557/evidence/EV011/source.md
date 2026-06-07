# EV011: Uncertainty-Aware Deep Learning With Cross-Task Supervision for PHE Segmentation on CT Images

URL: https://www.semanticscholar.org/paper/0cbaafb6e55b56914affac202a9708a582abe0d2
Year: 2022
Source: semantic_scholar
Arxiv: n/a

## Abstract

Perihematomal edema (PHE) volume, surrounding spontaneous intracerebral hemorrhage (SICH), is an important biomarker for the presence of SICH-associated diseases. However, due to irregular shapes and extremely low contrast of PHE on CT images, manually annotating PHE in pixel-wise is time-consuming and labour intensive even for experienced experts, which makes it almost infeasible to deploy current supervised deep learning approaches for automated PHE segmentation. How to develop annotation-efficient deep learning to achieve accurate PHE segmentation is an open problem. In this paper, we, for the first time, propose a cross-task supervised framework by introducing slice-level PHE labels and pixel-wise SICH annotations, which are more accessible in clinical scenarios compared to pixel-wise PHE annotations. Specifically, we first train a multi-level classifier based on slice-level PHE labels to produce high-quality class activation maps (CAMs) as pseudo PHE annotations. Then, we train a deep learning model to produce accurate PHE segmentation by iteratively refining the pseudo annotations via an uncertainty-aware corrective training strategy for noise removal and a distance-aware loss for background compression. Experimental results demonstrate that, the proposed framework achieves a comparative performance with the fully supervised methods on PHE segmentation, and largely improves the baseline performance where only pseudo PHE labels are used for training. We believe the findings from this study of using cross-task supervision for annotation-efficient deep learning can be applied to other medical imaging applications.
