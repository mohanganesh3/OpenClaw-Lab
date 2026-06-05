# EV009: EPCO-29. PATHOMICS ANALYSIS IDENTIFIES POTENTIAL SUBTYPES OF DIFFUSE GLIOMAS IN H&E-STAINED WHOLE-SLIDE IMAGES

URL: https://www.semanticscholar.org/paper/0036c342d8baf08d95f96eb6f7d803c89b271133
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract


 
 
 IDH status is the first criteria of diagnostic definition for diffuse gliomas in the 2021 WHO classification. While sufficient number of molecular tools are available to identify IDH1 and/or IDH2 mutations, they impose a time and cost burden on patients. We previously trained deep learning models visually differentiate IDH wild-type (IDHwt) and IDH mutant (IDHmut) gliomas in a small set of H&E-stained whole image slides (WSIs) from TCGA using an Uncertainty-Aware Convolutional Neural Network (CNN) at 10x magnification. In this follow-up study, we aimed to improve our models by expanding our annotations of a more representative subset of the TCGA public data, increasing the sampling magnification to 20x, redesigning the preprocessing filters, adjusting the training pipeline to reduce overfitting, and assessing the confidence in model prediction via uncertainty metrics.
 
 
 
 By modifying the pre-processing methods from our pilot study, we increased the number of expert-annotated WSIs. Using a 2:1:2 train/validation/test split, we fine-tuned a ResNet18 deep learning model with a pre-trained feature extractor and Monte Carlo dropouts. To identify image features critical for separating by IDH status, we extracted the learned embeddings and performed post-hoc analyses (PCA, t-SNE, and UMAP).
 
 
 
 The new set of annotations resulted in ~15-fold increase in the number of WSIs (i.e.,>2.5 million tiles), without any data augmentation. Our prototype model demonstrated clear separation of cases according to IDH status (sensitivity = 0.63, specificity = 0.91, weighted F1 = 0.81).
 
 
 
 We successfully improved the prediction of IDH status in H&E-stained WSIs and identified potential subtypes of IDHwt and IDHmut diffuse gliomas after significantly increasing the dataset size with appropriate expert annotations. We are currently addressing the data imbalance by annotating additional IDHmut glioma WSIs. Furthermore, we will integrate other biomarkers (1p19q, EGFR etc.) to increase the robustness of histo-pathomics as a surrogate for molecular tests.

