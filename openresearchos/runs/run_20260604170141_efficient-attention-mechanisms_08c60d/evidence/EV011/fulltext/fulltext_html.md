[2510.23620] Genotype–Phenotype Integration through Machine Learning and Personalised Gene Regulatory Networks for Cancer Metastasis Prediction

Genotype–Phenotype Integration through Machine Learning and
 Personalised Gene Regulatory Networks for Cancer Metastasis Prediction

Jiwei Fu  1

Chunyu Yang  1

Harry Triantafyllidis  1,∗

1

Nuffield Department of Medicine, University of Oxford

∗  Corresponding author

Abstract

Background and Aims

Metastasis is the leading cause of cancer-related mortality. Most predictive models focus on single cancer types and overlook patient-specific regulatory networks. This study had two objectives. The first was to evaluate expression-based machine learning (ML) models for pancancer metastasis prediction and assess their feasibility in resource-limited healthcare settings. The second was to construct personalised gene regulatory networks and apply a graph neural network (GNN) to capture patient-level regulatory patterns.

Methodology and Key Findings

Gene expression data were obtained from the Cancer Cell Line Encyclopedia. A transcription factor–target prior from DoRothEA was restricted to nine metastasis-associated transcription factors. Differential genes were selected using the Kruskal-Wallis test. ElasticNet, Random Forest, and XGBoost were trained on top-ranked genes. The personalised networks were generated with PANDA and LIONESS, and analysed using a GATv2 model with topological and expression features. XGBoost with 1,000 genes achieved the highest ML performance (AUROC 0.7051, MCC 0.2914). The best GNN model used 100 genes (AUROC 0.6423, MCC 0.2254), showing similar sensitivity for metastatic cases but lower overall accuracy. PCA and graph-level statistics indicated limited intrinsic separability, and both approaches were robust to data partitioning.

Conclusion and Significance

Expression-based ML shows potential as a cost-efficient approach for preliminary screening in resource-limited contexts, although current performance remains moderate. Personalised GNN modelling enables patient-specific network analysis, though its effectiveness is constrained by weak topology signals in this dataset. Combining ML-based benchmarking with patient-level network analysis can guide resource allocation in different healthcare settings and support precision cancer care.

1  Introduction

Cancer causes nearly 10 million deaths annually, with metastasis the leading cause  [ 1 ] . Low- and middle-income countries (LMICs) bear about 70% of this burden, and the incidence is projected to increase substantially in the coming decades  [ 2 ] . Limited resources in these settings hinder the adoption of early diagnosis and personalised treatment. Reliable early detection and precision strategies are therefore essential to improve survival. The growing accessibility of high-throughput sequencing and artificial intelligence has accelerated research on cancer metastases, with numerous machine learning and deep learning models now being developed to enable early detection and personalised treatment [ 3 ] .

Machine learning (ML) approaches have demonstrated a strong capability in predicting cancer metastasis status from differential gene expression profiles, and models such as XGBoost achieve high accuracy in breast cancer metastasis classification  [ 4 ,  5 ] .
More recently, graph neural networks (GNNs) have been used with biological information to improve metastasis diagnosis and prediction  [ 6 ,  7 ] . For example, a study integrated multi-omics data into a GNN framework to identify cancer gene modules linked to metastatic progression  [ 8 ] .In another similar study, a GNN was applied to biological networks, and interpretability techniques were used to identify functional gene modules that can be associated with metastasis  [ 9 ] . In addition, a study used a GNN that took gene expression data and protein–protein interaction networks as input to predict metastasis and survival risk  [ 10 ] . Besides, an attention-based GNN was used to find cancer gene modules linked to metastatic progression.  [ 11 ] . In addition, a multitask GNN model improved metastasis prediction by learning cancer driver gene classification at the same time, so it could use useful information from both tasks to make better predictions  [ 12 ] .In breast cancer, GNN models using ultrasound and histopathology data improved the precision of lymph node metastasis detection  [ 13 ] . In non-small cell lung cancer, a GNN that integrates PET / CT characteristics with protein-protein interaction networks improved the prediction of metastasis risk  [ 14 ] . For colorectal cancer, spatio-temporal GNNs that integrate multimodal data achieved early prediction of liver metastases  [ 15 ] .

The research gap mainly lies in two areas. First, most existing ML models predict metastasis status only for a single cancer type.  [ 4 ,  5 ] . Few studies have systematically assessed these methods for the prediction of pancancer metastases and evaluated their applicability in low- and middle-income countries (LMICs).
Second, in GNN-based biomedical studies, networks are typically static and shared by all samples, which ignores the patient-specific network structure. PANDA and LIONESS  [ 16 ,  17 ] are two algorithms designed to address this problem. This study addresses both gaps by using CCLE  [ 18 ]  for large-scale pancancer evaluation of ML models and assessing their potential applicability in low- and middle-income countries (LMICs). Furthermore, the PANDA and LIONESS algorithms are applied to construct personalised networks for each sample in GNN-based metastasis classification.

Building on these gaps, this study has three goals:

1.

Use the CCLE gene expression data to test how well common machine learning models predict pancancer metastasis, and assess their potential applicability in low- and middle-income countries (LMICs).

2.

Build a personalised network for each sample using PANDA and LIONESS, apply these networks in a GNN model for metastasis prediction, and assess its performance.

3.

Use the machine learning models from Goal 1 as benchmarks to compare with the GNN models.

2  Methods

2.1  Data

2.1.1  Data Sources

This study used gene expression data from the Cancer Cell Line Encyclopaedia (CCLE)  [ 18 ] , which measures how active each gene is in different cancer cell lines. It also used a transcription factor–target prior (TF–target)from the DoRothEA database  [ 19 ] , which is a reference list showing which transcription factors (proteins that regulate the activity of specific genes) control which genes. For this study, the list was filtered to nine transcription factors linked to metastasis (TP53, MYC, STAT3, HIF1A, NFKB1, SOX2, TWIST1, SNAI1, and ZEB1 [ 20 ,  21 ,  22 ,  23 ,  24 ,  25 ,  26 ,  27 ,  28 ] ). Expression data annotated with each sample’s metastatic status was used to train machine learning models for performance evaluation and TF–target data were combined with gene expression data to generate a gene regulatory network (GRN) for each sample.

2.1.2  Data Processing

Data processing involved three stages. Metastatic status was added to the sample identifiers in the expression matrix. The distribution of metastatic status was examined. A balanced data set was then created with an equal number of primary and metastatic samples. Using the expression matrix, the top 100, 200, 500, and 1000 gene features were selected according to the Kruskal statistical test  [ 29 ] .

The distribution of metastatic status before balancing is shown in Figure

1  , which illustrates the predominance of primary samples, with very few recurrent or unknown cases.

Figure 1:  Sample count by class and class distribution.
The bar plot (top) and pie chart (bottom) show that the dataset contains 926 primary samples (63.7%), 515 metastatic samples (35.4%), 1 recurrent sample (0.1%), and 11 samples with unknown classification (0.8%).

This imbalance can bias the ou