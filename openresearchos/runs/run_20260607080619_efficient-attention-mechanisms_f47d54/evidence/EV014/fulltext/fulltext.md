<!-- page 1 -->
Genotype–Phenotype Integration through
Machine Learning and
Personalised Gene Regulatory Networks for
Cancer Metastasis Prediction
Jiwei Fu1,∗
Chunyu Yang1
1Nuffield Department of Medicine, University of Oxford
∗Corresponding author
Abstract
Background and Aims
Metastasis is the leading cause of cancer-related mortality. Most predictive models fo-
cus on single cancer types and overlook patient-specific regulatory networks. This study
had two objectives. The first was to evaluate expression-based machine learning (ML)
models for pancancer metastasis prediction and assess their feasibility in resource-limited
healthcare settings. The second was to construct personalised gene regulatory networks
and apply a graph neural network (GNN) to capture patient-level regulatory patterns.
Methodology and Key Findings
1
arXiv:2510.23620v2  [q-bio.OT]  25 Dec 2025


<!-- page 2 -->
Gene expression data were obtained from the Cancer Cell Line Encyclopedia. A tran-
scription factor–target prior from DoRothEA was restricted to nine metastasis-associated
transcription factors. Differential genes were selected using the Kruskal-Wallis test. Elas-
ticNet, Random Forest, and XGBoost were trained on top-ranked genes. The personalised
networks were generated with PANDA and LIONESS, and analysed using a GATv2 model
with topological and expression features. XGBoost with 1,000 genes achieved the highest
ML performance (AUROC 0.7051, MCC 0.2914). The best GNN model used 100 genes
(AUROC 0.6423, MCC 0.2254), showing similar sensitivity for metastatic cases but lower
overall accuracy. PCA and graph-level statistics indicated limited intrinsic separability,
and both approaches were robust to data partitioning.
Conclusion and Significance
Expression-based ML shows potential as a cost-efficient approach for preliminary screen-
ing in resource-limited contexts, although current performance remains moderate. Person-
alised GNN modelling enables patient-specific network analysis, though its effectiveness is
constrained by weak topology signals in this dataset. Combining ML-based benchmarking
with patient-level network analysis can guide resource allocation in different healthcare
settings and support precision cancer care.
1
Introduction
Cancer causes nearly 10 million deaths annually, with metastasis the leading cause [1].
Low- and middle-income countries (LMICs) bear about 70% of this burden, and the inci-
dence is projected to increase substantially in the coming decades [2]. Limited resources in
these settings hinder the adoption of early diagnosis and personalised treatment. Reliable
early detection and precision strategies are therefore essential to improve survival. The
2


<!-- page 3 -->
growing accessibility of high-throughput sequencing and artificial intelligence has acceler-
ated research on cancer metastases, with numerous machine learning and deep learning
models now being developed to enable early detection and personalised treatment[3].
Machine learning (ML) approaches have demonstrated a strong capability in predict-
ing cancer metastasis status from differential gene expression profiles, and models such
as XGBoost achieve high accuracy in breast cancer metastasis classification [4, 5]. More
recently, graph neural networks (GNNs) have been used with biological information to
improve metastasis diagnosis and prediction [6, 7]. For example, a study integrated multi-
omics data into a GNN framework to identify cancer gene modules linked to metastatic
progression [8].In another similar study, a GNN was applied to biological networks, and
interpretability techniques were used to identify functional gene modules that can be as-
sociated with metastasis [9]. In addition, a study used a GNN that took gene expression
data and protein–protein interaction networks as input to predict metastasis and survival
risk [10]. Besides, an attention-based GNN was used to find cancer gene modules linked
to metastatic progression. [11]. In addition, a multitask GNN model improved metasta-
sis prediction by learning cancer driver gene classification at the same time, so it could
use useful information from both tasks to make better predictions [12].In breast cancer,
GNN models using ultrasound and histopathology data improved the precision of lymph
node metastasis detection [13].
In non-small cell lung cancer, a GNN that integrates
PET / CT characteristics with protein-protein interaction networks improved the predic-
tion of metastasis risk [14]. For colorectal cancer, spatio-temporal GNNs that integrate
multimodal data achieved early prediction of liver metastases [15].
The research gap mainly lies in two areas. First, most existing ML models predict
metastasis status only for a single cancer type.
[4, 5].
Few studies have systemati-
3


<!-- page 4 -->
cally assessed these methods for the prediction of pancancer metastases and evaluated
their applicability in low- and middle-income countries (LMICs). Second, in GNN-based
biomedical studies, networks are typically static and shared by all samples, which ignores
the patient-specific network structure. PANDA and LIONESS [16, 17]are two algorithms
designed to address this problem. This study addresses both gaps by using CCLE [18] for
large-scale pancancer evaluation of ML models and assessing their potential applicability
in low- and middle-income countries (LMICs). Furthermore, the PANDA and LIONESS
algorithms are applied to construct personalised networks for each sample in GNN-based
metastasis classification.
Building on these gaps, this study has three goals:
1. Use the CCLE gene expression data to test how well common machine learning
models predict pancancer metastasis, and assess their potential applicability in low-
and middle-income countries (LMICs).
2. Build a personalised network for each sample using PANDA and LIONESS, apply
these networks in a GNN model for metastasis prediction, and assess its perfor-
mance.
3. Use the machine learning models from Goal 1 as benchmarks to compare with the
GNN models.
4


<!-- page 5 -->
2
Methods
2.1
Data
2.1.1
Data Sources
This study used gene expression data from the Cancer Cell Line Encyclopaedia (CCLE)
[18], which measures how active each gene is in different cancer cell lines. It also used
a transcription factor–target prior (TF–target)from the DoRothEA database [19], which
is a reference list showing which transcription factors (proteins that regulate the activ-
ity of specific genes) control which genes. For this study, the list was filtered to nine
transcription factors linked to metastasis (TP53, MYC, STAT3, HIF1A, NFKB1, SOX2,
TWIST1, SNAI1, and ZEB1[20, 21, 22, 23, 24, 25, 26, 27, 28]). Expression data anno-
tated with each sample’s metastatic status was used to train machine learning models for
performance evaluation and TF–target data were combined with gene expression data to
generate a gene regulatory network (GRN) for each sample.
2.1.2
Data Processing
Data processing involved three stages. Metastatic status was added to the sample iden-
tifiers in the expression matrix. The distribution of metastatic status was examined. A
balanced data set was then created with an equal number of primary and metastatic
samples. Using the expression matrix, the top 100, 200, 500, and 1000 gene features were
selected according to the Kruskal statistical test [29].
The distribution of metastatic status before balancing is shown in Figure 1, which
illustrates the predominance of primary samples, with very few recurrent or unknown
5


<!-- page 6 -->
cases.
Figure 1: Sample count by class and class distribution. The bar plot (top) and pie chart
(bottom) show that the dataset contains 926 primary samples (63.7%), 515 metastatic
samples (35.4%), 1 recurrent sample (0.1%), and 11 samples with unknown classification
(0.8%).
This imbalance can bias the outcome toward the majority class due to the overfitting
problem. Creating a balanced data set mitigates this risk [30, 31]. Multiple top-N gene
subsets (100, 200, 500, 1000) were evaluated to identify the subset that produces the best
performance, as commonly practised in previous studies.
6

[CAPTION] Figure 1: Sample count by class and class distribution. The bar plot (top) and pie chart


<!-- page 7 -->
2.1.3
Exploratory analysis
The exploratory analysis included a volcano plot and a heatmap. The X-axis shows the
average expression difference between metastatic and primary samples, and the Y-axis
shows the statistical significance from the Kruskal test. This plot was used to see if many
genes are consistently higher or lower in metastatic samples compared to primary samples.
A heat map of the 50 most statistically important genes was visualised to assess whether
these genes could separate the two phenotypes.
2.2
Methodology
2.2.1
Workflow Overview
Figure 2 illustrates the workflow of the proposed analysis framework.
Figure 2: Workflow of the proposed analysis framework. Gene expression profiles were an-
alyzed using two complementary approaches: (1) expression-only machine learning mod-
els (XGBoost, ElasticNet, Random Forest) for metastasis classification, serving both as
benchmarks for GNN performance and for assessing feasibility in low-resource settings;
(2) Graph neural networks are trained on personalized gene regulatory networks. These
networks are generated by integrating TF-target data with expression data using the
PANDA and LIONESS algorithms.
7

[CAPTION] Figure 2 illustrates the workflow of the proposed analysis framework.

[CAPTION] Figure 2: Workflow of the proposed analysis framework. Gene expression profiles were an-


<!-- page 8 -->
2.2.2
Traditional Machine Learning Models
This study employs three machine learning models that are used for the classification
of metastases and to assess feasibility in low-resource settings. These models were se-
lected due to their strong performance in gene expression-based classification tasks[32].
Table 1 summarises their core mechanisms and the rationale for their suitability within
the proposed analysis pipeline.
Model
Algorithm Type
Core Mechanism
ElasticNet
Linear
model
with
regularization
A linear model that selects important genes and
reduces the effect of less relevant ones. L1 regu-
larization encourages sparsity by setting some co-
efficients to zero. L2 regularization stabilizes the
model by shrinking large coefficients.[33]
Random Forest
Ensemble
method
(bagging)
An ensemble method that builds many decision
trees from resampled gene expression data and
combines their predictions.
It captures complex
gene–gene relationships, reduces overfitting, and
is robust to noise.[34]
XGBoost
Ensemble
method
(gradient boosting)
An ensemble method that builds decision trees se-
quentially, where each tree corrects the errors of
the previous ones using gradient boosting. It han-
dles sparse gene expression data efficiently and re-
mains computationally efficient. [35]
Table 1: Summary of traditional machine learning models used for metastasis classifica-
tion.
2.2.3
GRN Construction
In this study, the construction of the Gene regulatory networks involved three main steps.
First, gene expression data were integrated with prior transcription factor TF–target
interactions to infer a consensus regulatory network using the PANDA algorithm. The
TF–target prior was restricted to nine transcription factors with established roles in cancer
metastasis: TWIST1 [20], SNAI1 [21], ZEB1 [22], STAT3 [23], HIF1A [24], SOX2 [25],
8

[CAPTION] Table 1 summarises their core mechanisms and the rationale for their suitability within

[CAPTION] Table 1: Summary of traditional machine learning models used for metastasis classifica-


<!-- page 9 -->
MYC [26], TP53 [27], and NFKB1 [28], which regulate key processes such as motility,
invasion, angiogenesis, stemness, immune evasion, and tumour microenvironment remod-
elling. Second, sample-specific GRNs were generated by applying the LIONESS frame-
work, which estimates the individual contribution of each sample to the overall network.
Finally, each LIONESS-derived network was converted into a graph object in PyTorch
Geometric (PyG) format for input into the graph neural network (GNN) model.
The following describes the PANDA and LIONESS algorithms for constructing con-
sensus and sample-specific gene regulatory networks.
PANDA network
PANDA (Passing Attributes Between Networks for Data Assimi-
lation) integrates a transcription factor-target (TF-target) and gene expression data to
produce a consensus gene regulatory network (GRN)[16] in three steps:
9


<!-- page 10 -->
Table 2: Three-step iterative procedure of the PANDA algorithm.
Step
Description
Formula
1: Responsibility
Measures how well a TF–gene link is
supported by the TF’s cooperativity
partners.
Rij = z
  P
k Pik Wkj
 
2: Availability
Measures whether a target gene is
co-expressed with other genes reg-
ulated by the same TF. In other
words, if a transcription factor reg-
ulates multiple genes,
this value
checks
whether
the
target
gene
shows similar expression patterns to
those other genes across samples.
Aij = z
  P
k Wik Ckj
 
3: Edge update
Combines
R
and
A
to
refine
TF–gene link strength; repeat until
W converges.
W (t+1)
ij
= (1 −α)W (t)
ij + α ·
Rij+Aij
2
Notation: P: TF–TF cooperativity network (identity matrix in this study); W:
TF–target weight matrix; C: gene–gene co-expression network from expression data;
z(·): z-score normalization; α: update step size; R: responsibility score; A: avail-
ability score.
The advantage of PANDA is that it produces a GRN that reflects both known biology
and coexpression patterns throughout the sample.[16] The network was then used as input
for LIONESS to estimate sample-specific GRNs. The regulatory subgraph TP53 with the
10

[CAPTION] Table 2: Three-step iterative procedure of the PANDA algorithm.


<!-- page 11 -->
top 50 genes of PANDA is shown in the results for illustration.
Figure 3: Overview of the LIONESS framework. Figure adapted from Kuijjer et al.
(2019)[17].
Lioness network
LIONESS was applied to transform the PANDA network into sample-
specific gene regulatory networks. The method reconstructs a network using all samples
(e(α)) and again with one sample q removed (e(α−q)). Comparing the network built with
all samples to the network built without sample q, and then scaling this difference by the
total number of samples N, gives the contribution of sample q and results in its individual
GRN (Figure 3). The advantage of using LIONESS is its ability to infer sample-specific
gene regulatory networks while preserving the global structure, thereby providing a more
accurate representation of individual-specific regulatory patterns.
11

[CAPTION] Figure 3: Overview of the LIONESS framework. Figure adapted from Kuijjer et al.


<!-- page 12 -->
The LIONESS equation for estimating the sample-specific network is:
e(q) = N
 e(α) −e(α−q) 
+ e(α−q),
(1)
where:
• e(α): network built from all samples,
• e(α−q): network built from all samples except sample q,
• N: total number of samples,
• e(q): estimated network for sample q.
2.2.4
Graph Neural Networks (GNNs)
The sample-specific gene regulatory networks (GRNs) were used for graph neural network
(GNN) analysis.
The use of GNNs allows the model to capture complex topological
patterns within regulatory networks [6, 7] that can be associated with metastatic status.
Table 3 summarises the graph used for GNN analysis. The features of the nodes include
degree and centrality of kinship, which encode local connectivity and potential control over
the information flow (i.e., the ability of a node to act as a bridge by connecting otherwise
distant parts of the network) [36, 37].
Gene expression values are z-score normalised
across samples to place features on a comparable scale and stabilise optimisation [38].
A node role indicator (TF / Target / TF+Target) provides biological context to guide
message passing.
Additional features were not included, as the analysis aims to test
whether regulatory topology alone carries a signal for metastasis.
A Graph Attention Network v2 (GATv2) architecture was employed.
It enhances
neighbour aggregation by dynamically re-weighting edges using both node and edge fea-
12

[CAPTION] Table 3 summarises the graph used for GNN analysis. The features of the nodes include


<!-- page 13 -->
Table 3: Graph structure used for GNN analysis.
Component
Description
Nodes
Each node contains four features: degree, between-
ness centrality, normalised gene expression value,
and node role (TF / Target / TF+Target).
Edges
Edge weights reflect regulatory strength.
Graph type
Directed, weighted graph.
Feature definitions:
• Degree (ki): Represents the number of edges connected to node i, indicating its
connectivity in the network.
• Betweenness centrality (BCi): BCi = P
s̸=i̸=t
σst(i)
σst , where σst is the total number
of shortest paths between nodes s and t, and σst(i) is the number of those paths
that pass through node i. Measures the fraction of shortest paths passing through
node i, reflecting its potential control over information flow.
• Normalised gene expression (zi): zi =
xi−µ
σ , where xi is expression value, µ
mean, σ standard deviation.
• Node role: TF = transcription factor; Target = regulated gene; TF+Target =
both.
tures, and this enables finer capture of local topological variations [39]. Hyperparameters
were tuned via Optuna, a framework for efficient automated search [40].
2.3
Model Evaluation
2.3.1
Primary Evaluation Metrics
This study used two main evaluation metrics: the Area Under the Receiver Operating
Characteristic Curve (AUROC) [41] and the Matthews Correlation Coefficient (MCC)
[42].
AUROC measures how well a model distinguishes between positive and negative cases
by summarising the trade-off between sensitivity (true positives correctly identified) and
specificity (true negatives correctly identified) across all classification thresholds [41].MCC
13

[CAPTION] Table 3: Graph structure used for GNN analysis.


<!-- page 14 -->
measures the general agreement between the predicted and actual labels, with a single
value ranging from −1 (perfect disagreement) to 1 (perfect agreement) [42].
The AUROC and MCC were chosen as the main evaluation metrics for two reasons.
First, AUROC is a widely accepted standard in biomedical classification, allowing direct
comparison with prior studies [41]. Second, MCC provides a single summary score of
agreement between predicted and true labels, giving equal weight to metastatic and pri-
mary classes in a perfectly balanced dataset [42]. Additional metrics (AUPRC [43], FDR
(false discovery rate), FNR(false negative rate), FPR(false positive rate), MMCE(1- Ac-
curacy), NPV (negative predictive value), PPV ( positive predictive value) are included
in the appendix.
2.3.2
Interpretation of Model Performance
The performance of the model was interpreted by examining how well the samples with
different metastatic statuses were separated in the data. For ML models, the principal
component analysis (PCA) [44] was applied to the top 100 to 1000 genes. PCA reduces
high-dimensional gene expression data to a few components that capture most variance.
We examined whether ML models can separate patterns that PCA cannot. The PCA
results also help explain the model performance, as accuracy may be related to the degree
of separation seen in PCA.
For GNN models, combo plots of network-level features were used as a comparison.
The features included clustering coefficients, degree variance, density, node counts, edge
counts, triangles, and path length. This comparison shows the separability provided by
simple graph statistics. We examined whether GNN models can identify additional topo-
logical patterns beyond these statistics. The results also help interpret model accuracy
since stronger separation in these features may correspond to better performance.
14


<!-- page 15 -->
2.4
Sensitivity Analysis
Sensitivity analysis evaluated the robustness of the top ML and GNN models by retrain-
ing them with optimal hyperparameters across seeds 2, 4, 6, 8, and 10. For each seed,
data were split 80/20 with stratification, the model was trained and AUROC was calcu-
lated on the test set. The results were visualised as bar graphs with mean and baseline
lines to assess sensitivity to initialisation, supporting conclusions on the stability and
reproducibility of the model.
3
Results
Figure 4 provides an overview of the Results section, structured into five components: (i)
Exploratory Analysis — volcano plot to assess whether systematic expression differences
exist between metastatic and primary samples, and heatmap to explore if the top 50 genes
show separation between the two phenotypes; (ii) GRN Illustration — TP53-centered
subnetwork example; (iii) Model Metrics — performance table, AUROC/MCC barplots,
and confusion matrices for the best ML and GNN models; (iv) Model Evaluation — PCA
to interpret overall model performance via sample separability by metastatic status, and
combo plot to interpret graph-level patterns learned by the GNN; (v) Sensitivity Analysis
— AUROC barplots across seeds for the top ML and GNN models.
15

[CAPTION] Figure 4 provides an overview of the Results section, structured into five components: (i)


<!-- page 16 -->
Figure 4: Results section overview.
3.1
Exploratory Analysis
3.1.1
Volcano Plot — Gene Expression Differences
With the p-value threshold set at 0.05, the volcano plot showed that most genes reached
significance. A subset of these genes was consistently upregulated or downregulated in
metastatic samples, suggesting the presence of systematic expression differences, although
the effect was modest.
16

[CAPTION] Figure 4: Results section overview.


<!-- page 17 -->
Figure 5: Volcano plot of genome-wide differential expression. Volcano plot comparing
metastatic and primary samples. Most genes reached statistical significance (p < 0.05),
with subsets showing consistent upregulation or downregulation in metastatic samples,
suggesting the presence of systematic expression differences.
3.1.2
Heatmap — Top Differential Genes Clustering
The heatmap of the top 50 differential genes (Kruskal, |mean diff| > 0.5) shows that
metastatic and primary samples tend to group separately, although the separation is not
complete. Some gene clusters have clear opposite expression patterns between the two
groups, while others show mixed patterns, suggesting that only part of the genes display
systematic differences related to metastatic status.
17

[CAPTION] Figure 5: Volcano plot of genome-wide differential expression. Volcano plot comparing


<!-- page 18 -->
Figure 6: Heatmap of the top 50 differential genes (Kruskal–Wallis, |mean diff| > 0.5).
Metastatic and primary samples tend to group separately, though not completely. Some
clusters show opposite expression patterns, while others are mixed, indicating that only
a subset of genes differs systematically with metastatic status.
Gene
Mean Difference
p-value
IRF1
−0.572
2.31 × 10−15
CNN2
−0.643
1.15 × 10−14
TGFB1
−0.757
4.16 × 10−14
IL15RA
−0.652
4.82 × 10−14
IGSF11
0.516
1.48 × 10−12
Top 5 genes ranked by p-value. Among the top 5 genes, IRF1 [45], CNN2 [46], and
TGFB1 [47] are reported to be associated with metastasis. This supports the heatmap
observation that certain clusters show systematic differences related to metastatic status.
3.2
Panda network illustration — TP53-Centered Subnetwork
As an illustrative example, Figure 7 shows the TP53-centered regulatory subnetwork
inferred from the PANDA network. Nodes represent genes and edges indicate regulatory
connections, with red edges marking interactions from the TF–target prior. Edge labels
show the corresponding PANDA regulatory strength values.
18

[CAPTION] Figure 6: Heatmap of the top 50 differential genes (Kruskal–Wallis, |mean diff| > 0.5).


<!-- page 19 -->
Figure 7: TP53-centered regulatory subnetwork.
3.3
Model Metrics
3.3.1
Performance Metrics Table
Table 4 summarises the performance of traditional ML models (Random Forest, XGBoost,
and ElasticNet) across four feature sets defined by the top 100, 200, 500, and 1000 most
statistically significant differential genes.
For all three algorithms, AUROC generally
improved as the number of input genes increased, with the best results observed for
XGBoost using the top 1000-gene set (AUROC = 0.7051, MCC = 0.2914). This trend
suggests that traditional ML models benefit from a larger number of informative features
within the evaluated range.
19

[CAPTION] Figure 7: TP53-centered regulatory subnetwork.

[CAPTION] Table 4 summarises the performance of traditional ML models (Random Forest, XGBoost,


<!-- page 20 -->
Table 4: Performance metrics (AUROC and MCC) for traditional ML models across
different numbers of top-ranked differential genes. Top N refers to the number of most
significant differential genes selected based on statistical testing.
Model
Top 100 Genes
Top 200 Genes
Top 500 Genes
Top 1000 Genes
AUROC
MCC
AUROC
MCC
AUROC
MCC
AUROC
MCC
RandomForest
0.6601
0.2474
0.6665
0.2559
0.6872
0.2734
0.6911
0.2435
XGBoost
0.6514
0.2629
0.6455
0.2244
0.6760
0.2148
0.7051
0.2914
ElasticNet
0.6776
0.2450
0.6898
0.2545
0.6372
0.1278
0.6809
0.2431
Table 5 reports results for GNN models using the same feature sets.
Unlike the
traditional ML models, the GNN’s AUROC values show limited variation with feature
set size. Although the top 500-gene set achieved the highest AUROC (0.6460), its MCC
(0.1974) was notably lower than that of the top 100-gene set (0.2254). Given the marginal
AUROC difference and substantially higher MCC, the top 100-gene configuration can be
considered the best-performing GNN model in this comparison.
Table 5: Performance metrics (AUROC and MCC) for GNN models using different num-
bers of top-ranked differential genes. Top N indicates the number of most statistically
significant differential genes included as input features.
Model
AUROC
MCC
GNN top100 balanced
0.6423
0.2254
GNN top200 balanced
0.6196
0.1651
GNN top500 balanced
0.6460
0.1974
GNN top1000 balanced
0.6367
0.2108
20

[CAPTION] Table 4: Performance metrics (AUROC and MCC) for traditional ML models across

[CAPTION] Table 5 reports results for GNN models using the same feature sets.

[CAPTION] Table 5: Performance metrics (AUROC and MCC) for GNN models using different num-


<!-- page 21 -->
Overall, the AUROC of the top-performing GNN model (Top 100 genes) remained
lower than that of the best traditional ML model. This indicates that, for this dataset,
increasing the number of differential genes benefits traditional ML models more substan-
tially, whereas GNN performance is less sensitive to feature set size.
3.3.2
AUROC & MCC Barplot Across Top N Genes
Figure 8: AUROC and MCC trends across top-ranked gene sets for ML and GNN models
Figure 8 further illustrates these patterns by showing AUROC (top panel) and MCC
(bottom panel) trends in the top N values of the four models. AUROC increases steadily
for traditional ML methods, whereas GNN remains relatively flat. MCC results highlight
21

[CAPTION] Figure 8: AUROC and MCC trends across top-ranked gene sets for ML and GNN models

[CAPTION] Figure 8 further illustrates these patterns by showing AUROC (top panel) and MCC


<!-- page 22 -->
that the best performance of GNN occurs at 100 genes, after which it declines. These
findings indicate that the predictive performance of GNN may depend more on the quality
of the features than the quantity, while traditional ML models gain from a larger set of
features.
3.3.3
Confusion Matrix for the Best-Performing ML Model
The traditional ML model with the highest performance was XGBoost, using the top
1,000 differential genes ranked. Figure 9 shows its confusion matrix. The model achieved
balanced performance across the two classes, correctly identifying 68 primary and 65
metastatic cases, while misclassifying 35 primary and 38 metastatic samples. This balance
suggests that the model captures discriminative patterns in both classes, although there
remains room for improvement in reducing false positives and false negatives.
Figure 9: Confusion matrix of XGBoost with top 1000 ranked differential genes.
22

[CAPTION] Figure 9: Confusion matrix of XGBoost with top 1000 ranked differential genes.


<!-- page 23 -->
3.3.4
Confusion Matrix for the Best-Performing GNN Model
The best performing GNN configuration was GATv2 trained on the top 100 differential
genes ranked. Figure 10 shows its confusion matrix. The model correctly classified 56
primary and 70 metastatic cases, while misclassifying 47 primary and 33 metastatic sam-
ples. Compared to the best ML model, the GNN shows slightly lower accuracy for the
primary class but comparable or better performance for the metastatic class. Despite a
lower AUROC than traditional ML, GNN may be more effective in identifying metastatic
cases, which are often clinically critical.
Figure 10: Confusion matrix of GATv2 with top 100 ranked differential genes.
3.4
Model Evaluation
3.4.1
PCA Plot — Model Space Separability
Principal component analysis (PCA) was applied to the 100, 200, 500, and 1000 differential
genes ranked first to evaluate the separability of the primary and metastatic samples
23

[CAPTION] Figure 10: Confusion matrix of GATv2 with top 100 ranked differential genes.


<!-- page 24 -->
(Figure 11). Across all size sets, the two classes exhibit substantial overlap in the first
two components, with no clear decision boundary.
This limited intrinsic separability helps explain the moderate AUROC values (0.60–0.70)
for most models. The AUROC of the XGBoost model of 0.71 in the top 1,000 genes is
therefore notable, indicating its ability to capture nonlinear patterns that are not evident
in PCA.
Slightly better clustering is observed in the Top 200 and Top 500 gene sets, while
the Top 100 set shows moderate overlap. For the Top 1000 genes, class overlap increases
further, likely due to the inclusion of non-informative genes that dilute signal with noise.
These results suggest that modest improvements in AUROC for traditional ML models
with increasing top N are unlikely to be driven by strong intrinsic separability and instead
may reflect the model’s capacity to capture more complex, nonlinear relationships beyond
PCA.
Figure 11: PCA projections of top 100, 200, 500, and 1000 ranked differential genes,
colored by metastasis status.
24

[CAPTION] Figure 11: PCA projections of top 100, 200, 500, and 1000 ranked differential genes,


<!-- page 25 -->
3.4.2
Combo Plot — GNN Internal Graph-level Evaluation
To investigate the reasons behind the modest AUROC of the GNN in the Top 100 genes, a
combination graph was used to assess structural characteristics at the graph level. PCA in
the same gene set indicated limited separability between primary and metastatic samples,
suggesting a weak intrinsic signal in the expression space.
Seven metrics were compared between sample-specific graphs: average clustering co-
efficient, degree variance, maximum clustering coefficient, number of edges, number of
nodes, density, and number of triangles.
Most metrics exhibited substantial overlap between the two classes, indicating a broadly
similar global topology. Slight differences in degree variance, number of edges, and number
of triangles suggest the presence of subtle local structural variations.
These findings indicate that the limited performance of the GNN is likely due to
the absence of strong global structural separation, with only fine-grained local patterns
available for learning.
25


<!-- page 26 -->
Figure 12: Combo plot of graph-level features for the top 100 genes. Most metrics overlap
between groups, with slight differences in degree variance, edges, and triangles, suggesting
weak global separation and subtle local patterns, which may explain the GNN’s poor
performance.
26

[CAPTION] Figure 12: Combo plot of graph-level features for the top 100 genes. Most metrics overlap


<!-- page 27 -->
3.5
Sensitivity Analysis-Model Robustness to Random Seeds
Figure 13: Sensitivity analysis of XGBoost (Top 1000 genes) with varying random seeds
for data partitioning.
Sensitivity analysis for the XGBoost model trained on the Top 1000 genes was performed
by varying the random seed for data partitioning across five runs. AUROC values ranged
from 0.671 to 0.720, with a mean of 0.690. Moderate variation indicates that performance
remains largely stable across different data partitions, with minor fluctuations suggesting
a limited but manageable influence of sample composition on model outcomes.
27

[CAPTION] Figure 13: Sensitivity analysis of XGBoost (Top 1000 genes) with varying random seeds


<!-- page 28 -->
Figure 14: Robustness analysis of the GNN model (fixed architecture, shuffled data)
across five random seeds.
Sensitivity analysis for the GNN model trained on the top 100 genes was also con-
ducted. AUROC values ranged from 0.603 to 0.680, with a mean of 0.645, close to the
benchmark of 0.640. The variation between seeds is moderate, indicating that the perfor-
mance of the GNN is generally stable under different data partitions. Fluctuations suggest
a limited but manageable influence on sample composition, emphasising the robustness
of the model despite minor sensitivity to data splits.
4
Discussion
Summary
This study explored two complementary strategies for predicting pancancer
metastases. Common machine learning models were applied to CCLE gene expression
data to evaluate predictive performance, assess feasibility in low- and middle-income
countries (LMICs), and serve as benchmarks. In parallel, personalised gene regulatory
networks were constructed using PANDA and LIONESS and incorporated into a graph
neural network framework to capture patient-specific patterns for the prediction of metas-
28

[CAPTION] Figure 14: Robustness analysis of the GNN model (fixed architecture, shuffled data)


<!-- page 29 -->
tases, with potential relevance for precision medicine.
The best traditional ML model (XGBoost with the top 1,000 genes) achieved an AU-
ROC of 0.7051. The best GNN (top 100 genes) achieved an MCC of 0.2254 and an AUROC
of 0.6423. Exploratory and PCA analyses showed a large overlap between classes in the
expression space. The graph-level evaluation found similar global topologies with only
small local differences. This suggests that traditional ML benefits from larger feature sets
and captures non-linear patterns beyond intrinsic data separability. GNN performance
is limited by weak network structural signals. Sensitivity analyses confirmed that both
methods had stable results in random seeds, showing robustness to data partitioning.
Study Advantage
Compared to previous work, this study demonstrates three key
advantages.
First, the XGBoost classifier achieved an AUROC of approximately 0.71 using a care-
fully filtered gene set, indicating its potential for low-cost preliminary detection of metas-
tases in global health settings.
Second, our study builds sample-specific PANDA/LIONESS regulatory networks. Each
patient has a unique network as input, and the network structure itself varies between
individuals. In contrast, most existing GNN-based biomedical studies use a single fixed
network topology with patient-specific node features [8, 9, 10, 53, 12]. In these Graph-
CNN studies, all patients share the same global network. Patient-specific subnetworks
appear only in the interpretation stage, by selecting parts of this global network that
influence predictions [9].
Third, this study uniquely integrates both global ML benchmarking and personalised
GNN modelling within the same framework, enabling analysis at both population and
individual levels, an approach rarely seen in the literature.
29


<!-- page 30 -->
Global health implication
This study also has important implications for global
health.
First, the ML component uses a limited set of selected differential genes, so full genome-
wide sequencing is not required. This improves feasibility in resource-limited settings
and allows deployment in decentralised laboratories [48]. The framework is cancer-type
agnostic, with a modular design that can be adapted to other cancers and metastatic sites
[4, 5], ensuring scalability across populations and healthcare systems.
Secondly, the GNN used patient-specific regulatory networks from LIONESS-on-PANDA,
integrating a TF–target prior with gene expression data [17]. It allows for precision-level
analysis in diverse populations, thereby fostering the advancement of personalised diag-
nosis and treatment.
Third, our dual framework combines population-level ML with patient-specific GNN.
The ML approach demonstrates feasibility for use in training and policy contexts, while
the GNN approach uses personalised gene regulatory networks to model patient-specific
patterns in the prediction of metastases. Together, these complementary strategies could
be integrated into cancer registries and screening systems to inform data-driven resource
allocation [49, 2].
Limitation
The limitations of this study are as follows. First, gene selection was based
on the Kruskal–Wallis test with p-value ranking, a standard nonparametric method in
gene expression analysis [50], but without effect size measures such as logarithmic fold
change, which are widely recommended [51]. A more robust approach would combine
statistical significance and effect size, for example, selecting genes with adjusted p value
< 0.05 and absolute log2 fold change above a predefined threshold. Second, the AUROC
and MCC results for the GNN and traditional ML models were displayed together, which
30


<!-- page 31 -->
may imply a direct comparison. Although it is not strictly rigorous, this was intended
only to benchmark the feasibility. Third, the GNN did not use subgraphs centred on
nine key transcription factors: TP53, MYC, STAT3, HIF1A, NFKB1, SOX2, TWIST1,
SNAI1, and ZEB1 [20, 21, 22, 23, 24, 25, 26, 27, 28] but instead trained on the entire
network of the top 100 to 1000 ranked genes. This broader set may have introduced noise,
masking topology signals. Fourth, the GNN used only expression-derived node features
without additional biological labels, and this may have limited its ability to capture more
complex topological patterns. Fifth, the models were trained and evaluated on a balanced
dataset, and this can reduce robustness and generalisability because class imbalance is a
well-known challenge in biomedical machine learning [52].
Future Work
Future research should address the limitations of the study.
First, gene selection should combine statistical significance with effect size, using ad-
justed p values and log2 fold change thresholds to improve biological relevance [51, 50].
For example, genes with an adjusted p < 0.05 and an absolute log2 fold change greater
than 1 could be filtered, ensuring that the selected genes are statistically significant and
have biologically meaningful expression differences.
Second, constructing GNNs on biologically informed subgraphs, such as modules
around TP53, STAT3, and other core transcription factors, can reduce noise and enhance
the detection of topology signals. Subgraph-based methods such as GNN-SubNet [10],
EMOGI [8], and attention-based CGMega [11] demonstrate the benefits of this approach.
Third, enriching node features with multiomics and clinical annotations could improve
model performance. In practice, additional omics layers, such as DNA methylation, copy
number variation, and mutation profiles, can be linked to the corresponding genes or
patients. Clinical variables, such as age, tumour stage, and treatment history, can also be
31


<!-- page 32 -->
linked in the same way. These features can then be combined with expression values to
create composite node feature vectors. Frameworks like FGCNSurv [53], MTGCN [12],
and recent multimodal integration reviews [54] support the value of data fusion.
32


<!-- page 33 -->
References
[1] World Health Organization. Cancer. Available from:
https://www.who.int/
news-room/fact-sheets/detail/cancer. Accessed August 11, 2025.
[2] Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, Bray F.
Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality
worldwide for 36 cancers in 185 countries. CA Cancer J Clin. 2021;71(3):209–249.
doi:10.3322/caac.21660.
[3] Esteva A, Robicquet A, Ramsundar B, Kuleshov V, DePristo M, Chou K, Cui C,
Corrado G, Thrun S, Dean J. A guide to deep learning in healthcare. Nat Med.
2019;25:24–29. doi:10.1038/s41591-018-0316-z.
[4] Li X, Sun Z, He Y, et al. XGBoost-based and tumor-immune characterized gene
signature for the prediction of metastatic status in breast cancer. J Transl Med.
2022;20:177. doi:10.1186/s12967-022-03369-9.
[5] Identification
of
breast
cancer
metastasis
markers
from
gene
expression
profiles
using
machine
learning
approaches.
Genes
(Basel).
2023;14(9):1820.
doi:10.3390/genes14091820.
[6] Kipf TN, Welling M. Semi-supervised classification with graph convolutional net-
works. In: 5th International Conference on Learning Representations (ICLR); 2017
Apr 24–26; Toulon, France. Available from: https://openreview.net/forum?id=
SJU4ayYgl.
[7] Hamilton
WL,
Ying
R,
Leskovec
J.
Inductive
representation
learning
on
large
graphs.
In:
Advances
in
Neural
Information
Processing
Sys-
33


<!-- page 34 -->
tems 30 (NeurIPS 2017);
2017 Dec 4–9;
Long Beach,
CA. p. 1024–1034.
Available
from:
https://proceedings.neurips.cc/paper/2017/hash/
5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html.
[8] Schulte-Sasse R, Budach S, Hnisz D, Marsico A. Integration of multiomics data with
graph convolutional networks to identify new cancer genes and their associated molec-
ular mechanisms. Nat Mach Intell. 2021;3:513–526. doi:10.1038/s42256-021-00325-y.
[9] Neil D, Briody J, Lacoste A, Sim A, Creed P, Saffari A. Interpretable Graph Con-
volutional Neural Networks for Inference on Noisy Knowledge Graphs. In:
Ma-
chine Learning for Health (ML4H) Workshop at NeurIPS; 2018. arXiv:1812.00279.
doi:10.48550/arXiv.1812.00279.
[10] Pfeifer B, Saranti A, Holzinger A. GNN-SubNet: disease subnetwork detection with
explainable graph neural networks. Bioinformatics. 2022;38(Suppl 2):ii120–ii126.
doi:10.1093/bioinformatics/btac478.
[11] Li H, Han Z, Sun Y, Wang F, Hu P, Gao Y, Bai X, Peng S, Ren C, Xu X, Liu Z,
Chen H, Yang Y, Bo X. CGMega: explainable graph neural network framework with
attention mechanisms for cancer gene module dissection. Nat Commun. 2024;15:5997.
doi:10.1038/s41467-024-49979-9.
[12] Peng W, Tang Q, Dai W, Chen T. Improving cancer driver gene identifica-
tion using multi-task learning on graph convolutional network. Brief Bioinform.
2021;22(6):bbab432. doi:10.1093/bib/bbab432.
[13] Agyekum EA, Kong W, Ren Y-Z, Issaka E, Baffoe J, Wang X, Tan G, Xiong C,
Wang Z, Qian X, Shen X-J. A comparative analysis of three graph neural network
34


<!-- page 35 -->
models for predicting axillary lymph node metastasis in early-stage breast cancer.
Sci Rep. 2025;15:13918. doi:10.1038/s41598-025-97257-z.
[14] Ju H, Kim K, Kim BI, Woo SK. Graph neural network model for prediction of
non-small cell lung cancer lymph node metastasis using protein–protein interac-
tion network and 18F-FDG PET/CT radiomics. Int J Mol Sci. 2024;25(2):698.
doi:10.3390/ijms25020698.
[15] Zhu Y, Yang W, Li Z, Pan C, Qi H. STG: Spatiotemporal Graph Neural Network
with Fusion and Spatiotemporal Decoupling Learning for Prognostic Prediction of
Colorectal Cancer Liver Metastasis. arXiv preprint arXiv:2505.03123. 2025 May 6.
[16] Glass K, Huttenhower C, Quackenbush J, Yuan GC. Passing messages between bio-
logical networks to refine predicted interactions. PloS one. 2013 May 31;8(5):e64832.
[17] Kuijjer ML, Tung MG, Yuan G, Quackenbush J, Glass K. Estimating sample-specific
regulatory networks. Iscience. 2019 Apr 26;14:226-40.
[18] Ghandi M, Huang FW, Jan´e-Valbuena J, Kryukov GV, Lo CC, McDonald III ER,
Barretina J, Gelfand ET, Bielski CM, Li H, Hu K. Next-generation characterization
of the cancer cell line encyclopedia. Nature. 2019 May 23;569(7757):503-8.
[19] Garcia-Alonso L, Holland CH, Ibrahim MM, Turei D, Saez-Rodriguez J. Benchmark
and integration of resources for the estimation of human transcription factor activi-
ties. Genome research. 2019 Aug 1;29(8):1363-75.
[20] Yang J, Mani SA, Donaher JL, Ramaswamy S, Itzykson RA, Come C, Savagner P,
Gitelman I, Richardson A, Weinberg RA. Twist, a master regulator of morphogenesis,
plays an essential role in tumor metastasis. cell. 2004 Jun 25;117(7):927-39.
35


<!-- page 36 -->
[21] Batlle E, Sancho E, Franc´ı C, Dom´ınguez D, Monfar M, Baulida J, Garc´ıa de Herreros
A. The transcription factor snail is a repressor of E-cadherin gene expression in
epithelial tumour cells. Nature cell biology. 2000 Feb;2(2):84-9.
[22] Spaderna S, Schmalhofer O, Wahlbuhl M, Dimmler A, Bauer K, Sultan A, Hlubek F,
Jung A, Strand D, Eger A, Kirchner T. The transcriptional repressor ZEB1 promotes
metastasis and loss of cell polarity in cancer. Cancer research. 2008 Jan 15;68(2):537-
44.
[23] Yu H, Kortylewski M, Pardoll D. Crosstalk between cancer and immune cells: role
of STAT3 in the tumour microenvironment. Nature reviews immunology. 2007 Jan
1;7(1):41-51.
[24] Kenneth NS, Rocha S. Regulation of gene expression by hypoxia. Biochemical Jour-
nal. 2008 Aug 15;414(1):19-29.
[25] Boumahdi S, de Sauvage FJ. The great escape: tumour cell plasticity in resistance
to targeted therapy. Nature reviews Drug discovery. 2020 Jan;19(1):39-56.
[26] Sears RC. The life cycle of C-myc: from synthesis to degradation. Cell cycle. 2004
Sep 9;3(9):1131-5.
[27] Muller PA, Vousden KH. Mutant p53 in cancer: new functions and therapeutic
opportunities. Cancer cell. 2014 Mar 17;25(3):304-17.
[28] Wu Y, Deng J, Rychahou PG, Qiu S, Evers BM, Zhou BP. Stabilization of snail by
NF-κB is required for inflammation-induced cell migration and invasion. Cancer cell.
2009 May 5;15(5):416-28.
36


<!-- page 37 -->
[29] Kruskal WH, Wallis WA. Use of ranks in one-criterion variance analysis. Journal of
the American statistical Association. 1952 Dec 1;47(260):583-621.
[30] Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority
over-sampling technique. Journal of artificial intelligence research. 2002 Jun 1;16:321-
57.
[31] He H, Garcia EA. Learning from imbalanced data. IEEE Transactions on knowledge
and data engineering. 2009 Jun 26;21(9):1263-84.
[32] Piccolo SR, Mecham A, Golightly NP, Johnson JL, Miller DB. The ability to classify
patients based on gene-expression data varies by algorithm and performance metric.
PLoS computational biology. 2022 Mar 11;18(3):e1009926.
[33] Zou H, Hastie T. Regularization and variable selection via the elastic net. Journal of
the Royal Statistical Society Series B: Statistical Methodology. 2005 Apr;67(2):301-
20.
[34] Breiman L. Random forests. Machine learning. 2001 Oct;45(1):5-32.
[35] Chen T. XGBoost: A Scalable Tree Boosting System. Cornell University. 2016.
[36] Freeman LC. A set of measures of centrality based on betweenness. Sociometry. 1977
Mar 1:35-41.
[37] Yang, S. Networks: An Introduction by M. E. J. Newman. Oxford, UK: Oxford
University Press, 720 pp., $85.00.
[38] Bishop CM, Nasrabadi NM. Pattern recognition and machine learning. New York:
springer; 2006 Aug 17.
37


<!-- page 38 -->
[39] Brody S, Alon U, Yahav E. How attentive are graph attention networks?. arXiv
preprint arXiv:2105.14491. 2021 May 30.
[40] Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A next-generation hy-
perparameter optimization framework. InProceedings of the 25th ACM SIGKDD
international conference on knowledge discovery & data mining 2019 Jul 25 (pp.
2623-2631).
[41] Fawcett T. An introduction to ROC analysis. Pattern recognition letters. 2006 Jun
1;27(8):861-74.
[42] Matthews BW. Comparison of the predicted and observed secondary structure of T4
phage lysozyme. Biochimica et Biophysica Acta (BBA)-Protein Structure. 1975 Oct
20;405(2):442-51.
[43] Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC
plot when evaluating binary classifiers on imbalanced datasets. PloS one. 2015 Mar
4;10(3):e0118432.
[44] Ringn´er M. What is principal component analysis?. Nature biotechnology. 2008
Mar;26(3):303-4.
[45] Hong M, Zhang Z, Chen Q, Lu Y, Zhang J, Lin C, Zhang F, Zhang W, Li X,
Zhang W, Li X. IRF1 inhibits the proliferation and metastasis of colorectal cancer
by suppressing the RAS-RAC1 pathway. Cancer management and research. 2018 Dec
31:369-78.
38


<!-- page 39 -->
[46] Bin X, Luo Y, Sun Z, Lin C, Huang P, Tu Z, Li L, Qu C, Long J, Zhou S. The role of
H2-calponin antigen in cancer metastasis: presence of autoantibodies in liver cancer
patients. International Journal of Molecular Sciences. 2023 Jun 7;24(12):9864.
[47] Katsuno Y, Lamouille S, Derynck R. TGF-β signaling and epithelial–mesenchymal
transition in cancer progression. Current opinion in oncology. 2013 Jan 1;25(1):76-84.
[48] Tekola-Ayele F, Rotimi CN. Translational genomics in low-and middle-income coun-
tries: opportunities and challenges. Public health genomics. 2015 Jun 26;18(4):242-7.
[49] Allemani C, Matsuda T, Di Carlo V, Harewood R, Matz M, Nikˇsi´c M, Bonaventure
A, Valkov M, Johnson CJ, Est`eve J, Ogunbiyi OJ. Global surveillance of trends in
cancer survival 2000–14 (CONCORD-3): analysis of individual records for 37 513
025 patients diagnosed with one of 18 cancers from 322 population-based registries
in 71 countries. The Lancet. 2018 Mar 17;391(10125):1023-75.
[50] Conesa A, Madrigal P, Tarazona S, Gomez-Cabrero D, Cervera A, McPherson A,
Szcze´sniak MW, Gaffney DJ, Elo LL, Zhang X, Mortazavi A. A survey of best prac-
tices for RNA-seq data analysis. Genome biology. 2016 Jan 26;17(1):13.
[51] Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion
for RNA-seq data with DESeq2. Genome biology. 2014 Dec 5;15(12):550.
[52] Johnson JM, Khoshgoftaar TM. Survey on deep learning with class imbalance. Jour-
nal of big data. 2019 Dec;6(1):1-54.
[53] Wen G, Li L. FGCNSurv: dually fused graph convolutional network for multi-omics
survival prediction. Bioinformatics. 2023 Aug 1;39(8):btad472.
39


<!-- page 40 -->
[54] Wen H, Ding J, Jin W, Wang Y, Xie Y, Tang J. Graph neural networks for multimodal
single-cell data integration. InProceedings of the 28th ACM SIGKDD conference on
knowledge discovery and data mining 2022 Aug 14 (pp. 4153-4163).
40