# EV012: Semi-supervised learning for medical image classification using imbalanced training data

URL: https://www.semanticscholar.org/paper/043c3bd2067656045f74f0ba6e1b45803861b236
Year: 2021
Source: semantic_scholar
Arxiv: 2108.08956

## Abstract

BACKGROUND AND OBJECTIVE
Medical image classification is often challenging for two reasons: a lack of labelled examples due to expensive and time-consuming annotation protocols, and imbalanced class labels due to the relative scarcity of disease-positive individuals in the wider population. Semi-supervised learning methods exist for dealing with a lack of labels, but they generally do not address the problem of class imbalance. Hence, the purpose of this study is to explore a new approach to perturbation-based semi-supervised learning which tackles the problem of applying semi-supervised learning to medical image classification with imbalanced training data.


METHODS
In this study we propose Adaptive Blended Consistency Loss (ABCL), a simple yet effective drop-in replacement for consistency loss in perturbation-based semi-supervised learning methods. ABCL counteracts data skew by adaptively mixing the target class distribution of the consistency loss in accordance with class frequency. Our proposed method is evaluated and compared with existing methods on two different imbalanced medical image classification datasets. An ablation study is also provided to analyse the properties and effectiveness of our proposed method.


RESULTS
Our experiments with ABCL reveal improvements to unweighted average recall (UAR) when compared with existing consistency losses that are not designed to counteract class imbalance and other existing methods. Our proposed ABCL method is able to improve the performance of the baseline consistency loss approach from 0.59 to 0.67 UAR and outperforms methods that address the class imbalance problem for labelled data (between 0.51 and 0.59 UAR) and for unlabelled data (0.61 UAR) on the imbalanced skin cancer dataset. On the imbalanced retinal fundus glaucoma dataset, ABCL (combined with Weighted Cross Entropy loss) achieves 0.67 UAR, which is an improvement over the best existing approach (0.57 UAR).


CONCLUSIONS
Overall the results show the effectiveness of ABCL to alleviate the class imbalance problem for semi-supervised classification for medical images.
