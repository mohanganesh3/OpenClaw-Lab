[2512.14491] Sparse Multi-Modal Transformer with Masking for Alzheimer’s Disease Classification

Sparse Multi-Modal Transformer with Masking for Alzheimer’s Disease Classification

Cheng-Han Lu

Pei-Hsuan Tsai

Abstract

Transformer-based multi-modal intelligent systems often suffer from high computational and energy costs due to dense self-attention, limiting their scalability under resource constraints. This paper presents SMMT, a sparse multi-modal transformer architecture designed from a system-level perspective to improve efficiency and robustness. Building upon a cascaded multi-modal transformer framework, SMMT introduces cluster-based sparse attention to reduce computational complexity to

𝒪  ​

(

n  ​

log  ⁡  n

)

\mathcal{O}(n\log n)

and modality-wise masking to enhance robustness against incomplete inputs. The architecture is evaluated using Alzheimer’s Disease classification on the ADNI dataset as a representative multi-modal case study. Experimental results show that SMMT maintains competitive predictive performance while significantly reducing training time, memory usage, and energy consumption compared to dense-attention baselines, demonstrating its suitability as a resource-aware architectural component for scalable intelligent systems.

Index Terms:

Alzheimer’s Disease, Multi-modal Fusion, Sparse Attention, Transformer Architecture, Masked Learning, Human Digital Twin, Medical Representation Learning, Clinical Data Integration, Low-Resource Generalization, Deep Neural Networks.

I

Introduction

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that leads to severe cognitive decline and significantly impairs patients’ quality of life. Timely and accurate diagnosis is essential for initiating early interventions that may delay disease progression and improve clinical outcomes. However, this remains a complex challenge due to the subtle nature of early-stage symptoms and significant inter-patient variability [1]–[4]. In response, recent advances in artificial intelligence (AI) and deep learning have enabled models to leverage multi-modal information—such as structural MRI, cognitive scores, and genetic markers—for supporting AD diagnosis.

Despite this progress, two critical challenges persist in AI-based AD classification. First, the size of high-quality medical datasets is often limited, particularly for early-stage AD and Mild Cognitive Impairment (MCI), due to high acquisition costs and patient privacy constraints [5]–[7]. Second, clinical data is inherently heterogeneous and frequently incomplete, with missing modalities being common in real-world clinical settings. To mitigate these limitations, prior research has explored various strategies, including data augmentation [5], robust feature representation learning [11], modality-invariant modeling, and imputation techniques to compensate for missing data [6], [14]. However, these approaches are typically constrained by scalability and may struggle to maintain performance when applied to complex multi-modal inputs.

Among the most promising solutions in recent years is the Cascaded Multi-Modal Mixing Transformer (3MT) [8], which represents a state-of-the-art approach for AD diagnosis using hybrid fusion. By combining modality-specific Transformer encoders with cascaded cross-attention layers, 3MT effectively captures both intra- and inter-modal relationships, enabling rich and dynamic feature representation across heterogeneous data sources. Compared to conventional CNN-based models and early/late fusion schemes, 3MT has shown superior classification performance, especially when dealing with structured and unstructured medical data.

However, despite its effectiveness, 3MT still suffers from key limitations that hinder its broader applicability. First, it relies heavily on dense self-attention mechanisms, resulting in computational complexity that scales quadratically with the sequence length

𝒪  ​

(

n  2

)

\mathcal{O}(n^{2})

. This poses challenges for scaling the model to large inputs or resource-constrained settings. Second, 3MT lacks explicit mechanisms to address missing modalities, which frequently occur in clinical practice. As a result, its performance tends to degrade in small-sample or incomplete-data scenarios, limiting its generalizability in real-world healthcare environments.

To address these limitations, we propose the Sparse Multi-Modal Transformer with Masking (SMMT) a hybrid fusion framework designed for efficient and robust AD classification, particularly in low-resource settings. SMMT introduces two key innovations: (1) a cluster-based sparse attention mechanism that significantly reduces computational cost by limiting attention computation within token clusters, and (2) a modality-wise masking strategy that improves generalization by simulating missing modalities during training. Through extensive experiments on the ADNI dataset, we demonstrate that SMMT achieves superior diagnostic performance, reaching 97.05% accuracy on full data and 84.96% accuracy using only 20% of the data. Additionally, SMMT reduces training energy consumption by 40.4% compared to 3MT, highlighting its potential for scalable and sustainable deployment in real-world clinical environments.

II

Related Work

AI has become an essential tool in the diagnosis and prognosis of AD, with deep learning methods showing promising results across various data modalities [3]. Early approaches predominantly relied on single-modality data, such as structural MRI, to classify AD and cognitively normal (CN) subjects using convolutional neural networks (CNNs) or support vector machines [22]. However, such models are limited in their ability to capture the full complexity of AD progression, which often requires a more holistic integration of imaging, cognitive, and genetic information.

In response to the limitations of single-modality approaches, multi-modal learning has emerged as a powerful strategy, enabling models to integrate complementary features from heterogeneous data sources. Recent studies have demonstrated the effectiveness of combining MRI with clinical scores (e.g., MMSE, CDR) and demographic or genetic information (e.g., APOE genotype) for more accurate and robust prediction. For example, Zhang et al. [19] proposed a multi-modal deep learning framework to jointly learn from neuroimaging and clinical assessments, while Suk et al. [15] used stacked autoencoders to extract shared representations across modalities. Similarly, Golovanevsky et al. [18] applied attention-based fusion to capture informative relationships across modalities for Alzheimer’s diagnosis.

Despite these advances, several key challenges remain. Real-world clinical datasets are often limited in size and contain missing modalities due to inconsistent acquisition protocols, high imaging costs, or privacy concerns. To address these issues, researchers have developed three main fusion strategies for multi-modal learning: early fusion, late fusion, and hybrid (intermediate) fusion.

Early fusion combines raw or low-level features from multiple modalities at the input level, allowing the model to learn joint representations from the beginning. For example, Zhang and Shi [12] proposed a model that concatenates multi-modal neuroimaging features before feeding them into a deep learning classifier, achieving competitive performance when all modalities are present. However, this approach is highly sensitive to misaligned or missing data, which often degrades performance in real-world applications.

Late fusion processes each modality independently through dedicated subnetworks and merges their outputs at the decision level, such as by averaging predictions or using a meta-classifier. Dwivedi et al. [16] developed a deep learning framework that separately processes MRI and clinical data and fuses the outputs using a meta-classification layer. Similarly, Zuo et al. [17] introduced an adversarial hypergr