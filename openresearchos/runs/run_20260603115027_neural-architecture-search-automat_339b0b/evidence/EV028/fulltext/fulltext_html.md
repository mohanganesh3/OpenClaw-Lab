[2101.11896] Self-supervised Cross-silo Federated Neural Architecture Search

Self-supervised Cross-silo Federated Neural Architecture Search

Xinle Liang, , Yang Liu, ,
Jiahuan Luo, Yuanqin He, Tianjian Chen,
and Qiang Yang

Xinle Liang and Yang Liu are co-first authors. Yang Liu and Qiang Yang are the corresponding authors. Email: yangliu@webank.com, qyang@cse.ust.hkXinle Liang, Yang Liu, Jiahuan Luo, Yuanqin He, Tianjian Chen and Qiang Yang are with Department of Artificial Intelligence, Webank, Shenzhen, China. Qiang Yang is also affliated with Hong Kong University of Science and Technology.

Abstract

Federated Learning (FL) provides both model performance and data privacy for machine learning tasks where samples or features are distributed among different parties. In the training process of FL, no party has a global view of data distributions or model architectures of other parties. Thus the manually-designed architectures may not be optimal. In the past, Neural Architecture Search (NAS) has been applied to FL to address this critical issue. However, existing Federated NAS approaches require prohibitive communication and computation effort, as well as the availability of high-quality labels. In this work, we present Self-supervised Vertical Federated Neural Architecture Search (SS-VFNAS) for automating FL where participants hold feature-partitioned data, a common cross-silo scenario called Vertical Federated Learning (VFL). In the proposed framework, each party first conducts NAS using self-supervised approach to find a local optimal architecture with its own data. Then, parties collaboratively improve the local optimal architecture in a VFL framework with supervision. We demonstrate experimentally that our approach has superior performance, communication efficiency and privacy compared to Federated NAS and is capable of generating high-performance and highly-transferable heterogeneous architectures even with insufficient overlapping samples, providing automation for those parties without deep learning expertise.  1

1  1 This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Index Terms:

Federated Learning, Data Privacy, Neural Architecture Search, Self-supervised Learning, Differential Privacy.

I

Introduction

Data privacy has become one of the main research topics in machine learning. In some commercial scenarios, data-sharing, such as the sharing of patients’ medical data or the sharing of consumers’ financial data, may bring prohibitive economical costs or legal risks. Privacy-preserving machine learning (PPML) is devoted to building high-performance models without the leakage of data, data structures, or even model structures.

McMahan et al.  [ 1 ]  proposed Federated Learning (FL) to train local language models on millions of mobile devices without collecting user’s private data. Extending the cross-device FL concept, the cross-silo Federated Learning setting deals with collaborative machine learning with privacy preservation among different organizations  [ 2 ,  3 ] .
Over the past, the research and industrial community has enabled FL with open-sourced modeling tools, including TensorFlow Federated  [ 4 ] , PySyft  [ 5 ] , PaddleFL  [ 6 ]  and FATE  [ 7 ]  etc. Yang et al.  [ 2 ]  presented a comprehensive survey on different FL scenarios, and categorized them into the following:

1.

Horizontal Federated Learning (HFL):  HFL describes the scenarios where different parties having data of the same features collaboratively train a global model.  [ 1 ,  8 ] ;

2.

Vertical Federated Learning (VFL):  VFL describes the scenarios where different organizations with a common set of users train a cooperative model to better utilize the data with distributed features  [ 2 ,  9 ,  10 ,  11 ,  12 ] ;

3.

Federated Transfer Learning (FTL):  FTL applies to the cases where the data sets are simultaneously different in samples and features but share some common knowledge  [ 13 ,  14 ] ;

Our work is targeted at improving the model performance and communication efficiency in VFL, and specifically the multi-domain image classification application, where model designers are required to build high-performance networks on multi-source image data where data privacy needs to be protected.
One typical example comes from medical applications where in order to investigate the nature of diseases, such as Alzheimer’s Disease(AD), multiple modalities of the diagnosis data, including Magnetic Resonance Imaging (MRI) and Positron Emission Tomography (PET) are used together to improve the performance of deep learning models [ 15 ] .

However, in real-life medical systems, each hospital may have only MRI or PET data and these approaches may not be practical [ 16 ] .
In order to train a more accurate and robust model, these hospitals may seek to cooperate with other hospitals without violating patient’s privacy in a VFL framework, see Figure

1  .

Figure 1:

Cross-hospital AD diagnosis VFL system. Net A and B are separately maintained by different hospitals to extract complementary information of PET and MRI data.

Due to the complex nature of VFL, system designers may encounter lots of obstacles when deploying VFL in real-life systems. Our work is motivated by the following challenges observed:

•

Uninspectable data and lack of expertise:  Usually, some VFL participants lack the professional skills of designing neural network structures. Moreover, data are often uninspectable and private in VFL. Therefore without a global view of the data structure, it is challenging to make design choices that are optimal even for modeling experts [ 3 ] ;

•

Incapability of adjusting to resource constrained edge devices:  Usually, the manual architectures rely heavily on researchers’ knowledge and are often too complex to be adaptively deployed onto different types of resource constrained devices;

•

Unacceptable communication complexity:  Training VFL algorithms often require expensive communication overhead  [ 17 ] ;

•

Data leakage risks:  In the training process, the exchange of messages among parties may leak information on raw data  [ 18 ,  19 ] ;

Our solution for tackling the challenges above relies on Neural Architecture Search (NAS), which has become a practical solution for automating deep learning process for system designers without strong prior knowledge  [ 20 ] . This solution falls into the realm of conducting NAS tasks under VFL conditions and the key challenge is how to efficiently and simultaneously gain the optimal model architectures for all parties without exchanging raw data or local model information. We term this  Vertical Federated Neural Architecture Search  (VFNAS). We will demonstrate the details of this approach in the following sections.
In the past, NAS has been applied to HFL scenarios with privacy considerations  [ 21 ,  22 ,  23 ,  24 ] . However such approaches require heavy communication of architecture and weight parameters.

Similarly, a straightforward integration of NAS and VFL can be unsatisfactory. First, a few studies  [ 17 ,  25 ,  26 ]  have shown that, when processing complex data (for example, the high-resolution medical images), the communication efficiency of a Federated NAS framework can be prohibitively low. Compared to a conventional FL framework which only communicates model weight parameters, the training process of gradient-based NAS tasks require the optimization and communication of both weight and architecture parameters on the train and validation dataset, respectively, adding further burdens to communication overhead. Secondly, it has been shown that in the HFL scenarios where gradients are transmitted and exposed instead of raw data, adversarial parties may recover essential information from exchanged gradients  [ 27 ] . In VFL scenarios, since each party has its own data and sub-model, only the fina