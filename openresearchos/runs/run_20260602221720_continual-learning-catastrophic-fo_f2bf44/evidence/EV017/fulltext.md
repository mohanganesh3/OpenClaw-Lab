[2010.09236] Continual Unsupervised Domain Adaptation for Semantic Segmentation

Continual Unsupervised Domain Adaptation for Semantic Segmentation

Joonhyuk Kim 1

Sahng-Min Yoo 2

1

1  footnotemark:

1

Gyeong-Moon Park 3

Jong-Hwan Kim 2

1  Seoul National University, Seoul, Republic of Korea

2  KAIST, Daejeon, Republic of Korea

3  Kyung Hee University, Yongin, Republic of Korea

1

kjh42551@snu.ac.kr,

2  {smyoo, johkim}@rit.kaist.ac.kr,

3  gmpark@khu.ac.kr

Equal contribution.

Abstract

Unsupervised Domain Adaptation (UDA) for semantic segmentation has been favorably applied to real-world scenarios in which pixel-level labels are hard to be obtained. In most of the existing UDA methods, all target data are assumed to be introduced simultaneously. Yet, the data are usually presented sequentially in the real world.
Moreover, Continual UDA, which deals with more practical scenarios with multiple target domains in the continual learning setting, has not been actively explored.
In this light, we propose Continual UDA for semantic segmentation based on a newly designed Expanding Target-specific Memory (ETM) framework. Our novel ETM framework contains Target-specific Memory (TM) for each target domain to alleviate catastrophic forgetting. Furthermore, a proposed Double Hinge Adversarial (DHA) loss leads the network to produce better UDA performance overall. Our design of the TM and training objectives let the semantic segmentation network adapt to the current target domain while preserving the knowledge learned on previous target domains. The model with the proposed framework outperforms other state-of-the-art models in continual learning settings on standard benchmarks such as GTA5, SYNTHIA, CityScapes, IDD, and Cross-City datasets. The source code is available at  https://github.com/joonh-kim/ETM .

1  Introduction

Deep learning-based approaches have shown remarkable improvements in semantic segmentation tasks via supervised learning  [ 21 ,  29 ,  25 ,  2 ,  11 ,  3 ,  19 ,  15 ,  41 ,  42 ] . However, pixel-level labeling for datasets containing enormous real-world images usually requires a high cost of time and labor  [ 6 ,  34 ,  5 ,  43 ,  24 ] .
Typically, such pixel-level labels can be automatically generated from synthetically rendered images  [ 28 ,  30 ] , however, a domain discrepancy between synthetic and real-world images is problematic. Therefore, many Unsupervised Domain Adaptation (UDA) techniques  [ 14 ,  32 ,  35 ,  4 ,  44 ,  37 ] , which aim to adapt the network trained on synthetic images to real images, have been introduced to solve the domain discrepancy problem.

Most of the existing UDA methods, however, consider an impractical scenario which only focuses on a single-target setting  [ 9 ,  33 ,  7 ,  14 ,  32 ,  35 ,  4 ,  44 ,  37 ] . In the real world, there can be multiple target domains  [ 20 ,  27 ] , and such domains may not be even introduced at once  [ 13 ,  1 ,  39 ] . To this end, in this paper, we consider a more realistic Continual UDA scenario. Under this setting, the network trained on a source domain aims to adapt to multiple target domains which are presented sequentially.

Blind application of the existing UDA methods to this setting leads to sub-optimal results. We observe that notorious catastrophic forgetting  [ 22 ,  8 ,  23 ,  16 ]  occurs for the previous target domain as the network is trained on the current target domain (see Fig.

1  (c)).
Recently, Wu  et al .

[ 38 ]  introduced a method for adapting to changing environments such as varying weather and lighting conditions, which can be viewed as Continual UDA for semantic segmentation. By using the style-memory of each environment, the method transfers the style of the source environment into that of each target environment. It achieves superior performance over the previous UDA methods under the specific setting. However, the experiments were conducted within one synthetic dataset  [ 30 ] , and accordingly, we observe inferior performance when applied to multiple real-world datasets (see Sec.

3.2  ). We see that the style transfer is not enough to overcome the catastrophic forgetting problem when considerable domain discrepancy exists.

Figure 1:  (a) When the traditional UDA method is applied to the Continual UDA task, it suffers from catastrophic forgetting for the previously learned target domain (Target 1). (b) Our proposed ETM framework alleviates such forgetting by expanding a lightweight sub-memory called TM. (c) Qualitative results show that our framework actually mitigates forgetting on the previous target domain. AdaptSegNet  [ 32 ]  is used as the baseline UDA method.

To this end, we propose a novel  Expanding Target-specific Memory  (ETM) framework for Continual UDA for semantic segmentation on real-world datasets. In the framework, we introduce  Target-specific Memory  (TM) for each target domain. Inspired by the previous works  [ 36 ,  40 ,  26 ]  in the continual learning field, it is considered that the constant capacity of the existing networks may not be enough to handle multiple target domains as it faces a huge domain discrepancy. The proposed framework is illustrated in Fig.

1  (b). Specifically, a lightweight sub-memory called TM is initiated, trained, and stored for each target domain. Each TM contains unique information corresponding to each domain discrepancy by designing the TM in consideration of the structure, the forwarding path and the expanded location (see Section

2.1  ) .
When testing the network over the previous domains, the stored TM of the corresponding previous domain is used. In this way, the network overcomes the catastrophic forgetting problem.
In addition, we design a  Double Hinge Adversarial  (DHA) loss that enhances the overall UDA performance. By optimizing the DHA loss function, the segmentation network aligns the source and target domain data while considering geometric relations between them. We observe that the DHA loss is more suitable for the UDA objective. Without loss of generality, our framework can be applied to other adversarial learning-based UDA methods.

The main contribution of this paper is three-fold:

•

To the best of our knowledge, we address Continual UDA for semantic segmentation on real-world datasets for the first time, which considers more practical scenarios.

•

We propose the ETM framework for Continual UDA. We deal with the catastrophic forgetting problem by expanding a little amount of model capacity (TM), which is the way that is firstly introduced in this field. Moreover, we propose the DHA loss function to enhance the performance of UDA with adversarial learning.

•

We validate our framework by conducting experiments using two synthetic datasets (GTA5  [ 28 ] , SYNTHIA  [ 30 ] ), and three real-world datasets (CityScapes  [ 6 ] , IDD  [ 34 ] , Cross-City  [ 5 ] ) with large domain discrepancy. The model trained with the ETM framework outperforms other state-of-the-art models under the same conditions.

2  Approach

We first formalize the Continual UDA problem by defining the following notations. Let

𝒮  =

{

(

x  1  s

,

y  1  s

)

,  …  ,

(

x

N  s

s

,

y

N  s

s

)

}

𝒮

superscript

subscript  𝑥  1

𝑠

superscript

subscript  𝑦  1

𝑠

…

superscript

subscript  𝑥

subscript  𝑁  𝑠

𝑠

superscript

subscript  𝑦

subscript  𝑁  𝑠

𝑠

\mathcal{S}=\{(x_{1}^{s},y_{1}^{s}),\ldots,(x_{N_{s}}^{s},y_{N_{s}}^{s})\}

denote the source domain data, which consists of

N  s

subscript  𝑁  𝑠

N_{s}

images (

x  1  s

,  …  ,

x

N  s

s

superscript

subscript  𝑥  1

𝑠

…

superscript

subscript  𝑥

subscript  𝑁  𝑠

𝑠

x_{1}^{s},...,x_{N_{s}}^{s}

) and corresponding labels (

y  1  s

,  …  ,

y

N  s

s

superscript

subscript  𝑦  1

𝑠

…

superscript

subscript  𝑦

subscript  𝑁  𝑠

𝑠

y_{1}^{s},...,y_{N_{s}}^{s}

). Multiple

T

𝑇

T

target domains without any annotations are defined as

{

𝒯  i

}

i  =  1

T

superscript