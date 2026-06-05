[2303.16322] FMAS: Fast Multi-Objective SuperNet Architecture Search for Semantic Segmentation

FMAS: Fast Multi-Objective SuperNet Architecture Search for Semantic Segmentation

Zhuoran Xiong

zhuoran.xiong@mail.mcgill.ca

McGill University  address  Montreal  Canada

,

Marihan Amein

marihan.amein@mail.mcgill.ca

McGill University  address  Montreal  Canada

,

Olivier Therrien

olivier.therrian@mail.mcgill.ca

McGill University  address  Montreal  Canada

,

Warren J. Gross

warren.gross@mcgill.ca

McGill University  address  Montreal  Canada

and

Brett H. Meyer

brett.meyer@mcgill.ca

McGill University  address  Montreal  Canada

(2023)

Abstract.

We present FMAS, a fast multi-objective neural architecture search framework for semantic segmentation.
FMAS subsamples the structure and pre-trained parameters of DeepLabV3+, without fine-tuning, dramatically reducing training time during search.
To further reduce candidate evaluation time, we use a subset of the validation dataset during the search.
Only the final, Pareto non-dominated, candidates are ultimately fine-tuned using the complete training set.
We evaluate FMAS by searching for models that effectively trade accuracy and computational cost on the PASCAL VOC 2012 dataset.
FMAS finds competitive designs quickly, e.g., taking just 0.5 GPU days to discover a DeepLabV3+ variant that reduces FLOPs and parameters by 10

%

percent

\%

and 20

%

percent

\%

respectively, for less than 3% increased error. We also search on an edge device called GAP8 and use its latency as the metric. FMAS is capable of finding 2.2

×

\times

faster network with 7.61% MIoU loss.

NAS, Segmantic Segmentation, Edge Computing, TinyML

†

†  copyright:  rightsretained

†

†  journalyear:  2023

1.  Introduction

Semantic image segmentation  (Liu et al . ,  2019b )  is one of the fundamental applications in computer vision:
it helps us understand scenes by identifying the various objects in an image, and their corresponding locations, by predicting an independent class label for each pixel.
Image segmentation is essential to many applications that run on resource-constrained embedded hardware, such as:
autonomous driving, medical imaging, and biometric authentication.

Convolutional neural networks (CNN) that achieve state-of-the-art (SOTA) results in image segmentation have sophisticated structures that are generally optimized for accuracy.
They also often require larger feature maps than image classification tasks to be able to produce pixel-wise labels, resulting in a large memory footprint.
In some cases, e.g., autonomous vehicles, image segmentation must be performed in real-time, which makes the development of efficient models for deployment to edge devices critical.

Multi-objective network architecture search has been proposed for the purpose of finding efficient models, but the time required to train candidates is prohibitive. DPC  (Chen et al . ,  2018a ) , for instance, requires 2,590 GPU days to converge to good designs.
SqueezeNAS  (Shaw et al . ,  2019 )  and TASC  (Nekrasov et al . ,  2020 )  require 14 and 16 GPU days, respectively.
Fully training a DeepLabV3+ variant with a Modified Xception backbone requires 0.88 GPU days on average, and an additional 0.014 GPU days for evaluation on the validation data set.
This makes it challenging to search for efficient variants of off-the-shelf networks in a practical amount of time.
Once-for-all  (Cai et al . ,  2019 )  (OFA) solves this problem by essentially training a supernetwork and all its possible subnetworks simultaneously, at an additional computational cost.

To accelerate NAS for image segmentation, we propose Fast Multi-objective Architectural Search (FMAS), a fast multi-objective NAS framework for semantic image segmentation at the edge.
FMAS uses DeepLabV3+ as a supernet to search for computationally-efficient models.
DeepLabV3+  (Chen et al . ,  2018b )  (DL3+) is a SOTA encoder-decoder CNN which employs backbones like Modified Xception  (Chen et al . ,  2018b )  or MobileNetV2  (Sandler et al . ,  2019 )  for feature extraction.
FMAS addresses the computational complexity of supernet architectural search in two key ways.
First, candidate networks evaluated by FMAS sub-sample the pre-trained weights of DL3+.
FMAS uses the resulting model performance— without fine-tuning —and computational cost, to direct the search toward models with advantageous accuracy-cost trade-offs;
fine-tuning is only performed on final set of Pareto-optimal models.
Second, FMAS evaluates candidates on a subset of the validation set. Neither optimization significantly affects NAS decision making, but together they dramatically reduce search time.

We conducted experiments with PASCAL VOC 2012  (Everingham et al . ,  2010 )  to evaluate the effectiveness of our FMAS.
While reducing search time by 99

%

percent

\%

by using weight sharing and a subset of the validation set, our method is capable of finding models with less than a

3  %

percent  3

3\%

increase in MIoU error.
Moreover, we demonstrate the deployment of image segmentation to TinyML-class  (Ray,  2022 )  systems by targeting the ultra-low-power GAP8  (Flamand et al . ,  2018 )  platform.
FMAS finds a sub-network model that is 2.2

×

\times

faster on GAP8 with a loss in MIoU of 7.61% compared to the supernet.

Figure 1.  The supernet DeepLabV3+ of FMAS. The base structure is an encoder-decoder preceded with a Modified Xception backbone  (Chen et al . ,  2018b ) . The search space operations and how they are decoded into the supernet structure are highlighted.

Specifically, we make the following contributions:
1) We employ NSGA-II  (Deb et al . ,  2002 ) , an elitist genetic algorithm to quickly search for efficient sub-structures in a pre-trained supernet that has been optimized for the targeted task;
2) To accelerate the search, we use a subset of the validation dataset as a proxy; and b) only fine-tune Pareto-optimal models; 3) We optimize semantic segmentation for edge inference, deploying models on the GAP8 platform. To the best of our knowledge, this is the first work to perform NAS for semantic segmentation on ultra-low-power hardware.

2.  Related Work

To cut search time, multi-objective NAS methods for image segmentation represent the search space in either a  hierarchical

(Liu et al . ,  2019a ) ,
or a  template-based  way  (Chen et al . ,  2018a ; Shaw et al . ,  2019 ; Nekrasov et al . ,  2020 ) . A hierarchical search performs an alternating bi-level optimization, starting by first optimizing the high-level design, then optimizing the internal structure.
We adopt a template-based search, which tends to converge faster than hierarchical search, as it constrains the high-level structure of the candidates to a pre-defined architecture.

A number of Reinforcement Learning (RL) NAS approaches for image segmentation have been proposed  (Nekrasov et al . ,  2020 ,  2019b ,  2019a ) . RL-based NAS tends to converge slowly, as it prioritizes long-term reward.
Other approaches  (Chen et al . ,  2018a ; Liu et al . ,  2019a ; Shaw et al . ,  2019 )  relax the architectural parameters in the search space into a continuous, differentiable form, as gradient descent converges faster. It is difficult, however, to represent candidate cost, whether latency or FLOPs, in differentiable form.
Consequently, some approaches  (Calisto and Lai-Yuen,  2019 ,  2021 ; Lu et al . ,  2019 )  use elitist GAs that consider Pareto-dominance in selection.
We likewise employ NSGA-II.

Several approaches have been proposed to quickly evaluate the accuracy of candidate models. Chen et al.  (Chen et al . ,  2018a )  designed proxy networks, simplified variants of the candidates that are faster to train and representative of the original accuracy.
Although this saves some time, they still require 2,590 GPU days.
Alternatively, we use weight sharing, adopting a supernet design space structure, as in  (Bae et al . ,  2019 ; Ph