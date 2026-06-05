[2108.06859] Probeable DARTS with Application to Computational Pathology

Probeable DARTS with Application to Computational Pathology

Sheyang Tang  1  , Mahdi S. Hosseini  2

1

1  footnotemark:

1

, Lina Chen  3  , Sonal Varma  4  , Corwyn Rowsell

5  ,

Savvas Damaskinos  6 , Konstantinos N. Plataniotis  7 , and Zhou Wang 1

1 University of Waterloo, Canada

2 University of New Brunswick, Canada

3 Sunnybrook health science center, Canada

4 Kingston Health Sciences Center, Canada

5 St. Michaels Hospital, Canada

6 Huron Digital Pathology, Canada

7 University of Toronto, Canada

https://github.com/mahdihosseini/DARTS-ADP

Equal contribution

Abstract

AI technology has made remarkable achievements in computational pathology (CPath), especially with the help of deep neural networks. However, the network performance is highly related to architecture design, which commonly requires human experts with domain knowledge. In this paper, we combat this challenge with the recent advance in neural architecture search (NAS) to find an optimal network for CPath applications. In particular, we use differentiable architecture search (DARTS) for its efficiency. We first adopt a probing metric to show that the original DARTS lacks proper hyperparameter tuning on the CIFAR dataset, and how the generalization issue can be addressed using an adaptive optimization strategy. We then apply our searching framework on CPath applications by searching for the optimum network architecture on a histological tissue type dataset (ADP). Results show that the searched network outperforms state-of-the-art networks in terms of prediction accuracy and computation complexity. We further conduct extensive experiments to demonstrate the transferability of the searched network to new CPath applications, the robustness against downscaled inputs, as well as the reliability of predictions.

1  Introduction

Recent years have witnessed great advances in AI-based Computational Pathology (CPath)  [ 22 ,  15 ] . The emerging AI techniques have shown their superiority in more accurate, efficient, and large-scale medical diagnoses  [ 4 ] . In particular, Convolutional Neural Networks (CNNs) have been widely employed to extract meaningful information from medical images for various pathology applications, including disease diagnoses  [ 5 ,  38 ] , medical image segmentation  [ 27 ,  31 ] , etc. Yet designing the network architectures has long been a manual process that requires adequate domain knowledge. As a result, it has become a common standard that architectures from CV applications (such as ResNet  [ 8 ]  and GoogLeNet  [ 32 ] ) are transferred for technical developments in other fields, including CPath  [ 29 ,  34 ] .

Table 1:  Comparison between CV and CPath datasets.

General Stats

CV

CPath

CIFAR-10

CIFAR-100

ADP

Training size

50000

50000

14134

Validation size

-

-

1767

Test size

10000

10000

1767

Resolution

32x32

32x32

272x272

# classes

10

100

33

Label form

single-label

single-label

multi-label

Background

various

various

white

(a)

CIFAR

(b)

ADP

Figure 1:  Sample images from CV and CPath datasets.

The ultimate question is whether transferring architectures between the two domains is an efficient strategy. To answer this question, we first demonstrate how CV and CPath datasets are different. Here we compare the CIFAR  [ 19 ]  and ADP  [ 11 ]  datasets. Besides different data structures shown in Table.

1  , the nature of images from both sides is also different, which makes CV datasets more complicated.  First , the pixel resolution in CPath is fixed, corresponding to a fixed field of view (FOV) size. The root cause of such uniformity is the acquisition of whole slide images by a scanner in a much more controlled environment from both optics and illumination viewpoint  [ 11 ] . In contrast, the pixel resolution in CV is randomly distributed across different images due to different image setup and configurations. CV images are captured in natural scenes where the distance has much variance. Examples from each imaging modality are shown in Figure

1(a)  , where the ship images are taken from a further distance than the dog ones, resulting in larger pixel size and lower resolution.  Second , target objects in CV images only occupy part of the whole FOV and the rest are background which is irrelevant to the class label. Note that the diversity of the background in Figure

1(a)

is very high. This is quite different in CPath where the background information is obtained from an empty area of the sample using uniform white light illumination  [ 11 ]  –leading to more uniform and homogeneous images. This is illustrated in Figure

1(b)  , where the white part denotes the background. In the light of this difference, we form a hypothesis that such simplified imaging modality in CPath translates to simpler network architecture compared to CV. To this end, new network architectures should be designed for CPath applications.

Neural architecture search (NAS) has recently been proposed to automate the design of neural networks by searching for the optimal network structure on a given dataset. In many CV applications, NAS has outperformed state-of-the-art manually designed networks in terms of prediction accuracy and computation complexity  [ 7 ] . In medical image analysis, it has been utilized to find suitable networks for various applications, such as image segmentation for Magnetic Resonance Imaging (MRI)  [ 17 ,  36 ,  3 ] , ultrasound imaging  [ 35 ] , disease diagnoses from Computed Tomography (CT) scans  [ 16 ,  9 ] , etc. In pathology, however, NAS is not fully explored. There is a lack of a general framework that can be easily extended to various CPath applications.

In this work, we propose an architecture search platform based on differentiable architecture search (DARTS)  [ 23 ] . We choose DARTS because it is gradient-based and thus much more efficient and computation-friendly than other searching strategies including reinforcement learning  [ 39 ]  and evolutionary algorithms  [ 26 ] . DARTS achieves this by relaxing the search space to be continuous and dividing the whole pipeline into a search phase and an evaluation phase. However, in CV applications, it is reported that DARTS tends to exhibit overfitting issues, and the searched architecture does not generalize well in the evaluation phase  [ 21 ,  37 ] . To combat these challenges, we first conduct searching on CIFAR  [ 19 ]  and utilize a probing metric  stable rank

[ 12 ]  for each layer. In this way, we can better monitor the searching process and show that the overfitting issue comes from improper hyperparameter tuning. In addition, we use an adaptive optimizer  Adas

[ 12 ]  that automatically tunes the learning rates for each layer based on their probing metrics, so that the generalization ability of the searched architecture is improved. We then apply this searching framework on ADP  [ 11 ] , which contains a great variety of histological tissue types that are representative enough, so that the searched architecture can generalize well in different CPath applications. The searched network outperforms the state-of-the-art architectures in the speed-accuracy trade-off, which is crucial for real-time high-throughput CPath applications. We further conduct extensive experiments to show the transferability of the searched architecture on new CPath datasets, demonstrate its robustness against decreased input images, and verify its superiority in extracting label-pertinent features. Our main contributions are listed below:

•

We use a probing metric to show that the existing DARTS framework lacks proper hyperparameter tuning, and use an adaptive optimizer to improve the generalization ability of the searched model;

•

We apply the proposed searching platform on CPath applications and show the superiority of the searched model in prediction accuracy and computation complexity;

•

We demo