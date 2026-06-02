[2003.11818] Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection

Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection

Jianyuan Guo  1,2  ,
Kai Han  2  , Yunhe Wang  2  ,
Chao Zhang  1  ,
Zhaohui Yang  1

Han Wu  1  , Xinghao Chen  2  , Chang Xu  3

1  Key Lab of Machine Perception (MOE), Dept. of Machine Intelligence, Peking University.

2  Noah’s Ark Lab, Huawei Technologies.
 3  School of Computer Science, Faculty of Engineering, University of Sydney.

{jyguo, zhaohuiyang, wuhancs}@pku.edu.cn, chzhang@cis.pku.edu.cn
 c.xu@sydney.edu.au, {kai.han, yunhe.wang, xinghao.chen}@huawei.com

Corresponding author.

Abstract

Neural Architecture Search (NAS) has achieved great success in image classification task. Some recent works have managed to explore the automatic design of efficient backbone or feature fusion layer for object detection. However, these methods focus on searching only one certain component of object detector while leaving others manually designed. We identify the inconsistency between searched component and manually designed ones would withhold the detector of stronger performance. To this end, we propose a hierarchical trinity search framework to simultaneously discover efficient architectures for all components ( i.e .

backbone, neck, and head) of object detector in an end-to-end manner.
In addition, we empirically reveal that different parts of the detector prefer different operators. Motivated by this, we employ a novel scheme to automatically screen different sub search spaces for different components so as to perform the end-to-end search for each component on the corresponding sub search space efficiently. Without bells and whistles, our searched architecture, namely Hit-Detector, achieves 41.4% mAP on COCO minival set with 27M parameters. Our implementation is available at  https://github.com/ggjy/HitDet.pytorch .

1  Introduction

Object detection is a fundamental task in computer vision and has been widely applied in the real world, such as autonomous vehicles and surveillance video. The advancement of deep learning results in a number of convolutional neural network based solutions of object detection task. Typically, deep learning based detectors can be divided into two categories: (i) one-stage methods including YOLO  [ 39 ]  and SSD  [ 32 ]  which directly utilize CNNs to predict the bounding boxes of interest; and (ii) two-stage approaches such as Faster R-CNN  [ 41 ]  that generates the bounding boxes after extracting region proposals upon a region proposal network (RPN). The advantage of single-stage methods lies in the high detection speed whereas two-stage methods dominate in detection accuracy.

A series of either one-stage  [ 40 ,  26 ,  21 ]  or two-stage approaches  [ 8 ,  17 ,  5 ]  have been developed to continuously boost the detection speed and accuracy. However, the manually designed architectures heavily rely on the expert knowledge  [ 24 ]  while still might be suboptimal. Thus neural architecture search (NAS) that automates the design of network architectures and minimizes human labor has drawn much attention and made impressive progress, especially in image classification tasks  [ 56 ,  28 ,  44 ,  30 ,  42 ,  45 ,  47 ,  54 ,  16 ] . Compared with classification tasks that simply determine  what  the image is, detection tasks need to further figure out  where  the objects are.
NAS for object detection therefore requires more careful design and is much more challenging.

Table 1:  Comparing our model against some typical two-stage detectors on COCO benchmark. “4c” indicates four convolution layers.  †  means that NAS-FPN is originally searched for one-stage RetinaNet, we replace the neck in FPN with NAS-FPN to construct the two-stage detector.

Model

Backbone

Neck

Head

mAP

(#params/M)

(#params/M)

(#params/M)

(%)

FPN baseline

[  25  ]

Res50 (23.5)

FPN (3.3)

2fc (14.3)

36.2

NAS-FPN

[  12  ]

†

Res50 (23.5)

NAS-FPN (30.4)

2fc (14.3)

38.9

Backbone baseline

Res50 (23.5)

FPN (3.3)

4c1fc (15.6)

36.8

DetNAS

[  7  ]

DetNet (9.4)

FPN (2.8)

4c1fc (15.6)

40.2

NAS-FPN + DetNAS

DetNet (9.4)

NAS-FPN (29.7)

4c1fc (15.6)

39.4

Hit-Detecotr (ours)

Ours (13.9)

Ours (2.7)

Ours (9.9)

41.4

Modern object detection systems usually consist of four components: (a) backbone for extracting semantic features,  e.g .

ResNet-50  [ 18 ]  and ResNeXt-101  [ 48 ] ; (b) neck for fusing multi-level features,  e.g .

feature pyramid networks (FPN)  [ 25 ] ; (c) RPN for generating proposals (usually in two-stage detector); and (d) head for object classification and bounding box regression.
Recently, there are works that explore NAS in object detection tasks to search for a good architecture of the backbone  [ 36 ,  7 ]  or FPN  [ 12 ,  50 ] . With the searched backbone or FPN architectures, these works have achieved higher accuracy than the manually designed baselines with similar numbers of parameters and FLOPs.

Nevertheless, exploiting only one part of the detector at a time cannot fulfill the the potential of each component, and the separately searched backbone and neck may not be optimal or compatible with each other. As shown in Table

1  , NAS-FPN  [ 12 ]  for neck searching achieves a

38.9  %

percent  38.9

38.9\%

mAP which is higher than that of vanilla FPN, and DetNAS  [ 7 ]  for backbone searching outperforms vanilla ResNet-50 backbone with

40.2  %

percent  40.2

40.2\%

mAP. However, a straightforward
combination of NAS-FPN and DetNAS leads to a worse mAP,  i.e .  ,

39.4  %

percent  39.4

39.4\%

, let alone outperforms two models. This insightful observation motivates us to take the detector as a whole in NAS.

In this paper, we propose to simultaneously search all components of the detector in an end-to-end manner. Due to the differences among optimum space for each component and the difficulty in optimizing within large search space, we introduce a hierarchical way to mine the proper sub search space from the large volume of operation candidates. In particular, our proposed  Hit-Detector  framework consists of two key procedures as shown in Fig.

1  . First, given a large search space containing all the operation candidates, we screen out the customized sub search space suitable for each part of detector with the help of group sparsity regularization. Secondly, we search the architectures for each part within the corresponding sub search space by adopting the differentiable manner.
Extensive experiments demonstrate that our Hit-Detector achieves state-of-the-art results on the benchmark dataset, which validates the effectiveness of the proposed method.

Our main contributions can be summarized as follows:

•

This is the first time that architectures of backbone, neck and head are searched altogether in an end-to-end manner for object detection.

•

We show that different parts prefer different operations, and propose a hierarchical way to specify appropriate sub search space for different components in detection system to improve the sampling efficiency.

•

Our Hit-Detector outperforms either hand-crafted or automatically searched networks by a large margin with much less computational complexity.

2  Related Work

Figure 1:

Overview of our Hit-Detector architecture search framework. Our method focuses on searching better architectures of the trinity,  i.e .

backbone, neck, and head for object detector. (a) is the whole search space; (b) indicates three sub search spaces for different components; and (c) shows the end-to-end searching for object detector. “TBS” denotes the layer to be searched.

Object detection aims at determining  what  and  where  the object is when given an image. Riding the wave of convolutional neural networks, noticeable improvements in accuracy have been made in both one-stage  [ 39 ,  32 ,  26 ,  21 ,  10 ,  55 ]  and two-stage  [ 41 ,  17 ,  8 ,  25 ,  19 ,  20 ,  5 ,  31 ]  detectors. Generally,