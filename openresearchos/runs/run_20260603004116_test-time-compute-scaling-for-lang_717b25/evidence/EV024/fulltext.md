[2207.13600] Lightweight and Progressively-Scalable Networks for Semantic Segmentation

∎

1

1  institutetext:

Yiheng Zhang

2

2  institutetext:  JD Explore Academy, Beijing, China.

2

2  email:  yihengzhang.chn@gmail.com

3

3  institutetext:  Ting Yao

4

4  institutetext:  JD Explore Academy, Beijing, China.

4

4  email:  tingyao.ustc@gmail.com

5

5  institutetext:  Zhaofan Qiu

6

6  institutetext:  JD Explore Academy, Beijing, China.

6

6  email:  zhaofanqiu@gmail.com

7

7  institutetext:  Tao Mei

8

8  institutetext:  JD Explore Academy, Beijing, China.

8

8  email:  tmei@live.com

Lightweight and Progressively-Scalable Networks

for Semantic Segmentation

Yiheng Zhang

Ting Yao

Zhaofan Qiu

Tao Mei

(Received: date / Accepted: date)

Abstract

Multi-scale learning frameworks have been regarded as a capable class of models to boost semantic segmentation. The problem nevertheless is not trivial especially for the real-world deployments, which often demand high efficiency in inference latency. In this paper, we thoroughly analyze the design of convolutional blocks (the type of convolutions and the number of channels in convolutions), and the ways of interactions across multiple scales, all from lightweight standpoint for semantic segmentation. With such in-depth comparisons, we conclude three principles, and accordingly devise Lightweight and Progressively-Scalable Networks (LPS-Net) that novelly expands the network complexity in a greedy manner. Technically, LPS-Net first capitalizes on the principles to build a tiny network. Then, LPS-Net progressively scales the tiny network to larger ones by expanding a single dimension (the number of convolutional blocks, the number of channels, or the input resolution) at one time to meet the best speed/accuracy tradeoff. Extensive experiments conducted on three datasets consistently demonstrate the superiority of LPS-Net over several efficient semantic segmentation methods. More remarkably, our LPS-Net achieves 73.4% mIoU on Cityscapes test set, with the speed of 413.5FPS on an NVIDIA GTX 1080Ti, leading to a performance improvement by 1.5% and a 65% speed-up against the state-of-the-art STDC. Code is available at  https://github.com/YihengZhang-CV/LPS-Net .

Keywords:  Convolutional Neural Networks Semantic Segmentation Lightweight Scalable

1  Introduction

Semantic segmentation is to assign semantic labels to every pixel of an image or a video frame. With the development of deep neural networks, the state-of-the-art networks have successfully pushed the limits of semantic segmentation with remarkable performance improvements. For example, DeepLabV3+  (Chen et al.,  2018b )  and Hierarchical Multi-Scale Attention  (Tao et al.,  2020 )  achieve 82.1% and 85.4% mIoU on Cityscapes test set, which are almost saturated on that dataset. The recipe behind these successes originates from multi-scale learning. In the literature, the recent advances involve utilization of multi-scale learning for semantic segmentation along three different dimensions: U-shape  (Chen et al.,  2020 ; Peng et al.,  2017 ) , pyramid pooling  (Chen et al.,  2018b ; Zhao et al.,  2017 ) , and multi-path framework  (Chen et al.,  2016 ; Tao et al.,  2020 ) . The U-shape structure hierarchically fuses the features to gradually increase the spatial resolution and naturally produce multi-scale features. The pyramid pooling methods delve into multi-scale information through executing spatial or atrous spatial pyramid pooling at multiple scales. Unlike the former two research schemes, the multi-path frameworks resize the input image to multiple resolutions or scales and feed each scale into an individual path of a deep network. By doing so, the multi-path design places the input resolutions from high to low in parallel and explicitly maintains the high-resolution information rather than recovering from low-scale feature maps. As a result, the learnt features are potentially more capable of classifying and localizing each pixel.

We employ this elegant recipe of multi-path framework and further evolve such type of architectures with good accuracy/speed tradeoff for semantic segmentation. Our philosophies are from two perspectives: (1) lightweighting computational units for semantic segmentation, and (2) progressively scaling up the network while balancing accuracy and inference latency. We propose to explore the first problem by probing the basic unit of convolutional blocks, including the type of convolutions and the number of channels in convolutions, on the basis of several uniqueness (e.g., large feature maps, thin channel widths) in efficient semantic segmentation. Moreover, we further elaborate different ways of interaction across multiple paths with respect to the accuracy/speed tradeoff. Based on these lightweight practice, we build a tiny model, and then progressively expand the tiny model along multiple possible dimensions and select a single dimension that achieves the best tradeoff in each step to alleviate the second issue of accuracy/speed balance.

Figure 1 :

LPS-Net executes the growth of network complexity through expanding a single dimension of the number of convolutional blocks (Depth), the number of channels (Width), or the input resolution (Resolution) at one time.

Figure 2 :

Comparisons of inference speed/accuracy tradeoff on Cityscapes validation set. LPS-Net (-S, -M, and -L) which are progressively expanded along multiple dimensions demonstrate a good balance between accuracy and inference speed compared to other manually-/auto-deigned models.

To materialize the idea, we present Lightweight and Progressively-Scalable Networks (LPS-Net) for efficient semantic segmentation. Specifically, LPS-Net bases the multi-path design upon the low-latency regime. Each path takes the resized image as the input to an individual network, which consists of stacked convolutional blocks. The networks across paths share the same structure but have independent learnable weights. The outputs from all the paths are aggregated to produce the score maps, which are upsampled via bilinear upsampling for pixel-level predictions.
In an effort to achieve a lightweight and efficient architecture, we look into the basic unit of convolutional blocks and empirically suggest to purely use

3  ×  3

3  3

3\times 3

convolutions with

2  n

superscript  2  𝑛

2^{n}

-divisible channel widths.
Furthermore, we capitalize on a simple yet effective way of bilinear interpolation to encourage mutual exchange and interactions between paths. With these practical guidelines, LPS-Net first builds a tiny network and then scales the tiny network to a family of larger ones in a progressive manner. Technically, LPS-Net executes the growth of network complexity through expanding a single dimension of the number of convolutional blocks, the number of channels, or the input resolution at one time, as depicted in Figure

1  . In our case, LPS-Net ensures a good balance between accuracy and inference speed during expansion, and shows the superiority over other manually-/auto-deigned models, as shown in Figure

2  .

In summary, we have made the following contributions: (1) The lightweight design of convolutional blocks and the way of path interactions in multi-path framework are shown capable of regarding as the practical principles for efficient semantic segmentation; (2) The exquisitely devised LPS-Net is shown able to progressively expand the network complexity while striking the right accuracy-efficiency tradeoff; (3) LPS-Net has been properly verified through extensive experiments over three datasets, and superior capability is observed on both NVIDIA GPUs and embedded devices in our experiments.

2  Related Works

Semantic Segmentation. 
With the success of convolutional neural networks (CNNs), FCN  (Long et al.,  2015 )  which deploys CNN in a fully convolutional manner enables dense semantic prediction and end-to-end training for semantic segmenta