[2112.07133] 1 Introduction

CLIP-Lite: Information Efficient Visual Representation Learning with Language Supervision

Aman Shrivastava

Ramprasaath R. Selvaraju

Nikhil Naik

Vicente Ordonez

University of Virginia

Salesforce Research

Salesforce Research

Rice University

Abstract

We propose CLIP-Lite, an information efficient method for visual representation learning by feature alignment with textual annotations. Compared to the previously proposed CLIP model, CLIP-Lite requires only one negative image-text sample pair for every positive image-text sample during the optimization of its contrastive learning objective. We accomplish this by taking advantage of an information efficient lower-bound to maximize the mutual information between the two input modalities. This allows CLIP-Lite to be trained with significantly reduced amounts of data and batch sizes while obtaining better performance than CLIP at the same scale. We evaluate CLIP-Lite by pretraining on the COCO-Captions dataset and testing transfer learning to other datasets. CLIP-Lite obtains a +14.0% mAP absolute gain in performance on Pascal VOC classification, and a +22.1% top-1 accuracy gain on ImageNet, while being comparable or superior to other, more complex, text-supervised models. CLIP-Lite is also superior to CLIP on image and text retrieval, zero-shot classification, and visual grounding. Finally, we show that CLIP-Lite can leverage language semantics to encourage bias-free visual representations that can be used in downstream tasks. Implementation:  https://github.com/4m4n5/CLIP-Lite

1  Introduction

Figure 1:  Given a batch of

n

𝑛

n

image-caption pairs

{

(

I  i

,

T  i

)

}

subscript  𝐼  𝑖

subscript  𝑇  𝑖

\{(I_{i},T_{i})\}

, CLIP requires a large number of negative pairs

{

(

I  i

,

T  j

)

∣

i  ≠  j

}

conditional-set

subscript  𝐼  𝑖

subscript  𝑇  𝑗

𝑖  𝑗

\{(I_{i},T_{j})\mid i\neq j\}

due to the need to pair every image in the batch with captions from other images. Whereas, CLIP-Lite can learn representations using a single negative pair (in red) for every positive pair (in green).

Pretraining image classification networks on the Imagenet dataset has led to visual representations that transfer to other tasks  (Girshick et al.,,  2014 ; Long et al.,,  2015 ; Vinyals et al.,,  2015 ; Antol et al.,,  2015 ; Zhu et al.,,  2016 ) . However, such classification based pretraining requires a large amount of human-annotated data which is hard to obtain at scale. In contrast, captioned image data is an information-dense source of supervision that is relatively cheap to collect and plentiful on the internet. Therefore, recent methods have used joint vision-language pretraining to learn representations from image-caption pairs  (Desai and Johnson,,  2021 ; Sariyildiz et al.,,  2020 ) . However, methods such as VirTex  (Desai and Johnson,,  2021 )  which train on complex language modeling tasks such as masked language modeling, token classification, and captioning fail to align features in a common latent space.

Recently, CLIP  (Radford et al.,,  2021 ) , a vision-language pretraining model, was developed using contrastive learning between the two modalities on an Internet-sized dataset of 400 million image-caption pairs. Contrastive learning methods work by pulling closer the representations of independent views of the same datum  i.e.  a positive or matching image-caption pair and pushing apart the representations of independent views of different data  i.e.  negative or non-matching image-caption pairs. However, contrastive learning in vision-language pretraining still has some limitations as it seems to be most effective only with large scale data, and it requires a large number of negative image-caption pairs during training. Our work aims to address and explore these two limitations by proposing CLIP-Lite, an information efficient variation of CLIP that is useful even in smaller data regimes, does not rely in as many negative sample pairs during training, and provides comparable or superior performance on standard benchmarks against other methods trained at the same scale. Our work is motivated by the observation that multiple contrastive objectives maximize a lower-bound on the mutual information between two or more views of the same datum  (Wu et al.,,  2020 ) .

CLIP particularly maximizes the mutual information between the image and its caption by using a mutual information lower bound based on InfoNCE  (Oord et al.,,  2018 ) . The InfoNCE bound has seen wide adoption due to its favorable properties such as stability and low variance. However, the the bound is theoretically loose in cases when the true mutual information is larger than

log  ⁡  K

𝐾

\log K

where

(

K  −  1

)

𝐾  1

(K-1)

is the number of negative samples used for training. The negative pairs can be randomly sampled but usually a large amount of negative pairs are required to have a good estimate of the mutual information between the two input streams, and hence the need for rather large batch sizes  (Bachman et al.,,  2019 ; Chen et al.,,  2015 )  or memory-banks  ( Chen et al., 2020b,  ; Tian et al.,,  2019 ; He et al.,,  2020 ) .

We instead adopt a lower bound based on Jenssen Shannon Divergence to maximize the mutual information  (Hjelm et al.,,  2018 ; Nowozin et al.,,  2016 ) , thus requiring no more than one negative example pair for each positive example pair. This reduces the number of negative examples in a training batch to

O  ​

(  n  )

𝑂  𝑛

O(n)

, where

n

𝑛

n

is the batch size. In contrast, CLIP uses

O  ​

(

n  2

)

𝑂

superscript  𝑛  2

O(n^{2})

negative example pairs per batch. Figure

1

illustrates this difference. We implement this strategy and demonstrate thoroughly the efficacy of CLIP-Lite through experiments on several tasks and datasets at various scales. Our method demonstrates impressive data efficiency and is able to outperform CLIP trained on the entire COCO-Captions dataset while only training on

20  %

percent  20

20\%

of the same dataset. We also demonstrate that CLIP-Lite can be used as a good source of pretrained features by showing good generalization on Pascal VOC and Imagenet classification. We also show that the visual feature backbone of CLIP-Lite can be finetuned in the iNaturalist dataset to match top performances on this benchmark with caption supervision pretraining. Furthermore, we show that CLIP-Lite leads to good visual features for image retrieval compared to regular CLIP trained on COCO Captions. We also demonstrate that CLIP-Lite enables the removal of concepts from visual representations which we show can be applied in bias mitigation. Our work extends and complements the work using contrastive learning, especially addressing the computational requirements of the original CLIP model in terms of memory overhead through minimizing the number of negative sample image-text pairs required during training and shows its effectiveness in smaller data regimes including for zero-shot learning on CIFAR-10, image-text retrieval and unsupervised object localization.

2  Related Work

Our work is related to several strands of research on visual pretraining without full-supervision.

Vision-Language Pretraining:  Research on learning visual representations by using textual labels or annotations has a long history. In  (Quattoni et al.,,  2007 ) , the authors learn data-efficient image representations using manifold learning in the weight space of classifiers trained to predict tokens in image captions. Following this work,  (Joulin et al.,,  2016 )  used convolutional neural networks to predict words in image captions to learn image representations.

This approach was later extended in  (Lei Ba et al.,,  2015 )  where the model learns to predict phrase n-grams, which demonstrated impressive zero-shot performance on downstream classification tasks.
Recently, VirTex  (Desai and Johnson,,  2021 )  used proxy langu