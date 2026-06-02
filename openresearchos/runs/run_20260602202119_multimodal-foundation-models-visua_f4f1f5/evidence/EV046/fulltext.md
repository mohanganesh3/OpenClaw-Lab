[2204.03647] Adapting CLIP For Phrase Localization Without Further Training

Adapting CLIP For Phrase Localization Without Further Training

Jiahao Li

Greg Shakhnarovich

Raymond A. Yeh
 Toyota Technological Institute at Chicago

{jiahao, greg, yehr}@ttic.edu

Abstract

Supervised or weakly supervised methods for phrase localization (textual grounding) either rely on human annotations or some other supervised models,  e.g .  , object detectors. Obtaining these annotations is labor-intensive and may be difficult to scale in practice. We propose to leverage recent advances in contrastive language-vision models, CLIP, pre-trained on image and caption pairs collected from the internet. In its original form, CLIP only outputs an image-level embedding without any spatial resolution. We adapt CLIP to generate high-resolution spatial feature maps. Importantly, we can extract feature maps from both ViT and ResNet CLIP model while maintaining the semantic properties of an image embedding. This provides a natural framework for phrase localization. Our method for phrase localization requires no human annotations or additional training. Extensive experiments show that our method outperforms existing no-training methods in zero-shot phrase localization, and in some cases, it even outperforms supervised methods. Code is available at  https://github.com/pals-ttic/adapting-CLIP .

1  Introduction

Phrase Localization (a.k.a. textual grounding) is the task of localizing bounding boxes referred by textual phrases in a given image. It has many down-stream applications,  e.g .  , visual question answering, image caption, and human computer interaction.

With the advancement in deep learning models, supervised training of models has emerged as an dominant approach to build effective phrase localization systems  [ 1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ,  10 ] . Critically, these methods require human annotations specific to textual grounding,  i.e .  , triplets of (image, phrases, bounding boxes). The annotation process to collect such a training set is expensive, labor intensive, and error-prone, especially, when a large dataset is necessary to effectively train these deep models. To address this, some efforts  [ 11 ,  12 ,  13 ,  14 ,  15 ]  have considered weakly-supervised/unsupervised methods that rely on pre-trained supervised models,  e.g .  , object detectors, from other tasks  [ 16 ,  17 ,  18 ] .

Such pre-trained components are naturally biased towards the few commonly seen object categories, such as person, car,  etc . These biases from the pre-trained model potentially lead to poor performance for uncommon or unseen objects in the training/pre-training set. To study these biases, ZSGNet  [ 19 ]  propose zero-shot phrase localization, where the train and test sets have “non-overlapping” nouns hence evaluating models’ zero-shot capability. In more details, they proposed different levels of “non-overlapping”, at the word-level or at the category level. For example, “Toyota” and “Honda” would be non-overlapping at the word level, but considered overlapping at the category level. However, ZSGNet remains a supervised approach which requires human annotations on phrase localization.

To address the aforementioned issues, we propose a method that  does not rely on  any textual grounding, image classification or bounding box annotations. Instead, we leverage recent advances in large-scale contrastive language-vision model, CLIP  [ 20 ] , trained on a dataset of 400M image-text pairs collected from the internet.
We propose a method to adapt/re-purpose CLIP for phrase localization. Importantly, our method can do so  without any extra supervision or training ,  i.e .  , our method immediately improves with more advanced CLIP models.

At a high-level, our method extracts high-resolution pixel-wise semantic features from images. By construction, these features lie in the same semantic space as the text embedding extracted using CLIP. As a result, we can compute per-pixel similarity scores with a given text query to obtain a heatmap. Localizing the described object in the text query becomes a score maximization problem,  i.e .  , finding a bounding box which achieves the highest score characterized by the heatmap.

We demonstrate the effectiveness of our model on zero-shot phrase localization using Flickr30k Entities  [ 21 ]  and Visual Genome (VG)  [ 22 ]  datasets following the zero-shot setting proposed in ZSGNet  [ 19 ] . Our approach outperforms ZSGNet by an absolute

5  %

percent  5

5\%

on three out of four zero-shot splits over Flickr30k and VG, despite having never seen any textual grounding or object detection annotations. We also achieve comparable performance in long-tailed object categories compared to no-training methods that utilize pre-trained object detectors.

Summary of our contributions:

•

We propose a method for textual grounding entirely from a pre-trained language-vision model (CLIP) without any bounding box supervision.

•

We design methods for extracting high-resolution spatial feature maps from CLIP, for both ViT and ResNet architectures.

•

We conduct extensive experiments and ablation studies demonstrating the effectiveness of our approach in zero-shot phrase localization.

2  Related Works

Phrase Localization. 
To study and evaluate the progress of phrase localization, numerous datasets, such as
Flickr30k Entities  [ 21 ] , Visual Genome  [ 23 ]  and ReferItGame  [ 24 ]  have been proposed. These datasets contain a rich set of annotations covering a diverse set of objects and phrases.

Numerous approaches have been proposed to tackle the task of phrase localization  [ 1 ,  2 ,  3 ,  4 ,  5 ,  7 ,  8 ,  9 ,  10 ] . These methods can be roughly divided into two groups: (a) Two-stage methods based on the the classical proposal-classification paradigm of object detection  [ 3 ,  25 ,  26 ] . These take object proposals from the image and associated them with the corresponding query text,  e.g .  , ranking these proposals based on embedding similarity to the text query. (b) In contrast, one-stage methods  [ 27 ,  28 ]  build an end-to-end pipeline without an intermediate proposal stage and are directly trained via bounding box regression.

Another line of work in phrase localization aims to reduce the requirement for textual grounding annotations  [ 12 ,  14 ]  in a weakly supervised manner. A few efforts exist  [ 13 ,  15 ]  to build phrase localization models without any phrase localization data at all. Specifically, they utilized various off-the-shelf components such as detectors, word embedding models to create such systems. However, we note that these off-the-shelf detectors,  etc .  , are themselves trained with human-annotated bounding boxes.

Figure 1:  Illustration of different zero-shot phrase localization splits. ZSGNet  [ 19 ]  proposes different level of semantic overlaps, specifically, in Flickr-S0 only the query phrase needs to be unseen  e.g .  , “ a used car ” and “ a cab ” can both refer to a car. However, this is not allowed in Flickr-S1, where the object category is also required to be unseen during training,

e.g .  , “ red bucket ” is not seen during training; both the category and the phrase is novel.

Finally, most relevant to our work is ZSGNet  [ 19 ]  where they propose to evaluate the zero-shot generalization of phrase localization systems. Specifically, they re-split Flickr30K Entities and Visual Genome such that they have different levels of semantic overlap; see an illustrating example in Fig.

1  . This helps quantify how well the model can generalize to completely unseen objects. The proposed ZSGNet remains a supervised model,  i.e .  , they still train on examples of (image, phase, bounding box).
Different from ZSGNet, we propose to adapt a pre-trained CLIP for phrase localization. Our method does not require any annotations nor training (besides optional tuning of hyper-parameters on a validation set)