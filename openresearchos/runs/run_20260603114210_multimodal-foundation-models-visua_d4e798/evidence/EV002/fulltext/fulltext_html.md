[2409.17531] SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-modal Fusion

SimVG: A Simple Framework for Visual Grounding with Decoupled Multi-modal Fusion

Ming Dai  1  , Lingfeng Yang  2  , Yihao Xu  1  , Zhenhua Feng  3  , Wankou Yang  1

1  SouthEast University

2  Nanjing University of Science and Technology,

3  JiangNan University

{mingdai, 220211848, wkyang}@seu.edu.cn, yanglfnjust}@njust.edu.cn

fengzhenhua@jiangnan.edu.cn

Corresponding authors.

Abstract

Visual grounding is a common vision task that involves grounding descriptive sentences to the corresponding regions of an image.
Most existing methods use independent image-text encoding and apply complex hand-crafted modules or encoder-decoder architectures for modal interaction and query reasoning.
However, their performance significantly drops when dealing with complex textual expressions.
This is because the former paradigm only utilizes limited downstream data to fit the multi-modal feature fusion. Therefore, it is only effective when the textual expressions are relatively simple.
In contrast, given the wide diversity of textual expressions and the uniqueness of downstream training data, the existing fusion module, which extracts multimodal content from a visual-linguistic context, has not been fully investigated.
In this paper, we present a simple yet robust transformer-based framework, SimVG, for visual grounding.
Specifically, we decouple visual-linguistic feature fusion from downstream tasks by leveraging existing multimodal pre-trained models and incorporating additional object tokens to facilitate deep integration of downstream and pre-training tasks.
Furthermore, we design a dynamic weight-balance distillation method in the multi-branch synchronous learning process to enhance the representation capability of the simpler branch.
This branch only consists of a lightweight MLP, which simplifies the structure and improves reasoning speed.
Experiments on six widely used VG datasets,  i.e. , RefCOCO/+/g, ReferIt, Flickr30K, and GRefCOCO, demonstrate the superiority of SimVG.
Finally, the proposed method not only achieves improvements in efficiency and convergence speed but also attains new state-of-the-art performance on these benchmarks.
Codes and models will be available at  https://github.com/Dmmm1997/SimVG .

1  Introduction

Figure 1 :

An overview of visual grounding structures:
(a) Two-Stage: Applying a detector for proposals, followed by image-text encoding and feature similarity calculation for region matching.
(b) One-Stage: Grounding in the fused features through dense prediction.
(c) Transformer-based: Employing an encoder-decoder structure in the head.
(d) Proposed SimVG: Utilizing Multi-Modality Encoder for multimodal interaction among object, image, and text tokens, directly applies a lightweight MLP for grounding.

Visual grounding (VG) aims to predict the corresponding regions of an image through linguistic expressions. The task necessitates a comprehensive understanding of each modality, as well as the modeling of consistency between image context and text. Some benchmarks focus on addressing  phrase localization

[ 24 ,  48 ] , which entails locating all objects mentioned in a sentence within an image. Another aspect emphasizes resolving  referring expression comprehension  (REC)  [ 73 ,  46 ,  45 ] , characterized by only one target corresponding to a sentence. Recently, a new type of  general referring expression comprehension  (GREC)  [ 35 ,  16 ]  task has emerged. GREC is similar to REC, but in which a sentence can have multiple targets or no target at all.

Existing visual grounding models can be roughly divided into three categories: two-stage, one-stage, and transformer-based.
Among them, as shown in Fig.

1  (a), two-stage methods  [ 18 ,  75 ,  72 ,  42 ,  36 ]  require a pre-trained detector to generate proposals and perform localization through region-text retrieval. These methods rely on a complex module with manually designed mechanisms to achieve query reasoning and multi-modal fusion.
One-stage methods  [ 69 ,  34 ,  44 ,  67 ] , on the other hand, employ an end-to-end architecture, as shown in Fig.

1  (b). Most of them primarily perform dense prediction on multimodal fusion features defined in the form of anchors.
Some recent algorithms  [ 7 ,  23 ,  79 ,  54 ] , depicted in Fig.

1  (c), adopt an encoder-decoder architecture to perform multimodal fusion in the encoder and then decode the response target position using an object query similar to DETR [ 2 ] .
The existing methods share a commonality: they adopt architectures that independently encode each modality before merging them, with multimodal fusion intricately linked to each visual grounding task.
The feature extraction part of these methods generally employs specific classification  [ 6 ,  53 ]  or autoregressive  [ 8 ,  29 ]  tasks in each modality for pre-training. However, the alignment and mutual understanding between modalities only utilize a limited amount of downstream data, which undoubtedly underestimates the difficulty of achieving mutual understanding between modalities.
Another observed trend, as noted in  [ 23 ,  79 ] , is the notable enhancement in the performance of visual grounding with a significant augmentation of pretraining data on large corpora. This implicitly suggests that leveraging a small amount of downstream data does not fully capitalize on the potential for mutual understanding between images and text.
Nevertheless, this type of pretraining undoubtedly increases the burden of training resources.

Figure 2 :

The expression length and relative improvement between Dynamic MDETR  [ 54 ]  and SimVG.

Furthermore, the mutual understanding between multiple modalities is crucial for downstream tasks. As shown in Fig.

2  , for a dataset with long sentence characteristics (like RefCOCOg  [ 46 ] ), adopting the decoupled multimodal understanding method can significantly improve model performance, while the improvement is relatively modest on datasets with short sentences. This observation aligns with our expectation that shorter captions pose less challenge for inter-modal understanding, whereas SimVG’s decoupling approach proves advantageous for challenging longer descriptions that require intricate multi-modal comprehension.

To be specific, we delve into existing multimodal understanding methods in the form of multimodal pre-training. Existing approaches can be broadly categorized into three categories.
Dual-stream structures  [ 49 ,  22 ,  32 ]  encode image-text modalities independently and supervise them with contrastive learning. One-stream models  [ 5 ,  25 ,  29 ,  21 ]  concatenate multimodal features for feature extraction. Other works  [ 30 ,  1 ,  55 ,  60 ]  use a dual-stream design with a fusion encoder to balance complexity and computational cost, fusing multimodal features in intermediate layers.
Some methods also have applied multimodal pre-training to VG tasks, such as Dynamic MDETR  [ 54 ] , which uses an image-text encoder pretrained on CLIP  [ 49 ]  to enhance the performance. Recently, works like CPT  [ 70 ] , ReCLIP  [ 56 ] , and FGVP  [ 65 ]  have improved the performance of zero-shot visual grounding using dual-stream pre-trained models that employ a two-stage-like architecture and prompt engineering techniques. However, these approaches focus on multimodal alignment rather than mutual understanding. Thus, the dual-stream with fusion encoder architecture  [ 30 ,  1 ,  55 ,  60 ]  has been thrown into our sight. Specifically, based on BEiT-3  [ 60 ] , we propose a simple framework called SimVG that decouples multimodal fusion from downstream tasks and simultaneously encodes image, text, and object tokens, as illustrated in Fig.

1

(d).

The decoder structure used for query reasoning, similar to DETR  [ 2 ] , is effective but inevitably increases the model’s complexity and computational overhead.
We aim to develop a mo