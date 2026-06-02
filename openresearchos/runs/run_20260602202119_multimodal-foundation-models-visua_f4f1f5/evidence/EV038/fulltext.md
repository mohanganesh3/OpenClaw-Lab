[2205.02655] Language Models Can See: Plugging Visual Controls in Text Generation

Language Models Can See:
 Plugging Visual Controls in Text Generation

Yixuan Su  ♠,

Tian Lan  ♢,

Yahui Liu  ♣,

2

2  footnotemark:

2

Fangyu Liu  ♠,

2

2  footnotemark:

2

Dani Yogatama  ♡

Yan Wang  ♢

Lingpeng Kong  ▷

Nigel Collier  ♠,

♠ University of Cambridge

♢ Tencent AI Lab

♣ University of Trento

♡ DeepMind

▷ The University of Hong Kong

Project Lead,  ys484@cam.ac.uk Co-second AuthorsCorresponding Author,  nhc30@cam.ac.uk

Abstract

Generative language models (LMs) such as GPT-2/3 can be prompted to generate text with remarkable quality. While they are designed for text-prompted generation, it remains an open question how the generation process could be guided by modalities beyond text such as images. In this work, we propose a training-free framework, called MAGIC (i MA ge- G uided text generat I on with  C LIP), for plugging in visual controls in the generation process and enabling LMs to perform multimodal tasks (e.g., image captioning) in a zero-shot manner.
MAGIC is a simple yet efficient plug-and-play framework, which directly combines an off-the-shelf LM (i.e., GPT-2) and an image-text matching model (i.e., CLIP) for image-grounded text generation. During decoding, MAGIC influences the generation of the LM by introducing a CLIP-induced score, called  magic score , which regularizes the generated result to be semantically related to a given image while being coherent to the previously generated context. Notably, the proposed decoding scheme does not involve any gradient update operation, therefore being computationally efficient. On the challenging task of zero-shot image captioning, MAGIC outperforms the state-of-the-art method by notable margins with a nearly 27 times decoding speedup. MAGIC is a flexible framework and is theoretically compatible with any text generation tasks that incorporate image grounding. In the experiments, we showcase that it is also capable of performing visually grounded story generation given both an image and a text prompt. Our code, models and other related resources are publicly released at  https://github.com/yxuansu/MAGIC .

\doparttoc  \faketableofcontents

1  Introduction

Since the introduction of GPT-2  [ 60 ] , generative language models (LMs), which are pre-trained on enormous amount of unstructured text, have produced unmatched performances on a wide range of NLP tasks  [ 4 ,  11 ] .
Given a textual prompt, LMs can continuously generate texts with the next-token prediction decoding scheme.
Although controlling the outputs of LMs have become possible by inserting textual prompts, it is still unknown how the decoding process could be guided by information beyond texts, such as images.

Recently, multimodal representation learning of text and images have been rejuvenated by pre-trained image-text joint embedding models, such as CLIP  [ 59 ]  and ALIGN  [ 28 ] . They leverage large-scale nosiy image-text pairs with weak correspondence for contrastive embedding learning and the learned joint model achieves impressive zero-shot performance competitive to supervised models on tasks such as image classification and image-text retrieval. However, they are still under-explored for image-grounded text generation.  1

1  1 Note that while such noisy weak image-text pair supervision is sufficient for learning embeddings, they could not be directly used to train image captioning model due to the data’s extreme level of noise  [ 78 ] .

How can we combine the best of both the pre-trained LMs and image-text embedding models for visually grounded text generation? Existing supervised methods combine multimodal encoders by further training them on human-annotated paired image-text data  [ 51 ,  6 ] .
Differently, weakly supervised approaches  [ 2 ,  19 ,  37 ]  rely on pre-trained object detectors to identify visual concepts and create pseudo image-text pairs. Instead of training on annotated image-text pairs, they directly train on the pseudo data. However, such methods are usually limited by the object detectors that are trained with a fixed set of labels.
The closest to our proposal is ZeroCap  [ 78 ]  which is an unsupervised image captioning method by combining frozen CLIP and GPT-2. One of the advantages of ZeroCap is it performs  ex post facto  in the activation space without re-training or fine-tuning the CLIP and GPT-2 models. However, ZeroCap relies on gradient update and optimization over the context cache, which significantly slows down the inference and hindering its use in real-world scenarios.

In this paper, we propose to solve this challenging task in a completely new perspective
by designing a novel text decoding scheme, called MAGIC (i MA ge- G uided text generat I on with  C LIP). During inference, MAGIC does not rely on  any  additional training or parameters and utilizes explicit “control knobs” to select desired outputs following the guidance of both the GPT-2 and CLIP models.
Different from the standard decoding process of GPT-2, we insert a CLIP-induced term, called  magic score , in the next token search to encourage the predicted result to demonstrate information that is close to a given image. Our experiments show that such a framework enables zero-shot image captioning and also visually grounded story generation under a simple plug-and-play principle.

To verify the qualitative and quantitative performance of the proposed MAGIC method, we conduct comprehensive experiments on two commonly used benchmarks (Section

section

4  ): MS-COCO  [ 42 ]  and Flickr30k  [ 58 ] . To our surprise, MAGIC achieves state-of-the-art (SOTA) performance across different evaluation metrics, especially outperforming all unsupervised and weakly supervised baselines notably. Moreover, since MAGIC involves no gradient update, the inference speed accelerates upon previous zero-shot image captioning SOTA by around 27 times. Beyond image captioning, we also test our approach on visually grounded story generation (Section

section

5  ). In this task, given an image and a text prompt, MAGIC can generate high-quality stories that outperform strong baseline methods on both human and automatic evaluations.

In summary, we make the following contributions:

•

To the best of our knowledge, we are the first to propose a zero-shot method, called MAGIC, to utilize explicit “control knobs” to efficiently select desired outputs following the guidance of both the pre-trained GPT-2 and CLIP models;

•

We empirically show that MAGIC is extremely effective on zero-shot image captioning, achieving SOTA across different benchmarks;

•

We demonstrate that MAGIC could be used in creative ways: it can perform complex multimodal generation tasks such as visually grounded story generation and reaches near-human performances on a wide range of evaluation metrics.

2  Background

In this section, we briefly introduce previous work related to our research.

2.1  Image Captioning

Our work is closely related to the literature of image captioning, where the goal is to describe images with meaningful and syntactically correct sentences.
Although this topic has been extensively explored in the past few years, it is still far from being considered as a solved task. Given the training strategies (e.g., the type of training data), we can roughly classify the previous methods into two categories: (1) Supervised and (2) Weakly-/Un-Supervised approaches. The former heavily depends on manually labelled image-text datasets.
In contrast, the latter tries to create pseudo image-text pairs
(i.e., weakly supervised) or even avoid using any paired image-text data (i.e., unsupervised).

Supervised Approaches.  With the development of deep learning, most of the existing models use one CNN to encode the input image and one RNN to generate the corresponding sentence describing the image  [ 47 ,  81 ] . These models are trained to maximize the probability of