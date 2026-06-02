[2311.17945] Contrastive Vision-Language Alignment Makes Efficient Instruction Learner

Contrastive Vision-Language Alignment Makes Efficient Instruction Learner

Lizhao Liu  1,2

1

1  1 Equal contribution.

Xinyu Sun  1

1

1  1 Equal contribution.

Tianhang Xiang 1

Zhuangwei Zhuang 1

Liuren Yin  3

Mingkui Tan  1,2

2

2  2 Corresponding author.

1 South China University of Technology

2 PengCheng Laboratory

3 Duke University

{selizhaoliu, csxinyusun, sexiangtianhang, z.zhuangwei}@mail.scut.edu.cn,

liuren.yin@duke.edu, mingkuitan@scut.edu.cn

Code is available at:  https://github.com/lizhaoliu-Lec/CG-VLM

Abstract

We study the task of extending the large language model (LLM) into a vision-language instruction-following model. This task is crucial but challenging since the LLM is trained on text modality only, making it hard to effectively digest the visual modality.
To address this, existing methods typically train a visual adapter to align the representation between a pre-trained vision transformer (ViT) and the LLM by a generative image captioning loss.
However, we find that the generative objective can only produce weak alignment for vision and language, making the aligned vision-language model very hungry for the instruction fine-tuning data. In this paper, we propose CG-VLM that applies both  C ontrastive and  G enerative alignment objectives to effectively align the representation of  V iT and L LM . Different from image level and sentence level alignment in common contrastive learning settings, CG-VLM aligns the image-patch level features and text-token level embeddings, which, however, is very hard to achieve as no explicit grounding patch-token relation provided in standard image captioning datasets. To address this issue, we propose to maximize the averaged similarity between pooled image-patch features and text-token embeddings. Extensive experiments demonstrate that the proposed CG-VLM produces strong vision-language alignment and is an efficient instruction learner. For example, using only 10% instruction tuning data, we reach 95% performance of state-of-the-art method LLaVA  [  30  ]  on the zero-shot ScienceQA-Image benchmark.

1  Introduction

Figure 1 :

Comparisons to state-of-the-art visual instruction methods on zero-shot ScienceQA  [  34  ]  image set w.r.t.  different amount of instruction tuning data.

Visual instruction-following model has been considered as the bedrock for the general-purpose assistant  [  26  ,

11  ]  due to its ability to understand vision-language information and follow language instruction, drawing a large amount of attention from researchers  [  26  ,

11  ,

15  ,

52  ,

53  ,

32  ,

30  ,

29  ,

6  ] . Early attempts at vision instruction following models are restricted to image captioning  [  47  ,

22  ]  and visual question answering  [  27  ,

33  ]  tasks, limiting their application scenarios such as open-ended vision-language assistant  [  52  ] . Recently, attracted by the appealing instruction-following ability of large language models (LLM)  [  37  ,

45  ,

9  ] , researchers turn to adapt the LLM into a vision-language instruction following models, with the help of pre-trained vision transformer (ViT)  [  12  ]  from CLIP  [  39  ] . The main focus is on how to combine the pre-trained ViT and LLM in order to complete the complex visual instruction task.

To this end, existing methods often require a vision-language alignment step or mechanism that aligns the vision and text modalities by learning a visual adapter. There are mainly two lines of work: heavyweight adaptation  [  26  ,

53  ,

11  ,

1  ,

2  ]  and lightweight adaptation  [  15  ,

52  ,

32  ,

30  ] . BLIP-2  [  26  ] , MiniGPT4  [  53  ]  and InstructBLIP  [  11  ]  leverage a heavy visual adapter Q-former, a BERT  [  23  ]  model, to align different modalities. Flamingo  [  1  ]  propose to use perceiver  [  21  ]  and gated xattn-dense layers to respectively extract and inject vision-language features into a pre-trained Chinchilla LLM  [  19  ] . Different from them, lightweight adaptation methods such as LLaMA-Adapter  [  15  ,

52  ]  and LLaVA  [  32  ,

30  ]  propose to learn a linear or MLP projector that connects the pre-trained ViT and LLM. During the vision-language alignment step, the visual features from ViT are projected to the embedding space of LLM by the visual adapter. Then, the projected visual features are fed into LLM and are optimized by the image captioning loss in a generative ( i.e.,  auto-regressive) way.

However, we suspect that training the visual adapter in a generative manner may not align vision-language modalities effectively. To verify this, we visualize the cosine similarities between the image patch and text token, and the results are shown in Figure

2

. We find that the text token features often have a large cosine similarity to a majority of vision patch features, showing the weak alignment between vision-language modalities. Moreover, this weak alignment makes the model very data-hungry for the vision instruction fine-tuning data. From Figure

1  , with a generative aligned visual adapter, the accuracy of state-of-the-art method LLaVA  [  30  ]  on Science-QA  [  34  ]  image set drops significantly as the amount of instruction tuning samples decreases. On one hand, instruction tuning data is very crucial for generalization to vision-language tasks  [  11  ] . On the other, high-quality instruction data is very hard to collect since the process is time-consuming and less well-defined when human crowd-scouring is considered  [  32  ] . In this sense, how to align the vision language effectively remains an important but unsolved question.

Figure 2 :

Comparisons of the image features projected by different visual adapters. With generative objective only, the projected visual features of LLaVA  [  30  ]  have large similarities to all visual concepts. In contrast, with the proposed contrastive and generative aligned visual adapters, our CG-VLM provides  only  large similarities to the corresponding concepts.  2

2

2  More visualization results are put in the supplementary.

In this paper, we propose Contrastive and Generative Aligned VLM (CG-VLM) to effectively align the vision-language modalities from pre-trained ViT and LLM. To be specific, we use contrastive loss to align the vision and text features from the visual adapter and LLM embedding layer, respectively.
However, as the averaged token embeddings of LLM layers are very ambiguous among different sentences  [  13  ] ,
how to align the visual features to a sequence of token embeddings from the caption remains an unknown question. To this end, we propose to average the similarities between pooled image features and token features from one sentence as the image-sentence similarities. Then, we optimize the image-text similarities from batch data like CLIP  [  39  ] . Though the proposed contrastive objective provides better vision-language alignment performance, the average mechanism of image-sentence alignment inevitably focuses on the dominant visual concepts, resulting in sub-optimal vision-language alignment of the nondominant concepts or semantics. To address this issue, we preserve the original generative objective that should attend to the detailed and rich visual concept in order to complete the dense image captioning task. With our proposed CG-VLM, we make better visual instruction tuning in two aspects:  1)  a visual instruction tuning data efficient vision-language model that achieves most of the performance even trained with only 10% amount of the original data.  2)  preserves more text ability from the original LLM and achieves better results on the text-only instruction or question. We summarize our main contributions as follows:

•

To the best of our knowledge, we are the first to analyze the vision-language representation alignment issue in the visual instruction tuning task and find that existing