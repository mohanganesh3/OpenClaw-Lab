[2410.02615] LoGra-Med: Long Context Multi-Graph Alignment for Medical Vision-Language Model

LoGra-Med: Long Context Multi-Graph Alignment for Medical Vision-Language Model

Duy M. H. Nguyen  1,2,3  , Nghiem T. Diep

∗3  , Trung Q. Nguyen  ∗3,4  , Hoang-Bao Le  3  , Tai Nguyen  3

,

Tien Nguyen  5,6  , TrungTin Nguyen  8  , Nhat Ho  9  , Pengtao Xie  10  , Roger Wattenhofer  11

,

James Zhou  12  , Daniel Sonntag  † 3,7  , Mathias Niepert  † 1,2

1

Max Planck Research School for Intelligent Sytems (IMPRS-IS),

2

University of Stuttgart,

3

German Research Centre for Artificial Intelligence (DFKI),

4

Technical University of Munich,

5

University Medical Center Göttingen,

6

Max Planck Institute for Multidisciplinary Sciences,

7

Oldenburg University,

8

University of Queensland,

9

University of Texas at Austin,

10

University of California San Diego,

11

ETH Zurich,

12

Stanford University.

Abstract

State-of-the-art medical multi-modal large language models (med-MLLM), such as  LLaVA-Med  or  BioMedGPT , leverage instruction-following data in their pre-training stages.
However, those models primarily focus on scaling the  model size  and  data volume  to boost performance while mainly relying on the autoregressive learning objectives. Surprisingly, we reveal that such learning schemes might result in a weak alignment between vision and language modalities, making these models highly reliant on extensive pre-training datasets — a significant challenge in medical domains due to the expensive and time-consuming nature of curating high-quality instruction-following instances.
We address this challenge with a new multi-graph alignment algorithm, namely  LoGra-Med , which enforces triplet correlations on the latent embedding space among image modalities, conversation-based descriptions, and extended contextual captions. Owing to this technique, the model is encouraged to capture the semantic meaning of the context, handle linguistic variability where the captions or questions may differ from training instances, and learn cross-modal associations, linking visual elements with various textual interpretations.
To scale our algorithm to the med-MLLM setting, we also design an efficient end-to-end learning scheme based on advanced black-box gradient-estimation techniques that permit fast forward and backward steps through the LLM model (LLaMa 7B). Empirical results show that we can match the performance of LLAVA-Med pre-trained on 600K image-text pairs from PMC-15M for Medical VQA tasks and significantly outperform it when trained on only

10  %

percent  10

10\%

of the data. For instance, on VQA-RAD, we exceed LLAVA-Med (both trained on

10  %

percent  10

10\%

) by

20.13  %

percent  20.13

20.13\%

and achieve near parity with the

100  %

percent  100

100\%

pre-training setting (

72.52  %

percent  72.52

72.52\%

vs.

72.64  %

percent  72.64

72.64\%

).
Additionally, we also surpass other SOTA pre-training methods and med-MLLM such as  BiomedGPT  on  visual chatbot  or  RadFM  on  zero-shot image classification with VQA , showcasing the power of multi-graph alignment in improving vision-language integration for medical-MLLM.

†

†

∗

Co-second contribution,

†

Co-senior authors.

1  Introduction

Generic Multi-Modal Large Language Models (MLLM) are an emerging field integrating processing and generation across text, images, and audio. Models like GPT-4V  (Achiam et al.,  2023 ) , LLaVa  (Liu et al.,  2024b ) , and Next-GPT  (Wu et al.,  2023b )  handle tasks from image captioning to complex visual reasoning. A critical component in training these models is instruction-following (IF) data  (Lou et al.,  2023 ) , which involves complex, multi-turn interactions  (Sun et al.,  2024 )  where the model is expected to respond to specific instructions or questions about the image. In the medical domain, current efforts have been focused on building medical MLLM by curating specialized IF datasets encompassing medical images, clinical notes, and diagnostic criteria  (Xie et al.,  2024 ) . These datasets are used to adapt general-purpose MLLM, aiming to transfer foundational knowledge of generic concepts and reduce computation costs required in training steps. For instance, in LLAVA-Med,  (Li et al.,  2024 )  600K biomedical image-text pairs are sampled from PMC-15M  (Zhang et al.,  2023c )  and GPT-4 is used to create instruction data from the text inputs, resulting in approximately 60K multi-modal IF data points. In the next step, (i) vision encoders and language decoders are taken from LLaVa and are first aligned through trainable projection layers before (ii) the models (with the exception of the vision encoders) are trained together on collected medical IF data. Both steps are called the  pre-training  phase, where the auto-regressive function is used as the primary objective. The model can then be fine-tuned to various downstream tasks.

Figure 1:

Illustration of the data-hungry behavior of auto-regressive modeling  in LLaVA-Med when varying pre-training IF data size. Models are fine-tuned and performance is reported on VQA-RAD.

Following the above approach, most later works have focused on scaling up the amount of medical IF data  (Xie et al.,  2024 ; Zhang et al.,  2023a ; He et al.,  2024 )  or increasing the model size by incorporating larger vision encoders or language decoders  (Wu et al.,  2023a ; Jiang et al.,  2024 )  while relying on the same standard autoregressive learning scheme. Contrary to this, we question the effectiveness of autoregressive objective functions when learning medical-MLLM with IF data. Surprisingly,  our findings reveal that autoregressive learning is highly data-hungry during pre-training , i.e., without sufficient medical IF samples, model performance plummets for downstream tasks,  even after fine-tuning . To illustrate this,
we pre-trained LLAVA-Med using only

10  %

percent  10

10\%

of the data and compared it to the version trained on

100  %

percent  100

100\%

. Both models were fine-tuned on two medical visual question-answering tasks - VQA-RAD  (Lau et al.,  2018 )  and PathVQA  (He et al.,  2020 )  - and their average performance on open- and close-ended questions compared. The results show a dramatic decline: from

72.64  %

percent  72.64

72.64\%

to

52.39  %

percent  52.39

52.39\%

on VQA-RAD and from

64.06  %

percent  64.06

64.06\%

to

56.15  %

percent  56.15

56.15\%

on PathVQA (Figure

1  ). This underscores the instability of medical-MLLM trained with autoregressive methods and highlights the problem that these methods require the curation of enough medical IF data to achieve satisfactory performance.

To address this challenge, we present a novel multi-graph alignment algorithm, namely  LoGra-Med , that improves the model’s ability to learn complex interactions between vision and language modalities, mitigating the limitations of autoregressive functions when trained on limited instruction-following data. Specifically, given pairs of input images with instruction data, we use GPT-4  (Achiam et al.,  2023 )  to  form a longer version of the instruction , with more in-context explanations for concepts and correlations among entities while preserving the same meaning. These data pairs are fed into the MLLM, where a vision encoder extracts features for the image while the language model (LLaMa) computes latent embedding of the instruction data and its extended versions. The embedding vectors obtained from different IF samples in a batch are then used to construct three graphs. The first graph’s vertices are the image features, while the vertices of the two other graphs are the embeddings of the instructions and their extended versions. We subsequently learn feature representation by solving the combinatorial multi-graph alignment problem between these three graphs such that the input embedding of the graph alignment will output the triplet coupling among the image