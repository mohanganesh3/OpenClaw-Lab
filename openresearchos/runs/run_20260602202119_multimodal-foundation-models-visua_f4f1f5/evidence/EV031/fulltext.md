[2603.29692] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition

SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot

Skeleton-based Action Recognition

Ning Wang  1,2  , Tieyue Wu  2  , Naeha Sharif  3  , Farid Boussaid  3  , Guangming Zhu  2  ,

Lin Mei  4  , Mohammed Bennamoun  3  , Liang Zhang  2

1  Chang’an University,  2  Xidian University,

3  University of Western Australia,  4  Donghai Lab

ning@chd.edu.cn,

gxyzd648@gmail.com,

{gmzhu, liangzhang}@xidian.edu.cn,

{naeha.sharif,farid.boussaid,mohammed.bennamoun}@uwa.edu.au,

meilin@donghailab.com

Part of this work was completed during the Ning’s visit to The University of Western Australia.Corresponding author.

Abstract

Zero-shot skeleton-based action recognition aims to recognize unseen actions by transferring knowledge from seen categories through semantic descriptions. Most existing methods typically align skeleton features with textual embeddings within a shared latent space. However, the absence of contextual cues, such as objects involved in the action, introduces an inherent gap between skeleton and semantic representations, making it difficult to distinguish visually similar actions.
To address this, we propose SkeletonContext, a prompt-based framework that enriches skeletal motion representations with language-driven contextual semantics.
Specifically, we introduce a Cross-Modal Context Prompt Module, which leverages a pretrained language model to reconstruct masked contextual prompts under guidance derived from LLMs.
This design effectively transfers linguistic context to the skeleton encoder for instance-level semantic grounding and improved cross-modal alignment.
In addition, a Key-Part Decoupling Module is incorporated to decouple motion-relevant joint features, ensuring robust action understanding even in the absence of explicit object interactions.
Extensive experiments on multiple benchmarks demonstrate that SkeletonContext achieves state-of-the-art performance under both conventional and generalized zero-shot settings, validating its effectiveness in reasoning about context and distinguishing fine-grained, visually similar actions. Our project is available at  https://github.com/NingWang2049/skeletoncontext .

1  Introduction

Figure 1 :

Comparison between existing ZSSAR methods and our proposed SkeletonContext. 
Conventional ZSSAR methods directly align skeleton features with textual descriptions, but the absence of contextual cues creates a semantic gap that hinders discrimination between similar actions. In contrast, SkeletonContext reconstructs language-driven contextual semantics (e.g., objects) and injects them into skeleton representations, enabling fine-grained, context-aware zero-shot action recognition.

Human action recognition plays a crucial role in numerous applications, including human-computer interaction  [ 29 ,  18 ] , intelligent surveillance  [ 34 ,  33 ] , and healthcare monitoring  [ 38 ,  45 ] .
Skeleton-based representations  [ 51 ,  44 ,  43 ,  24 ] , which model human motion through joint coordinates, are valuable in privacy-sensitive and real-world scenarios due to their robustness to appearance variations, background clutter, and view changes.
Most existing skeleton-based action recognition methods  [ 11 ,  8 ,  25 ,  42 ,  40 ]  rely on fully supervised learning with pre-defined categories, limiting their performance to diverse and unseen action categories.
To overcome this limitation, zero-shot skeleton-based action recognition (ZSSAR) has emerged, enabling models to recognize unseen actions using knowledge from seen action categories and semantic descriptions.

Existing ZSSAR approaches  [ 15 ,  14 ,  48 ,  21 ,  6 ,  49 ,  22 ,  50 ]  align skeleton and text embeddings within a shared latent space to enhance cross-modal generalization.
They mainly focus on leveraging additional external knowledge  [ 6 ,  49 ] , augmenting training data  [ 14 ,  50 ] , or designing more powerful skeleton encoders  [ 48 ,  21 ,  22 ] .
However, skeleton sequences, unlike raw videos, lack detailed appearance and contextual cues, such as the keyboard in the action category “typing on a keyboard” shown in Fig.

1  .
As a result, the lack of contextual cues in skeleton data inherently hampers its alignment with semantic representations, limiting the model’s ability to capture fine-grained semantic distinctions between similar actions, such as “typing on a keyboard” and “writing on paper”.
Although fully supervised fine-grained skeleton-based action recognition methods [ 46 ,  26 ]  have explored learning discriminative representations to distinguish ambiguous actions, they rely heavily on class-specific supervision and contrastive calibration between known categories.
As a result, these methods cannot be directly extended to zero-shot settings, where unseen classes lack labeled instances and discriminative prototypes  [ 26 ]  cannot be established.

To address this, we argue that enhancing skeleton representations by reconstructing contextual semantics, such as associated objects, is crucial for improving cross-modal alignment and distinguishing visually similar actions.
In this paper, we propose SkeletonContext, a prompt-based framework that enriches skeletal motion representations with language-driven contextual knowledge.
As illustrated in Fig.

1  , our framework injects reconstructed contextual semantics into the skeleton modality, achieving more reliable alignment with textual semantics.
Specifically, we first extract action-related contextual descriptions from large language models (LLMs), capturing external knowledge such as interacted-with objects and environments that are absent in the skeleton modality.
Then, our Cross-Modal Context Prompt Module reconstructs masked contextual prompts under the supervision of generated semantics (e.g., objects and environments). This reconstruction process transfers contextual knowledge from a pretrained language model, resulting in context-enhanced skeleton representations.
We further enhance the Cross-Modal Context Prompt Module by designing a differential joint encoder to model fine-grained motion dynamics across joints, and a progressive partial masking strategy to progressively refine contextual learning by encouraging robustness under incomplete motion cues.
Different from prior prompt-based ZSSAR methods (e.g., SCoPLe  [ 50 ] , Neuron  [ 5 ] ) that enhance text encoders for better semantic alignment, SkeletonContext enriches the skeleton encoder itself with language-driven contextual prompts, thus fundamentally improving motion-side representation and cross-modal grounding.
Moreover, to handle actions without explicit object interactions, we also propose a Key Part Decoupling Module that disentangles motion-related joint features and highlights the most relevant body parts.

To summarize, our contributions are three-fold :

•

We propose SkeletonContext, a novel context-aware framework for zero-shot skeleton-based action recognition. It introduces language-driven contextual knowledge into the skeleton modality, effectively bridging the semantic gap between motion and language.

•

We demonstrate that injecting contextual semantics into skeleton representations via a cross-modal prompt learning mechanism significantly enhances the model’s ability to distinguish visually similar actions and generalize to unseen categories.

•

We conduct extensive experiments on three benchmark datasets. SkeletonContext consistently achieves state-of-the-art performance under both conventional and generalized zero-shot settings, validating the effectiveness and robustness of our approach.

2  Related Work

2.1  Zero-Shot Skeleton Action Recognition

Zero-shot skeleton action recognition (ZSSAR) seeks to recognize unseen action categories by transferring knowledge from seen classes through shared semantic embeddings. Early studies focused on global alignment