[2505.20975] DreamBoothDPO: Improving Personalized Generation using Direct Preference Optimization

DreamBoothDPO: Improving Personalized Generation using Direct Preference Optimization

Shamil Ayupov

HSE University

shiayupov@edu.hse.ru

&amp;Maksim Nakhodnov  ∗

AIRI

nakhodnov17@gmail.com

&amp;Anastasia Yaschenko

Sber AI

ASeYashchenko@sberbank.ru

&amp;Andrey Kuznetsov

AIRI, Sber, Innopolis

kuznetsov@airi.net

&amp;Aibek Alanov

HSE University, AIRI

alanov.aibek@gmail.com

The first two authors contributed equally.

Abstract

Personalized diffusion models have shown remarkable success in Text-to-Image (T2I) generation by enabling the injection of user-defined concepts into diverse contexts.
However, balancing concept fidelity with contextual alignment remains a challenging open problem.
In this work, we propose an RL-based approach that leverages the diverse outputs of T2I models to address this issue.
Our method eliminates the need for human-annotated scores by generating a synthetic paired dataset for DPO-like training using external quality metrics.
These better–worse pairs are specifically constructed to improve both concept fidelity and prompt adherence.
Moreover, our approach supports flexible adjustment of the trade-off between image fidelity and textual alignment.
Through multi-step training, our approach outperforms a naive baseline in convergence speed and output quality.
We conduct extensive qualitative and quantitative analysis, demonstrating the effectiveness of our method across various architectures and fine-tuning techniques. The source code can be found
at  github.com/ControlGenAI/DreamBoothDPO .

1  Introduction

Text-to-Image (T2I) diffusion models  ( ramesh2022hierarchical,  ;  saharia2022photorealistic,  ;  rombach2022high,  )  have recently achieved remarkable progress, generating diverse, high-fidelity images that closely align with textual prompts. In parallel, personalization techniques ( DB,  ;  TI,  )  have emerged that enable the integration of novel visual concepts into pre-trained models. However, this personalization often comes at the expense of prompt adherence, highlighting a fundamental trade-off between textual alignment and the fidelity of the injected concept. Resolving this trade-off remains a central challenge in Personalized Image Generation.

Reinforcement learning (RL) has shown promise in enhancing various aspects of T2I generation, including alignment with human preferences  ( ID-Aligner,  ) , visual quality  ( PPOD,  ) , diversity, and prompt fidelity  ( VersaT2I,  ) . While online RL methods based on direct score optimization can significantly improve target metrics  ( REFL,  ) , they frequently suffer from overfitting and diversity collapse  ( REFLAIG,  ) . Conversely, Direct Preference Optimization (DPO)  ( rafailov2024directpreferenceoptimizationlanguage,  )  and related methods typically require costly, manually curated datasets of paired "better–worse" examples, limiting their applicability.

In this work, we adapt the DPO framework to Personalized Generation and address its data requirements by introducing a fully automatic, customizable algorithm for pair generation. Our method exploits the intrinsic variability in outputs from T2I models and employs external scoring functions to assess concept fidelity and prompt alignment. It allows for flexible control over the trade-off between these two competing objectives.

Through systematic analysis, we optimize the computational efficiency of our approach using a multi-stage training scheme that enhances overall image quality. Extensive quantitative and qualitative evaluations, including a user study, demonstrate that our method improves Image Similarity (IS) and Text Similarity (TS)  ( DB,  )  in a fully automatic setup.

Our main contributions are as follows:

•

We adapt DPO-style training to the personalized generation setting and propose a fully automated dataset construction pipeline.

•

We analyze the sensitivity of key hyperparameters and identify configurations that balance performance and computational efficiency.

•

We demonstrate the effectiveness of our approach through comprehensive experiments and user studies, showing improvements across multiple baselines and architectures.

2  Related Work

Personalized Text-to-Image Generation. 
Personalized generation methods enable the injection of user-defined concepts into pre-trained diffusion models. DreamBooth  ( DB,  )  and Textual Inversion  ( TI,  )  pioneered this field by fine-tuning models with a unique identifier tied to reference images, enabling concept recontextualization. Subsequent works improved efficiency and multi-concept handling: SVDiff  ( SVDDiff,  )  reduced parameter space through singular value decomposition, while Custom Diffusion  ( CD,  )  enabled joint optimization of multiple concepts via constrained adaptation, and IP-adapter  ( IP-adapter,  ) , ELITE  ( ELITE,  ) , HyperDreamBooth  ( HDB,  ) , BLIP-Diffusion  ( BLIP-Diffusion,  ) , and Subject-Diffusion  ( Subject-Diffusion,  )  earned rapid personalization using hypernetworks. While these methods reduce computational costs, they retain the fundamental fidelity-alignment trade-off inherent to concept specialization. Our work addresses this limitation by introducing an optimization framework that controllably balances fidelity and alignment without architectural modifications, leveraging automated quality metrics rather than manual regularization.

Reinforcement Learning and Preference Optimization. 
Reinforcement learning and preference-based methods have emerged as tools for aligning diffusion models with complex objectives. ReFL  ( REFL,  ) , which directly optimizes diffusion models against a reward model, showed superior performance in human evaluations on different downstream tasks, including improving human alignment, while TexForce  ( TexForce,  )  and B2-DiffuRL  ( B2-DiffuRL,  )  optimize prompt adherence. However, ReFL-like approaches suffer from diversity collapse and overfitting  ( REFLAIG,  ) . Direct Preference Optimization (DPO)  ( DiffDPO,  )  bypasses reward modeling but depends on manually curated preference data for complex reward functions. Several methods overcome this issue using automatic pair-dataset creation. Multiple DPO-like methods generate training data using ranking from the reward function and successfully improve in various domains in an automated setup. Particularly, ID-Aligner  ( ID-Aligner,  ) , PPOD  ( PPOD,  )  improve human alignment, VersaT2I  ( VersaT2I,  )  improve multiple aspects ranging from text alignment to geometry. PSO  ( PSO,  )  optimizes the time-distilled models for personalized generation. However, it restricts the set of "winning" images as the reference set of concept images, which limits the model’s ability to adapt to diverse backgrounds and does not consider the trade-off between concept fidelity and prompt adherence. PatchDPO  ( patchdpo,  )  extends DPO to personalized generation using patch-level rewards obtained from pre-trained vision models. However, PatchDPO requires expensive fine-tuning and cannot directly control the trade-off between global image metrics. In contrast, our method generates synthetic preference pairs using CLIP-based image-text alignment and concept fidelity scores, enabling fully automated training.

3  Preliminaries

3.1  Diffusion Models

In this work, we adopt conditional Latent Diffusion Models (LDMs), specifically the Stable Diffusion family  ( sd,  ) , as our baseline architecture. LDMs operate in the latent space defined by a Variational Autoencoder (VAE)  ( vae,  ) , where an input image

x  x

is first encoded into its latent representation

z  =

E  ​

(  x  )

z=E(x)

. A Gaussian Markovian forward diffusion process progressively adds noise to the latent representation, following the forward kernel:

p  ​

(

z  t

∣

z

t  −  1

)

=

𝒩  ​

(

z  t

∣

α  t

​

z

t  −  1

,

σ  t  2

​  𝐈

)

=

α  t

​