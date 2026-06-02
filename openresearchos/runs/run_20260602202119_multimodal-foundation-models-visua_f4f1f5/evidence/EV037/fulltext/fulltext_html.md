[2310.01107] Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models

Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models

Hyeonho Jeong &amp; Jong Chul Ye
 Kim Jaechul Graduate School of AI, KAIST

{hyeonho.jeong, jong.ye}@kaist.ac.kr

Abstract

Recent endeavors in video editing have showcased promising results in single-attribute editing or style transfer tasks, either by training
text-to-video (T2V) models on text-video data or adopting training-free methods.
However, when confronted with the complexities of multi-attribute editing scenarios, they exhibit shortcomings such as omitting or overlooking intended attribute changes, modifying the wrong elements of the input video, and failing to preserve regions of the input video that should remain intact.
To address this, here we present a novel grounding-guided video-to-video translation framework called Ground-A-Video for multi-attribute video editing.
Ground-A-Video attains temporally consistent multi-attribute editing of input videos in a training-free manner without aforementioned shortcomings.
Central to our method is the introduction of Cross-Frame Gated Attention which incorporates groundings information into the latent representations in a temporally consistent fashion, along with Modulated Cross-Attention and optical flow guided inverted latents smoothing.
Extensive experiments and applications demonstrate that Ground-A-Video’s zero-shot capacity outperforms other baseline methods in terms of edit-accuracy and frame consistency.
Further results and codes are provided at our  project page .

Figure 1:

Ground-A-Video achieves multi-attribute editing, video style transfer with attribute change, and text-to-video generation with pose guidance, all in a time-consistent and training-free fashion. The boxes in the right-bottom images visualize the series of pose guidance.

1  Introduction

Coupled with massive text-image datasets  (Schuhmann et al.,  2022 ) , the advent of diffusion models  (Ho et al.,  2020 ; Song et al.,  2020b )  has revolutionized text-to-image (T2I) generation, making it increasingly accessible to generate high-quality images from text descriptions.
Additionally, the domain has seen profound expansion into several subfields, including controlled generation and real-world image editing.
On the other hand, the endeavor to extend the success to the video domain poses a significant computational hurdle.
Attaining time-consistent and high-quality results necessitates training on expensive video datasets—an endeavor beyond the means of most researchers, particularly given the absence of publicly available, generic text-to-video models.

As such, pioneering approaches exhibit promise in text-to-video generation  (Ho et al.,  2022b ;  a )  and video editing  (Esser et al.,  2023 )  by repurposing T2I diffusion model weights for extensive video data training.
Specifically, in pursuit of cost-effective video generation,  Wu et al. ( 2022 )  suggests fine-tuning the T2I model on a single video, which enables generating variations of the video.
Similar to the practice of manipulating attention maps within the realm of image editing  (Hertz et al.,  2022 ; Tumanyan et al.,  2023 ; Parmar et al.,  2023 ) , various methods guide the denoising process by self-attention maps  (Ceylan et al.,  2023 ) , cross-attention maps  (Liu et al.,  2023 ; Wang et al.,  2023 ) , or both  (Qi et al.,  2023 ) , which are obtained during the input video inversion stage.
Recently, to incorporate the denoising process with additional structural cues, ControlNet  (Zhang &amp; Agrawala,  2023 )  has been transferred to video domain, achieving structure-consistent output frames in video generation  (Khachatryan et al.,  2023a )  and translation  (Hu &amp; Xu,  2023 ; Chu et al.,  2023 ; Chen et al.,  2023 ; Zhang et al.,  2023 ) .

Nonetheless, in the scenario of fine-grained video editing involving multiple attribute changes, i.e.

Δ  ​  τ

=

{

τ  a

→

τ

a  ′

,

τ  b

→

τ

b  ′

,

τ  c

→

τ  c

,  …

,

∅  →

τ

n  ​  e  ​  w

}

Δ  𝜏

formulae-sequence

→

subscript  𝜏  𝑎

subscript  𝜏

superscript  𝑎  ′

formulae-sequence

→

subscript  𝜏  𝑏

subscript  𝜏

superscript  𝑏  ′

formulae-sequence

→

subscript  𝜏  𝑐

subscript  𝜏  𝑐

…

→

subscript  𝜏

𝑛  𝑒  𝑤

\Delta\tau{=}\{\tau_{a}{\shortrightarrow}\tau_{a^{\prime}},\tau_{b}{\shortrightarrow}\tau_{b^{\prime}},\tau_{c}{\shortrightarrow}\tau_{c},\ldots,\varnothing{\shortrightarrow}\tau_{new}\}

, where

τ

𝜏

\tau

represents a specific attributed indexed by subscript,
they encounter issues of degraded frame consistency, severe semantic misalignment  (Park et al.,  2023 ) , or both, as depicted in Fig.

2  - Left  and Sec.

C  .
In specific, instances of neglecting intended attribute edits (

τ  a

→

τ  a

→

subscript  𝜏  𝑎

subscript  𝜏  𝑎

\tau_{a}{\shortrightarrow}\tau_{a}

),
modifying the wrong elements (

τ  a

→

τ

b  ′

→

subscript  𝜏  𝑎

subscript  𝜏

superscript  𝑏  ′

\tau_{a}{\shortrightarrow}\tau_{b^{\prime}}

),
mixing two separate edits (

τ  a

→

τ

a  ′

⋅

τ

b  ′

→

subscript  𝜏  𝑎

⋅

subscript  𝜏

superscript  𝑎  ′

subscript  𝜏

superscript  𝑏  ′

\tau_{a}{\shortrightarrow}\tau_{a^{\prime}}{\cdot}\tau_{b^{\prime}}

),
and struggling to preserve regions that should remain unchanged (

τ  c

→

τ

c  ′

→

subscript  𝜏  𝑐

subscript  𝜏

superscript  𝑐  ′

\tau_{c}{\shortrightarrow}\tau_{c^{\prime}}

) are observed.
This is because in the existing models, the Cross-Attention layer is the single domain where the complex semantic changes wield their influence,
where the list of intricate changes being entangled as a form of “one-sentence target prompt”
makes the problem worse.

The key to address this issue lies in spatially-disentangled layout information, comprising bounding box coordinates and textual captions, namely ‘groundings’.
Groundings disentangle the complex semantic combination by localizing each semantic element with precise location.
Recently, grounding has been successfully employed to text-to-image generation tasks.
 Li et al. ( 2023b )  and  Yang et al. ( 2023 )  finetune existing T2I models to adhere to grounding conditions using box-image paired datasets, while  Xie et al. ( 2023 )  achieves training-free box-constrained image generation by injecting binary spatial masks into the cross-attention space.
However, unlike the literature on single-image synthesis, guiding video generation process with groundings alone could significantly detriment frame consistency.
One major problem is that localizing the bounding box is insufficient for the smooth frame transition (Fig.

2  - right ), which calls the need for additional structural guidance.
Consequently, we propose to integrate two distinct modalities:
1) spatially-continuous conditions, including depth and optical flow maps, are employed to maintain consistent structure across frames;
2) spatially-discrete conditions, specifically ‘groundings’, enable precise localization and editing of each attribute within the source video.
The principal contributions of  Ground-A-Video  are summarized as follows:

•

To our knowledge, we present the first groundings-driven video editing framework, also marking the first instance of integrating both spatially-continuous and discrete conditions.

•

We propose a novel Modulated Cross-Attention mechanism which efficiently enables interactions between differently optimized unconditional embeddings.

•

To further enhance consistency, we suggest smoothing inverted latent representations using optical flow information, which can be employed following any type of inversion.

•

Extensive experiments and applications demonstrate the effectiveness of our method in achieving time-consistent and precise multi-attribute editing of input videos.

Figure 2:

Left :
Failure cases of multi-attribute video editing by various methods,
driven by