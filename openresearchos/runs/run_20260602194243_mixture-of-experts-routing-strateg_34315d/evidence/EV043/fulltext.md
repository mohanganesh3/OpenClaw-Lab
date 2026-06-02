[2512.15702] End-to-End Training for Autoregressive Video Diffusion via Self-Resampling

1]The Chinese University of Hong Kong
2]ByteDance Seed
3]ByteDance
 \contribution [*]Work done at ByteDance Seed
 \contribution [†]Corresponding authors

End-to-End Training for Autoregressive Video
 Diffusion via Self-Resampling

Yuwei Guo

Ceyuan Yang

Hao He

Yang Zhao

Meng Wei

Zhenheng Yang

Weilin Huang

Dahua Lin

[

[

[

( January 6, 2026 )

Abstract

Autoregressive video diffusion models hold promise for world simulation but are vulnerable to exposure bias arising from the train-test mismatch. While recent works address this via post-training, they typically rely on a bidirectional teacher model or online discriminator. To achieve an end-to-end solution, we introduce  Resampling Forcing , a teacher-free framework that enables training autoregressive video models from scratch and at scale. Central to our approach is a self-resampling scheme that simulates inference-time model errors on history frames during training. Conditioned on these degraded histories, a sparse causal mask enforces temporal causality while enabling parallel training with frame-level diffusion loss. To facilitate efficient long-horizon generation, we further introduce history routing, a parameter-free mechanism that dynamically retrieves the top-

k  k

most relevant history frames for each query. Experiments demonstrate that our approach achieves performance comparable to distillation-based baselines while exhibiting superior temporal consistency on longer videos owing to native-length training.

\checkdata

[Project Page] https://guoyww.github.io/projects/resampling-forcing/

1  Introduction

Recent advances in generative video models have demonstrated strong potential for world modeling by approximating physical dynamics and predicting future states conditioned on current observations  [ 7 ,  2 ,  36 ,  28 ] . Realizing this vision necessitates an autoregressive video generation paradigm that predicts the next frame conditioned on past context, thereby mirroring the  strict causal  nature of the physical world. Beyond world simulation, such a paradigm empowers a diverse array of applications, spanning game simulation  [ 62 ,  1 ,  78 ,  85 ] , interactive content creation  [ 42 ,  54 ] , and temporal reasoning  [ 68 ] .

Despite its conceptual elegance, autoregressive video generation poses significant challenges. The primary hurdle is exposure bias  [ 48 ,  53 ] : under teacher forcing, the model conditions on ground truth histories during training, yet must rely on its own generated outputs during inference. This train–test mismatch can induce error accumulation, where small artifacts in model predictions are amplified across the autoregressive rollout, potentially leading to catastrophic video collapse (see

figure ˜ 1

top). Furthermore, the ever-expanding historical context in autoregressive generation exacerbates attention complexity, posing practical obstacles for both training and inference over long horizons.

Figure 1 :

We introduce  Resampling Forcing , an end-to-end, teacher-free training framework for autoregressive video diffusion models.  Top : The teacher forcing accumulates errors and leads to video collapse.  Middle : Distilled from a short bidirectional teacher, Self Forcing suffers from the degraded quality on longer videos.  Bottom : Our method offers stable quality by native training on long videos.

To mitigate the train–test mismatch, recent works  [ 32 ,  42 ]  employ post-training strategies aimed at aligning the generated video distribution with real data. For instance, Self Forcing  [ 32 ]  first autoregressively rolls out full videos, subsequently applying distillation or adversarial objectives to enforce distribution matching. By simulating inference during training, they reduce the discrepancy between training and test conditions. However, the reliance on a bidirectional teacher or an online discriminator impedes scalable training of autoregressive video models from scratch. A bidirectional teacher can also leak future information, compromising the strict temporal causality of the student model. Additionally, extensions to longer sequences typically use simple sliding-window attentions that disregard the varying importance of historical context, which may undermine long-term consistency.

In this work, we present  Resampling Forcing , an end-to-end training framework for autoregressive video diffusion models. Drawing inspiration from the next-token prediction objective in LLMs, we condition each frame on its clean history and train in parallel via a per-frame diffusion loss under causal masking. We posit that, to mitigate error propagation and amplification, the model must be trained for robustness against input perturbations while retaining a clean prediction objective. To this end, the core of our method is a self-resampling mechanism: the model first induces errors in the history frames, then utilizes this degraded history to condition next-frame prediction. To simulate inference-time model errors, we autoregressively resample the latter segment of each frame’s denoising trajectory with the online model weights. This process is detached from gradient backpropagation to avoid shortcut learning. In addition, we introduce a history routing mechanism that dynamically retrieves the top-

k  k

most relevant history frames via a parameter-free router, maintaining a near-constant attention complexity in long-horizon rollout.

Empirical results demonstrate that  Resampling Forcing  effectively mitigates error accumulation in autoregressive video diffusion models, achieving generation quality comparable to state-of-the-art distilled models. Leveraging native long-video training, our approach outperforms extrapolated baselines in longer video generation. Furthermore, our model exhibits stricter adherence to causal dependencies compared to distillation baselines. We also demonstrate that our history routing mechanism attains a sparse context with negligible quality loss, offering a viable memory design for long-horizon generation. We anticipate that this work will advance scalable training and long-term memory for future video world models.

2  Related Works

Bidirectional Video Generation.  Bidirectional video generation refers to non-causal models that synthesize all frames jointly, allowing each frame to attend to both past and future context. Early efforts leveraged GANs  [ 61 ,  55 ]  or adapted UNet-based text-to-image diffusion models  [ 6 ,  25 ,  5 ,  3 ] . Inspired by SoRA  [ 7 ] , the field shifted towards 3D autoencoders and scalable Diffusion Transformers (DiT)  [ 49 ] , where all video tokens interact via self-attention, with text conditions injected through MMDiT-style fusion  [ 18 ,  37 ,  26 ]  or separate cross-attention layers. State-of-the-art systems include commercial models such as Veo  [ 23 ] , Seedance  [ 21 ] , Kling  [ 38 ] , as well as open-source ones like CogVideoX  [ 74 ] , LTX-Video  [ 27 ] , HunyuanVideo  [ 37 ] , and WAN  [ 64 ] . Our approach is built with a DiT-based video diffusion backbone.

Autoregressive Video Generation.  Recently, autoregressive video generation  [ 41 ,  44 ,  67 ,  86 ,  79 ,  51 ,  29 ,  24 ,  45 ]  has gained significant traction due to its potential in world and game simulation. It generates a video sequentially under a causal factorization, conditioning each frame on its historical context.

A pivotal challenge for autoregressive video diffusion is the train–test mismatch that leads to error accumulation  [ 66 ] . Earlier attempts that directly use teacher forcing suffer from degraded quality as video length increases  [ 20 ,  31 ,  84 ] . To counteract this, prior work injects small noise into history frames to approximate inference-time degradation  [ 62 ,  67 ] . Another avenue adopts Diffusion Forcing  [ 10 ,  57 ,  11 ] , assigning each frame an independent noise level to enable condi