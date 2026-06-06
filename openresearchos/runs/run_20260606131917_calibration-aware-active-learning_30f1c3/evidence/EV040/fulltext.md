[2505.09089] Generating time-consistent dynamics with discriminator-guided image diffusion models

Generating time-consistent dynamics

with discriminator-guided image diffusion models

Philipp Hess  1,2  ,  Maximilian Gelbrecht  1,2  ,  Christof Schötz  1,2  ,  Michael Aich  1,2

Yu Huang  1,2

,

Shangshang Yang  1,2

,

Niklas Boers  1,2

1 Technical University of Munich,  2 Potsdam Institute for Climate Impact Research,

{philipp.hess, maximilian.gelbrecht, christof.schoetz,

{  michael.aich, y.huang, shangshang.yang, n.boers}@tum.de

Abstract

Realistic temporal dynamics are crucial for many video generation, processing and modelling applications, e.g. in computational fluid dynamics, weather prediction, or long-term climate simulations.
Video diffusion models (VDMs) are the current state-of-the-art method for generating highly realistic dynamics. However, training VDMs from scratch can be challenging and requires large computational resources, limiting their wider application.
Here, we propose a time-consistency discriminator that enables pretrained image diffusion models to generate realistic spatiotemporal dynamics. The discriminator guides the sampling inference process and does not require extensions or finetuning of the image diffusion model.
We compare our approach against a VDM trained from scratch on an idealized turbulence simulation and a real-world global precipitation dataset. Our approach performs equally well in terms of temporal consistency, shows improved uncertainty calibration and lower biases compared to the VDM, and achieves stable centennial-scale climate simulations at daily time steps.

1  Introduction

Generating time-consistent sequences of images is important to many video generation and synthesis tasks

ho_video_2022  ;  ho_imagen_2022  ;  zheng_open-sora_2024  ;  wang_zero-shot_2024  ;  daras_warped_2024

, for example in computational fluid dynamics

du_conditional_2024  ;  li_synthetic_2024  ;  lienen_zero_2024

, probabilistic weather forecasts

price_probabilistic_2025  ;  li_generative_2024

or climate simulations

srivastava_precipitation_2024  ;  bassetti_diffesm_2024  ;  ruhling_cachay_probablistic_2024  ;  couairon_archesweather_2024

.
 The success of image diffusion models (IDMs)

song_generative_2019  ;  ho_denoising_2020  ;  song_score-based_2021

has sparked a large interest in extending their generation to time-consistent videos, achieving remarkable results

ho_video_2022  ;  ho_imagen_2022  ;  gupta_photorealistic_2023  ;  wang_videocomposer_2023  ;  yang_cogvideox_2024  ;  zheng_open-sora_2024  ;  deng_autoregressive_2025  ;  kong_hunyuanvideo_2025

.
 However, training video diffusion models (VDMs) from scratch is challenging and requires large amounts of computational resources

deng_efficiency-optimized_2023

. Moreover, recent state-of-the-art VDMs are not always released open source

zheng_open-sora_2024

, limiting their adaptability to a wider scientific community.

Therefore, efforts have been made to leverage pretrained image models for video editing tasks such as style-transfer or inpainting

sun_diffusion_2024  ;  daras_warped_2024  ;  chang_how_2023  ;  ceylan_pix2video_2023  ;  zhang_controlvideo_2023  ;  qi_fatezero_2023

.
Video editing relies on full or partial temporal information in the source video that can then be combined with inference-level guidance techniques to preserve temporal consistency during the editing process.
Such video processing tasks are also important to many scientific applications, for example, in data reconstruction using inpainting methods or downscaling applications using super-resolution techniques in fluid dynamics

wan_debias_2023  ;  bischoff_unpaired_2024

, meteorology

mardani_residual_2024

and climate science

plesiat_artificial_2024  ;  bischoff_unpaired_2024  ;  hess_fast_2025  ;  aich_conditional_2024  ;  ling_diffusion_2024  ;  addison_machine_2024

.
 Generating videos with IDMs without relying on a source video or a given encoding of the dynamics is much more challenging. Most approaches rely on finetuning an IDM on video data, e.g., by inserting additional temporal layers into the architecture

singer_make–video_2022  ;  esser_structure_2023  ;  wu_tune–video_2023  ;  bar-tal_lumiere_2024

, which can still be computationally demanding and requires a deep understanding of the architecture.

We propose a novel guidance approach, inspired by temporal discriminators in generative adversarial networks

clark_adversarial_2019  ;  ravuri_skilful_2021  ;  das_hybrid_2024

, for the generation of realistic, time-consistent, and stable spatiotemporal dynamics with pretrained IDMs. Our discriminator guidance is lightweight and efficient, adding only about 3%-8% to the generation time, and is trained independently of the IDM, making extensions of different IDMs to new downstream tasks straightforward.
We perform a comprehensive evaluation on challenging datasets with high-dimensional chaotic dynamics, including 2D Navier-Stokes turbulence simulations and global precipitation reanalysis, using the extensive catalog of established metrics from fluid dynamics and Earth system science.
We find that our method performs similarly well as a VDM trained from scratch in terms of temporal dynamics, while achieving better uncertainty calibration and lower biases. Moreover, our guidance approach enables stable climate simulations for more than 100 years, while the VDM exhibits unstable drifts in global averages.

2  Related work

Video GANs.

Generative adversarial networks (GANs) have been widely explored for synthesizing temporally-consistent videos. Earlier work

vondrick_generating_2016  ;  mathieu_deep_2016

introduced the idea of using an adversarial discriminator to distinguish between real and generated video frames, which was improved in following studies

saito_temporal_2017  ;  tulyakov_mocogan_2018  ;  saito_train_2020

.
DVD-GAN

clark_adversarial_2019

proposed two separate discriminators for spatial and time domains, the latter being similarly motivated as our time-consistency discriminator.
 Video prediction GANs with temporal discriminators have shown great success in turbulence modelling

xie_tempogan_2018

and probabilistic weather predictions

ravuri_skilful_2021  ;  das_hybrid_2024

.
However, while temporal discriminators provide powerful tools that enable the generation of dynamically consistent videos in GANs, adversarial training is generally prone to instabilities and mode collapse, making GANs challenging to optimize.

Video diffusion models.

Generative diffusion models (DMs)

song_generative_2019  ;  ho_denoising_2020  ;  NEURIPS2021_49ad23d1  ;  song_score-based_2021

, have largely superseded GANs owing to their improved training stability, high-fidelity output and iterative sampling process, which enables downstream tasks without retraining

song_generative_2019  ;  ho_denoising_2020

.
 Video diffusion models

ho_video_2022

have achieved state-of-the-art performance

ho_imagen_2022  ;  gupta_photorealistic_2023  ;  wang_videocomposer_2023  ;  yang_cogvideox_2024  ;  zheng_open-sora_2024  ;  deng_autoregressive_2025  ;  kong_hunyuanvideo_2025

, e.g., through latent VDMs

he_latent_2023  ;  blattmann_stable_2023  ;  ma_latte_2024

, and improved training strategies

gupta_photorealistic_2023  ;  jin_pyramidal_2024  ;  deng_autoregressive_2025

. Classifier-free guidance has also been explored to enable variable-length conditioning on past video frames with VDMs

song_history-guided_2025

.
 The ability of VDMs to model uncertainties and to produce sharp outputs makes them powerful tools, e.g., for weather prediction

price_probabilistic_2025  ;  li_generative_2024  ;  yang_generative_2025

, super-resolution (downscaling)

srivastava_precipitation_2024

, reconstructing spatiotemporal dynamics from sparse sensor measurements

li_learning_2024

, emulating precipitation dynamics directly from remote sensing observations

stock_d