[2511.11627] SA-EMO: Structure-Aligned Encoder Mixture of Operators for Generalizable Full-waveform Inversion

SA-EMO: Structure-Aligned Encoder Mixture of Operators for Generalizable Full-waveform Inversion

Zhenyu Wang 2 ,
Peiyuan Li 1

1

1  footnotemark:

1

,
Yongxiang Shi 1

1

1  footnotemark:

1

,
Ruoyu Wu 3 ,
Lei Zhang 1

1 School of Science, China University of Mining and Technology, Beijing

2 School of Artificial Intelligence, China University of Mining and Technology, Beijing

3 City University of Hong Kong (Dongguan)

These authors contributed equally.Corresponding author.

Abstract

Full-waveform inversion (FWI) can produce high-resolution subsurface models, yet it remains inherently ill-posed, highly nonlinear, and computationally intensive. Although recent deep-learning and numerical-acceleration methods have improved speed and scalability, they often rely on single-CNN architectures or single neural operators, which struggle to generalize in unknown or complex geological settings and are ineffective at distinguishing diverse geological types. To address these issues, we propose an Structure-Aligned Encoder–Mixture-of-Operators (SA-EMO) architecture for velocity-field inversion under unknown subsurface structures. First, a structure-aligned encoder maps high-dimensional seismic wavefields into a physically consistent latent space, thereby eliminating spatio-temporal mismatch between the waveform and velocity domains, recovering high-frequency components, and enhancing feature generalization. Then, an adaptive routing mechanism selects and fuses multiple neural-operator experts—including spectral, wavelet, multiscale, and local operators—to predict the velocity model. We systematically evaluate our approach on the OpenFWI benchmark and the Marmousi2 dataset. Results show SA-EMO significantly outperforms traditional CNN or single-operator methods, achieving an average MAE reduction of approximately 58.443% and an improvement in boundary resolution of about 10.308%. Ablation studies further reveal that the structure-alignment encoder, the expert-fusion mechanism, and the routing module each contribute markedly to the performance gains. This work introduces a new paradigm for efficient, scalable, and physically interpretable full-waveform inversion.

K  eywords  Full Waveform Inversion

⋅  \cdot

Neural Operators

⋅  \cdot

Structure-Aligned Encoder

⋅  \cdot

Seismic Imaging

⋅  \cdot

MoE

1  Introduction

Figure 1:  Cross-domain SSIM comparison of data-driven inversion frameworks.

Full waveform inversion (FWI) aims to reconstruct high-fidelity subsurface velocity structures from multi-source seismic wavefields. Despite remarkable speedups achieved by deep learning, existing models still exhibit inconsistent cross-type generalization and degraded high-frequency reconstruction under complex geological conditions. We trace this limitation to an overlooked factor: the  temporal–spatial and structural misalignment  between the waveform domain (time

×  \times

receivers) and the velocity domain (2D structure). This mismatch causes uneven spectral energy distribution, degrading both learning stability and boundary fidelity.

Spectral analysis reveals that naive geometric interpolation ( e.g ., resizing wavefields onto the velocity grid) superficially aligns shapes but distorts frequency components and erases fine details, leading to over-smoothing and poor boundary recovery. Consequently, even powerful single-operator architectures such as FNO or WNO struggle to generalize across diverse geological structures.

To address this challenge, we present the Structure-Aligned Encoder–Mixture-of-Operators (SA-EMO) framework. The design consists of three tightly coupled components.
First, a  structure-aligned encoder  learns a physically consistent projection from raw wavefields to a

70  ×  70

70{\times}70

latent space that aligns with the target velocity structure.
Second, four  complementary neural operator experts —FNO, WNO, MNO, and LNO—jointly capture global, multiscale, hierarchical, and local physical priors within this latent space.
Third, an  adaptive routing mechanism  fuses experts via type-weighted and group-weighted aggregation with  strong–weak activation , dynamically balancing dominant and auxiliary experts to adapt to different geological regimes.
The entire system is trained with a hybrid spatial–spectral loss for unified optimization.

Empirically , the encoder primarily improves per-type inversion accuracy, whereas the routing mechanism enhances cross-type generalization. On OpenFWI  [ openfwi ] , coupling FNO with our encoder improves MAE/RMSE/SSIM from

0.104  /  0.147  /  0.903

0.104/0.147/0.903

to

0.085  /  0.125  /  0.925

0.085/0.125/0.925

on CurveVelB. Strong–weak routing further enhances robustness across all ten sub-datasets and preserves structural continuity on the Marmousi2  [ marmousi ]  dataset, demonstrating strong out-of-distribution (OOD) generalization.

Contributions.

•

We propose a  structure-aligned encoder  that explicitly aligns waveform and velocity domains, achieving balanced spectral energy and improved single-operator accuracy.

•

We introduce an  operator-level mixture-of-experts  that integrates FNO, WNO, MNO, and LNO to jointly capture global, multiscale, and local physical priors in a unified latent space.

•

We design a  strong–weak activation routing mechanism  within a hierarchical type- and group-weighted framework, improving cross-type generalization and interpretability without explicit type labels.

•

We conduct extensive experiments on ten OpenFWI sub-datasets and the OOD Marmousi2 dataset, showing that the encoder improves accuracy while routing enhances generalization.

2  Related Work

In this section, we review prior work related to our method, covering physics-based full waveform inversion, deep learning-based inversion, neural operator learning and its applications to FWI, encoder–neural-operator architectures, and mixture-of-experts models. We highlight their limitations and explain how our framework builds upon and extends these directions.

2.1  Physics-based Full Waveform Inversion

Classical full waveform inversion (FWI) iteratively minimizes the data misfit between observed and simulated wavefields under a physical wave equation  [ Virieux2009 ] .
Although these approaches provide physically interpretable reconstructions, they suffer from non-convexity, sensitivity to initial models, and high computational cost  [ Geng2018 ,  Song2023 ,  DesigningFWI2023 ] .
Recent reviews  [ Operto2022 ,  UnlockingFWI2023 ,  ElasticFWI2025 ]  summarize extensions beyond the Born approximation and emphasize the need for scalable, data-efficient solutions.
However, traditional FWI still struggles in complex geological settings due to the ill-posed nature of the inversion problem and strong nonlinearity of wave propagation.

2.2  Deep Learning for Seismic Inversion

With the rise of data-driven modeling, deep learning has been introduced to approximate the nonlinear FWI mapping directly from seismic data to velocity fields  [ araya2018deep ] .
CNN-based architectures such as InversionNet  [ wu2019inversionnet ] , VelocityGAN  [ zhang2019velocitygan ] , and UPFWI  [ yang2023fwigan ]  achieve remarkable inference speed and robustness.
However, they often exhibit poor generalization to unseen geological patterns  [ zhang2020data ] .
Hybrid frameworks that embed physical constraints or forward modeling within neural networks  [ jin2021unsupervised ,  chen2020seismic ,  jin2024empirical ,  schuster2024review ]  improve stability and interpretability but remain limited in scalability and adaptability across multiple geologies.

2.3  Neural Operators and Applications to Full Waveform Inversion

Neural operators (NOs) generalize function mappings between infinite-dimensional spaces, enabling resolution-independent PDE solvers  [ kovachki2023neural ] .
Representative families include the