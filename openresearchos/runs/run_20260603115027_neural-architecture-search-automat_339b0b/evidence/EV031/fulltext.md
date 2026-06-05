[2112.04719] Learning with Nested Scene Modeling and Cooperative Architecture Search for Low-Light Vision

Learning with Nested Scene Modeling and Cooperative Architecture Search
 for Low-Light Vision

Risheng Liu, 
Long Ma,
Tengyu Ma,
Xin Fan, 
and Zhongxuan Luo

R. Liu is with DUT-RU International School of Information Science &amp; Engineering, Dalian University of Technology, Dalian, 116024, China. (Corresponding author, e-mail: rsliu@dlut.edu.cn).
L. Ma is with the School of Software Technology, Dalian University of Technology, Dalian, 116024, China. (e-mail: malone94319@gmail.com).
T. Ma is with the School of Software Technology, Dalian University of Technology, Dalian, 116024, China. (e-mail:matengyu@mail.dlut.edu.cn).
X. Fan is with the DUT-RU International School of Information Science &amp; Engineering, Dalian University of Technology, Dalian, 116024, China. (email: xin.fan@dlut.edu.cn).
Z. Luo is with the School of Software Technology, Dalian University of Technology, Dalian, 116024, China. (email: zxluo@dlut.edu.cn).
Manuscript received April 19, 2005; revised August 26, 2015.

Abstract

Images captured from low-light scenes often suffer from severe degradations, including low visibility, color cast and intensive noises, etc. These factors not only affect image qualities, but also degrade the performance of downstream Low-Light Vision (LLV) applications. A variety of deep learning methods have been proposed to enhance the visual quality of low-light images. However, these approaches mostly rely on significant architecture engineering to obtain proper low-light models and often suffer from high computational burden. Furthermore, it is still challenging to extend these enhancement techniques to handle other LLVs. To partially address above issues, we establish Retinex-inspired Unrolling with Architecture Search (RUAS), a general learning framework, which not only can address low-light enhancement task, but also has the flexibility to handle other more challenging downstream vision applications. Specifically, we first establish a nested optimization formulation, together with an unrolling strategy, to explore underlying principles of a series of LLV tasks. Furthermore, we construct a differentiable strategy to cooperatively search specific scene and task architectures for RUAS. Last but not least, we demonstrate how to apply RUAS for both low- and high-level LLV applications (e.g., enhancement, detection and segmentation). Extensive experiments verify the flexibility, effectiveness, and efficiency of RUAS.

Index Terms:

Low-light vision, nested optimization, Retinex-inspired unrolling, cooperative architecture search.

1

Introduction

Low-Light Vision (LLV) mainly refers to various visual enhancement and perception tasks in low-light scenarios and is always more challenging than classical computer vision problems. This is because that the poor visual quality of low-light observations (e.g., low visibility, color cast, and intensive noises caused by diverse degraded factors) could heavily affect information extraction and transformation in these vision tasks  [ 1 ,  2 ,  3 ] .

(a) Image Enhancement

(b) Object Detection

(c) Semantic Segmentation

Figure 1:  Illustrating averaged quantitative results of RUAS on various LLV tasks. In (a), we first compare our RUAS (with three kinds of variations) to state-of-the-art low-light enhancement approaches on LOL dataset  [ 4 ] . We then demonstrate the performance of RUAS for objective detection (on DARK FACE dataset  [ 5 ] ) and semantic segmentation (on ACDC dataset  [ 6 ] ) in (b) and (c), respectively. For two high-level tasks, we also demonstrate the overall performance of the two-stage techniques, i.e., integrate these enhancement methods with state-of-the-art detection (e.g., DSFD  [ 7 ] ) and segmentation (e.g., DeepLab-v3+  [ 8 ] ) techniques. More detailed analysis can also be found in our experimental part.

In the past few years, researchers have made a considerable effort on designing deep learning models to enhance the visual quality of low-light observations. Learning with paired data  [ 4 ,  9 ,  10 ,  11 ]  is the prevalent pattern to acquire the desired model, although the physical principle (with high generality) is what they utilize in most works, which is because of the distribution limitation of supervised learning that these techniques manifest the poor generalization in the unknown real-world scenarios. Unsupervised learning paradigm for low-light image enhancement  [ 12 ,  13 ,  14 ]  emerges as time requires, to provide a strong adaptation ability towards diverse low-light conditions. Indeed, these works realize better performance than those based on supervised learning. However, currently employed methods have mostly been developed manually, thus requiring significant architecture engineering and large-size models to explore the underlying low-light scene information. More importantly, these works with supervised/unsupervised learning are specifically designed for the purpose of improving the visual quality, hardly be extended to other types of LLVs (e.g., detection and segmentation).

Very recently, benefiting from the development of intelligent techniques (e.g., autonomous driving), several perception tasks have also been considered in low-light scenarios  [ 2 ,  15 ,  3 ] .
Existing works mostly devote themselves to shrinking the gap between low-light observations and clear images, and further fine-tune perception algorithms that are designed on the regular natural environments. However, in this way, it pays too much attention to the visual quality of low-light observations, causing unfavorable outcomes of the perception algorithms because of ignoring the intrinsic interactions for scenes and tasks. In other words, how to explore the latent correspondence between low-light scenes and perception tasks may be more important.

To partially address the above issues, we propose a general learning framework, named Retinex-inspired Unrolling with Architecture Search (RUAS)  1

1  1 A preliminary version of this work has been published in  [ 1 ] .

. As shown in Fig.

1  , our method realizes the best task performance with the lowest computational resources both in low-light image enhancement and other extended high-level vision tasks (e.g., detection and segmentation).
Concretely, taking low-light scenes into consideration, we first reformulate the low-light vision as a scene energy constrained learning formulation. Then by integrating the Retinex rule to obtain an unrolling scheme for training, we establish two kinds of training strategies from the training objective that is with an unsupervised scene-oriented loss. Finally, we provide a cooperative architecture search strategy for the scene and task module. Our contributions can be summarized as follows  2

2  2 Code is available at  https://github.com/vis-opt-group/RUAS .

:

•

RUAS establishes a general and principled learning framework, which can effectively formulate the underlying low-light scene information for LLV tasks, thus not only has the ability to enhance the visual quality of low-light images, but also is flexible enough to solve other challenging perception-type vision applications.

•

We establish a nested optimization model to simultaneously formulate the low-light scene feature, vision task and their intrinsic relationships. By designing a Retinex-inspired unrolling scheme and introducing an unsupervised scene-oriented loss, we obtain a prior and data aggregated training strategy for RUAS.

•

For architecture search, we develop a differentiable strategy, which is able to discover cooperative scene and task architectures from a compact search space, thus has the ability to not only obtain memory and computation efficient models, but also offer flexibility for different kinds of LLVs.

•

We experimentally demonstrate that RUAS can be effectively applied to both low- and high-level LLV applications (