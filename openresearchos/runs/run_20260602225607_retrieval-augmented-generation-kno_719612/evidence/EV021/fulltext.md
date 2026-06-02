[2507.21072] Snap, Segment, Deploy: A Visual Data and Detection Pipeline for Wearable Industrial Assistants

Snap, Segment, Deploy: A Visual Data and Detection Pipeline for Wearable Industrial Assistants

Di Wen  1  , Junwei Zheng  1  , Ruiping Liu  1  , Yi Xu  1  , Kunyu Peng  1,†  , Rainer Stiefelhagen  1

†

Corresponding authorAll authors are with the Computer Vision for Human-Computer Interaction Lab (cv:hci),
Institute for Anthropomatics and Robotics (IAR),
Karlsruhe Institute of Technology, 76131 Karlsruhe, Germany.
Emails:  {di.wen, junwei.zheng, ruiping.liu, kunyu.peng, rainer.stiefelhagen}@kit.edu , yi.xu@student.kit.edu

Abstract

Industrial assembly tasks increasingly demand rapid adaptation to complex procedures and varied components, yet are often conducted in environments with limited computing, connectivity, and strict privacy requirements. These constraints make conventional cloud-based or fully autonomous solutions impractical for factory deployment. This paper introduces a mobile-device-based assistant system for industrial training and operational support, enabling real-time, semi-hands-free interaction through on-device perception and voice interfaces.
The system integrates lightweight object detection, speech recognition, and Retrieval-Augmented Generation (RAG) into a modular on-device pipeline that operates entirely on-device, enabling intuitive support for part handling and procedure understanding without relying on manual supervision or cloud services.
To enable scalable training, we adopt an automated data construction pipeline and introduce a two-stage refinement strategy to improve visual robustness under domain shift. Experiments on our generated dataset,  i.e. , Gear8, demonstrate improved robustness to domain shift and common visual corruptions.
A structured user study further confirms its practical viability, with positive user feedback on the clarity of the guidance and the quality of the interaction. These results indicate that our framework offers a deployable solution for real-time, privacy-preserving smart assistance in industrial environments. We will release the Gear8 dataset and source code upon acceptance.

I

INTRODUCTION

The complexity of manufacturing assembly tasks is escalating, driven by the increasing diversity of components, intricate assembly procedures, and the growing need for customization in contemporary production settings  [ 1 ] .
While full automation is still unfeasible in many contexts, newly onboarded or rotating workers often struggle with steep learning curves, resulting in extended training periods and a higher incidence of assembly mistakes  [ 2 ] .
To address these challenges, vision-based assistance systems have emerged as a promising solution, offering real-time, step-by-step guidance that helps reduce both task completion time and error rates  [ 3 ] .

Figure 1:

Overview of our three-stage pipeline:  Snap ,  Segment , and  Deploy .
In the  Snap  stage, we collect multi-angle part videos and real-world background images from industrial environments.
In the  Segment  stage, we extract instance masks using SAM2  [ 4 ]  and generate a synthetic dataset (Gear8) by compositing parts into factory scenes. In the  Deploy  stage, a lightweight object detector trained on Gear8 is integrated into a wearable assistant system featuring speech input, visual recognition, knowledge retrieval, and audio feedback for interactive on-site guidance. The bottom row shows corrupted test images simulating real-world conditions for robustness evaluation.

However, the development of multimodal, on-device assistants for industrial use remains underexplored. Many existing solutions rely on cloud infrastructure, require extensive manual annotation, or fail to operate under practical constraints such as limited compute, offline usage, and strict data privacy  [ 5 ,  6 ,  7 ] .

We present a mobile-device-based industrial assistance system that leverages computer vision and Large Language Models (LLMs) to deliver real-time, context-aware guidance during assembly tasks. The system comprises a lightweight object detection module for part identification, a speech-to-text interface for natural language input, and a Retrieval-Augmented Generation (RAG) engine  [ 8 ]  that generates part-specific responses, which are conveyed through text-to-speech output.
This end-to-end, on-device pipeline supports intuitive and semi-hands-free interaction, eliminating the need for cloud-based processing.
To facilitate training without manual annotations, we propose a fully automated data generation pipeline that synthesizes part-level training images using multi-view video captures and compositing over diverse industrial backgrounds.
Additionally, to improve the detector’s generalization under domain shift, we introduce a two-stage training approach termed Background-Agnostic Refinement (BAR), wherein the model is fine-tuned on plain-background object crops to prioritize object-centric features.

The system is modular, allowing components such as the vision model, knowledge base, or language model to be updated independently for new tasks or deployment settings.

Bridging real-time object detection, vision-language interaction, and mobile industrial AI, our framework offers a deployable solution for contextual task assistance in constrained factory environments.

Extensive experiments on the Gear8 dataset, along with a structured user study, demonstrate the system’s effectiveness in improving detection robustness, task efficiency, and overall usability.

Our contributions are summarized as follows:

•

We design a real-time, multimodal assistant system for mobile industrial deployment, integrating visual detection, voice interaction, and semantic retrieval for hands-free part recognition and instruction.

•

We propose a fully automated dataset generation pipeline that requires no manual annotation and supports diverse industrial backgrounds using consumer-grade equipment.

•

We introduce a two-stage training strategy, Background-Agnostic Refinement (BAR), which improves detection robustness under domain shift without modifying model architecture.

II

Related Work

II-A

Synthetic Data Generation for Vision Tasks

Synthetic data has proven crucial for modern vision, initially in object detection and now across segmentation, pose estimation, and beyond. Early work used 3D renders: Peng et al.  [ 9 ]  fine‑tuned networks pre‑trained on CAD images with minimal real data for PASCAL VOC. Game‑engine datasets like “Playing for Data”/“Playing for Benchmarks”  [ 10 ,  11 ]  and Virtual KITTI  [ 12 ]  provided pixel‑perfect urban scenes, while URSA  [ 13 ]  and ProcSy  [ 14 ]  scaled this to millions of driving images.

Domain randomization  [ 15 ,  16 ]  addresses the sim‑to‑real gap by randomizing textures, lighting, and viewpoints. GAN‑based refinement improves realism: Nogues et al.  [ 17 ]  and Hu et al.  [ 18 ]  apply CycleGAN variants to industrial parts; Lin et al.  [ 19 ]  use multi‑task GANs for traffic‑sign detection. BigDatasetGAN  [ 20 ]  synthesizes pixel‑annotated ImageNet, and InstaGen  [ 21 ]  employs diffusion models for open‑vocabulary detection. Copy‑paste augmentation remains simple yet effective  [ 22 ] . Synthetic data has also enabled progress in diverse tasks such as urban-scene parsing  [ 23 ] , remote-sensing detection  [ 24 ] , and medical or robotic vision. Moreover, domain-adaptive detectors  [ 25 ]  and self-supervised learners benefit significantly from combining synthetic and real data.

We build on the copy‑paste paradigm by combining factory‑specific background images with instance masks extracted from full‑view object captures, producing diverse, domain‑adapted synthetic images that strike a balance between simplicity and high realism for industrial inspection.

II-B

Real-Time Egocentric Vision in Wearable Systems

Assistive technology  [ 26 ,  27 ] , especially real-time egocentric vision  [ 28 ,