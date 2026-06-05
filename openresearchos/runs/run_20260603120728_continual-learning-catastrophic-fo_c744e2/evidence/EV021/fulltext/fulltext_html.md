[2509.07993] Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization

Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization

Federico Fontana
 Sapienza University of Rome

federico.fontana@uniroma1.it

Anxhelo Diko
 Sapienza University of Rome

anxhelo.diko@uniroma1.it

Romeo Lanzino
 Sapienza University of Rome

romeo.lanzino@uniroma1.it

Marco Raoul Marini
 Sapienza University of Rome

marco.marini@uniroma1.it

Bachir Kaddar
 University Ibn Khaldoun of Tiaret

b.kaddar@univ-tiaret.dz

Gian Luca Foresti
 University of Udine

gianluca.foresti@uniud.it

Luigi Cinque
 Sapienza University of Rome

luigi.cinque@uniroma1.it

Abstract

The rapid evolution of deepfake generation technologies poses critical challenges for detection systems, as non-continual learning methods demand frequent and expensive retraining. We reframe deepfake detection (DFD) as a Continual Learning (CL) problem, proposing an efficient framework that incrementally adapts to emerging visual manipulation techniques while retaining knowledge of past generators. Our framework, unlike prior approaches that rely on unreal simulation sequences, simulates the real-world chronological evolution of deepfake technologies in extended periods across 7 years. Simultaneously, our framework builds upon lightweight visual backbones to allow for the real-time performance of DFD systems. Additionally, we contribute two novel metrics: Continual AUC (C-AUC) for historical performance and Forward Transfer AUC (FWT-AUC) for future generalization. Through extensive experimentation (over 600 simulations), we empirically demonstrate that while efficient adaptation (+155 times faster than full retraining) and robust retention of historical knowledge is possible, the generalization of current approaches to future generators without additional training remains near-random (FWT-AUC

≈  \approx

0.5) due to the unique imprint characterizing each existing generator. Such observations are the foundation of our newly proposed Non-Universal Deepfake Distribution Hypothesis.
 Code will be released upon acceptance.

1  Introduction

Figure 1 :

Illustration of the simulation framework spanning from 2018 to 2025. Given a time

t  t

, a dataset is selected according to a Time protocol followed by a custom batch extraction which simulates a real-world data stream scenario at the specified time

t  t

.

The evolution of deepfake technology has profound implications for the authenticity of digital media. With novel deepfake generators emerging continuously, traditional deepfake detection (DFD) methods—which rely on static training paradigms—struggle to keep pace. This leads not only to significant computational overhead but also to reduced effectiveness in dynamic, real-world environments. Such challenges are especially critical on social media, where the rapid spread of manipulated content can exacerbate misinformation.
Prior works  [  29  ,

26  ,

53  ]  have explored the application of continual learning (CL) for DFD.  [  29  ]  introduces a continual learning benchmark for DFD with three evaluation sequences based on difficulty or length, applying CL methods in class-incremental (CIL) and domain-incremental (DIL) settings.  [  53  ]  builds on this by adding a new generator and evaluating two CL methods, while  [  26  ]  reframes DFD as a CIL, focusing on DFD generalization.

However, current approaches prove useful only in a small number of CL methods  [  53  ] , utilize a very limited evaluation set  [  26  ]  (

≤  \leq

2K samples), or employ a not chronological sequence of deepfake generators  [  29  ,

26  ,

53  ] . To our knowledge, no prior work addresses the DFD as a real-world scenario; rather, the sequence of generators fed to the CL methods is chosen either randomly, based on generator affinity, or according to the difficulty detectors face in recognizing them.
As a result, existing research cannot be directly related to real-world settings, as the experimental assumptions for training and evaluation do not reflect real conditions.
Furthermore, no in-depth study on data efficiency has been conducted for CL methods, nor has a study on training and inference efficiency been made. Without efficient training and inference, detection models risk becoming obsolete and unable to keep pace with the continuous advancements. Optimizing efficiency ensures maintainability, making DFD systems viable for deployment in high-stakes environments such as social media monitoring.

Motivated by these works, we propose a pipeline that chronologically simulates a deepfake scenario over extended periods using datasets ordered by time, along with a strategy to simulate the sequence of tasks as a random sample from a datastream in any moment(e.g., social media) as illustrated in Fig.

1  . This design aims to better capture a real-world deepfake scenario for training and evaluation while avoiding the choice bias on the generator sequence. Additionally, we frame the problem as a DIL task to maintain a detector with consistent size, speed, and memory usage throughout its lifetime. Our focus also extends to the need for data, training, and inference efficiency. Therefore, the proposed method employs lightweight models that can achieve real-time performance on most smartphones and requires significantly lower data to be trained due to the CL strategies, while maintaining competitive performance.

We conducted a total of over 600 full simulations on the presented chronological framework and observed a striking pattern: while some methods successfully retained knowledge of past deepfake generators, none exhibited strong generalization to future ones. To systematically capture this phenomenon, we introduce two new evaluation metrics to measure the retention of past knowledge in highly imbalanced classes and one to quantify a model’s generalizability to chronologically unseen deepfake generators on unbalanced classes.
Our analysis using these metrics revealed a fundamental decorrelation between past and future generalization, challenging the assumption that deepfake detection models can generalize across time. This leads us to formally propose the Non-Universal Deepfake Distribution Hypothesis, which states that deepfake detection cannot be effectively generalized through static training because each deepfake generator imprints a unique, non-transferable signature. We empirically validate this hypothesis.

In summary, our main contributions are:

•

We propose a novel  Chronological CL framework  for DFD that mirrors real-world deepfake evolution.

•

We evaluated  8 different CL strategies and 4 models  within the proposed framework, demonstrating adaptation while efficiently preserving historical knowledge.

•

We introduce  Continual AUC (C-AUC)  and  Forward Transfer AUC (FWT-AUC)  metrics to measure the stability and transferability of the learned detection capabilities in a chronological and unbalanced classes setting.

•

We propose the  Non-Universal Deepfake Distribution Hypothesis , suggesting that each deepfake generator leaves a unique, non-transferable signature, requiring continuous model updates. We support this hypothesis with empirical evaluation on more than 600 simulations.

2  Related Work

Figure 2 :

The lines represent the 8 CL methods evaluated on MobileNetV4, each configured with its optimal hyperparameters found in the ablation study. The C-AUC (Eq.

5  ) indicates the classifiers’ performance on the test data available up to that point. The red horizontal line represents the mean FWT-AUC (Eq.

6  ) across all experiments, reflecting the methods’ generalization performance on unseen datasets, with the red bars indicating the standard deviation. The blue horizontal line denotes the Eval AUC achieved by the full-retraining method.

2.1  Continual Learning

CL tackles the challenge of catastrophic forgetting  [  15  ] , where mo