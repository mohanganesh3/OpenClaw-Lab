[2205.12594] Heterogeneous Reservoir Computing Models for Persian Speech Recognition

Heterogeneous Reservoir Computing Models for Persian Speech Recognition

Zohreh Ansari

Biomedical Engineering Department
 Meybod University
 Meybod, Iran

z_ansari@meybod.ac.ir

&amp;Farzin Pourhoseini

Biomedical Engineering Department
 Meybod University
 Meybod, Iran

pourhoseinifarzin@gmail.com

&amp;Fatemeh Hadaeghi

Institute of Computational Neuroscience
 University Medical Center Hamburg-Eppendorf (UKE)
 Hamburg, Germany

f.hadaeghi@uke.de

Abstract

Over the last decade, deep-learning methods have been gradually incorporated into conventional automatic speech recognition (ASR) frameworks to create acoustic, pronunciation, and language models. Although it led to significant improvements in ASRs’ recognition accuracy, due to their hard constraints related to hardware requirements (e.g., computing power and memory usage), it is unclear if such approaches are the most computationally- and energy-efficient options for embedded ASR applications. Reservoir computing (RC) models (e.g., echo state networks (ESNs) and liquid state machines (LSMs)), on the other hand, have been proven inexpensive to train, have vastly fewer parameters, and are compatible with emergent hardware technologies. However, their performance in speech processing tasks is relatively inferior to that of the deep-learning-based models. To enhance the accuracy of the RC in ASR applications, we propose heterogeneous single and multi-layer ESNs to create non-linear transformations of the inputs that capture temporal context at different scales. To test our models, we performed a speech recognition task on the Farsdat Persian dataset. Since, to the best of our knowledge, standard RC has not yet been employed to conduct any Persian ASR tasks, we also trained conventional single-layer and deep ESNs to provide baselines for comparison. Besides, we compared the RC performance with a standard long-short-term memory (LSTM) model. Heterogeneous RC models (1) show improved performance to the standard RC models; (2) perform on par in terms of recognition accuracy with the LSTM, and (3) reduce the training time considerably.

K  eywords  Automatic speech recognition

⋅

⋅

\cdot

Deep echo state networks

⋅

⋅

\cdot

Heterogeneity

⋅

⋅

\cdot

Farsdat

⋅

⋅

\cdot

Persian (Farsi) language

⋅

⋅

\cdot

Recurrent neural networks

⋅

⋅

\cdot

Reservoir computing (RC)

1  Introduction

1.1  Practical relevance and motivation

Over decades of research, hidden Markov models and different variations of neural networks have been extensively exploited to perform speech recognition, speaker identification, text to speech conversion, and other tasks that are relevant for speech processing applications  [ 1 ,  2 ,  3 ,  4 ] . During the past decade, and thanks to remarkable advancements in graphics processing unit (GPU) and cloud computing technologies, a large variety of deep-learning-based methods have been designed and tested to accomplish challenging speech processing tasks on large datasets  [ 5 ,  6 ,  7 ,  8 ] . In conventional ASR systems, these models are primarily employed to extract features, to uncover the relationships between the audio signal and the phonemes (or other linguistic units), to learn pronunciation lexicons, and to recognize (and use) sequences of words. These developments, however, have been mainly focused on improving recognition accuracy by training models on a multitude of collected data. Imprecision and false interpretation, although momentous, are not the only challenges faced by practical ASR systems. Since algorithms have to be computationally efficient, less data-hungry, and compatible with emerging micro-device technologies, further investigations are needed to develop inexpensive yet accurate processing methods. Besides, it is of crucial importance to make speech recognition available for more languages and dialects.

In contrast to deep-learning-based methods, reservoir computing (RC) models (e.g., echo state networks (ESNs)  [ 9 ]  and liquid state machines (LSMs)  [ 10 ] ) have been proven inexpensive to train, have vastly fewer parameters, and reported to perform well in processing complex temporal and spatio-temporal data in real-world applications  [ 11 ,  12 ,  13 ] . It has also been suggested that RC models can provide accurate subject-specific classifiers that are adaptable to the unique characteristic features of temporal data recorded from a target person and do not rely on a vast amount of data collected from other individuals  [ 14 ] . Besides, the essential ingredient of a reservoir computing model is a random excitable medium that non-linearly projects an input signal into a higher-dimensional signal space. Therefore, researchers from computing theory and microchip technologies have considered RC as a computational scheme compatible with “unconventional” physical or computational
platforms such as analog electrical circuits  [ 15 ,  16 ] , optical media  [ 17 ] , and chemical (molecular) substrates  [ 18 ] . It is, therefore, promising to design and implement functional sensors, processors, and controllers based on this computational framework.

However, up to this point, the performance of RC in speech processing tasks has been relatively inferior to that of the deep-learning-based models  [ 19 ]  and needs further improvements to fit conventional or end-to-end ASR systems with practical exploitation. In this regard, a major upgrade to shallow RC systems was introducing deep echo state networks (deep ESNs) that are able to capture temporal context of the input signal at different time-scales through several successively stacked RC layers  [ 20 ,  21 ] . It enhanced the performance of RC in time-series prediction  [ 20 ] , short-term memory capacity (MC) task, and classification of experimental data in the field of computational biology  [ 22 ] . However, it has remained to be further assessed if this alteration could also strengthen RC-based speech recognition models. In this study, therefore, we chose to explore applicability of RC models in speech recognition. Besides, since to the best of our knowledge, neither standard RC nor deep ESN have been employed to conduct any Persian ASR tasks, we trained conventional single-layer and deep ESNs to perform a speech recognition task on the Farsdat Persian speech dataset.

1.2  Related works

The FARSDAT dataset  [ 23 ]  was collected in 1996 and since then has been used as the standard benchmark for developing Persian ASR systems such as Shenava  [ 24 ]  and Nevisa  [ 25 ] . Similar to other ASR models, both systems comprise three modules for feature extraction, and acoustic and language modelling. Dimension reduction techniques such as principle component analysis, wavelet transforms, filter banks, Cepstral analysis, Mel frequency cepstrum, and kernel based methods are commonly employed to extract relevant features from each 20-30 ms frames (segments) of the speech signal. While standard Mel frequency Cepstral coefficient (MFCC) analysis was dominantly utilized in the feature extraction modules of the first editions of Persian ASRs, it has been recently suggested that LHCB (Logarithm of Hamming Critical Band filter banks)  [ 26 ] , convolutional neural networks (CNNs), and deep-belief-networks (DBNs) may return features that further increase the accuracy of Persian speech recognition in neural network models  [ 27 ] .

In the acoustic and language modelling modules, combinations of Gaussian mixture models (GMMs) and deep neural networks with hidden Markov models have been extensively investigated to convert the extracted features to a probability over characters in the Persian alphabet, and to turn these probabilities into words  [ 28 ,  29 ,  30 ,  31 ] . In addition to conventional Persian ASR platforms, end-to-end systems based on modular deep neural architectures, and uni- and bi-directional long-short-term-me