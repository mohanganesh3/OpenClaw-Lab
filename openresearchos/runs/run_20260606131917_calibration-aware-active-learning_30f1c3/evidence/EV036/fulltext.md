[2510.00528] Quantum Probabilistic Label Refining: Enhancing Label Quality for Robust Image Classification

Quantum Probabilistic Label Refining: Enhancing Label Quality for

Robust Image Classification

Fang Qi,
Lu Peng,
Zhengming Ding

Abstract

Learning with softmax cross-entropy on one-hot labels often leads to overconfident predictions and poor robustness under noise or perturbations. Label smoothing mitigates this by redistributing some confidence uniformly, but treats all samples equally, ignoring intra-class variability. We propose a hybrid quantum–classical framework that leverages quantum non-determinism to refine data labels into probabilistic ones, offering more nuanced, human-like uncertainty representations than label smoothing or Bayesian approaches. A variational quantum circuit (VQC) encodes inputs into multi-qubit quantum states, using entanglement and superposition to capture subtle feature correlations. Measurement via the Born rule extracts probabilistic soft labels that reflect input-specific uncertainty. These labels are then used to train a classical convolutional neural network (CNN) with soft-target cross-entropy loss. On MNIST and Fashion-MNIST, our method improves robustness—achieving up to 50% higher accuracy under noise—while maintaining competitive clean-data accuracy. It also enhances model calibration and interpretability, as CNN outputs better reflect quantum-derived uncertainty. This work introduces  Quantum Probabilistic Label Refining , bridging quantum measurement and classical deep learning for robust training via refined, correlation-aware labels without architectural changes or adversarial techniques.

Introduction

Deep learning has transformed image classification, yet even top models remain brittle under minor distribution shifts like noise, blur, or rotation  (Hendrycks and Dietterich  2019 ; Minderer et al.  2021 ) . In contrast, human perception remains robust and calibrated under similar conditions  (Muttenthaler et al.  2022 ; Geirhos et al.  2021 ; Hermann, Chen, and Kornblith  2020 ; Minderer et al.  2021 ) . A key reason is the use of one-hot labels in training, which treat classification as deterministic, forcing full certainty on a single class, even for ambiguous inputs  (Zhang et al.  2017 ) . This overconfidence is problematic in high-stakes settings. In autonomous driving, misclassifying a road sign due to occlusion or lighting can be catastrophic. In medical imaging, confident errors on borderline cases may lead to misdiagnoses. In such domains, recognizing and conveying uncertainty is essential  (Hendrycks et al.  2019 ; Ghoshal et al.  2021 ) .

Figure 1:  Conceptual overview of our QPLR approach, where an input digit is encoded into a variational quantum circuit and outputs a  probabilistic label distribution  over the

K  =  10

K=10

classes. This soft label supervises a downstream DNN to capture more realistic crowd-source knowledge.

Label smoothing is a common solution that redistributes a fraction of confidence from the true class to others to reduce overconfidence  (Szegedy et al.  2016 ) . However, it applies the same softened distribution to all samples, ignoring differences in ambiguity among inputs. In reality, class overlaps and uncertain samples are poorly represented in standard datasets. What’s missing is a data-dependent method to reflect sample-specific uncertainty—something static label smoothing can’t achieve.

To address this, we turn to quantum non-determinism as a natural source of uncertainty. Quantum systems yield probabilistic outputs upon measurement—a principle formalized by the Born rule  (Hall  2013 ) , which assigns outcome probabilities based on the squared amplitude of quantum states  (Nielsen and Chuang  2010 ) . By encoding images into a variational quantum circuit (VQC), we harness quantum superposition and entanglement to produce rich, sample-specific distributions over class labels  (Huang et al.  2021b ; McArdle et al.  2020 ; Cong, Choi, and Lukin  2019 ) . Specifically, entanglement encodes input-feature correlations, and superposition represents diverse configurations. Quantum measurement then extracts input-dependent probabilistic labels that reflect these correlations, offering a more expressive alternative to classical heuristics.

We introduce Quantum Probabilistic Label Refining (QPLR), a novel framework that uses a VQC to generate soft labels from quantum measurement distributions, shown in Fig.

1  . These are then used to train a classical convolutional neural network (CNN) via soft-target cross-entropy loss. Unlike one-hot or uniform-smoothed labels, these targets reflect input-specific ambiguity and class overlap. To our knowledge, this is the first work to leverage quantum measurement distributions as soft labels for classical deep learning. Our contributions are summarized as follows:

•

QPLR framework:  We propose QPLR, a hybrid method using quantum measurement from VQCs to generate refined, sample-specific probabilistic labels for training classical networks.

•

Improved robustness:  CNNs trained with QPLR show up to 50% better accuracy under image corruptions on MNIST and Fashion-MNIST, without degrading clean-data performance.

•

Better calibration:  QPLR enhances uncertainty estimation and interpretability, critical for safety-sensitive applications like autonomous driving and medical AI.

Related Work

Quantum Deep Learning . Variational Quantum Circuits (VQCs) have emerged as a novel approach for realizing near-term quantum advantage in supervised learning tasks  (Cerezo et al.  2021 ; Blance and Spannowsky  2021 ; Maheshwari, Sierra-Sosa, and Garcia-Zapirain  2021 ; Kapoor, Wiebe, and Svore  2016 ; Khairy et al.  2020 ) .
Compared to traditional quantum neural networks (QNNs)  (Abbas et al.  2021 ; Henderson et al.  2020 ; Cong, Choi, and Lukin  2019 )  that rely on mid-circuit measurements or deep architectures, VQCs prioritize resource efficiency on NISQ devices by using shallow parameterized circuits without intermediate state collapse, thereby mitigating decoherence and measurement noise  (Cerezo et al.  2021 ; Benedetti et al.  2019 ) .
Early works successfully demonstrated VQCs on small-scale benchmarks such as MNIST, achieving feasibility on simulators and noisy hardware  (Havlíček et al.  2019 ; Caro et al.  2022 ; Chalumuri, Kune, and Manoj  2021 ; Zhang et al.  2023 ; Schuld et al.  2020 ; McArdle et al.  2019 ) . However, these efforts have focused on using the quantum model itself as the final classifier—i.e., collapsing the measurement outcome into a single label or expectation value. In contrast, our work exploits the  full probabilistic output  of the VQC (i.e., the distribution over measurement outcomes) as a  teaching signal  for a classical network.

Soft Labels and Probabilistic Supervision . In classical deep learning,  soft labels  or probability targets are well known to improve generalization and calibration. The seminal work of Hinton et al. introduced  knowledge distillation , where a large teacher network’s predictive distribution (soft targets) guides a smaller student model, yielding better performance than hard labels alone  (Hinton, Vinyals, and Dean  2015 ) . Label smoothing extends this idea by blending the one-hot target with a uniform distribution over classes, thus preventing overconfidence and improving calibration  (Szegedy et al.  2016 ; Pereyra et al.  2017 ) . More recent studies have shown that soft targets encode inter-class similarity information that hard labels lack, making models more robust to input perturbations  (Müller, Kornblith, and Hinton  2019 ; Pereyra et al.  2017 ) . Our approach can be viewed as a form of distillation where the  teacher is a quantum model —a fundamentally different source of uncertainty stemming from quantum mechanics rather than a larger neural network.

Robust Image Classification . Robustness to input noise and geometric distortions remains a cen