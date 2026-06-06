[2509.11800] Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics

\useunder

\ul

1

1  institutetext:  Department of Electrical and Computer Engineering, University of British Columbia, Vancouver, Canada

1

1  email:  guangnan@ece.ubc.ca

2

2  institutetext:  Division of Cardiology, Vancouver General Hospital, Vancouver, Canada

1

1  1 T. Tsang and P. Abolmaesumi are joint senior authors.

Pseudo-D: Informing Multi-View Uncertainty Estimation with Calibrated Neural Training Dynamics

Ang Nan Gu

Michael Tsang

Hooman Vaseli

Purang Abolmaesumi

Teresa Tsang

Abstract

Computer-aided diagnosis systems must make critical decisions from medical images that are often noisy, ambiguous, or conflicting, yet today’s models are trained on overly simplistic labels that ignore diagnostic uncertainty. One-hot labels erase inter-rater variability and force models to make overconfident predictions, especially when faced with incomplete or artifact-laden inputs. We address this gap by introducing a novel framework that brings uncertainty back into the label space. Our method leverages neural network training dynamics (NNTD) to assess the inherent difficulty of each training sample. By aggregating and calibrating model predictions during training, we generate uncertainty-aware pseudo-labels that reflect the ambiguity encountered during learning. This label augmentation approach is architecture-agnostic and can be applied to any supervised learning pipeline to enhance uncertainty estimation and robustness. We validate our approach on a challenging echocardiography classification benchmark, demonstrating superior performance over specialized baselines in calibration, selective classification, and multi-view fusion.

1  Introduction

Medical image-based diagnosis faces two key challenges. First, it is safety-critical, where diagnostic errors can have serious consequences. Second, image acquisition is inherently imperfect. For example, ultrasound imaging depends heavily on the sonographer’s skill and patient-specific factors, making it difficult to capture images at the optimal angle consistently. These limitations can lower image quality and, in turn, compromise diagnostic accuracy.
As a result, evaluating these systems requires more than measuring classification accuracy; it must also account for the model’s uncertainty estimates. In challenging cases or when image quality is poor, the system should know to abstain from making a prediction and instead refer the case to a human expert. This capability is assessed through the task of selective classification.
Moreover, arriving at a diagnosis often requires integrating information from multiple sources, such as different imaging modalities or several views of the same anatomy. Crucially, the fusion process must account for the varying degrees of uncertainty associated with each source. This capability is evaluated through the task of multi-view fusion.

We propose a method to enhance selective classification and multi-view fusion by improving uncertainty estimation (UE). We focus on aleatoric uncertainty (AU), the irreducible uncertainty caused by incomplete or ambiguous input data. Although AU is less common in traditional vision tasks, it is critical in medical imaging, where anatomical features can be obscured or ambiguous because of patient variability, the imaging process, and modality-specific factors.

A core challenge in estimating AU is the lack of ground-truth labels that faithfully capture uncertainty. Classification datasets typically provide a single, definitive label, even though expert assessments often reflect borderline cases or ambiguity. Moreover, the common use of “one-hot” labels fails to convey the nuanced reasoning needed in uncertain cases. To achieve this, the model’s confidence should be better aligned with task difficulty.

We leverage Neural Network Training Dynamics (NNTD) to generate pseudo-labels that quantify uncertainty based on how confidently and consistently the model learns each sample during training. Rather than relying on a fixed label, we track the model’s evolving predictions across epochs and treat this trajectory as a measure of sample difficulty. NNTD-based methods have proven effective in detecting label noise  [ 23 ,  26 ] , improving classification  [ 24 ] , and producing more reliable uncertainty estimates  [ 9 ] .

We propose  Pseudo-D , a novel technique which combines NNTD information from both the training and validation sets to created pseudo-labels which are calibrated at the sub-class level. This is particularly useful when certain sub-classes are harder to distinguish than others.
We evaluate Pseudo-D on a challenging multi-view ultrasound dataset for aortic stenosis (AS) classification, which requires integrating information from multiple scanning planes and handling patient-specific variability in image quality.
We demonstrate that training with  Pseudo-D  improves uncertainty estimation in standard deep learning classifiers, and outperforms specialized methods on selective classification and multi-view fusion tasks. Compared to existing approaches, our method better aligns model uncertainty with input-specific factors like image quality and anatomical visibility. Furthermore,  Pseudo-D  is agnostic to model architecture and requires minimal changes to integrate to existing customized workflows.

Figure 1:  We augment the training phase by first recording the history of predicted logits. The magnitude of the correct class logit (shown in blue) relative to other classes (in black) varies with task difficulty. We use the training history to generate pseudo-labels that align with the difficulty of each example. Our proposed training technique,  Pseudo-D , yields a model with predicted probabilities that strongly correspond to image quality. Additionally, the pseudo-labels help mitigate overfitting by assigning lower confidence values to difficult training examples.

2  Related Works

2.0.1  Selective Classification.

The task of selective classification (SC), or prediction with a “reject” option, was initially extended to deep learning by Geifman et al.  [ 8 ] . DeVries and Taylor  [ 6 ]  proposed explicitly learning uncertainty as an additional output of the model, using a modified loss function.
Rabanser et al.  [ 24 ]  suggested using model checkpoints from different training epochs to form an ensemble for SC. Huang et al.  [ 14 ]  explored how label augmentation can improve SC, and Feng et al.  [ 7 ]  showed that the softmax response can out-perform specialized scoring functions in existing SC approaches.

2.0.2  Multi-view Fusion.

Multi-view fusion involves the combination of classifier predictions, where each prediction stems from a unique view of the same underlying object. The fusion of predicted probabilities can be through averaging  [ 25 ] , majority voting  [ 19 ] , or by learned weighting scheme  [ 31 ] . Zhang et al.  [ 33 ]  establishes a theoretical link between uncertainty estimation and multi-view classification performance for logit-based approaches.
Evidential Neural Networks (ENNs)  [ 27 ]  are trained to output belief masses instead of logits. Belief masses can be aggregated using Dempster’s Rule of Combination, a mathematically rigorous method for combining multiple predictions  [ 12 ] .

2.0.3  Aortic Stenosis Severity Classification.

Aortic stenosis (AS) is a heart valve disease characterized by restricted blood flow through the aortic valve. The clinical standard for AS diagnosis relies on measuring blood flow volume through the left ventricular outflow tract, typically derived from spectral Doppler  [ 21 ] . However, Doppler-based diagnosis is sensitive to measurement variability  [ 28 ,  22 ]  and is often unavailable on newer, lightweight ultrasound devices  [ 10 ] . Recent clinical works  [ 1 ,  20 ]  proposed assessing AS severity through B-mode ultrasound interpretation by human cl