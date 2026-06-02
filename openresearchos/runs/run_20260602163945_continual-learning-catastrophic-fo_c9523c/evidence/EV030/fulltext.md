[2201.12604] Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System

\sidecaptionvpos

figuret

Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System

Elahe Arani , Fahad Sarfraz *  &amp; Bahram Zonooz

Advanced Research Lab, NavInfo Europe, Eindhoven, Netherlands

{elahe.arani, fahad.sarfraz}@navinfo.eu, bahram.zonooz@gmail.com

Contributed equally.

Abstract

Humans excel at continually learning from an ever-changing environment whereas it remains a challenge for deep neural networks which exhibit catastrophic forgetting. The complementary learning system (CLS) theory suggests that the interplay between rapid instance-based learning and slow structured learning in the brain is crucial for accumulating and retaining knowledge. Here, we propose CLS-ER, a novel dual memory experience replay (ER) method which maintains short-term and long-term semantic memories that interact with the episodic memory. Our method employs an effective replay mechanism whereby new knowledge is acquired while aligning the decision boundaries with the semantic memories. CLS-ER does not utilize the task boundaries or make any assumption about the distribution of the data which makes it versatile and suited for “general continual learning”. Our approach achieves state-of-the-art performance on standard benchmarks as well as more realistic general continual learning settings.

1

1  1 The code is avaiable at:  https://github.com/NeurAI-Lab/CLS-ER

1  Introduction

Continual learning (CL) refers to the ability of a learning agent to continuously interact with a dynamic environment and process a stream of information to acquire new knowledge while consolidating and retaining previously obtained knowledge  (Parisi et al.,  2019 ) . This ability to continuously learn from a changing environment is a hallmark of intelligence and a critical missing component in our quest towards making our models truly intelligent.
The major challenge towards enabling CL in deep neural networks (DNNs) is that the continual acquisition of incrementally available information from non-stationary data distributions leads to catastrophic forgetting whereby the performance of the model on previously learned tasks drops drastically  (McCloskey &amp; Cohen,  1989 ) .

Several approaches have been proposed to address the issue of catastrophic forgetting in CL. These can be broadly categorized into regularization-based methods  (Farajtabar et al.,  2020 ; Kirkpatrick et al.,  2017 ; Ritter et al.,  2018 ; Zenke et al.,  2017 )  which penalizes changes in the network weights, network expansion-based methods  (Rusu et al.,  2016 ; Yoon et al.,  2017 )  which dedicate a distinct set of network parameters to distinct tasks, and rehearsal-based methods  (Chaudhry et al.,  2018 ; Lopez-Paz &amp; Ranzato,  2017 )  which maintains a memory buffer and replays samples from previous tasks. Amongst these, rehearsal-based methods have proven to be more effective in challenging CL tasks  (Farquhar &amp; Gal,  2018 ) . However, an optimal approach for replaying memory samples and constraining the model update to efficiently consolidate knowledge remains an open question.

In the brain, the ability to continually acquire, consolidate, and transfer knowledge over time is mediated by a rich set of neurophysiological processing principles  (Parisi et al.,  2019 ; Zenke et al.,  2017 )  and multiple memory systems  (Hassabis et al.,  2017 ) .
In particular, the CLS theory  (Kumaran et al.,  2016 )  posits that efficient learning requires two complementary learning systems: the hippocampus exhibits short-term adaptation and rapid learning of episodic information which is then gradually consolidated to the neocortex for slow learning of structured information.
Furthermore, a recent study by  Hayes et al. ( 2021 )  identified the missing elements of biological reply in the replay mechanisms employed in DNNs for CL. They highlight that many existing approaches only focus on modeling the prefrontal cortex directly and do not have a fast learning network which plays a critical role in enabling efficient CL in the brain.
Inspired by these studies, we hypothesize that mimicking the slow and rapid adaptation of information and having an efficient mechanism for incorporating them into the working memory can enable better CL in DNNs.

Figure 1:  CLS-ER employs a dual-memory learning mechanism whereby the episodic memory stores the samples and the semantic memories build short-term and long-term memories of the learned representations of the working model. The two memories interact to enforce a consistency loss on the working model which prevents rapid changes in the parameter space and enables the alignment of the decision boundary with semantic memories for effective knowledge consolidation.

To this end, we propose a novel dual memory experience replay method based on the complementary learning systems theory in the brain, dubbed as CLS-ER. In addition to a small episodic memory, our method builds long-term and short-term semantic memories which mimic the rapid and slow adaptation of information (Figure

1  ). As the network weights encode the learned representations of the tasks

(Krishnan et al.,  2019 ) , the semantic memories are maintained by taking the exponential moving average of the working model’s weights to consolidate information across the tasks with varying time windows and frequencies. The semantic memories interact with the episodic memory to extract consolidated replay activation patterns and enforce a consistency loss on the update of the working model so that new knowledge is acquired while aligning the decision boundary of the working model with the decision boundaries of semantic memories. This maintains a balance between the plasticity and stability of the model for effective knowledge consolidation.

CLS-ER provides a general CL method that does not utilize the task boundaries or make any strong assumption regarding the distribution of the data and tasks. We demonstrate the versatility and effectiveness of our method on a wide range of CL benchmark tasks as well as more challenging scenarios which simulate the complexities of CL in the real world.

2  Related Work

The base method for the rehearsal-based approach, Experience Replay (ER)  (Riemer et al.,  2018 )  combines the memory samples with the task samples into the training batch. Several techniques have since been employed on top of ER.
Meta Experience Replay (MER)  (Riemer et al.,  2018 )  considers replay as a meta-learning problem for maximizing the transfer from previous tasks and minimizing the interference.
iCARL  (Rebuffi et al.,  2017 )  uses the nearest average representation of past exemplars to classify in an incrementally learned representation space.
Gradient Episodic Memory (GEM)  (Lopez-Paz &amp; Ranzato,  2017 )  formulates optimization constraints on the exemplars in memory.
Gradient-based Sample Selection (GSS)  (Aljundi et al.,  2019 )  aims for memory sample diversity in the gradient space and provides a greedy selection approach.
Function Distance Regularization (FDR)  (Benjamin et al.,  2018 )  saves the network response at the task boundaries and adds a consistency loss on top of ER.
Dark Experience Replay (DER++) applies knowledge distillation  (Sarfraz et al.,  2021 )  and regularization on logits sampled during the optimization trajectory.

CLS has been used as a source of inspiration for dual memory learning systems in earlier works  (French,  1999 ; Robins,  1993 )  but they have not been shown to scale to current computer vision tasks  (Parisi et al.,  2019 ) . Recently,  Rostami et al. ( 2019 )  utilizes a generative model to couple sequential tasks in a latent embedding space.  Kamra et al. ( 2017 )  utilizes two generative models in a dual memory architecture. However, they utilize the task boundaries and generative replay ha