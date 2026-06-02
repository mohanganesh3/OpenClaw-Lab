[2603.28211] Explaining CLIP Zero-shot Predictions Through Concepts

Explaining CLIP Zero-shot Predictions Through Concepts

Onat Ozdemir 1,2

Anders Christensen 3,4,5

Stephan Alaniz 6

Zeynep Akata 7,8,9,10

Emre Akbas 2,8,11

1 School of Informatics, University of Edinburgh

2 Dept. of Computer Eng., Middle East Technical University (METU)

3 Orbital

4 DTU Compute, Technical University of Denmark

5 Dept. of Biology, University of Copenhagen

6 LTCI, Télécom Paris, Institut Polytechnique de Paris

7 Technical University of Munich (TUM)

8 Helmholtz Munich

9 MCML

10 MDSI

11 Robotics &amp; AI Center (ROMER), METU

Abstract

Large-scale vision-language models such as CLIP have achieved remarkable success in zero-shot image recognition, yet their predictions remain largely opaque to human understanding. In contrast, Concept Bottleneck Models provide interpretable intermediate representations by reasoning through human-defined concepts, but they rely on concept supervision and lack the ability to generalize to unseen classes. We introduce EZPC that bridges these two paradigms by explaining CLIP’s zero-shot predictions through human-understandable concepts. Our method projects CLIP’s joint image-text embeddings into a concept space learned from language descriptions, enabling faithful and transparent explanations without additional supervision. The model learns this projection via a combination of alignment and reconstruction objectives, ensuring that concept activations preserve CLIP’s semantic structure while remaining interpretable. Extensive experiments on five benchmark datasets, CIFAR-100, CUB-200-2011, Places365, ImageNet-100, and ImageNet-1k, demonstrate that our approach maintains CLIP’s strong zero-shot classification accuracy while providing meaningful concept-level explanations. By grounding open-vocabulary predictions in explicit semantic concepts, our method offers a principled step toward interpretable and trustworthy vision-language models. Code is available at  https://github.com/oonat/ezpc .

1  Introduction

Figure 1 :

Overview of EZPC.  CLIP image and text embeddings are projected into a shared concept space using a learnable matrix

A  A

. The projected representations

c  x

c_{x}

and

c  k

c_{k}

provide (i) concept-based explanations via a Hadamard product and (ii) class logits via a dot-product in concept space. Training jointly optimizes a matching loss and a reconstruction loss to preserve CLIP’s predictive behavior.

The rapid integration of machine learning into real-world systems has intensified the demand for models that are not only accurate but also transparent and trustworthy. In high-stakes domains such as medical imaging, autonomous driving, and scientific discovery, understanding why a model makes a particular prediction is as critical as the prediction itself. Despite their impressive capabilities, modern deep networks remain largely black boxes, making it difficult to interpret their internal reasoning or diagnose their failures.

Concept Bottleneck Models (CBMs)  [ 13 ]  address this issue by introducing an intermediate layer composed of human-understandable concepts. These models decompose the prediction process into two stages: (1) mapping inputs to concept activations, and (2) predicting the final class label based on these activations. This structure provides an interpretable interface between perception and decision-making, allowing users to inspect, validate, or even modify concept activations to understand or correct model behavior.

However, classical CBMs are constrained by two major limitations. First, they require dense concept supervision, which is often expensive or infeasible to collect. Second, they operate under a closed-world assumption: CBMs are trained and evaluated on a fixed set of classes and thus fail to generalize to unseen categories or new domains. These constraints limit their scalability and applicability to open-vocabulary recognition problems. Recent efforts to mitigate these issues by leveraging vision-language models  [ 22 ,  35 ,  37 ]  reduce annotation costs, but still require task-specific training and cannot generalize to unseen classes.

In contrast, vision-language models (VLMs) such as CLIP  [ 24 ] , ALIGN  [ 10 ] , and SigLIP  [ 38 ]  demonstrate strong open-vocabulary generalization by aligning images and text in a shared semantic space. CLIP, for instance, learns to associate visual and textual information at scale, enabling zero-shot classification by comparing an image embedding with textual embeddings of candidate labels. Without task-specific training, CLIP can accurately recognize objects from natural-language descriptions, which is a significant leap toward flexible, general-purpose perception.

Yet, this generalization comes at the cost of interpretability. CLIP’s embeddings are high-dimensional and entangled, offering little insight into what visual or semantic properties drive a particular decision. As a result, users cannot easily understand why CLIP associates an image with a given label, nor can they trace these predictions back to human-interpretable reasoning.

In this paper, we aim to bridge the gap between the interpretability of CBMs and the generalization ability of CLIP. We propose a method that explains CLIP’s zero-shot predictions through human-understandable concepts. Instead of retraining CLIP or relying on annotated concepts, our method introduces a lightweight decomposition that projects CLIP’s image-text embeddings into a shared concept space. This enables faithful, concept-level explanations of CLIP’s predictions while maintaining its zero-shot capabilities.

Our method, “Explaining CLIP Zero-shot Predictions Through Concepts” ( EZPC ), aligns CLIP’s representations with a predefined concept basis using two complementary objectives: (i) a matching loss that enforces alignment between learned and known concept embeddings, and (ii) a reconstruction loss that preserves CLIP’s similarity structure in the concept space. The resulting model not only interprets CLIP’s predictions through meaningful concepts but also retains high zero-shot accuracy across diverse datasets.

Contributions. 
Our key contributions are as follows:

•

We propose a novel method that decomposes CLIP’s image-text embeddings into a shared concept space, enabling interpretable zero-shot predictions.

•

We introduce two training objectives, matching and reconstruction, that jointly align concept projections with CLIP’s latent space while preserving semantic fidelity.

•

We demonstrate through quantitative and qualitative experiments on five benchmarks that EZPC provides human-interpretable explanations of CLIP predictions with minimal performance loss.

2  Related Work

Our work lies at the intersection of zero-shot learning, vision-language modeling, and concept-based interpretability. Below, we review the most relevant research in each area and discuss how our approach relates to prior work.

Zero-shot Learning. 
Zero-shot learning (ZSL) aims to recognize unseen categories without explicit training data for those classes. Early work achieved this by leveraging human-defined attributes as semantic intermediaries  [ 6 ,  16 ,  15 ] , enabling recognition of novel classes through shared attributes. Later approaches replaced attributes with distributed word embeddings such as Word2Vec, GloVe, and BERT to establish semantic correspondences between visual and linguistic spaces  [ 20 ,  23 ,  7 ] .

Embedding-based ZSL methods  [ 21 ,  8 ,  1 ,  26 ]  projected both images and labels into a shared latent space, where similarity was computed for classification. Other approaches  [ 5 ,  31 ,  11 ,  18 ]  instead aim to learn weights for zero-shot classification. More recently, large-scale vision-language models such as CLIP  [ 24 ] , ALIGN  [ 10 ] , and SigLIP  [ 38 ]  have shown remarkable zero-shot generalization by aligning visual and textual modalities through contrast