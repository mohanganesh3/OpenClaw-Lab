[2010.04683] Smooth Variational Graph Embeddings for Efficient Neural Architecture Search The authors acknowledge support by the German Federal Ministry of Education and Research Foundation via the project DeToL.

Smooth Variational Graph Embeddings for Efficient Neural Architecture Search

†

†  thanks:  The authors acknowledge support by the German Federal Ministry of Education and Research Foundation via the project DeToL.

Jovita Lukasik 1 ,
David Friede 1 ,
Arber Zela 2 ,
Frank Hutter 2,  3 ,
Margret Keuper 1

1 University of Mannheim,
 2 University of Freiburg,
 3 Bosch Center for Artificial Intelligence

{jovita, david}@informatik.uni-mannheim.de, keuper@uni-mannheim.de
 {zelaa, fh}@cs.uni-freiburg.de

Abstract

Neural architecture search (NAS) has recently been addressed from various directions, including discrete, sampling-based methods and efficient differentiable approaches. While the former are notoriously expensive, the latter suffer from imposing strong constraints on the search space. Architecture optimization from a learned embedding space for example through graph neural network based variational autoencoders builds a middle ground and leverages advantages from both sides. Such approaches have recently shown good performance on several benchmarks. Yet, their stability and predictive power heavily depends on their capacity to reconstruct networks from the embedding space. In this paper, we propose a two-sided variational graph autoencoder, which allows to smoothly encode and accurately reconstruct neural architectures from various search spaces. We evaluate the proposed approach on neural architectures defined by the ENAS approach, the NAS-Bench-101 and the NAS-Bench-201 search space and show that our smooth embedding space allows to directly extrapolate the performance prediction to architectures outside the seen domain (e.g. with more operations).
Thus, it facilitates to predict good network architectures even without expensive Bayesian optimization or reinforcement learning.

Index Terms:

representation learning, neural architecture search, graph neural network, deep learning

I

Introduction

Recent progress in computer vision is to a large extent coupled to the advancement of novel neural architectures  [ 1 ,  2 ] . In this context, the automated search of neural architectures  [ 3 ,  4 ,  5 ] 
is increasingly important, as it removes the fatiguing and time-consuming process of manual trial-and-error network design.

Figure 1:  Architecture of the proposed SVGe model.
It takes as input a neural architecture graph.
The encoder (left) uses two GNN modules, the forward encoder (green) and the backward encoder (red), to create an informative latent representation

𝐡  G

subscript  𝐡  𝐺

\mathbf{h}_{G}

. This latent vector is input to the decoder (right), which decodes forward (green) and backward (red) pass separately, generating two graphs in a sequential manner. Their union is the output of SVGe.

Neural Architecture Search (NAS) is intrinsically a discrete optimization problem and can be solved effectively using black-box methods such as reinforcement learning  [ 6 ,  4 ] , evolution  [ 7 ,  5 ] , Bayesian optimization (BO)  [ 8 ,  9 ,  10 ]  or local search  [ 11 ] . However, finding a good solution typically requires thousands of function evaluations, which is infeasible without company-scale compute infrastructure. Recent research in NAS focus as well on efficient methods via continuous relaxations of the discrete search space and weight-sharing  [ 12 ,  13 ,  14 ,  15 ] . However, such methods
yield efficient yet oftentimes sub-optimal results  [ 16 ] .

Therefore, we argue in favor of NAS on learned graph embeddings using encoder-decoder graph neural networks (GNN)  [ 17 ,  18 ,  19 ] . Zhang et al.  [ 20 ]  recently showed good performance with such a model, D-VAE, on the ENAS search space  [ 13 ]  in neural architecture performance prediction and BO - proving its ability to learn smooth continuous graph representations. D-VAE aggregates information in the architecture GNN alternatingly in the forward pass and in the backward pass to encode the neural network information flow. However, the D-VAE model imposes strong constraints on the graph structure, which limit its applicability to search spaces beyond ENAS. In addition, it has very long training times. In this paper, we propose a two-sided variational encoder-decoder GNN to learn smooth embeddings in various NAS search spaces, which we call  Smooth Variational Graph embedding  (SVGe). In contrast to D-VAE, SVGe aggregates node representations in the forward and backward pass separately and consequently decodes their joint representation into forward and backward pass separately (see Fig.

1  ). This yields a very high reconstruction ability without imposing any constraints on the search space and allows for a more efficient training.

Inheriting from variational autoencoders  [ 21 ] , it places structurally similar graphs close to one another in the embedding space and thus facilitates efficient black-box optimization to find high-performing architectures. The proposed model is not only three times faster than D-VAE but also shows improved BO results on the ENAS search space. In contrast to D-VAE, it can be directly applied to other search spaces such as NAS-Bench-101  [ 22 ]  and NAS-Bench-201  [ 23 ] .

Moreover, it allows to learn architecture performance prediction in a supervised way and extrapolate from the space of observed architectures at test time. This way, high performing architectures even outside of the original search space can be proposed at very low costs.

In summary, we make the following contributions:
(i) We introduce a novel graph variational autoencoder, SVGe, that builds a structurally smooth variational graph embedding by learning accurate representations of neural architectures (Sec.

III-A

and

III-B

).
(ii) We discuss theoretical properties of our approach (Sec.

III-D

).
(iii) We conduct extensive evaluations on the ENAS  [ 13 ] , NAS-Bench-101  [ 22 ]  and NAS-Bench-201 [ 23 ]  search spaces and show that our approach allows for competitive BO results in all three search spaces. Our experiments show that SVGe is able to extrapolate to larger unseen architectures. It finds an architecture with a best accuracy of

95.18  %

percent  95.18

95.18\%

when learning from the NAS-Bench-101 search space. This improves over the best architecture within this space. In addition, our top

1

1

1

found architecture improves over comparable architectures in terms of validation and test accuracy, when transferring to ImageNet16-120  [ 24 ]  (Sec.

IV  ).

II

Related Work

Graph Generative Models.

Recent years have shown huge progress in representation learning for graph-based data with Graph Neural Networks (GNNs)  [ 25 ,  18 ,  26 ,  27 ] . GNNs follow a message passing scheme, where node feature vectors aggregate information from their neighbors  [ 28 ]  and capture local structural information.
To obtain a graph-level representation these
feature vectors are pooled  [ 29 ] .
GNNs differ in their neighborhood node information as well as in their graph-level aggregation procedure  [ 30 ,  27 ,  18 ,  25 ,  31 ,  32 ,  33 ] .
Graph generation can be addressed globally by relaxing the adjacency matrix  [ 34 ,  35 ]  or sequentially by adding nodes and edges alternately using recurrent networks  [ 36 ,  37 ]  or GNNs  [ 38 ] . Our decoder model is similar to  [ 38 ]  in the aggregation procedure. Yet, while  [ 38 ]  treat forward and backward pass equally, our model aggregates node information for both separately to account for the order of network operations and the information flow. Zhang et al.  [ 20 ]  propose a less efficient, alternating message passing scheme for this purpose and reinstall the validity of decoded architectures using a heuristic which employs prior knowledge on the search space. The proposed method differs in both encode