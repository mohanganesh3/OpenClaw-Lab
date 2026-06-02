[2307.05956] Language-Routing Mixture of Experts for Multilingual and Code-Switching Speech Recognition

\interspeechcameraready  \name

Wenxuan Wang, Guodong Ma, Yuke Li ∗

†

†  thanks:

∗  Corresponding author

, Binbin Du

Language-Routing Mixture of Experts for Multilingual and Code-Switching Speech Recognition

Abstract

Multilingual speech recognition for both monolingual and code-switching speech is a challenging task. Recently, based on the Mixture of Experts (MoE), many works have made good progress in multilingual and code-switching ASR, but present huge computational complexity with the increase of supported languages. In this work, we propose a computation-efficient network named  L anguage- R outing  M ixture  o f  E xperts (LR-MoE) for multilingual and code-switching ASR. LR-MoE extracts language-specific representations through the Mixture of Language Experts (MLE), which is guided to learn by a frame-wise language routing mechanism. The weight-shared frame-level language identification (LID) network is jointly trained as the shared pre-router of each MoE layer. Experiments show that the proposed method significantly improves multilingual and code-switching speech recognition performances over baseline with comparable computational efficiency.

Index Terms : mixture of experts, language identification, multilingual, code-switch, speech recognition

1  Introduction

Multilingualism is a widespread phenomenon in the world. Multilingual speakers often communicate in multiple languages simultaneously, such as interspersing English in Mandarin. Therefore, a practical multilingual speech recognition system needs to support the recognition of monolingual and code-switching utterances in multiple languages.

End-to-end (E2E) ASR systems [ 1 ,  2 ,  3 ,  4 ,  5 ,  6 ,  7 ,  8 ,  9 ]  have become more and more popular recently due to the simple pipeline, excellent performance and less dependence on linguistic knowledge compared to traditional hybrid methods [ 10 ] . Prior works based on the E2E model have also made good progress in the field of multilingual ASR, including code-switching corpus synthesis [ 11 ,  12 ,  13 ] , multi-task training with joint language identification [ 14 ,  15 ] , self-supervised speech representation learning [ 16 ,  17 ,  18 ] , cross-lingual transfer learning [ 15 ,  19 ] , etc.
The MoE architecture is an effective method to improve the performance of multilingual speech recognition in both monolingual and code-switching scenarios, which has been widely concerned and studied recently. The existing MoE-based methods  [ 20 ,  21 ,  22 ,  23 ,  24 ,  25 ,  26 ]  extract language-specific representations separately by independent encoders and fuse them to decode. Mostly, the computational complexity of the models will increase significantly with the number of supported languages.

In this work, we propose a computation-efficient network named  L anguage- R outing  M ixture  o f  E xperts (LR-MoE) to improve the performance of the multilingual and code-switching ASR task. The LR-MoE architecture consists of a shared block and a Mixture-of-Language-Experts (MLE) block.
Unlike the sparsely-gated mixture of experts (sMoE)  [ 27 ,  28 ,  29 ,  30 ] , the expert layers in the MLE block are language-dependent, which is called Language-Specific Experts (LSE).
The shared block generates the global representation, while the LSE of the MLE block extracts language-specific representations. In the MLE block, we design a Frame-wise Language Routing (FLR) mechanism, which guides the expert layers to learn language specialization at the training stage. A weight-shared frame-level language identification (LID) network is jointly trained as the shared pre-router of each LSE layer, and the alignment of frame-wise LID will be used as the routing path of the LSE layers. We also compare utterance-wise and frame-wise language routing for LR-MoE in the multilingual and code-switching experiment. To distinguish them, we will name the two networks with different routing as ULR-MoE and FLR-MoE, respectively.
Our contributions are summarized as follows:

•

We propose a computation-efficient LR-MoE architecture, which is suitable to apply in more languages with little increase in computational complexity.

•

We investigate multiple routing strategies of MoE and propose an FLR mechanism to guide the expert layers to learn language specialization, which is compatible with both multiple monolingual and code-switched ASR.

•

In Mandarin-English code-switching and multilingual experiments, the proposed method significantly improves the performances of multilingual and code-switching speech recognition over the baseline with comparable computational efficiency and outperforms previous MoE-based methods with less computational complexity.

2  Related Works and Motivation

2.1  Previous MoE-based works

More recently, many works  [ 20 ,  21 ,  22 ,  23 ,  24 ]  focus on exploring MoE architectures to recognize monolingual and intra-sentence code-switching speech. The MoE-based methods mainly utilized language-specific expert encoders to generate parallel language-specific representations and fuse them, whose difference is primarily in the fusion mode of expert encoders and training strategy. For example, the Bi-encoder transformer network  [ 21 ]  uses a gated network to dynamically output the MoE interpolation coefficients to mix two encoding representations. The weights of expert encoders are initialized with the pretrained monolingual model, respectively. Conditional factorized neural transducer  [ 22 ]  defined the monolingual sub-tasks with label-to-frame synchronization to achieve unified modeling of code-switching and monolingual ASR. The language-aware encoder  [ 24 ,  23 ]  learned language-specific representations through language-aware training with the language-specific auxiliary loss instead of monolingual pretraining and used the frame-wise addition to fuse them.

2.2  Motivations

As mentioned above, the previous MoE-based works achieved considerable improvement in monolingual and code-switching ASR, but there are still the following problems:

•

The approach needs to compute all language-specific blocks. However, only one works in the monolingual scene. It means a large amount of redundant computational overhead. And the more languages are supported, the more redundant computational overhead is.

•

Language-specific blocks are isolated from each other and lack interaction. As a result, the cross-linguistic contextual information is easily lost in code-switching speech.

In order to alleviate the above two issues, we propose the LR-MoE architecture inspired by the sparsely-gated mixture of experts  [ 27 ,  28 ,  29 ,  30 ] . Please refer to Section 3 for more details.

(a)

(b)

Figure 1:  Schematic diagram of MoE modules. (a) Sparsely-Gated Mixture of Experts (sMoE), (b) Mixture of Language Experts (MLE).

Figure 2:  The structure of the LR-MoE Transformer Model.

N

𝑁

N

and

(

L  −  N

)

𝐿  𝑁

(L-N)

are the number of layers of the shared block and the MLE block, respectively.

3  Proposed Method

3.1  Sparsely-Gated Mixture of Experts

The sMoE module is shown in Fig.

1  (a). As a representative, Switch Transformer  [ 28 ]  adopts a top-1 expert routing strategy in the MoE architecture to route the data samples to the expert model with the highest probability in the gated network. The computational complexity of the whole network increases slightly as the number of experts increases, and the extra computational overhead only comes from the gated network. The inputs of the expert layer and the gated network are the outputs of the previous non-expert layer

o

n  ​  e

subscript  𝑜

𝑛  𝑒

o_{ne}

. The router probability

p

𝑝

p

can be expressed as follows:

p  =

S  ​  o  ​  f  ​  t  ​  m  ​  a  ​  x  ​

(

W  r

⋅

o

n  ​  e

+

b  r

)

𝑝

𝑆  𝑜  𝑓  𝑡  𝑚  𝑎  𝑥

⋅

subscript  𝑊  𝑟

subscript  𝑜

𝑛  𝑒