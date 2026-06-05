# EV003: Dynamic Short Convolutions Improve Transformers

URL: https://www.semanticscholar.org/paper/0008fb43a207d6b6abe7bbfcf54100bd2d62401a
Year: 2026
Source: semantic_scholar
Arxiv: 2606.03825

## Abstract

Transformers have become the dominant architecture for large language models, largely due to the scalability and flexibility of attention, feed-forward layers, residual connections, and normalization. This paper introduces dynamic short convolutions as an additional neural network primitive for improving Transformers. Unlike static short convolutions, dynamic convolutions use input-dependent filters, which preserves the locality bias of convolution while increasing expressivity. Motivating experiments show that applying dynamic short convolutions to key, query, and value representations improves performance on challenging associative recall tasks compared with static convolutional variants. Across language-modeling experiments ranging from 150M to 2B parameters, dynamic convolutions consistently outperform standard Transformers and Transformers augmented with static short convolutions. Fitting scaling laws indicates a 1.33$\times$ compute advantage over compute-matched Transformers when dynamic convolutions are applied to the key, query, and value vectors, and a 1.60$\times$ advantage when adding dynamic convolutions after every linear layer. Dynamic convolutions also offer improvements on linear RNNs (Mamba-2/Gated DeltaNet) and mixture-of-experts architectures. We make these gains practical with custom Triton kernels that enable efficient training with a manageable end-to-end slowdown. These results suggest that dynamic short convolutions are a scalable, hardware-efficient, and expressive primitive for advancing Transformer-based language models.
