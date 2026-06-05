# EV002: DPRNN-Former: An Efficient Way to Deal with Blind Source Separation

URL: https://www.semanticscholar.org/paper/0085379764b9036cf727bd1a72b40790247e85b8
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

Recent advancements in DL have indicated that time-domain methods are more successful than traditional time-frequency-based methods regarding speech separation. However, modeling very long sequences in time-domain separation systems presents some challenges. Recurrent neural networks and 1-D convolutional neural networks are not sufficient for modeling lengthy sequences by themselves. In this paper, a hybrid RNN is proposed, combining a pre-trained DPRNN and transformer. This strategy uses the transformer's ability to perceive context, allowing it to gain insight into the time-evolving data connected to audio signals. To handle extended input sequences, the network partitions them into more manageable sections, performing intra-section and inter-section operations iteratively. The proposed network surpasses current state-of-the-art algorithms, achieving SI-SNR of 11.129 and SDR of 11.285 dB on the public WSj0-3mix dataset.
