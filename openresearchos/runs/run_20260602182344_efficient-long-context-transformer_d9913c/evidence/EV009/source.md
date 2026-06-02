# EV009: Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation

URL: https://www.semanticscholar.org/paper/00272b4b7ae8abe36c74b0b04591a1fd678f7c3b
Year: 2025
Source: semantic_scholar
Arxiv: 2503.22051

## Abstract

Simultaneous or streaming machine translation generates translation while reading the input stream. These systems face a quality/latency trade-off, aiming to achieve high translation quality similar to non-streaming models with minimal latency. We propose an approach that efficiently manages this trade-off. By enhancing a pretrained non-streaming model, which was trained with a seq2seq mechanism and represents the upper bound in quality, we convert it into a streaming model by utilizing the alignment between source and target tokens. This alignment is used to learn a read/write decision boundary for reliable translation generation with minimal input. During training, the model learns the decision boundary through a read/write policy module, employing supervised learning on the alignment points (pseudo labels). The read/write policy module, a small binary classification unit, can control the quality/latency trade-off during inference. Experimental results show that our model outperforms several strong baselines and narrows the gap with the non-streaming baseline model.
