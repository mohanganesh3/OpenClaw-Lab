# EV007: From Packets to Pixels: A Vision Transformer & Few-Shot Learning Approach to Malicious Network Threat Detection

URL: https://www.semanticscholar.org/paper/012d5b9f0c5615ad890846a7735cf5d529ea5841
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Deep learning excels at deriving rich, high-dimensional features from modalities such as text, audio, and images. In network security, these methods offer a path to learn deep representations from the vast volumes of traffic that modern infrastructures carry. However, current malware-traffic analysis workflows rely on classical deep learning methods, such as convolutional or sequential architectures. While these methods are effective at capturing local byte-sequence patterns, they struggle to model dependencies that span long packet flows. Transformer models, with their use of self-attention, help address this limitation by attending globally to the input sequence. However, treating each network byte as an individual token leads to lengthy one-dimensional input sequences. In this paper, we introduce a compact, image-inspired encoding that converts contiguous byte segments into fixed patch sizes and arranges them on a two-dimensional grid. Each resulting patch functions as a single input token, dramatically reducing sequence length. Next, we leverage these packet images as input into a Vision Transformer (ViT) to efficiently attend across the entire packet. This patch-based representation preserves global context for self-attention while exploiting the ViT’s strength in modeling spatially structured inputs. Lastly, to address the limitation of retraining the model to generalize to newly introduced malware classes, we introduce few-shot learning to quickly adapt to novel classes in a data-restricted environment. On the CIC-IoT23 malware-traffic dataset, our approach achieves impressive accuracy across multiple malware families, outperforming baseline methods.
