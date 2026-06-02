# EV012: ECGMamba: Towards ECG Classification with State Space Models

URL: https://www.semanticscholar.org/paper/0020ade044802cc30de71eff66ce2eb2f49f1a98
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

Electrocardiogram (ECG) is essential for diagnosing cardiovascular diseases. The Transformer architecture, employing multi-head self-attention, has shown promise in ECG classification due to its strong sequence modeling capabilities. However, Transformers suffer from suboptimal inference efficiency. State Space Models (SSMs) have garnered attention for their efficient inference. This study introduces ECGMamba, a novel ECG classification model based on Mamba, a specialized SSM. ECGMamba comprises the ECG encoder and the Mamba layer centered on Mamba. The encoder uses three one-dimensional convolutions to extract ECG local information. The core of the Mamba layer is the Multi-Path Mamba-based block, which models global contextual semantic information. This architecture ensures efficient inference and significant performance improvements. Experimental results on PTB-XL and CPSC2018 databases demonstrate ECGMamba’s competitive performance and efficient inference capabilities. This study suggests a new approach for efficient and accurate ECG classification.
