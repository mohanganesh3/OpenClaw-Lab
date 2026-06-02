# EV003: DSAT-Net: Dual Spatial Attention Transformer for Building Extraction From Aerial Images

URL: https://www.semanticscholar.org/paper/00a3e8e05e7a2c2feae20df51b5acaed537724b6
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

Both local and global context dependencies are essential for building extraction from remote sensing (RS) images. Convolutional neural network (CNN) can extract local spatial details well but lacks the ability to model long-range dependency. In recent years, vision transformer (ViT) has shown great potential in modeling global context dependency. However, it usually brings huge computational cost, and spatial details cannot be fully retained in the process of feature extraction. To maximize the advantages of CNNs and ViTs, we propose dual spatial attention transformer net (DSAT-Net), which combines them in one model. In DSAT-Net, we design an efficient dual spatial attention transformer (DSAFormer) to solve the defects of standard ViT. It has a dual attention structure to complement each other. Specifically, the global attention path (GAP) conducts large-scale downsampling of the feature maps before the global self-attention (SA) computing, to reduce the computational cost. The local attention path (LAP) uses efficient stripe convolution to generate local attention, which can alleviate the loss of information caused by downsampling operation in the GAP and supplement the spatial details. In addition, we design a feature refining module called channel mixing feature refine module (CM-FRM) to fuse low- and high-level features. Our model achieved competitive results on three public building extraction datasets. The code will be available at https://github.com/stdcoutzrh/BuildingExtraction.
