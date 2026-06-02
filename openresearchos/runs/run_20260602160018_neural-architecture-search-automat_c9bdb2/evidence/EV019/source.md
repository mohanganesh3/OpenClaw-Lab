# EV019: LAMPS: A Layer-wised Mixed-Precision-and-Sparsity Accelerator for NAS-Optimized CNNs on FPGA

URL: https://www.semanticscholar.org/paper/025b8fea1cb30aa371fb5adb932bbf8052deac17
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

The increasing model size and computation load of convolutional neural networks (CNN) pose a grand challenge to deploy CNN models on edge computing devices. To further improve performance without significant accuracy loss, this paper developed a neural architecture search (NAS) method to achieve a layer-wise mixed-precision-and-sparsity (LAMPS) CNN. However, this optimization cannot be fully utilized and directly mapped to existing AI accelerators due to the irregu- lar computation of sparse and multi-precision data. To tackle this challenge, this work proposed a LAMPS vector systolic accelerator and demonstrated state-of-the-art results. Experi- mental results show that the LAMPS accelerator on Xilinx ZCU102 achieves an average performance of 756.83 GOPS and 470.25 GOPS when accelerating the NAS-optimized VGG16 and Resnet18, respectively, leading to 1.3-6.0x speed-up over the state- of-the-art accelerators on FPGA.
