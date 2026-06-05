# EV002: A Spatial–Temporal Adaptive Graph Convolutional Network with Multi-Sensor Signals for Tool Wear Prediction

URL: https://www.semanticscholar.org/paper/0005f6d4318b963c623ce471083c04236fd46b6c
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Tool wear monitoring is crucial for optimizing cutting performance, reducing costs, and improving production efficiency. Existing tool wear prediction models usually design integrated models based on a convolutional neural network (CNN) and recurrent neural network (RNN) to extract spatial and temporal features separately. However, the topological structures between multi-sensor networks are ignored, and the ability to extract spatial features is limited. To overcome these limitations, a novel spatial–temporal adaptive graph convolutional network (STAGCN) is proposed to capture spatial–temporal dependencies with multi-sensor signals. First, a simple linear model is used to capture temporal patterns in individual time-series data. Second, a spatial–temporal layer composed of a bidirectional Mamba and an adaptive graph convolution is established to extract degradation features and reflect the dynamic degradation trend using an adaptive graph. Third, multi-scale triple linear attention (MTLA) is used to fuse the extracted multi-scale features across spatial, temporal, and channel dimensions, which can assign different weights adaptively to retain important information and weaken the influence of redundant features. Finally, the fused features are fed into a linear regression layer to estimate the tool wear. Experimental results conducted on the PHM2010 dataset demonstrate the effectiveness of the proposed STAGCN model, achieving a mean absolute error (MAE) of 3.40 μm and a root mean square error (RMSE) of 4.32 μm in the average results across three datasets.
