# EV008: Compiler-Based Efficient CNN Model Construction for 5G Edge Devices

URL: https://www.semanticscholar.org/paper/004e787f43bdc21f9745d3e4e8e546f9fa15d01b
Year: 2021
Source: semantic_scholar
Arxiv: n/a

## Abstract

With the increasing demand to deploy convolutional neural networks (CNNs) on 5G mobile platforms, architecture designs with efficient sparse kernels (SKs) were proposed, which can save more parameters than the standard convolution while maintaining the high accuracy. Despite the great potential, neural network designs with SKs still require a lot of expert knowledge and take ample time. In this paper, we first propose a search scheme that effectively reduces the SK design space based on three aspects: composition, performance, and efficiency. Meanwhile, we completely eliminate the model training from our search scheme. Instead, an easily measurable quantity, the information field, is identified and used to predict the model accuracy in the searching process. Additionally, we provide a detailed efficiency analysis on the final designs found by our scheme. Second, based on the analysis we propose a model transformation scheme to better utilize the SK designs on existing models to either reduce the number of parameters or increase the accuracy. Last, considering the extra programming overhead and the expert knowledge required by the model transformation scheme, we develop a compiler prototype to automate the entire process, given the source code of an existing model. Experimental results show that models composed of the sparse kernel designs searched by our search scheme can beat state-of-the-art networks such as ResNets in terms of the accuracy and the efficiency. Also by using our model transformation scheme we can easily improve the accuracy (the same number of parameters) or the efficiency (the same accuracy) upon existing state-of-the-art models.
