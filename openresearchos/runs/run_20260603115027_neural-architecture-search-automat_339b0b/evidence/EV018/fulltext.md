[2203.13714] Searching for Network Width with Bilaterally Coupled Network

Searching for Network Width with Bilaterally Coupled Network

Xiu Su, 
Shan You, 
Jiyang Xie,

Fei Wang,
Chen Qian,
Changshui Zhang, 
Chang Xu

Xiu Su and Chang Xu are with the School of Computer Science, Faculty of Engineering, The University of Sydney, Australia. E-mail: xisu5992@uni.sydney.edu.au, c.xu@sydney.edu.au

Jiyang Xie is with the Pattern Recognition and Intelligent Systems Lab., Beijing University of Posts and Telecommunications, China. E-mail: xiejiyang2013@bupt.edu.cn

Shan You, Fei Wang, and Chen Qian are with the SenseTime Research Centre. E-mail: {youshan,wangfei,qianchen}@sensetime.com

Shan You and Changshui Zhang are with the Department of Automation, Tsinghua University, Institute for Artificial Intelligence, Tsinghua University (THUAI), Beijing National Research Center for Information Science and Technology (BNRist). E-mail: zcs@mail.tsinghua.edu.cn

Manuscript received April 19, 2005; revised August 26, 2015.

Abstract

Searching for a more compact network width recently serves as an effective way of channel pruning for the deployment of convolutional neural networks (CNNs) under hardware constraints. To fulfill the searching, a one-shot supernet is usually leveraged to efficiently evaluate the performance w.r.t. different network widths. However, current methods mainly follow a  unilaterally augmented  (UA) principle for the evaluation of each width, which induces the training unfairness of channels in supernet. In this paper, we introduce a new supernet called Bilaterally Coupled Network (BCNet) to address this issue. In BCNet, each channel is fairly trained and responsible for the same amount of network widths, thus each network width can be evaluated more accurately. Besides, we propose to reduce the redundant search space and present the BCNetV2 as the enhanced supernet to ensure rigorous training fairness over channels.
Furthermore, we leverage a stochastic complementary strategy for training the BCNet, and propose a prior initial population sampling method to boost the performance of the evolutionary search.
We also propose the first open-source width benchmark on macro structures named Channel-Bench-Macro for the better comparison of width search algorithms.
Extensive experiments on benchmark CIFAR-10 and ImageNet datasets indicate that our method can achieve state-of-the-art or competing performance over other baseline methods. Moreover, our method turns out to further boost the performance of NAS models by refining their network widths. For example, with the same FLOPs budget, our obtained EfficientNet-B0 achieves 77.53% Top-1 accuracy on ImageNet dataset, surpassing the performance of original setting by 0.65%.

Index Terms:

Deep neural network, channel number search, one-shot supernet, Bilaterally Coupled Network, prior initial population sampling, stochastic complementary strategy, evolutionary search.

1

Introduction

When deploying deep convolutional neural networks (CNNs) in the real world, it is important to take different hardware budgets into consideration  [ 1 ,  2 ] , e.g., floating point operations (FLOPs), latency, memory footprint, and energy consumption. Pruning redundant channels in CNNs is a natural way to derive a compact network that can simultaneously satisfy these different hardware constraints. Typical channel pruning usually leverages a pre-trained network and implements the pruning in an end-to-end  [ 3 ,  4 ]  or layer-by-layer  [ 5 ,  6 ]  manner. After pruning, the structure of the pre-trained model remains unchanged, so that the pruned network is friendly to off-the-shelf deep learning frameworks and can be further boosted by other techniques, such as quantification  [ 1 ]  and knowledge distillation  [ 7 ] .

Recently, the core of channel pruning has been suggested to learn a more compact  network width  instead of the retained weights  [ 8 ] . The number of channels or filters is taken as a direct measure of the network width  [ 9 ,  10 ,  11 ] . Recently, neural architecture search (NAS)  [ 12 ,  13 ]  and other AutoML techniques (such as MetaPruning  [ 9 ] , AutoSlim  [ 10 ] , and TAS  [ 11 ] ) have been explored to directly search for an optimal network width. In general, a one-shot supernet is usually adopted for the evaluation of different widths. If a certain layer is of the width

c

𝑐

c

, we need to assign

c

𝑐

c

channels (filters) for that layer and all the other layers follow the similar setup. All these assigned channels in the supernet thus form a sub-network from the supernet.

Existing methods  [ 9 ,  10 ,  11 ]  often follow a  unilaterally augmented  (UA) principle to produce a sub-network layer of different widths from the supernet, i.e., channels in a layer are counted from the left to the right. To obtain a sub-network layer of a width

c

𝑐

c

, the UA principle simply chooses the leftmost

c

𝑐

c

channels from the supernet. In this way, leftmost channels will be more frequently used to form the sampled sub-network of different widths, compared with the channels in the right side. For example, in Figure

1

(a), the leftmost channel will always be used to form the sub-network layer whose width ranges from 1 to 6, while the rightmost channel is only used in the sub-network layer that is of a width 6. This causes a  training unfairness  among the channels and their corresponding kernels, i.e,
left channels will be trained more than right ones. This training unfairness can affect the accuracy of sub-network evaluation and lead to an unconvincing optimization of the network widths.

Figure 1:  Comparison of unilaterally augmented (UA) principle and our proposed bilaterally coupled (BC) principle in supernet. In BC principle, each network width is indicated by two (left and right) paths, so that all channels get the same cardinality for evaluation different widths. However, in UA principle each width goes through one path, and training unfairness over channels and evaluation bias exist. Under uniform sampling strategy, for each channel the expectation of the times being evaluated is theoretically equal to the times being trained, since we simply sample each path and train it. For simplicity, we use  cardinality  to refer to the number of times that a channel is used for evaluation over all widths.

In this paper, we introduce a new supernet called Bilaterally Coupled Network (BCNet) to address the training and evaluation unfairness within UA principle. In BCNet, each channel is fairly trained wherever it is from the left side or the right side. Specifically, both in training and evaluation, the optimality of a sub-network width is determined symmetrically by the average performance of bilateral ( i.e. , both left and right) channels. As shown in Figure

1

(b), considering a supernet layer with six channels, both the leftmost channel and the rightmost channel can always be used to form a sub-network layer whose width ranges from 1 to 6. In this way, all the channels will be trained equally and bilaterally coupled in BCNet, which leads to a more fair evaluation than that induced by the UA principle. To encourage a rigorous training fairness over channels, we adopt a complementary training strategy as shown in Figure

2  .

A preliminary version of this work was presented earlier  [ 14 ] , namely BCNet. This journal version adds to the initial conference paper in significant ways. First, we follow the wisdom of existing successful neural architectures from network engineering, and pre-set the smallest and maximum width allowed in a sub-network layer. The search space can be shrunk around the premise that the channel training fairness is not destroyed. Second, a new iterative updating method is developed to save half of the memory cost of BCNetV2. Third, to promote a better comparison between our algorithm with other search methods, we open-source the first width benchmark on macro st