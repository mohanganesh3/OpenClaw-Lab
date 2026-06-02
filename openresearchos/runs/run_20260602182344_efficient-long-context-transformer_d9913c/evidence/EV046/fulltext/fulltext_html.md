[2307.01694] Spike-driven Transformer

Spike-driven Transformer

Man Yao  1,2,3  , Jiakui Hu  1,4∗  , Zhaokun Zhou  3,4∗  , Li Yuan  3,4  ,

Yonghong Tian  3,4

,  Bo Xu  1

,  Guoqi Li  1

1  Institute of Automation, Chinese Academy of Sciences, Beijing, China

2  School of Automation Science and Engineering, Xi’an Jiaotong University, Xi’an, Shaanxi, China

3  Peng Cheng Laboratory, Shenzhen, Guangzhou, China

4  Peking University, Beijing, China

Equal contribution. manyao@stu.xjtu.edu.cn; jiakuihu29@gmail.com; zhouzhaokun@stu.pku.edu.cnCorresponding author, guoqi.li@ia.ac.cn

Abstract

Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (i.e., spike-driven) paradigm. In this paper, we incorporate the spike-driven paradigm into Transformer by the proposed Spike-driven Transformer with  four  unique properties: i) Event-driven, no calculation is triggered when the input of Transformer is zero; ii) Binary spike communication, all matrix multiplications associated with the spike matrix can be transformed into sparse additions; iii) Self-attention with linear complexity at both token and channel dimensions; iv) The operations between spike-form Query, Key, and Value are mask and addition. Together, there are only sparse addition operations in the Spike-driven Transformer. To this end, we design a novel Spike-Driven Self-Attention (SDSA), which exploits only mask and addition operations without any multiplication, and thus having up to

87.2  ×

87.2\times

lower computation energy than vanilla self-attention. Especially in SDSA, the matrix multiplication between Query, Key, and Value is designed as the mask operation. In addition, we rearrange all residual connections in the vanilla Transformer before the activation functions to ensure that all neurons transmit binary spike signals. It is shown that the Spike-driven Transformer can achieve 77.1% top-1 accuracy on ImageNet-1K, which is the state-of-the-art result in the SNN field. The source code is available at  Spike-driven Transfromer .

1  Introduction

One of the most crucial computational characteristics of bio-inspired Spiking Neural Networks (SNNs)  [ 1 ]  is spike-based event-driven (spike-driven): i) When a computation is event-driven, it is triggered sparsely as events (spike with address information) occur; ii) If only binary spikes (0 or 1) are employed for communication between spiking neurons, the network’s operations are synaptic ACcumulate (AC). When implementing SNNs on neuromorphic chips, such as TrueNorth  [ 2 ] , Loihi  [ 3 ] , and Tianjic  [ 4 ] , only a small fraction of spiking neurons at any moment being active and the rest being idle. Thus, spike-driven neuromorphic computing that only performs sparse addition operations is regarded as a promising low-power alternative to traditional AI  [ 5 ,  6 ,  7 ] .

Although SNNs have apparent advantages in bio-plausibility and energy efficiency, their applications are limited by poor task accuracy. Transformers have shown high performance in various tasks for their self-attention  [ 8 ,  9 ,  10 ] . Incorporating the effectiveness of Transformer with the high energy efficiency of SNNs is a natural and exciting idea. There has been some research in this direction, but all so far have relied on “hybrid computing”. Namely, Multiply-and-ACcumulate (MAC) operations dominated by vanilla Transformer components and AC operations caused by spiking neurons both exist in the existing spiking Transformers. One popular approach is to replace some of the neurons in Transformer with spiking neurons to do a various of tasks  [ 11 ,  12 ,  13 ,  14 ,  15 ,  16 ,  17 ,  18 ,  19 ,  20 ,  21 ,  22 ] , and keeping the MAC-required operations like dot-product, softmax, scale, etc.

Though hybrid computing helps reduce the accuracy loss brought on by adding spiking neurons to the Transformer, it can be challenging to benefit from SNN’s low energy cost, especially given that current spiking Transformers are hardly usable on neuromorphic chips. To address this issue, we propose a novel Spike-driven Transformer that achieves the spike-driven nature of SNNs throughout the network while having great task performance. Two core modules of Transformer, Vanilla Self-Attention (VSA) and Multi-Layer Perceptron (MLP), are re-designed to have a spike-driven paradigm.

Figure 1:  Comparison Vanilla Self-Attention (VSA) and our Spike-Driven Self-Attention (SDSA).  (a)  is a typical Vanilla Self-Attention (VSA)  [ 8 ] .  (b)  are two equivalent versions of SDSA. The input of SDSA are binary spikes. In SDSA, there are only mask and sparse additions. Version 1: Spike

Q

𝑄

Q

and

K

𝐾

K

first perform element-wise mask, i.e., Hadamard product; then column summation and spike neuron layer are adopted to obtain the binary attention vector; finally, the binary attention vector is applied to the spike

V

𝑉

V

to mask some channels (features). Version 2: An equivalent version of Version 1 (see Section

3.3  ) reveals that SDSA is a unique type of linear attention (spiking neuron layer is the kernel function) whose time complexity is linear with both token and channel dimensions. Typically, performing self-attention in VSA and SDSA requires

2  ​

N  2

​  D

2

superscript  𝑁  2

𝐷

2N^{2}D

multiply-and-accumulate and

0.02  ​  N  ​  D

0.02  𝑁  𝐷

0.02ND

accumulate operations respectively, where

N

𝑁

N

is the number of tokens,

D

𝐷

D

is the channel dimensions, 0.02 is the non-zero ratio of the matrix after the mask of

Q

𝑄

Q

and

K

𝐾

K

. Thus, the self-attention operator between the spike

Q

𝑄

Q

,

K

𝐾

K

, and

V

𝑉

V

has almost no energy consumption.

The three input matrices for VSA are Query (

Q

𝑄

Q

), Key (

K

𝐾

K

), and Value (

V

𝑉

V

), (Fig.

1  (a)).

Q

𝑄

Q

and

K

𝐾

K

first perform similarity calculations to obtain the attention map, which includes three steps of matrix multiplication, scale and softmax. The attention map is then used to weight the

V

𝑉

V

(another matrix multiplication). The typical spiking self-attentions in the current spiking Transformers  [ 20 ,  19 ]  would convert

Q

𝑄

Q

,

K

𝐾

K

,

V

𝑉

V

into spike-form before performing two matrix multiplications similar to those in VSA. The distinction is that spike matrix multiplications can be converted into addition, and softmax is not necessary  [ 20 ] . But these methods not only yield large integers in the output (thus requiring an additional scale multiplication for normalization to avoid gradient vanishing), but also fails to exploit the full energy-efficiency potential of the spike-driven paradigm combined with self-attention.

We propose Spike-Driven Self-Attention (SDSA) to address these issues, including two aspects (see SDSA Version 1 in Fig.

1  (b)): i) Hadamard product replaces matrix multiplication; ii) matrix column-wise summation and spiking neuron layer take the role of softmax and scale. The former can be considered as not consuming energy because the Hadamard product between spikes is equivalent to the element-wise mask. The latter also consumes almost no energy since the matrix to be summed column-by-column is very sparse (typically, the ratio of non-zero elements is less than 0.02). We also observe that SDSA is a special kind of linear attention  [ 23 ,  24 ] , i.e., Version 2 of Fig.

1  (b). In this view, the spiking neuron layer that converts

Q

𝑄

Q

,

K

𝐾

K

, and

V

𝑉

V

into spike form is a kernel function.

Additionally, existing spiking Transformers  [ 20 ,  12 ]  usually follow the SEW-SNN residual design  [ 25 ] , whose shortcut is spike addition and thus outputs multi-bit (integer) spikes. This shortcut can satisfy event-driven, but introduces integer multiplication. We modified the residual connections throughout the Transformer architecture as shortcuts between membrane potentials  [ 26 ,  27 ]  to address this is