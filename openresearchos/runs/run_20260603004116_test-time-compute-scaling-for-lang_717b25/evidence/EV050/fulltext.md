[2305.17346] Input-Aware Dynamic Timestep Spiking Neural Networks for Efficient In-Memory Computing This work was supported in part by CoCoSys, a JUMP2.0 center sponsored by DARPA and SRC, Google Research Scholar Award, the NSF CAREER Award, TII (Abu Dhabi), the DARPA AI Exploration (AIE) program, and the DoE MMICC center SEA-CROGS (Award #DE-SC0023198).

Input-Aware Dynamic Timestep Spiking Neural Networks for Efficient In-Memory Computing

†

†  thanks:  This work was supported in part by CoCoSys, a JUMP2.0 center sponsored by DARPA and SRC, Google Research Scholar Award, the NSF CAREER Award, TII (Abu Dhabi), the DARPA AI Exploration (AIE) program, and the DoE MMICC center SEA-CROGS (Award #DE-SC0023198).

Yuhang Li, Abhishek Moitra, Tamar Geller, Priyadarshini Panda

Department of Electrical Engineering, Yale University

New Haven, CT 06511, USA

Abstract

Spiking Neural Networks (SNNs) have recently attracted widespread research interest as an efficient alternative to traditional Artificial Neural Networks (ANNs) because of their capability to process sparse and binary spike information and avoid expensive multiplication operations.
Although the efficiency of SNNs can be realized on the In-Memory Computing (IMC) architecture, we show that the energy cost and latency of SNNs scale linearly with the number of timesteps used on IMC hardware.
Therefore, in order to maximize the efficiency of SNNs, we propose input-aware Dynamic Timestep SNN (DT-SNN), a novel algorithmic solution to dynamically determine the number of timesteps during inference on an input-dependent basis.
By calculating the entropy of the accumulated output after each timestep, we can compare it to a predefined threshold and decide if the information processed at the current timestep is sufficient for a confident prediction. We deploy DT-SNN on an IMC architecture and show that it incurs negligible computational overhead. We demonstrate that our method only uses 1.46 average timesteps to achieve the accuracy of a 4-timestep static SNN while reducing the energy-delay-product by 80%.

Index Terms:

Spiking neural networks, in-memory computing, dynamic inference

I

Introduction

Deep learning has revolutionized many challenging computational tasks such as computer vision and natural language processing  [ 10 ]  using Artificial Neural Networks (ANNs). These successes, however, have come at the cost of tremendous computing resources and high latency  [ 6 ] . Over the past decade, Spiking Neural Networks (SNNs) have gained popularity as an energy-efficient alternative to ANNs  [ 14 ,  17 ] .
SNNs are different from ANNs in that they process inputs over a series of timesteps, whereas ANNs infer over what can be considered a single timestep.
The biologically-plausible neurons in SNNs maintain a variable called membrane potential, which controls the behavior of the SNN over a series of timesteps.
When the membrane potential exceeds a certain threshold, the neuron fires, creating a spike, and otherwise, the neuron remains inactive (neuron outputs a 0 or 1).
Such spike-based computing creates sparsity in the computations and replaces multiplications with additions.

Figure 1:  Energy estimation on our IMC
architecture using VGG-16 on CIFAR-10 dataset. (A) energy ratio of each unit, (B) energy/latency vs. timesteps.

Although the binary spike nature of SNNs eliminates the need for multiplications, compared to ANNs, SNNs require significantly more memory access due to multi-timestep computations on traditional von-Neumann architectures (called the “memory wall problem”)  [ 21 ] .
To alleviate this problem, In-Memory Computing (IMC) hardware is used to perform analog dot-product operations to achieve high memory bandwidth and compute parallelism  [ 15 ] .
In this work, we mainly focus on achieving lower energy and latency in the case of IMC-implemented SNNs while maintaining iso-accuracy.

Fig.

1  (A) shows a component-wise energy distribution for the CIFAR10-trained VGG-16 network on 64

×

\times

64 4-bit RRAM IMC architecture. Among these, the digital peripherals (containing crossbar input switching circuits, buffers, and accumulators) entail the highest energy cost (45%). The IMC-crossbar and analog-to-digital converter (ADC) consumes the second highest energy (25%). Efforts to lower energy consumption have been made by previous works. As an example, prior IMC-aware algorithm-hardware co-design techniques  [ 13 ,  22 ] , have used pruning, quantization to reduce the ADC and crossbar energy, and area cost. However, in the case of SNNs, the improvement is rather limited because the crossbar and ADC only occupy 25% of the overall energy cost.

Unlike ANNs, the number of timesteps in SNNs plays an important role in hardware performance, which is orthogonal to data precision or sparsity. In Fig.

1  (B) we investigate how timesteps affect the energy consumption and latency of an SNN.
Note that both metrics are normalized to the performance of a 1-timestep SNN.
We find that both energy consumption and latency scale linearly with the number of timesteps, up to

4.9  ×

4.9\times

more energy and

8  ×

8\times

more latency when changing the number of timesteps from 1 to 8.
More importantly, if one can reduce the number of timesteps in SNNs, then all parts in Fig.

1  (A) can benefit from the energy and latency savings.
These findings highlight the tremendous potential to optimize SNNs’ performance on IMC hardware.

In fact,  [ 8 ,  12 ,  3 ]  have explored ways to reduce the number of timesteps from an algorithmic perspective. They all train an SNN with a high number of timesteps first and then finetune the model with fewer timesteps later.
However, their method decreases the number of timesteps for all input samples, thereby inevitably leading to an accuracy-timestep trade-off.
In this paper, we tackle this problem with another solution.
 We view the number of timesteps during inference as a variable conditional to each input sample. 
We call our method Dynamic Timestep Spiking Neural Network (DT-SNN) as it varies the number of timesteps based on each input sample.
In particular, we use entropy thresholding to determine the appropriate number of timesteps for each sample.
To further optimize our algorithm in practice, we design a new training loss function and implement our algorithm on an IMC architecture.

The main contributions of our work are summarized below:

1.

Based on what we have seen thus far, this is the first work that changes the number of timesteps in SNNs based on the input, reducing computational overhead and increasing inference efficiency without compromising task performance.

2.

To achieve that goal, we propose using entropy thresholding to distinguish the number of timesteps required. Meanwhile, we also provide a new training loss function and an IMC implementation of our method.

3.

Extensive experiments are carried out to demonstrate the efficacy and efficiency of DT-SNN. For example, the DT-SNN ResNet-19 achieves the same accuracy as the 4-timestep SNN ResNet-19 with only an average of 1.27-timestep on the CIFAR-10 dataset, reducing 84% energy-delay-product.

II

Preliminaries

We start by introducing the basic background of SNNs.
We denote the overall spiking neural network as a function

f  T

​

(  𝒙  )

subscript  𝑓  𝑇

𝒙

f_{T}({\bm{x}})

(

𝒙

𝒙

{\bm{x}}

is the input image), its forward propagation can be formulated as

𝒚  =

f  T

​

(  𝒙  )

=

1  T

​

∑

t  =  1

T

h  ∘

g  L

∘

g

L  −  1

∘

g

L  −  2

∘  ⋯

​

g  1

​

(  𝒙  )

,

𝒚

subscript  𝑓  𝑇

𝒙

1  𝑇

superscript

subscript

𝑡  1

𝑇

ℎ

superscript  𝑔  𝐿

superscript  𝑔

𝐿  1

superscript  𝑔

𝐿  2

⋯

superscript  𝑔  1

𝒙

{\bm{y}}=f_{T}({\bm{x}})=\frac{1}{T}\sum_{t=1}^{T}h\circ g^{L}\circ g^{L-1}\circ g^{L-2}\circ\cdots g^{1}({\bm{x}}),

(1)

where

g  ℓ

​

(  𝒙  )

=

LIF  ​

(

𝑾  ℓ

​  𝒙

)

superscript  𝑔  ℓ

𝒙

LIF

superscript  𝑾  ℓ

𝒙

g^{\ell}({\