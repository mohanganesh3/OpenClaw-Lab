[2501.07825] An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer This work was supported in part by the Shenzhen Science and Technology Program 2023A007. Corresponding author: Wendong Mao.

An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer

†

†  thanks:  This work was supported in part by the Shenzhen Science and Technology
Program 2023A007.
Corresponding author: Wendong Mao.

Zhengke Li  a  , Wendong Mao  a  , Siyu Zhang  b  , Qiwei Dong  b

and Zhongfeng Wang  a

,

b

a

School of Integrated Circuits, Sun Yat-Sen University, China

b School of Electronic Science and Engineering, Nanjing University, China

Email: lizhk37@mail2.sysu.edu.cn, maowd@mail.sysu.edu.cn, syzhang@smail.nju.edu.cn,

qiweidong@smail.nju.edu.cn, wangzf83@mail.sysu.edu.cn

Abstract

Recently, large models, such as Vision Transformer and BERT, have garnered significant attention due to their exceptional performance. However, their extensive computational requirements lead to considerable power and hardware resource consumption. Brain-inspired computing, characterized by its spike-driven methods, has emerged as a promising approach for low-power hardware implementation. In this paper, we propose an efficient sparse hardware accelerator for Spike-driven Transformer. We first design a novel encoding method that encodes the position information of valid activations and skips non-spike values. This method enables us to use encoded spikes for executing the calculations of linear, maxpooling and spike-driven self-attention. Compared with the single spike input design of conventional SNN accelerators that primarily focus on convolution-based spiking computations, the specialized module for spike-driven self-attention is unique in its ability to handle dual spike inputs. By exclusively utilizing activated spikes, our design fully exploits the sparsity of Spike-driven Transformer, which diminishes redundant operations, lowers power consumption, and minimizes computational latency. Experimental results indicate that compared to existing SNNs accelerators, our design achieves up to 13.24

×  \times

and 1.33

×  \times

improvements in terms of throughput and energy efficiency, respectively.

Index Terms:

Spiking Neuron Networks (SNNs), Hardware Accelerator, Spike-Driven Transformer.

I

Introduction

In recent decades, Artificial Neural Networks (ANNs), such as Convolutional Neural Networks (CNNs)  [ 1 ]  and Transformer  [ 2 ] , have achieved significant success in various fields. However, the continuous expansion of network structures has made ANNs with extensive parameters and computations challenging to implement on edge devices with limited hardware resources and power consumption.

Spiking Neural Networks (SNNs)  [ 3 ]  have emerged as a promising approach for energy-efficient implementation. By mimicking the neurodynamics of the human brain, SNNs use neurons that communicate between layers via spike-form signals and typically involve temporal accumulation operations. The binary nature of SNNs simplifies their operations, offering more potential directions for designing hardware accelerators tailored to SNNs.

Developing efficient hardware accelerators tailored to specific networks has been a prevalent approach to accelerating the inference of ANNs on edge devices  [ 4 ,  5 ,  6 ] . Although these designs optimize computations of convolution and attention through parallelism, sparsity, and data reuse, they are unable to efficiently exploit the characteristics of SNNs such as spiking data and timesteps in spatial and temporal dimensions. Thus, it is crucial to design dedicated accelerators for SNNs.

Currently, most existing SNN accelerators target driving types (time-driven, event-driven, or hybrid-driven), sparsity (reconfiguring computation kernels for different sparsity), data reshaping for spiking neurons, or data reuse with reconfigurable designs  [ 7 ,  8 ,  9 ,  10 ] . However, these accelerators only focus on the inherent characteristics of spiking neurons, without incorporating optimizations for specific layers like maxpooling and spike-driven self-attention (SDSA). Additionally, the dual spike inputs of self-attention are rarely considered in previous designs. To address this, we develop an efficient hardware accelerator for Spike-driven Transformer  [ 11 ]  to fully exploit its high sparsity and optimize spiking computations, thereby reducing hardware costs with lower power consumption and processing latency.
The main contributions of this paper are as follows:

•

We propose a novel spike redundancy elimination computation method to fully utilize the sparsity in the Spike-driven Transformer. This method encodes the position information of spikes and converts spiking computations into address comparison, which can effectively bypass zeros in the single and dual spike inputs.

•

Based on the proposed efficient computation method, we design low-complexity spike encoding and computing units to optimize multiple types of spiking computations, including maxpooling, SDSA, and linear. Benefiting from the binary nature of SNNs and our encoding method, all spiking calculations are implemented through simple addition and comparison instead of multiplication.

•

The proposed hardware accelerator is implemented on Xilinx Virtex UltarScale FPGA and evaluated with Spike-driven Transformer on the Cifar-10 dataset. Experimental results show that compared to other SNN accelerators, our design achieves up to 13.24

×  \times

and 1.33

×  \times

improvements in peak performance and energy efficiency, respectively.

II

Preliminary of Spike Neuron Model

Inspired by the way that neurons in the human brain process and transmit signals, SNNs have been proposed and considered as the third generation of neural networks  [ 3 ] . In SNNs, the Leaky Integrate-and-Fire (LIF) spiking neuron  [ 12 ]  is generally used to encapsulate the main characteristics of biological neurons, whose dynamic behaviors can be summarized by the following equations:

T  ​  e  ​  m  ​  p  ​

[  t  ]

\displaystyle Temp[t]

=

S  ​

[  t  ]

​

V

r  ​  e  ​  s  ​  e  ​  t

+

(

1  −

S  ​

[  t  ]

)

​

(

γ  ​  M  ​  e  ​  m  ​

[  t  ]

)

,

\displaystyle=S[t]V_{reset}+(1-S[t])(\gamma Mem[t]),

(1)

M  ​  e  ​  m  ​

[  t  ]

\displaystyle Mem[t]

=

S  ​  p  ​  a  ​

[  t  ]

+

T  ​  e  ​  m  ​  p  ​

[

t  −  1

]

,

\displaystyle=Spa[t]+Temp[t-1],

(2)

S  ​

[  t  ]

\displaystyle S[t]

=

ε  ​

(

M  ​  e  ​  m  ​

[  t  ]

−

V

t  ​  h

)

,

\displaystyle=\varepsilon(Mem[t]-V_{th}),

(3)

where

M  ​  e  ​  m  ​

[  t  ]

Mem[t]

,

S  ​  p  ​  a  ​

[  t  ]

Spa[t]

, and

T  ​  e  ​  m  ​  p  ​

[  t  ]

Temp[t]

represent the membrane potential, spatial input, and temporal input of the neuron at timestep

t  t

, respectively.

S  ​

[  t  ]

S[t]

is the output of the spiking neuron at timestep

t  t

.

ε  ​

(  x  )

\varepsilon(x)

denotes the step function, whose output is ‘1’ when

x  x

is greater than or equal to 0; otherwise, the output is ‘0’.

V

t  ​  h

V_{th}

represents the threshold voltage required for the spiking neuron to fire a spike.

γ  \gamma

is the time constant representing the decay of the membrane potential.

V

r  ​  e  ​  s  ​  e  ​  t

V_{reset}

represents the reset value of the membrane potential.

An intuitive understanding of spiking neuron’s behavior is as follows: spatial input comes from various sources such as convolution, linear or maxpooling layers, while temporal input is determined by whether the spiking neuron fires a spike in the previous timestep. If the neuron fires,

M  ​  e  ​  m  ​

[  t  ]

Mem[t]

resets to

V

r  ​  e  ​  s  ​  e  ​  t

V_{reset}

; otherwise, the membrane potential decays according to

γ  \gamma

, which is typically between 0 and 1. From Equations (  1  )-(  3  ), it is evident that the computational mechanism of spiking neurons involves timestep, meaning whether a spike is fired depen