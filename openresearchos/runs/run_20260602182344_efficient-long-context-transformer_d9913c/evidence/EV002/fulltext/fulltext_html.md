[2504.05729] Robust and Efficient Average Consensus with Non-Coherent Over-the-Air Aggregation

Robust and Efficient Average Consensus with Non-Coherent Over-the-Air Aggregation

Yuhang Deng, Zheng Chen, and Erik G. Larsson

This work was supported in part by Knut and Alice Wallenberg Foundation, ELLIIT, and Swedish Research Council.

Department of Electrical Engineering, Linköping University, Sweden

E-mail: {yuhang.deng, zheng.chen, erik.g.larsson}@liu.se

Abstract

Non-coherent over-the-air (OTA) computation has garnered increasing attention for its advantages in facilitating information aggregation among distributed agents in resource-constrained networks without requiring precise channel estimation.
A promising application scenario of this method is distributed average consensus in wireless multi-agent systems.
However, in such scenario, non-coherent interference from concurrent OTA transmissions can introduce bias in the consensus value.
To address this issue, we develop a robust distributed average consensus algorithm by formulating the consensus problem as a distributed optimization problem.
Using decentralized projected gradient descent (D-PGD), our proposed algorithm can achieve unbiased mean square average consensus even in the presence of non-coherent interference and noise.
Additionally, we implement transmit power control and receive scaling mechanisms to further accelerate convergence. Simulation results demonstrate that our method can significantly enhance the convergence speed of the D-PGD algorithm for OTA average consensus without compromising accuracy.

Index Terms:

Distributed average consensus, over-the-air aggregation, non-coherent transmission, power control.

I

Introduction

In multi-agent systems, distributed consensus algorithms allow networked agents to reach agreement on specific values of interest, which is essential for applications requiring decentralized coordination  [ 1 ] . A special case is the average consensus, where the goal is to agree on the average of agents’ initial state values. The interaction protocol and network topology can greatly affect the process of reaching consensus  [ 2 ] . The most classical interaction protocol is distributed linear iteration, where agents linearly combine their own state values with the values of their neighbors to obtain updated state values in the next iteration. The convergence conditions for achieving average consensus through distributed linear iteration are presented in  [ 3 ,  4 ] .

Earlier studies on average consensus consider either perfect communication or imperfect communication with link-level noise, packet losses or delay  [ 5 ,  6 ] .
In large-scale wireless multi-agent systems containing numerous agents, access control and interference management are critical for achieving efficient and accurate average consensus. Assigning orthogonal resource blocks to different nodes/links is highly inefficient for extremely large networks.
Over-the-air (OTA) computation offers a resource-efficient solution for data aggregation  [ 7 ,  8 ] . Typically, the superposition property of wireless channels is considered a source of multi-user interference, which can be mitigated by orthogonal channel access or multiple antenna techniques. OTA computation leverages the signal superposition to aggregate data from distributed nodes, allowing

N  N

senders to transmit simultaneously using the same frequency resources, thus increasing spectral efficiency by a factor of

N  N

.

Typical OTA approaches rely on coherent transmission, which requires perfect channel state information (CSI) to achieve phase alignment across different transmitters. However, in fully decentralized systems with multiple receivers, it is nearly impossible for them to achieve constructive phase-coherent combination simultaneously, even with perfect transmitter CSI. Furthermore, estimating CSI of all links in multi-agent systems also leads to substantial signaling overhead  [ 9 ] . These challenges have spurred research into non-coherent OTA aggregation  [ 10 ,  11 ] .
In  [ 10 ] , non-coherent OTA aggregation has been proposed for wireless multi-agent optimization systems employing a decentralized gradient descent (DGD) algorithm. The key idea is to use energy-based coding and decoding such that the phase information of signals is not needed for data processing.
Similarly, in  [ 11 ] , a non-coherent OTA-based average consensus algorithm is studied.
In a half-duplex setting, all active receivers can receive the aggregated transmission signals from all active transmitters simultaneously.
Following a well-designed encoding/decoding rule and consensus protocol, agents update their state values using the decoded information and the system achieves asymptotic consensus. However, the consensus value is not the exact average of the initial state values but includes a stochastic bias, which is caused by the non-coherent interference.

In this work, we propose an average consensus algorithm based on decentralized projected gradient descent (D-PGD) with non-coherent OTA aggregation. Our proposed algorithm can reach mean square consensus on the average of the agents’ initial state values without any bias. Moreover, we design a transmit power control and receive scaling method to accelerate the convergence of our algorithm. Our proposed algorithm is shown to achieve fast and robust convergence to average consensus, as validated by simulation results.

II

Preliminaries

In this section, we describe the fundamental concept of distributed average consensus, and possible mechanisms for solving the consensus problem.

II-A

Distributed Average Consensus

Consider a network represented by a time-invariant undirected graph

𝒢  =

(  𝒩  ,  ℰ  )

\mathcal{G}=\left(\mathcal{N},\,\mathcal{E}\right)

, where

𝒩  =

{  1  ,  …  ,  N  }

\mathcal{N}=\{1,\ldots,N\}

denotes the set of agents and

ℰ  ⊆

𝒩  ×  𝒩

\mathcal{E}\subseteq\mathcal{N}\times\mathcal{N}

denotes the set of links. A pair of agents

i  i

and

j  j

is said to be connected if

(  i  ,  j  )

∈  ℰ

(i,j)\in\mathcal{E}

.

Each agent has a real-valued state value

x  n

​

[  t  ]

x_{n}[t]

with

|

x  n

​

[  t  ]

|

≤

x  max

|x_{n}[t]|\leq x_{\max}

, where

t  ∈

ℕ  0

t\!\in\!\mathbb{N}_{0}

denotes the time index. The goal of distributed consensus is for all agents to reach agreement on a certain value,

μ  \mu

, in a distributed manner. This can be achieved by following an iteration process where agents exchange information with neighbors and update their state values in each iteration.
Let

𝐱  ​

[  t  ]

=

[

x  1

​

[  t  ]

,

x  2

​

[  t  ]

,  …  ,

x  N

​

[  t  ]

]

T

\mathbf{x}[t]=[x_{1}[t],\,x_{2}[t],\,\ldots,\,x_{N}[t]]^{\text{T}}

denotes the vector of agent state values.
The agents are considered to achieve asymptotic consensus if and only if

lim

t  →  ∞

𝐱  ​

[  t  ]

=

μ  ⋅  𝟏

,

\lim_{t\rightarrow\infty}\mathbf{x}[t]=\mu\cdot\mathbf{1},\vskip-5.69054pt

(1)

where

𝟏  \mathbf{1}

denotes the vector of all ones. A special case is that

μ  =

∑

n  ∈  𝒩

x  n

​

[  0  ]

/  N

\mu=\sum_{n\in\mathcal{N}}x_{n}[0]/N

, which is called the average consensus. The most commonly adopted mechanism for achieving average consensus is the distributed linear iteration written as

x  n

​

[

t  +  1

]

=

∑

m  ∈  𝒩

w

n  ,  m

⋅

x  m

​

[  t  ]

,

∀  n

∈  𝒩

,

x_{n}[t+1]=\sum\limits_{m\in\mathcal{N}}w_{n,m}\cdot x_{m}[t],\ \ \forall n\in\mathcal{N},\vskip-2.84526pt

(2)

where

w

n  ,  m

w_{n,m}

denotes the weight that agent

n  n

assigns to the information received from agent

m  m

. Due to the connectivity constraint in the network topology, we have

w

n  ,  m

=  0

w_{n,m}=0

if

n  ≠  m

n\neq m

and

(  n  ,  m  )

∉  ℰ

(n,m)\notin\mathcal{E}

. Let

𝐖  ∈

ℝ

N  ×  N

\mathbf{W}\in\mathbb{R}^{N\times N}

be a matrix where the

n  n

-th row and

m  m

-column is

{

w

n  ,  m

}

\{w_{n,m}\}

. Then, (