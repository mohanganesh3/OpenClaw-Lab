[2001.06901] Modeling of Deep Neural Network (DNN) Placement and Inference in Edge Computing

Modeling of Deep Neural Network (DNN) Placement and Inference in Edge Computing

Mounir Bensalem, Jasenka Dizdarević and Admela Jukan

Technische Universität Braunschweig, Germany

{mounir.bensalem, j.dizdarevic, a.jukan}@tu-bs.de

Abstract

With the edge computing becoming an increasingly adopted concept in system architectures, it is expected its utilization will be additionally heightened when combined with deep learning (DL) techniques. The idea behind integrating demanding processing algorithms in Internet of Things (IoT) and edge devices, such as Deep Neural Network (DNN), has in large measure benefited from the development of edge computing hardware, as well as from adapting the algorithms for use in resource constrained IoT devices. Surprisingly, there are no models yet to optimally place and use machine learning in edge computing. In this paper, we propose the first model of optimal placement of Deep Neural Network (DNN) Placement and Inference in edge computing. We present a mathematical formulation to the DNN Model Variant Selection and Placement (MVSP) problem considering the inference latency of different model-variants, communication latency between nodes, and utilization cost of edge computing nodes. We evaluate our model numerically, and show that for low load increasing model co-location decreases the average latency by 33% of millisecond-scale per request, and for high load, by 21%.

I

Introduction

The potential benefits of edge computing paradigm and related distributed system solutions, have been particularly linked with the breakthroughs achieved in the fast growing development of deep learning (DL) techniques designed to boost automation in all application domains. With that in mind, this vibrant research area has been more and more focusing on integrating edge computing with deep learning  [ 1 ]  and the associated challenges due to resource constraints  [ 2 ,  3 ] .
Recent hardware developments are making more and more possible to run highly computationally demanding algorithms in the edge  [ 4 ] .

Among myriad of open research issues, the models for machine learning (ML) inference latency and ML model selection optimization in edge computing, along with related task placement are of particular importance. This is because the related such models  [ 5 ]  developed for cloud computing cannot be directly applied in edge computing. The DNN placement problem in the edge needs to consider in particular the communication delay between nodes and the hardware heterogeneity of devices. To the best of our knowledge there has been no study of the DNN application selection, placement and inference serving problem in consideration of edge computing. This paper presents the first DNN Model Variant Selection and Placement (MVSP) in edge computing networks. We provide a mathematical formulation of the problems of ML placement and inference service, considering inference latency of different model-variants, communication latency between nodes and utilization cost of edge computing nodes (resources). Our model also includes a discussion on the potential effects of hardware sharing, with GPU edge computing nodes shared between different model-variants, on inference latency.

An illustration of the DNN application placement problem is presented in Figure

1

with the arrival of inference requests from IoT nodes to the edge computing layer. IoT nodes are assumed to be devices with processing and sensing capabilities, but not enough to run DNN models. In this system abstraction, edge computing layer, consisting of edge nodes with GPUs for running ML models, serves as an inference service system to the requests from IoT nodes. For the illustrated system we focus on designing a placement strategy of ML models, taking into account different possibilities of model-variants and how to forward requests coming from IoT nodes.

Figure 1:  A reference inference service system

Figure 2:  An illustrative example of DNN application placement problem and assignment of inference request in edge nodes

The remainder of this paper is organized as follows. Section

II

introduces a mathematical model for the MVSP problem. Section

III

numerically evaluates the proposed model. We conclude the paper in

IV  .

II

System Model

II-A

Reference Edge Computing Network Model

In order to analyze MVSP problem in edge computing network we define a system model which will consider inference latency of different model-variant with shared and unshared access to GPUs, node communication latency and utilization cost.
The considered system consists of

N  I

subscript  𝑁  𝐼

N_{I}

IoT nodes, e.g. smart phone, security camera and smart car cameras, and

N  E

subscript  𝑁  𝐸

N_{E}

edge nodes, e.g. access points. Let

𝒩  ℐ

=

{  1  ,  …  ,

N  I

}

subscript  𝒩  ℐ

1  …

subscript  𝑁  𝐼

\mathcal{N_{I}}=\{1,...,N_{I}\}

and

𝒩  ℰ

=

{  1  ,  …  ,

N  E

}

subscript  𝒩  ℰ

1  …

subscript  𝑁  𝐸

\mathcal{N_{E}}=\{1,...,N_{E}\}

denote the set of indexes of IoT nodes and edge nodes, respectively. Edge nodes are able to host various ML applications designated to serve the inference requests coming from IoT nodes. Every edge node

e  ∈

𝒩  ℰ

𝑒

subscript  𝒩  ℰ

e\in\mathcal{N_{E}}

has a computing unit specific for inference serving tasks e.g. CPU, GPU and TPU as well as memory capacity

C  e

superscript  𝐶  𝑒

C^{e}

. We assume that we have

M

𝑀

M

different ML models that can be used for different tasks such as face recognition and object detection. Each ML model

m

𝑚

m

can have

V  m

subscript  𝑉  𝑚

V_{m}

variants with different sizes and inference latencies per request and can be deployed via a VM or a container. We denote by

ℳ  =

{  1  ,  …  ,  M  }

ℳ

1  …  𝑀

\mathcal{M}=\{1,...,M\}

the set of ML models and

𝒱  m

=

{  1  ,  …  ,

V  m

}

subscript  𝒱  𝑚

1  …

subscript  𝑉  𝑚

\mathcal{V}_{m}=\{1,...,V_{m}\}

the set of variants of model

m

𝑚

m

. Each model variant (

m  ,  v

𝑚  𝑣

m,v

) has a minimum memory requirement

R

m  ​  v

subscript  𝑅

𝑚  𝑣

R_{mv}

to be loaded and can process at most

L  ​  o  ​  a  ​

d

m  ​  v

𝐿  𝑜  𝑎

subscript  𝑑

𝑚  𝑣

Load_{mv}

with a stable performance. Each IoT node

i

𝑖

i

can define its own latency requirement

L  ​  m  ​  a  ​

x  m  i

𝐿  𝑚  𝑎

superscript

subscript  𝑥  𝑚

𝑖

Lmax_{m}^{i}

for each infered model

m

𝑚

m

as well as the number of inference requests

r  m  i

superscript

subscript  𝑟  𝑚

𝑖

r_{m}^{i}

. The notations used in this paper are summarized in Table

I  .
We introduce a binary variable

x

m  ​  v

i  ​  e

superscript

subscript  𝑥

𝑚  𝑣

𝑖  𝑒

x_{mv}^{ie}

to indicate the forwarding decision of requests of model-variant (

m  ,  v

𝑚  𝑣

m,v

) from IoT node

i  ∈

𝒩  ℐ

𝑖

subscript  𝒩  ℐ

i\in\mathcal{N_{I}}

to edge node

e  ∈

𝒩  ℰ

𝑒

subscript  𝒩  ℰ

e\in\mathcal{N_{E}}

. The placement decision of model-variant (

m  ,  v

𝑚  𝑣

m,v

) in an edge node

e

𝑒

e

is defined by an integer variable

n

m  ​  v

e

superscript

subscript  𝑛

𝑚  𝑣

𝑒

n_{mv}^{e}

, which indicates the number of deployed instances.

Figure

2

shows an illustrative example of a network of

10

10

10

IoT nodes (

I  ​  1

,  …  ,

I  ​  10

𝐼  1

…

𝐼  10

I1,...,I10

) and

5

5

5

edge nodes (

E  ​  1

,  …  ,

E  ​  5

𝐸  1

…

𝐸  5

E1,...,E5

) for the above described system. In this example, each edge node stores the 3 ML models and can instantiate them during loading various model-variants by changing the batch size parameter, which affects the instance size and throughput. The figure shows how from this set of 3 models, the optimally selected model-variants would be placed in edge nodes after the placement decisions have been made, along with served inference requests. For example, after placement edge no