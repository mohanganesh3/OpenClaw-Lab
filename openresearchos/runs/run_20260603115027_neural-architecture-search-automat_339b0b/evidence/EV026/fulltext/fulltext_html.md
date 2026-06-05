[2107.12917] Experiments on Properties of Hidden Structures of Sparse Neural Networks

label=pruning]content/bib-pruning.bib

Experiments on Properties of Hidden Structures of Sparse Neural Networks

Julian Stier, Harshil Darji, Michael Granitzer

julian.stier@uni-passau.de, darji01@ads.uni-passau.de

2021/07/26

∼

similar-to

\sim

b785d11

Abstract

Sparsity in the structure of Neural Networks can lead to less energy consumption, less memory usage, faster computation times on convenient hardware, and automated machine learning.
If sparsity gives rise to certain kinds of structure, it can explain automatically obtained features during learning.

We provide insights into experiments in which we show how sparsity can be achieved through prior initialization, pruning, and during learning, and answer questions on the relationship between the structure of Neural Networks and their performance.
This includes the first work of inducing priors from network theory into Recurrent Neural Networks and an architectural performance prediction during a Neural Architecture Search.
Within our experiments, we show how magnitude class blinded pruning achieves 97.5% on MNIST with 80% compression and re-training, which is 0.5 points more than without compression, that magnitude class uniform pruning is significantly inferior to it and how a genetic search enhanced with performance prediction achieves 82.4% on CIFAR10.
Further, performance prediction for Recurrent Networks learning the Reber grammar shows an

R  2

superscript  𝑅  2

R^{2}

of up to 0.81 given only structural information.

1  Introduction

Understanding the structure of deep neural networks promises advances across many open problems such as energy-efficient hardware, computation times, and domain-specific performance improvements.
The structure is coupled with sparsity on different levels of the neural architecture, and if there is no sparsity, then there is also no structure:
a single hidden layered neural network is capable of universal approximation  [ 17 ] , but as soon as there exists a deeper structure, there naturally occurs sparsity.

Clearly, the structure between the input domain and the first hidden layer is tightly coupled with the structure within the data – correlations between the underlying random variables such as the spatial correlation of images or correlation in windows of time series data.
In theory and with perfectly fitting functions, that should be all there is, but in practice, neural architectures got deeper and deeper, and hidden structures seem to have an effect when neural networks are not just measured by their goodness of fit but also, e.g., on hardware efficiency or robustness  [ 3 ] .
Assuming such hidden structures exist for the better, we wonder how we can automatically find them, how they can be controlled during learning, and whether we can exploit given knowledge about them.

We give our definition for sparse neural networks and show experiments on automatic methods to obtain hidden structures:
pruning, neural architecture search, and prior initialization.
With structural performance prediction, we also show experiments on exploiting structural information to speed up neural architecture search methods.

Our  contributions  comprise a pytorch tool called  deepstruct  1

1

1

http://github.com/innvariant/deepstruct

which provides models and tools for Sparse Neural Networks, a  genetic neural architecture search  enhanced with structural performance prediction, a  comparison of magnitude-based pruning  on feed-forward and recurrent networks, an  original  correlation analysis on  recurrent networks  with different biologically plausible  structural priors  from social network theory, and  performance prediction  results on these recurrent networks.
Details on the experiments and code for reproducibility can be found at  github.com/innvariant/sparsity-experiments-2021 .

2  Sparse Neural Networks

Sparse Neural Networks (SNNs) are deep neural networks

f

𝑓

f

with a low proportion of connectivity

ξ  ​

(  f  )

𝜉  𝑓

\xi(f)

with respect to all possible connections.

Sparsity

Given a vector

x  ∈

ℝ  d

𝑥

superscript  ℝ  𝑑

x\in\mathbb{R}^{d}

with

d  ∈  ℕ

𝑑  ℕ

d\in\mathbb{N}

, its sparsity is

ξ  ​

(  x  )

=

‖  x  ‖

0

d

=

1  d

⋅

∑

i  =  0

d

|

x  i

|

0

𝜉  𝑥

subscript

norm  𝑥

0

𝑑

⋅

1  𝑑

superscript

subscript

𝑖  0

𝑑

superscript

subscript  𝑥  𝑖

0

\xi(x)=\frac{||x||_{0}}{d}=\frac{1}{d}\cdot\sum_{i=0}^{d}|x_{i}|^{0}

, given the cardinality function

|  |  ⋅  |

|  0

||\cdot||_{0}

(of which 0 refers to the case of

p  =  0

𝑝  0

p=0

of a

ℒ  p

subscript  ℒ  𝑝

\mathcal{L}_{p}

norm) and the size of the vector.
Density is defined as its complement with

1  −

ξ  ​

(  x  )

1

𝜉  𝑥

1-\xi(x)

.
The definition extends naturally to tensors and simply provides the proportion of non-zero elements in a tensor compared to the total number of its elements.
A tensor can be considered as sparse as soon as its sparsity is below a given threshold value, e.g.,

ξ  ​

(  x  )

&lt;  0.5

𝜉  𝑥

0.5

\xi(x)&lt;0.5

– as soon as more than 50% of its elements are zero.

What is the  motivation  for sparsity at all?
First, more sparsity implies a lower number of parameters which is desirable if the approximation and generalization capabilities are not heavily affected.
In theory, it also implies a lower number of computations.
From a technical perspective, sparse structures could lead to specialized hardware.
Further, sparsity means that there is space for compression that can affect the overall model memory footprint.
Memory requirements are an important aspect for limited capacity devices such as in mobile deployment.
In the feature transformation layers, sparsity explains data dependencies and provides room for explainability.

Neural Networks

A neural network is a function composed of non-linear transformation layers

σ  ​

(

W  ​  x

+  B

)

𝜎

𝑊  𝑥

𝐵

\sigma(Wx+B)

extended with transformations for skip-layer connections such that

z  l

=

∑

s  =  1

l  −  1

W

s  →  l

⋅

a  s

+

B  l

superscript  𝑧  𝑙

superscript

subscript

𝑠  1

𝑙  1

⋅

superscript  𝑊

→  𝑠  𝑙

superscript  𝑎  𝑠

superscript  𝐵  𝑙

z^{l}=\sum_{s=1}^{l-1}W^{s\rightarrow l}\cdot a^{s}+B^{l}

with

a  l

=

σ  ​

(

z  l

)

superscript  𝑎  𝑙

𝜎

superscript  𝑧  𝑙

a^{l}=\sigma(z^{l})

being the activation of layer

l

𝑙

l

with

σ

𝜎

\sigma

being e.g.  tanh  or  max(x, 0) .

W

s  →  l

superscript  𝑊

→  𝑠  𝑙

W^{s\rightarrow l}

describes the weights from layer

l  −  1

𝑙  1

l-1

to

l

𝑙

l

for a network with

l  ∈

{  1  ,  …  ,  L  }

𝑙

1  …  𝐿

l\in\{1,\dots,L\}

.
The input to the function

a  0

superscript  𝑎  0

a^{0}

is

x  ∈

ℝ

d  x

𝑥

superscript  ℝ

subscript  𝑑  𝑥

x\in\mathbb{R}^{d_{x}}

from the input domain.
Consecutive sizes of weight matrices

W

𝑊

W

need to be aligned and define the layer sizes.
The final weight matrices

W

s  →  L

superscript  𝑊

→  𝑠  𝐿

W^{s\rightarrow L}

map to the output domain

ℝ

d  y

superscript  ℝ

subscript  𝑑  𝑦

\mathbb{R}^{d_{y}}

with

B  L

∈

ℝ

d  y

superscript  𝐵  𝐿

superscript  ℝ

subscript  𝑑  𝑦

B^{L}\in\mathbb{R}^{d_{y}}

.

Given the weights of a neural network

f

𝑓

f

as a set of grouped vectors

W

𝑊

W

, we overload

ξ

𝜉

\xi

such that we obtain the sparsity of a neural network

ξ  ​

(  f  )

=

1

|  W  |

​

∑

x  ∈  W

ξ  ​

(  x  )

𝜉  𝑓

1

𝑊

subscript

𝑥  𝑊

𝜉  𝑥

\xi(f)=\frac{1}{|W|}\sum_{x\in W}\xi(x)

.
A  Sparse Neural Network  is a neural network

f

𝑓

f

with low sparsity, e.g.

ξ  ​

(  f  )

&lt;  0.5

𝜉  𝑓

0.5

\xi(f)&lt;0.5

.
The set of grouped vectors could, e.g., be all neurons with their weights from all possible incoming connections.

Sparse Neural Networks (SNN) naturally provide a directed acyclic graph

G  =

(  V  ,  E  )

𝐺

𝑉  𝐸

G=(V,E)

associated with them.
N