[2011.14632] Optimizing the Neural Architecture of Reinforcement Learning Agents

\tocauthor

Nina Mazyavkina, Samir Moustafa, Ilya Trofimov, Evgeny Burnaev

1

1  institutetext:  Skolkovo Institute of Science and Technology, Moscow, Russia,

1

1  email:  [n.mazyavkina, samir.mohamed, ilya.trofimov, e.burnaev]@skoltech.ru

Optimizing the Neural Architecture of Reinforcement Learning Agents

N. Mazyavkina

equal contribution

S. Moustafa  1

1  footnotemark:

1

I. Trofimov

E. Burnaev

Abstract

Reinforcement learning (RL) enjoyed significant progress over the last years. One of the most important steps forward was the wide application of neural networks. However, architectures of these neural networks are quite simple and typically are constructed manually. In this work, we study recently proposed  neural architecture search (NAS)  methods for optimizing the architecture of RL agents. We create two search spaces for the neural architectures and test two NAS methods: Efficient Neural Architecture Search (ENAS) and Single-Path One-Shot (SPOS). Next, we carry out experiments on the Atari benchmark and conclude that modern NAS methods find architectures of RL agents outperforming a manually selected one.

keywords:  AutoML, Neural Architecture Search, Reinforcement Learning, Atari

1  Introduction

Over the last several years, deep learning (DL) has experienced enormous growth in popularity among the researchers from both the academia and the industry.
Moreover, each of the separate tasks solved by the DL methods requires its own approach, one of the most important aspects of which is the choice of the neural network’s (NN) architecture. In this case, it is essential to demonstrate good expertise and experience in the problem’s field.
However, even then, the chosen architecture may not give any acceptable results until various heuristics and tricks will be applied to its construction. This motivated the emergence of the  neural architecture search  (NAS) field, which focuses on automating the ways to find the optimal architecture for the specific tasks.

Another family of methods that has been gaining popularity is reinforcement learning (RL) and deep reinforcement learning (deep RL), in particular. It consolidates a vast collection of machine learning methods, designed to solve a variety of Markov-Decision-Process-like problems.
Over the last several years the successes of RL and deep RL has been frequently demonstrated by the research community: from better-than-human performance in ATARI  [ 27 ] , DOTA 2  [ 24 ] , Go  [ 33 ]  to robotic manipulation  [ 14 ] .
In deep RL, neural networks are usually used to approximate a  value  function, in the case of the value-based methods, or a  policy  function, in the case of policy gradient methods. Moreover, actor-critic RL algorithms  [ 35 ,  25 ]  combine these two NN approximations, in order to gain even better performance. Consequently, finding a suitable NN architecture is also a vital part of designing RL experiments.

In this work, we are going to explore deep RL as a new application of NAS, i.e. deriving well-performing NN architectures for RL tasks.
The motivation for the research in this direction is the following:

1.

Only a single NN architecture is often chosen for many common benchmarks such as ATARI  [ 27 ]  and MuJoCo  [ 36 ] , despite of them consisting of a big number of different environments. Hence, automatically finding a suitable network for each of the environments may lead to better results;

2.

NAS can be useful in the cases of more complicated environments with bigger state and action spaces, where a more complicated deeper network might be required.

Early NAS methods  [ 43 ,  44 ]  required training of numerous neural architectures. However, even training of one RL agent takes a significant amount of time.
In this paper, we limit ourselves to fast  one-shot  methods, which perform architecture search in the time not significantly larger than the training time of a single neural network.
Most of the NAS methods were developed for computer vision applications, the major part – for the object classification problem.
At the same time, reinforcement learning is quite a different problem. The performance of an RL agent, that is, average reward, is not differentiable like the cross-entropy of object classification.
Thus, only few popular NAS methods are suited for RL. In this work, we evaluate ENAS  [ 44 ]  and SPOS  [ 15 ] .

The contribution of our paper is the following: we experimentally prove that modern one-shot NAS methods can be successfully applied for optimizing the neural architecture of RL agents.
The source code is publicly available from  https://github.com/NinaMaz/NAS˙RL˙torch .

2  Related work

Early neural architecture search (NAS) approaches
treated this problem as a black-box optimization, that is, search over a discrete domain of architectures. Such methods are quite general but require training of numerous architectures and vast computational resources. One of the first proposed methods of this kind  [ 43 ,  44 ]  used reinforcement learning for the optimization process itself. Architecture creation, layer by layer, was done by an RL agent. Thus, the reward was the performance of the constructed network.
Other works proposed evolutionary optimization  [ 29 ] , bayesian optimization based on Gaussian processes  [ 16 ] , bayesian performance predictors based on architecture features  [ 38 ,  32 ] . Black-box optimization enjoy speedup from multi-fidelity methods  [ 37 ] . Several benchmarks for NAS were developed  [ 42 ,  18 ,  9 ] .

The later family of methods –  one-shot NAS  – gone beyond black-box optimization and utilized the structure a neural network. These methods involve the  supernetwork , which contain all the architectures from the search space as its subnetworks. Thus, all the architectures share weights of some of the blocks.
The architecture search in the supernetwork is performed simultaneously with the training of networks themselves.
The one-shot methods are: ENAS  [ 28 ] , numerous modifications of DARTS  [ 23 ,  40 ,  7 ,  21 ,  10 ,  6 ] , single path one-shot  [ 15 ] , random search with weight sharing  [ 20 ,  3 ] .

Most of the existing research focuses on problems from computer vision and linguistics.
There are no papers about applications of modern NAS methods to RL to the best of our knowledge.

3  Reinforcement Learning Methods

In our experiments, we have used reinforcement learning for training both ENAS controller and sampled child networks. On the other hand, SPOS does not use a trainable controller for architecture sampling and, hence, the RL methods, mentioned in this section, do not concern it.

An LSTM controller, used in the ENAS framework, is trained with REINFORCE  [ 39 ]  algorithm. REINFORCE belongs to a group of policy-based methods, which focuses on the straightforward approximation of the optimal policy, via calculating the direct gradient of the parameterized objective function

J

𝐽

J

:

∇  θ

J  θ

=

𝔼  t

​

(

∇  θ

log

⁡

π  θ

​

(

a  t

|

s  t

)

​

R  t

)

.

subscript  ∇  𝜃

subscript  𝐽  𝜃

subscript  𝔼  𝑡

subscript  ∇  𝜃

subscript  𝜋  𝜃

conditional

subscript  𝑎  𝑡

subscript  𝑠  𝑡

subscript  𝑅  𝑡

\nabla_{\theta}J_{\theta}=\mathbb{E}_{t}\Big{(}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})R_{t}\Big{)}.

In our case,

θ

𝜃

\theta

are the parameters of the neural network, outputting the logits, from which the actions

a  t

subscript  𝑎  𝑡

a_{t}

can be derived;

π

𝜋

\pi

is the policy,

s  t

subscript  𝑠  𝑡

s_{t}

are the states,

R  t

subscript  𝑅  𝑡

R_{t}

is the sum of the discounted rewards collected so far.
Specifically, in the case of REINFORCE algorithm, the update to the parameters

θ

𝜃

\theta

takes the form:

θ  ←

θ  +

γ  t

​

R  t

​

∇  θ

log

⁡

π  θ

​

(

a  t

|

s  t

)

,

←  𝜃

𝜃

superscript  𝛾  𝑡

subscript  𝑅  𝑡

subscript  ∇  𝜃