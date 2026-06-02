[2210.05931] Explaining Online Reinforcement Learning Decisions of Self-Adaptive Systems

Explaining Online Reinforcement Learning Decisions of Self-Adaptive Systems

Felix Feit, Andreas Metzger, Klaus Pohl

paluno (The Ruhr Institute for Software Technology)
 University of Duisburg-Essen;
Essen, Germany
 f.m.feit@gmail.com, andreas.metzger@paluno.uni-due.de, klaus.pohl@paluno.uni-due.de

Abstract

Design time uncertainty poses an important challenge when developing a self-adaptive system.
As an example, defining how the system should adapt when facing a new environment state, requires understanding the precise effect of an adaptation, which may not be known at design time.
Online reinforcement learning, i.e., employing reinforcement learning (RL) at runtime, is an emerging approach to realizing self-adaptive systems in the presence of design time uncertainty.
By using Online RL, the self-adaptive system can learn from actual operational data and leverage feedback only available at runtime.
Recently, Deep RL is gaining interest.
Deep RL represents learned knowledge as a neural network whereby it can generalize over unseen inputs, as well as handle continuous environment states and adaptation actions. A fundamental problem of Deep RL is that learned knowledge is not explicitly represented.
For a human, it is practically impossible to relate the parametrization of the neural network to concrete RL decisions and thus Deep RL essentially appears as a black box.
Yet, understanding the decisions made by Deep RL is key to (1) increasing trust, and (2) facilitating debugging.
Such debugging is especially relevant for self-adaptive systems, because the reward function, which quantifies the feedback to the RL algorithm, must be defined by developers.
The reward function must be explicitly defined by developers, thus introducing a potential for human error.
To explain Deep RL for self-adaptive systems, we enhance and combine two existing explainable RL techniques from the machine learning literature.
The combined technique, XRL-DINE, overcomes the respective limitations of the individual techniques.
We present a proof-of-concept implementation of XRL-DINE, as well as qualitative and quantitative results of applying XRL-DINE to a self-adaptive system exemplar.

I

Introduction

A self-adaptive system  can modify its own structure and behavior at runtime based on its perception of the environment, of itself and of its requirements  [ 1 ] .
One key element of a self-adaptive system is its  self-adaptation logic  that encodes when and how the system should adapt itself.
When developing the adaptation logic, developers face the challenge of  design time uncertainty

[ 2 ,  3 ,  4 ] .
To define  when  the system should adapt, they have to anticipate all potential environment states.
However, this is infeasible in most cases due to incomplete information at design time.
As an example, the concrete services that may be dynamically bound during the execution of a service orchestration and thus their quality characteristics are typically not known at design time.
To define  how  the system should adapt itself, developers need to know the precise effect an adaptation action has.
However, the precise effect may not be known at design time.
As an example, while developers may know in principle that enabling more features will negatively influence the performance, exactly determining the performance impact is more challenging.
A recent industrial survey identified optimal design and design complexity together with design time uncertainty to be the most frequently observed difficulties in designing self-adaptation in practice  [ 4 ] .

Online reinforcement learning (Online RL)  is an emerging approach to realize self-adaptive systems in the presence of design time uncertainty.
Online RL means that reinforcement learning  [ 5 ]  is employed at runtime (see  [ 3 ]  for a discussion of existing solutions).
The self-adaptive system thereby can learn from actual operational data and thus leverages information only available at runtime.
A recent survey indicates that since 2019 the use of learning dominates over the use of predetermined and static policies or rules  [ 6 ] .

Online RL aims at learning suitable adaptation actions via the self-adaptive system’s interactions with its initially unknown environment  [ 7 ] .
During system operation, the RL algorithm receives a numerical reward based on actual runtime monitoring data for executing an adaptation action.
The reward expresses how suitable that adaptation action was in the short term.
The goal of Online RL is to maximize the cumulative reward.

Initially, research on self-adaptive systems leveraged RL algorithms that represent learned knowledge as a so-called  value function

[ 7 ] .
The value function quantifies how much cumulative reward can be expected if a particular adaptation is chosen in a given environment state.
Typically, this value function was represented as a table.
However, such tabular approaches exhibit key limitations.
First, they require a finite set of environment states and a finite set of adaptations and thus cannot be directly applied to continuous state spaces or continuous action spaces.
Second, they do not generalize over neighboring states, which leads to slow learning in the presence of continuous environment states  [ 7 ] .

Deep reinforcement learning  ( Deep RL ) addresses these disadvantages by representing the learned knowledge as a neural network.
Since neural network inputs are not limited to elements of finite or discrete sets, and neural networks can generalize well over inputs, deep RL has shown remarkable success in different application areas.
Recently, Deep RL is also being applied to self-adaptive systems  [ 8 ,  9 ,  7 ] .

A principal problem of Deep RL  is that learned knowledge is not explicitly represented.
Instead, it is “hidden” in the parametrization of the neural network.
For a human, it is practically impossible to relate this parametrization to concrete RL decisions.
Deep RL thus essentially appears as a black box  [ 10 ] .
Yet, understanding the decisions made by Deep RL systems is key to (1) increase trust in these systems, and (2) facilitate their debugging  [ 11 ,  12 ] .

Facilitating the debugging of Deep RL is especially relevant for self-adaptive systems, because Online RL does not completely eliminate manual development effort.
Since developers need to explicitly define the reward function, this introduces a potential source for human error.

To explain Deep RL systems, various  Explainable Reinforcement Learning  ( XRL ) techniques were recently put forward in machine learning research  [ 10 ,  13 ] .
Here, we set out to answer the question how existing XRL techniques can be applied for the explainability of Online RL for self-adaptive systems.
We follow XRL literature and use ”explainable” to also include ”interpretable”, even though one may consider ”interpretable” only as basis for ”explainable”.

Our contribution  is to enhance and combine two existing XRL techniques from the literature:  Reward Decomposition

[ 14 ]  and  Interestingness Elements

[ 15 ] .
Reward Decomposition uses a suitable decomposition of the reward function into sub-functions to explain the short-term goal orientation of RL, thereby providing contrastive explanations.

Reward composition is especially helpful for the typical problem of adapting a system while taking into account multiple quality goals.
Each of these quality goals could then be expressed as a reward sub-function.
However, no indication for the explanation’s relevance is provided, but instead it requires manually selecting relevant RL decisions to be explained.
In particular when RL decisions are taken at runtime, which is the case for Online RL for self-adaptive systems, monitoring all explanations to identify relevant ones introduces cognitive overhead for developers.
In contrast, Interestingness Elements collect and