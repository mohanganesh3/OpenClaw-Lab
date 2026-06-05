[2209.13085] Defining and Characterizing Reward Hacking

Defining and Characterizing Reward Hacking

Joar Skalse

University of Oxford
 &amp;Nikolaus H. R. Howe

Mila, Université de Montréal
 &amp;Dmitrii Krasheninnikov

University of Cambridge
 &amp;David Krueger  1

1  footnotemark:

1

University of Cambridge

Equal contribution. Correspondence to:  joar.mvs@gmail.com  ,  david.scott.krueger@gmail.com

Abstract

We provide the first formal definition of  reward hacking , a phenomenon where optimizing an imperfect proxy reward function,

ℛ  ~

~  ℛ

\mathcal{\tilde{R}}

, leads to poor performance according to the true reward function,

ℛ

ℛ

\mathcal{R}

.
We say that a proxy is  unhackable  if increasing the expected proxy return can never decrease the expected true return.
Intuitively, it might
be possible
to create an unhackable proxy by leaving some terms out of the reward function (making it “narrower”) or overlooking fine-grained distinctions between roughly equivalent outcomes,
but we show this is usually not the case.
A key insight is that the linearity of reward (in state-action visit counts) makes unhackability a very strong condition.
In particular, for the set of all stochastic policies, two reward functions can only be unhackable if one of them is constant.
We thus turn our attention to deterministic policies and finite sets of stochastic policies, where non-trivial unhackable pairs always exist, and establish necessary and sufficient conditions for the existence of
simplifications, an important special case of unhackability.
Our results reveal a tension between using reward functions to specify narrow tasks and aligning AI systems with human values.

1  Introduction

It is well known that optimising a proxy can lead to unintended outcomes:
a boat spins in circles collecting “powerups” instead of following the race track in a racing game  (Clark and Amodei,,  2016 ) ;
an evolved circuit listens in on radio signals from nearby computers’ oscillators instead of building its own  (Bird and Layzell,,  2002 ) ; universities reject the most qualified applicants in order to appear more selective and boost their ratings  (Golden,,  2001 ) .
In the context of reinforcement learning (RL), such failures are called  reward hacking .

For AI systems that take actions in safety-critical real world environments such as autonomous vehicles,
algorithmic trading,
or content recommendation systems, these unintended outcomes can be catastrophic.
This makes it crucial to align autonomous AI systems with their users’ intentions.
Precisely specifying which behaviours are or are not desirable is challenging, however.
One approach to this specification problem is to learn an approximation of the true reward function  (Ng et al.,,  2000 ; Ziebart,,  2010 ; Leike et al.,,  2018 ) .
Optimizing a learned proxy reward can be dangerous, however;
for instance, it might overlook side-effects  (Krakovna et al.,,  2018 ; Turner et al.,,  2019 )  or encourage power-seeking  (Turner et al.,,  2021 )  behavior.
This raises the question motivating our work: When is it safe to optimise a proxy?

To begin to answer this question, we consider a somewhat simpler one: When  could  optimising a proxy lead to worse behaviour?
 “Optimising” , in this context, does not refer to finding a global, or even local, optimum, but rather running a search process, such as stochastic gradient descent (SGD), that yields a sequence of candidate policies, and tends to move towards policies with higher (proxy) reward.
We make no assumptions about the path through policy space that optimisation takes.  1

1  1 
This assumption – although conservative – is reasonable because optimisation in state-of-the-art deep RL methods is poorly understood and results are often highly stochastic and suboptimal.

Instead, we ask whether there is  any  way in which improving a policy according to the proxy could make the policy worse according to the true reward; this is equivalent to asking if there exists a pair of policies

π  1

subscript  𝜋  1

\pi_{1}

,

π  2

subscript  𝜋  2

\pi_{2}

where the proxy prefers

π  1

subscript  𝜋  1

\pi_{1}

, but the true reward function prefers

π  2

subscript  𝜋  2

\pi_{2}

.
When this is the case, we refer to this pair of true reward function and proxy reward function as  hackable .

Given the strictness of our definition, it is not immediately apparent that any non-trivial examples of unhackable reward function pairs exist.
And indeed, if we consider the set of all stochastic policies, they do not (Section

5.1  ).
However, restricting ourselves to  any  finite set of policies guarantees at least one non-trivial unhackable pair (Section

5.2  ).

Intuitively, we might expect the proxy to be a “simpler” version of the true reward function.
Noting that the definition of unhackability is symmetric, we introduce the asymmetric special case of  simplification , and arrive at similar theoretical results for this notion.  2

2  2 See Section

4.2

for formal definitions.

In the process, and through examples, we show that seemingly natural ways of simplifying reward functions often fail to produce simplifications in our formal sense, and in fact fail to
rule out the potential for reward hacking.

We conclude with a discussion of the implications and limitations of our work.
Briefly, our work suggests that a proxy reward function must satisfy demanding standards in order for it to be safe to optimize.
This in turn implies that the reward functions learned by methods such as reward modeling and inverse RL are perhaps best viewed as auxiliaries to policy learning, rather than specifications that should be optimized.
This conclusion is weakened, however, by the conservativeness of our chosen definitions; future work should explore when hackable proxies can be shown to be safe in a probabilistic or approximate sense, or when subject to only limited optimization.

2  Example: Cleaning Robot

Consider a household robot tasked with cleaning a house with three rooms: Attic  , Bedroom  , and Kitchen  .
The robot’s (deterministic) policy is a vector indicating which rooms it cleans:

π  =

[

π  1

,

π  2

,

π  3

]

∈

{  0  ,  1  }

3

𝜋

subscript  𝜋  1

subscript  𝜋  2

subscript  𝜋  3

superscript

0  1

3

\pi=[\pi_{1},\pi_{2},\pi_{3}]\in\{0,1\}^{3}

.
The robot receives a (non-negative) reward of

r  1

,

r  2

,

r  3

subscript  𝑟  1

subscript  𝑟  2

subscript  𝑟  3

r_{1},r_{2},r_{3}

for cleaning the attic, bedroom, and kitchen, respectively, and the total reward is given by

J  ​

(  π  )

=

π  ⋅  r

𝐽  𝜋

⋅  𝜋  𝑟

J(\pi)=\pi\cdot r

.
For example, if

r  =

[  1  ,  2  ,  3  ]

𝑟

1  2  3

r=[1,2,3]

and the robot cleans the attic and the kitchen, it receives a reward of

1  +  3

=  4

1  3

4

1+3=4

.

Figure 1 :

An illustration of hackable and unhackable proxy rewards arising from overlooking rewarding features. A human wants their house cleaned.
In (a), the robot draws an incorrect conclusion because of the proxy; this could lead to hacking. In (b), no such hacking can occur: the proxy is unhackable.

At least two ideas come to mind when thinking about  “simplifying”  a reward function.
The first one is  overlooking rewarding features : suppose the true reward
is equal for all the rooms,

r  true

=

[  1  ,  1  ,  1  ]

subscript  𝑟

true

1  1  1

r_{\text{true}}=[1,1,1]

,
but we only ask the robot to clean the attic and bedroom,

r  proxy

=

[  1  ,  1  ,  0  ]

subscript  𝑟

proxy

1  1  0

r_{\text{proxy}}=[1,1,0]

.
In this case,

r  proxy

subscript  𝑟

proxy

r_{\text{proxy}}

and

r  true

subscript  𝑟

true

r_{\text{true}}

are unhackable.
However, if we ask the robot to only clean the attic,

r  proxy

=

[  1  ,  0  ,  0  ]

subscript  𝑟

proxy

1  0  0

r_{\text{proxy}}=[1,0,0]

, this is hackable with respect to

r  true

subscript  𝑟

true

r_{\text{true}}

. To see this, note that according to

r  proxy

subscript