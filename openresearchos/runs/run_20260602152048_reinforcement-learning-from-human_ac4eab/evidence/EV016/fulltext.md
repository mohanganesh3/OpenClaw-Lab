[2404.17546] Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo

Probabilistic Inference in Language Models

via Twisted Sequential Monte Carlo

Stephen Zhao

Rob Brekelmans

Alireza Makhzani

Roger Grosse

Abstract

Numerous capability and safety techniques of Large Language Models (LLMs), including RLHF, automated red-teaming, prompt engineering, and infilling, can be cast as sampling from an unnormalized target distribution defined by a given reward or potential function over the full sequence. In this work, we leverage the rich toolkit of Sequential Monte Carlo (SMC) for these probabilistic inference problems. In particular, we use learned  twist functions  to estimate the expected future value of the potential at each timestep, which enables us to focus inference-time computation on promising partial sequences. We propose a novel contrastive method for learning the twist functions, and establish connections with the rich literature of soft reinforcement learning. As a complementary application of our twisted SMC framework, we present methods for evaluating the accuracy of language model inference techniques using novel  bidirectional  SMC bounds on the log partition function. These bounds can be used to estimate the KL divergence between the inference and target distributions in both directions. We apply our inference evaluation techniques to show that twisted SMC is effective for sampling undesirable outputs from a pretrained model (a useful component of harmlessness training and automated red-teaming), generating reviews with varied sentiment, and performing infilling tasks.

Machine Learning, ICML

\doparttoc  \faketableofcontents

1  Introduction

A wide range of language model learning and inference tasks can be viewed as steering a model’s generations to satisfy a specified property.
In particular, traditional  reinforcement learning from human feedback ( RLHF )  pipelines  (Ziegler et al.,  2019 ; Stiennon et al.,  2020 ; Ouyang et al.,  2022 ; Bai et al.,  2022 ; Rafailov et al.,  2023 )  may be viewed as targeting an unnormalized target modulated by a terminal reward function which reflects human feedback
 (Korbak et al.,  2022b ) .
Red-teaming techniques such as prompt-engineering and infilling may seek target outputs with low reward or (high probability of) undesirable responses
 (Zou et al.,  2023 ; Perez et al.,  2022 ) .
In reasoning tasks, we may seek to target outputs which are likely to be deemed valid by a ‘verifier’  (Cobbe et al.,  2021 ; Anil et al.,  2021 ; Dohan et al.,  2022 ; Hu et al.,  2023 ) . Specific properties of the generated responses might also be enforced  (Khalifa et al.,  2020 ; Yang &amp; Klein,  2021 ; Lew et al.,  2023 ) .

We view the above tasks as instances of  probabilistic inference :
sampling from a target unnormalized density and estimating its intractable (log) normalization constant. Consider a pretrained base model

p  0

​

(

𝐬

1  :  T

|

𝐬  0

)

subscript  𝑝  0

conditional

subscript  𝐬

:  1  𝑇

subscript  𝐬  0

p_{0}(\mathbf{s}_{1:T}|\mathbf{s}_{0})

which generates responses

𝐬

1  :  T

subscript  𝐬

:  1  𝑇

\mathbf{s}_{1:T}

of maximum length

T

𝑇

T

based on a variable-length prompt

𝐬  0

subscript  𝐬  0

\mathbf{s}_{0}

.
We consider
defining the
target distribution of interest
using
the base model modulated by
a  potential  function

ϕ  ​

(

𝐬

1  :  T

)

italic-ϕ

subscript  𝐬

:  1  𝑇

\phi(\mathbf{s}_{1:T})

which evaluates full sequences,

σ  ​

(

𝐬

1  :  T

|

𝐬  0

)

𝜎

conditional

subscript  𝐬

:  1  𝑇

subscript  𝐬  0

\displaystyle\sigma(\mathbf{s}_{1:T}|\mathbf{s}_{0})

≔

1

𝒵  σ

​

(

𝐬  0

)

​

p  0

​

(

𝐬

1  :  T

|

𝐬  0

)

​  ϕ  ​

(

𝐬

1  :  T

)

,

≔  absent

1

subscript  𝒵  𝜎

subscript  𝐬  0

subscript  𝑝  0

conditional

subscript  𝐬

:  1  𝑇

subscript  𝐬  0

italic-ϕ

subscript  𝐬

:  1  𝑇

\displaystyle\coloneqq\frac{1}{{\mathcal{Z}}_{\sigma}(\mathbf{s}_{0})}p_{0}(\mathbf{s}_{1:T}|\mathbf{s}_{0})\phi(\mathbf{s}_{1:T}),

(1)

where  ​

𝒵  σ

​

(

𝐬  0

)

where

subscript  𝒵  𝜎

subscript  𝐬  0

\displaystyle\text{where}\,\,{\mathcal{Z}}_{\sigma}(\mathbf{s}_{0})

≔

∑

𝐬

1  :  T

σ  ~

​

(

𝐬

1  :  T

|

𝐬  0

)

=

∑

𝐬

1  :  T

p  0

​

(

𝐬

1  :  T

|

𝐬  0

)

​  ϕ  ​

(

𝐬

1  :  T

)

,

≔  absent

subscript

subscript  𝐬

:  1  𝑇

~  𝜎

conditional

subscript  𝐬

:  1  𝑇

subscript  𝐬  0

subscript

subscript  𝐬

:  1  𝑇

subscript  𝑝  0

conditional

subscript  𝐬

:  1  𝑇

subscript  𝐬  0

italic-ϕ

subscript  𝐬

:  1  𝑇

\displaystyle\coloneqq\sum_{\mathbf{s}_{1:T}}\tilde{\sigma}(\mathbf{s}_{1:T}|\mathbf{s}_{0})=\sum_{\mathbf{s}_{1:T}}p_{0}(\mathbf{s}_{1:T}|\mathbf{s}_{0})\phi(\mathbf{s}_{1:T}),

where

σ  ~

​

(

𝐬

1  :  T

|

𝐬  0

)

~  𝜎

conditional

subscript  𝐬

:  1  𝑇

subscript  𝐬  0

\tilde{\sigma}(\mathbf{s}_{1:T}|\mathbf{s}_{0})

denotes the unnormalized density.
We refer to

𝒵  σ

​

(

𝐬  0

)

subscript  𝒵  𝜎

subscript  𝐬  0

{\mathcal{Z}}_{\sigma}(\mathbf{s}_{0})

as the normalization constant or partition function, which is intractable due to the summation over

𝐬

1  :  T

subscript  𝐬

:  1  𝑇

\mathbf{s}_{1:T}

.
We drop dependence on

𝐬  0

subscript  𝐬  0

\mathbf{s}_{0}

to avoid clutter, but note that each prompt induces a different partition function.
In the context of the aforementioned applications,

ϕ  ​

(

𝐬

1  :  T

)

italic-ϕ

subscript  𝐬

:  1  𝑇

\phi(\mathbf{s}_{1:T})

may be derived from a human preference model (for RLHF), an indication of bad behavior (for automated red-teaming), or a verifier’s prediction of correctness (for reasoning tasks).
We refer to

Table

5

or  Korbak et al. ( 2022b ); Dohan et al. ( 2022 ); Phan et al. ( 2023 ); Hu et al. ( 2023 )  for further examples and discussion of probabilistic inference in language models.

Twisted Sequential Monte Carlo in Language Models

In this work, we leverage tools from (twisted)  Sequential Monte Carlo ( SMC )

(Doucet et al.,  2001 ; Del Moral et al.,  2006 ; Briers et al.,  2010 ; Chopin et al.,  2020 )  to perform and evaluate inference in the language modeling setting (  Sec.

3  ).
A particular challenge in sampling from

Eq.

1

is that the target distribution

σ  ​

(

𝐬

1  :  T

)

𝜎

subscript  𝐬

:  1  𝑇

\sigma(\mathbf{s}_{1:T})

is non-causal. In order to sample tokens sequentially, one needs to infer the marginal distribution

σ  ​

(

𝐬

1  :  t

)

=

∑

𝐬

t  +  1

:  T

σ  ​

(

𝐬

1  :  T

)

∝

∑

𝐬

t  +  1

:  T

p  0

​

(

𝐬

t  +  1

:  T

|

𝐬

1  :  t

)

​  ϕ  ​

(

𝐬

1  :  T

)

𝜎

subscript  𝐬

:  1  𝑡

subscript

subscript  𝐬

:

𝑡  1

𝑇

𝜎

subscript  𝐬

:  1  𝑇

proportional-to

subscript

subscript  𝐬

:

𝑡  1

𝑇

subscript  𝑝  0

conditional

subscript  𝐬

:

𝑡  1

𝑇

subscript  𝐬

:  1  𝑡

italic-ϕ

subscript  𝐬

:  1  𝑇

\sigma(\mathbf{s}_{1:t})=\sum_{\mathbf{s}_{t+1:T}}\sigma(\mathbf{s}_{1:T})\propto\sum_{\mathbf{s}_{t+1:T}}p_{0}(\mathbf{s}_{t+1:T}|\mathbf{s}_{1:t})\phi(\mathbf{s}_{1:T})

,
which
involves an intractable marginalization.

To address this
problem, we propose to learn  twist functions

ψ  t

​

(

𝐬

1  :  t

)

subscript  𝜓  𝑡

subscript  𝐬

:  1  𝑡

\psi_{t}(\mathbf{s}_{1:t})

which modulate the base model such that

p  0

​

(

𝐬

1  :  t

)

​

ψ  t

​

(

𝐬

1  :  t

)

subscript  𝑝  0

subscript  𝐬

:  1  𝑡

subscript  𝜓  𝑡

subscript  𝐬

:  1  𝑡

p_{0}(\mathbf{s}_{1:t})\psi_{t}(\mathbf{s}_{1:t})

matches the target marginals

σ  ​

(

𝐬

1  :  t

)

𝜎

subscript  𝐬

:  1  𝑡

\sigma(\mathbf{s}_{1:t})

, up to normalization.
The twist functions can
be used to focus each step of language model
generation
on promising partial sequences.

Evaluating Inference in Language Modeling

Sampling from the target distribution
is closely intertwined with bounding the log partition function.
Similarly to variational infere