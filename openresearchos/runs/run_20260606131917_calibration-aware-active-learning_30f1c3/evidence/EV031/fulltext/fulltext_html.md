[2509.05381] Murphy’s Laws of AI Alignment: Why the Gap Always Wins

Murphy’s Laws of AI Alignment: Why the Gap Always Wins

Madhava Gaikwad

mgaikwad@microsoft.com

Microsoft

This is a working paper open to collab, revision and critique. This work was done in individual capacity and does not represent the views of my employer.

Abstract

We study reinforcement learning from human feedback under misspecification. Contexts are drawn i.i.d. from a distribution

μ  \mu

over

𝒳  \mathcal{X}

. There exists a measurable subset

X  hard

⊂  𝒳

X_{\mathrm{hard}}\subset\mathcal{X}

with mass

α  =

μ  ​

(

X  hard

)

\alpha=\mu(X_{\mathrm{hard}})

on which the feedback channel is systematically biased. We consider pairwise preference feedback and allow adaptive query selection with a budget of

n  n

queries.

We construct two environments

w  ∈

{  +  ,  −  }

w\in\{+,-\}

with reward functions

r  w

r_{w}

that differ only on

X  hard

X_{\mathrm{hard}}

. Let

π  w

\pi^{w}

denote an optimal policy for

r  w

r_{w}

, and define the separation parameter

γ  =

𝔼

x  ∼  μ

​

[

|

r  +

​

(  x  ,

π  +

​

(  x  )

)

−

r  +

​

(  x  ,

π  −

​

(  x  )

)

|

⋅  𝟏

​

{

x  ∈

X  hard

}

]

&gt;

0

.

\gamma\;=\;\mathbb{E}_{x\sim\mu}\!\left[\big|r_{+}(x,\pi^{+}(x))-r_{+}(x,\pi^{-}(x))\big|\cdot\mathbf{1}\{x\in X_{\mathrm{hard}}\}\right]\;&gt;\;0.

On

X  hard

X_{\mathrm{hard}}

the pairwise label favors the truly better action with probability

1  2

+

w  ​  ε

\tfrac{1}{2}+w\varepsilon

for some

ε  ∈

(  0  ,

1  2

)

\varepsilon\in(0,\tfrac{1}{2})

. Outside

X  hard

X_{\mathrm{hard}}

feedback is symmetric. For any learner using at most

n  n

queries, Le Cam’s two point method combined with a transcript level Kullback–Leibler decomposition yields

inf

π  ^

sup

w  ∈

{  +  ,  −  }

(

V  w

​

(

π  w

)

−

𝔼  w

​

[

V  w

​

(

π  ^

)

]

)

≥

γ  4

​

exp

⁡

(

−

n  ​  α  ​  κ  ​

(  ε  )

)

,

κ  ​

(  ε  )

=

4  ​  ε  ​  atanh  ​

(

2  ​  ε

)

.

\inf_{\widehat{\pi}}\;\sup_{w\in\{+,-\}}\Big(V_{w}(\pi^{w})-\mathbb{E}_{w}[V_{w}(\widehat{\pi})]\Big)\;\geq\;\frac{\gamma}{4}\exp\!\big(-n\,\alpha\,\kappa(\varepsilon)\big),\qquad\kappa(\varepsilon)=4\varepsilon\,\mathrm{atanh}(2\varepsilon).

With access to a calibration oracle

h  ​

(  x  )

=

𝟏  ​

{

x  ∈

X  hard

}

h(x)=\mathbf{1}\{x\in X_{\mathrm{hard}}\}

that flags misspecified contexts, an adaptive procedure that concentrates queries on

X  hard

X_{\mathrm{hard}}

achieves expected gap at most

η  \eta

with

Q  ≤

1

2  ​  α  ​

ε  2

​

log  ⁡

γ  η

Q\;\leq\;\frac{1}{2\alpha\varepsilon^{2}}\log\!\frac{\gamma}{\eta}

queries, using a simple majority test on the flagged contexts. The constants arise from per hit Bernoulli KL and standard testing bounds. The results quantify how

α  \alpha

and

ε  \varepsilon

govern the sample complexity required to resolve misspecification.

1  Introduction

Reinforcement learning from human feedback (RLHF) is often instantiated via pairwise comparisons or scalar ratings that train a reward model used for policy optimization. In many deployments the feedback channel is imperfect and the number of informative queries is limited. This paper analyzes the fundamental limitations imposed by misspecification under bounded query budgets and identifies minimal additional structure that removes these limitations.

Setting.

Let

(  𝒳  ,  μ  )

(\mathcal{X},\mu)

be a context space with i.i.d. draws

x  ∼  μ

x\sim\mu

and a finite action set

𝒜  \mathcal{A}

. A policy

π  :

𝒳  →  𝒜

\pi:\mathcal{X}\to\mathcal{A}

has value

V  w

​

(  π  )

=

𝔼

x  ∼  μ

​

[

r  w

​

(  x  ,

π  ​

(  x  )

)

]

V_{w}(\pi)=\mathbb{E}_{x\sim\mu}[r_{w}(x,\pi(x))]

under world

w  w

. There is a measurable subset

X  hard

⊂  𝒳

X_{\mathrm{hard}}\subset\mathcal{X}

with

α  =

μ  ​

(

X  hard

)

\alpha=\mu(X_{\mathrm{hard}})

on which feedback is misspecified. Two worlds

w  ∈

{  +  ,  −  }

w\in\{+,-\}

are defined by reward functions

r  w

r_{w}

that differ only on

X  hard

X_{\mathrm{hard}}

. Let

π  w

\pi^{w}

denote an optimal policy for

r  w

r_{w}

. Define the separation parameter

γ  =

𝔼

x  ∼  μ

​

[

|

r  +

​

(  x  ,

π  +

​

(  x  )

)

−

r  +

​

(  x  ,

π  −

​

(  x  )

)

|

⋅  𝟏

​

{

x  ∈

X  hard

}

]

&gt;

0

.

\gamma\;=\;\mathbb{E}_{x\sim\mu}\!\left[\big|r_{+}(x,\pi^{+}(x))-r_{+}(x,\pi^{-}(x))\big|\cdot\mathbf{1}\{x\in X_{\mathrm{hard}}\}\right]\;&gt;\;0.

On

x  ∈

X  hard

x\in X_{\mathrm{hard}}

the pairwise label equals the indicator that the truly better action is preferred, flipped with Massart style bias of magnitude

ε  ∈

(  0  ,

1  2

)

\varepsilon\in(0,\tfrac{1}{2})

:

Pr  w

⁡

{

label favors the truly better action  ∣  x

∈

X  hard

}

=

1  2

+

w  ​  ε

.

\Pr_{w}\{\text{label favors the truly better action}\mid x\in X_{\mathrm{hard}}\}\;=\;\tfrac{1}{2}+w\varepsilon.

On

x  ∈

X  easy

=

𝒳  ∖

X  hard

x\in X_{\mathrm{easy}}=\mathcal{X}\setminus X_{\mathrm{hard}}

the channel is symmetric with probability

1  2

\tfrac{1}{2}

. The learner may adaptively select which queries to issue, up to a budget of

n  n

queries.

Lower bound.

Let

𝖯  w

\mathsf{P}^{w}

be the joint law of the full transcript under world

w  w

. The transcript level Kullback–Leibler divergence satisfies

D  KL

(

𝖯  +

∥

𝖯  −

)

=

∑

t  =  1

n

𝔼

𝖯  +

[

D  KL

(

𝖯  +

(

Y  t

∣

ℋ

t  −  1

)

∥

𝖯  −

(

Y  t

∣

ℋ

t  −  1

)

)

]

≤  n  α  κ

(  ε  )

,

D_{\mathrm{KL}}(\mathsf{P}^{+}\,\|\,\mathsf{P}^{-})\;=\;\sum_{t=1}^{n}\mathbb{E}_{\mathsf{P}^{+}}\!\left[D_{\mathrm{KL}}\!\big(\mathsf{P}^{+}(Y_{t}\mid\mathcal{H}_{t-1})\,\|\,\mathsf{P}^{-}(Y_{t}\mid\mathcal{H}_{t-1})\big)\right]\;\leq\;n\,\alpha\,\kappa(\varepsilon),

where each term is nonzero only when

x  t

∈

X  hard

x_{t}\in X_{\mathrm{hard}}

and

κ

(  ε  )

=

D  KL

(

Ber

(

1  2

+  ε  )

∥

Ber

(

1  2

−  ε  )

)

=

4  ε  atanh

(  2  ε  )

.

\kappa(\varepsilon)\;=\;D_{\mathrm{KL}}\!\left(\mathrm{Ber}\!\left(\tfrac{1}{2}+\varepsilon\right)\,\middle\|\,\mathrm{Ber}\!\left(\tfrac{1}{2}-\varepsilon\right)\right)\;=\;4\varepsilon\,\mathrm{atanh}(2\varepsilon).

A standard testing inequality converts the KL budget into a lower bound on the Bayes error for distinguishing the two worlds, which implies the finite sample value gap

inf

π  ^

sup

w  ∈

{  +  ,  −  }

(

V  w

​

(

π  w

)

−

𝔼  w

​

[

V  w

​

(

π  ^

)

]

)

≥

γ  4

​

exp

⁡

(

−

n  ​  α  ​  κ  ​

(  ε  )

)

.

\inf_{\widehat{\pi}}\;\sup_{w\in\{+,-\}}\Big(V_{w}(\pi^{w})-\mathbb{E}_{w}[V_{w}(\widehat{\pi})]\Big)\;\geq\;\frac{\gamma}{4}\exp\!\big(-n\,\alpha\,\kappa(\varepsilon)\big).

The bound holds for adaptive procedures since the chain rule is applied conditionally on the observed history.

Upper bound with a calibration oracle.

Suppose an oracle

h  :

𝒳  →

{  0  ,  1  }

h:\mathcal{X}\to\{0,1\}

indicates membership in

X  hard

X_{\mathrm{hard}}

. Draw contexts until

h  ​

(  x  )

=  1

h(x)=1

. The expected number of draws per hit is

1  /  α

1/\alpha

. On each hit, collect a bounded preference bit. A majority test over

T  T

hits on

X  hard

X_{\mathrm{hard}}

errs with probability at most

exp  ⁡

(

−

2  ​  T  ​

ε  2

)

\exp(-2T\varepsilon^{2})

by a bounded difference inequality. Setting

T  =

1

2  ​

ε  2

​

log  ⁡

(

γ  /  η

)

T=\tfrac{1}{2\varepsilon^{2}}\log(\gamma/\eta)

yields an expected value gap at most

η  \eta

with

Q  =

T  α

≤

1

2  ​  α  ​

ε  2

​

log  ⁡

γ  η

Q\;=\;\frac{T}{\alpha}\;\leq\;\frac{1}{2\alpha\varepsilon^{2}}\log\!\frac{\gamma}{\eta}

total queries. This matches the dependence on

α  \alpha

and

ε  \varepsilon

in the lower bound up to constants.

Discussion.

The analysis isolates three parameters that govern difficulty under misspecification: prevalence

α  \alpha

, bias magnitude

ε  \varepsilon

, and separation

γ  \gamma

. The lower bound follows from indist