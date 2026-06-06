[2412.05590] Active Sequential Posterior Estimation for Sample-Efficient Simulation-Based Inference

Active Sequential Posterior Estimation
 for Sample-Efficient Simulation-Based Inference

Sam Griesemer  1

Defu Cao  1

Zijun Cui  1,2

Carolina Osorio  3,4

Yan Liu 1

1 USC

2 MSU

3 Google Research

4 HEC Montréal

{samgriesemer,defucao,yanliu.cs}@usc.edu
 cuizijun@msu.edu
 osorioc@google.com

Work completed while at USC.

Abstract

Computer simulations have long presented the exciting possibility of scientific insight into complex real-world processes. Despite the power of modern computing, however, it remains
challenging to systematically perform inference under simulation models. This has led to the rise of simulation-based inference (SBI), a
class of machine learning-enabled techniques for approaching inverse problems with stochastic
simulators. Many such methods, however, require large numbers of simulation samples and face difficulty scaling to high-dimensional settings, often making inference prohibitive under resource-intensive simulators. To mitigate these
drawbacks, we introduce active sequential neural posterior estimation (ASNPE). ASNPE brings an active learning
scheme into the inference loop to estimate the utility of simulation parameter candidates to the underlying probabilistic model. The proposed acquisition scheme is easily integrated into existing posterior estimation pipelines, allowing for improved sample efficiency with low computational overhead.
We further demonstrate the effectiveness of the proposed method in the travel demand calibration setting, a
high-dimensional inverse problem commonly requiring computationally expensive traffic
simulators. Our method outperforms well-tuned benchmarks and state-of-the-art posterior estimation methods on a large-scale real-world traffic network, as well as demonstrates a performance advantage over non-active counterparts on a suite of SBI benchmark environments.

1  Introduction

High-fidelity computer simulations have been embraced across countless scientific domains,
furthering the ability to understand and predict behaviour in complex real-world systems.
Modern computing architectures and flexible programming paradigms have further lowered the
barrier to capturing approximate models for scientific study in silico, enabling
wide-spread use of computational experiments across disciplines. However, despite the
relative ease of capturing real-world generative processes programmatically, the resulting
black-box programs are often difficult to leverage for inverse problems. This is a
common challenge in practical applications; the simulator is often computationally
expensive to evaluate, its implicit likelihood function is generally intractable, and the
dimensionality of high-fidelity outputs is typically prohibitive. To address these issues,
likelihood-free inference methods have been introduced, operating under the broadly
applicable assumption that no tractable likelihood function is available. Early
success along this direction was achieved through easy-to-use methods like
Approximate Bayesian Computation (ABC)  [ 41 ,  6 ] , or extensions of kernel
density estimation. The scale of real-world applications demands more flexible and
scalable approaches, which has lead to the integration of aptly suited deep learning
methods in likelihood-free settings. Neural network-based methods
 [ 33 ,  29 ,  17 ]  have since been proposed, introducing greater flexibility when
approximating probabilistic components (e.g., the posterior, likelihood ratio, etc) in the
inference pipeline. The use of the term "simulation-based inference" (SBI) has since been
colloquially embraced  [ 11 ]  when referring to this emerging class of
techniques.

SBI methods primarily leverage deep learning through their use of neural density
estimators (NDE), neural network-based parametrizations of probability density functions.
Common choices of NDE in practice include mixture density networks  [ 8 ]  and
normalizing flows  [ 40 ,  34 ] , along with popular extensions (e.g., Real NVP
 [ 12 ] , MAE  [ 16 ] , MAF  [ 35 ] , etc). Methods also vary in the probabilistic
form they elect to approximate; the posterior  [ 33 ] , the likelihood  [ 36 ] , and
the likelihood ratio  [ 19 ,  13 ]  are all common choices for well-established
methods.

While many SBI techniques leverage basic principles from active learning (AL), they are mostly established as a helpful heuristic
for increased sample efficiency, rather than an explicit optimization over a defined
acquisition function. For example, methods like Sequential Neural Posterior Estimation
(SNPE)  [ 33 ]  boost sample efficiency (over non-sequential methods) by iteratively
updating a proposal prior

p  ~

​

(  θ  )

~  𝑝

𝜃

\tilde{p}(\theta)

, steering simulator parameters to values expected to
be more useful for learning the posterior under observations

x  o

subscript  𝑥  𝑜

x_{o}

of interest.

While sequential proposal updates are an effective first-order step to more informative
simulation runs, there are many key factors that remain overlooked. For example, the
updated proposal does not take into account the current parameters of the NDE itself, and
parameter samples in the batch are drawn from the current proposal independently. This fails to fully utilize myopic AL strategies and batch optimization, leading to large amounts
of simulation runs with high expected information overlap and wasted computation. To
address this issue, we formulate an active
learning scheme that (1) selects samples expected to target epistemic uncertainty in the underlying
probabilistic model, and (2) makes acquisition evaluation simple and efficient when using any Bayesian NDE.

We demonstrate the effectiveness of our method on the origin-destination (OD) calibration task. OD calibration aims to identify OD matrices that yield simulated traffic metrics that accurately reflect field-observed traffic conditions. It
can be seen as a parameter tuning process, akin to model fitting in machine learning. From
the machine learning perspective, OD calibration presents challenges due to the
requirement of calibrating specific unique samples from observed traffic
information, such as link flows, trip speeds, etc.

Our contributions are summarized as follows:

•

Active Sequential Neural Posterior Estimation (ASNPE), an SNPE variant that incorporates active acquisition of informative simulation parameters

θ

𝜃

\theta

to the underlying (direct) posterior estimation model, without the use of additional surrogate models. This helps to drive down uncertainty in parameters of the utilized NDE and improve sample efficiency, both of which are particularly important when interfacing with computationally costly simulation-based models.

•

An efficient approximation to the proposed acquisition function above, along with a means of training Bayesian flow-based generative models for density estimation
during posterior approximation (both with open source implementations  2

2  2  https://github.com/samgriesemer/seqinf

). Leveraging this class of models enables direct uncertainty quantification in
the acquisition function, and is more flexible, efficient to train, and scalable to high-dimensional data than many traditional Bayesian model choices (e.g., Gaussian processes).

•

A Bayesian formulation of the OD calibration problem and coupled statistical
framework for performing sequential likelihood-free inference with neural posterior estimation methods. We show ASNPE outperforms baseline methods across a wide variety of simulation scenarios on a large-scale traffic network. We also evaluate ASNPE on three broader SBI benchmark environments and find it acheives a performance advantage over non-active counterparts.

2  Background

2.1  Neural posterior estimation

Given observational data of interest

x  o

subscript  𝑥  𝑜

x_{o}

and a prior

p  ​

(  θ  )

𝑝  𝜃

p(\theta)

, we want to carry out
st