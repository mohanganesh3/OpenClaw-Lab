[2210.16365] Elastic Weight Consolidation Improves the Robustness of Self-Supervised Learning Methods under Transfer

\foreach  \x

in a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z

\foreach  \x in alpha,beta,gamma,delta,epsilon,varepsilon,zeta,eta,theta,vartheta,iota,kappa,varkappa,lambda,mu,nu,xi,pi,varpi,rho,varrho,sigma,varsigma,tau,upsilon,phi,varphi,chi,psi,omega

\foreach  \x in A,B,C,D,E,F,G,H,I,J,K,L,M,M,O,P,Q,R,S,T,U,V,W,X,Y,Z

\foreach  \x in Gamma,Delta,Theta,Lambda,Xi,Pi,Sigma,Upsilon,Phi,Psi,Omega

Elastic Weight Consolidation Improves the Robustness of Self-Supervised Learning Methods under Transfer

Andrius Ovsianas 1  Jason Ramapuram  2

Dan Busbridge  2

Eeshan Gunesh Dhekane  2

Russ Webb  2

University of Cambridge 1 , Apple 2

ao464@cam.ac.uk

{jramapuram, dbusbridge, eeshan, rwebb}@apple.com

Work done during Apple internship.

Abstract

Self-supervised representation learning (SSL)  methods provide an effective label-free initial condition for
fine-tuning downstream tasks. However, in numerous realistic scenarios, the
downstream task might be biased with respect to the target label distribution.
This in turn moves the learned fine-tuned model posterior away from the initial
(label) bias-free self-supervised model posterior. In this work, we re-interpret
 SSL  fine-tuning under the lens of Bayesian continual learning and consider
regularization through the  Elastic Weight Consolidation (EWC)  framework. We demonstrate that
self-regularization against an initial  SSL  backbone improves worst
sub-group performance in Waterbirds by 5% and
Celeb-A by 2% when using the ViT-B/16 architecture. Furthermore, to help simplify the
use of  EWC  with  SSL , we pre-compute and publicly release the  Fisher Information Matrix (FIM) , evaluated with 10,000 ImageNet-1K variates evaluated on large modern  SSL  architectures including
ViT-B/16 and ResNet50 trained with DINO.

1  Introduction

Self-supervised learning (SSL) methods for learning representations have recently gained
popularity within the deep learning community, bridging the gap with
supervised discriminative methods in vision
 (Caron et al.,  2021 ; Goyal et al.,  2021 ; Grill et al.,  2020 ; Chen et al.,  2020b ,  a ) .
While representations learned via SSL methods are free from label induced bias
 (Goyal et al.,  2021 ) , this can change during the process of
fine-tuning to a downstream task.

In the supervised regime  Sagawa et al. ( 2019 )  showed
that models trained with  empirical risk minimization (ERM)  tend to be biased towards label
population distributions that are disproportionately represented within the
training dataset. Our objective with this work is to mitigate this drift through
the use of Bayesian continual learning where we investigate regularizing downstream
tasks towards their robust initial representation produced by the SSL
pre-training procedure  (Goyal et al.,  2021 ) .

We consider regularizers based on the Fisher Information Matrix ( FIM ),
which constrain the model parameters towards their initial SSL values, as in
 Continual Learning (CL)  techniques, such as Elastic Weight Consolidation ( EWC )
 (Kirkpatrick et al.,  2017 ) .  CL  is a rich sub-field of machine
learning that seeks to minimize the effect of catastrophic forgetting
 (McCloskey &amp; Cohen,  1989 )  – the phenomenon where models trained in a
sequential manner tend to become biased towards the latest observed distribution.
To use  EWC , we compute the  FIM  for DINO  (Caron et al.,  2021 ) 
models pre-trained with ViT-B/16  (Dosovitskiy et al.,  2021 )  and
ResNet50  (He et al.,  2016 )  architectures on a subset of ImageNet1k
 (Deng et al.,  2009 )  images  3

3  3  FIM  available
at https://coming_soon

.

We validate the accuracy of the  FIM  by analyzing reverse transfer
performance from CIFAR10 fine-tuning. We observe that the
 FIM  for the ViT-B/16 is poorly conditioned, making it impossible to fully
recover the SSL performance on ImageNet1k, and propose a method to alleviate
this. Finally, we show that with ViT-B/16 this regularization can be used
to improve worst group accuracy on Waterbirds  (Sagawa et al.,  2019 )  and
Celeb-A  (Liu et al.,  2015 )  by 5% and
2% respectively.

2  Method

To keep fine-tuned representations close to their initial  SSL  values we
consider techniques used in  CL , in particular  EWC

(Kirkpatrick et al.,  2017 ) , and treat the SSL pre-training and fine-tuning
tasks as two distinct sequential tasks. To overcome catastrophic forgetting
 (McCloskey &amp; Cohen,  1989 ) , where models trained with stochastic
gradient descent become biased towards the latest task distribution, EWC
regularizes model parameters towards their optimal values on previous tasks.
The regularization uses the Fisher Information Matrix (FIM) and is based on the Laplace approximation  (Laplace,  1986 ; Smola et al.,  2003 ) .
An intuitive way to look at this regularization is through the
lens of online Bayesian inference. In particular, we consider a given SSL model
as a statistical Bayesian model

p  θ

​

(

y  |  x

)

subscript  𝑝  𝜃

conditional  𝑦  𝑥

p_{\theta}(y\;|\;x)

with prior

p  ​

(  θ  )

𝑝  𝜃

p(\theta)

.
Given two datasets,

𝒟  SSL

subscript  𝒟

SSL

\mathcal{D}_{\text{SSL}}

and

𝒟  FT

subscript  𝒟

FT

\mathcal{D}_{\text{FT}}

, observed one after another, our
objective is to estimate parameters

θ

𝜃

\theta

. While the full posterior
distribution

p  ​

(

θ  |

𝒟  SSL

,

𝒟  FT

)

𝑝

conditional  𝜃

subscript  𝒟

SSL

subscript  𝒟

FT

p(\theta\;|\;\mathcal{D}_{\text{SSL}},\mathcal{D}_{\text{FT}})

might be intractable, a point
estimate can be computed using Laplace’s approximation.

First, the posterior with respect to the SSL task,

p  ​

(

θ  |

𝒟  SSL

)

𝑝

conditional  𝜃

subscript  𝒟

SSL

p(\theta\;|\;\mathcal{D}_{\text{SSL}})

,
is approximated with a Normal distribution using a Taylor’s expansion as shown in Eq (  1  ):

log  ⁡  p

​

(

θ  |

𝒟  SSL

)

≈

−

1  2

​

(

θ  −

θ  SSL

)

⊤

​  H  ​

(

θ  SSL

)

​

(

θ  −

θ  SSL

)

=

−

1  2

​

‖

θ  −

θ  SSL

‖

H  ​

(

θ  SSL

)

2

.

𝑝

conditional  𝜃

subscript  𝒟

SSL

1  2

superscript

𝜃

subscript  𝜃

SSL

top

𝐻

subscript  𝜃

SSL

𝜃

subscript  𝜃

SSL

1  2

superscript

subscript

norm

𝜃

subscript  𝜃

SSL

𝐻

subscript  𝜃

SSL

2

\displaystyle\log p(\theta\;|\;\mathcal{D}_{\text{SSL}})\approx-\frac{1}{2}(\theta-\theta_{\text{SSL}})^{\top}H(\theta_{\text{SSL}})(\theta-\theta_{\text{SSL}})=-\frac{1}{2}\|\theta-\theta_{\text{SSL}}\|_{H(\theta_{\text{SSL}})}^{2}.

(1)

Here

θ  SSL

subscript  𝜃

SSL

\theta_{\text{SSL}}

is the  Maximum a Posteriori (MAP)  estimate of

log  ⁡  p

​

(

θ  |

𝒟  SSL

)

𝑝

conditional  𝜃

subscript  𝒟

SSL

\log p(\theta\;|\;\mathcal{D}_{\text{SSL}})

and

H  ​

(

θ  SSL

)

𝐻

subscript  𝜃

SSL

H(\theta_{\text{SSL}})

is the Hessian. A point estimate to

p  ​

(

θ  |

𝒟  SSL

,

𝒟  FT

)

𝑝

conditional  𝜃

subscript  𝒟

SSL

subscript  𝒟

FT

p(\theta\;|\;\mathcal{D}_{\text{SSL}},\mathcal{D}_{\text{FT}})

can then
be derived via Bayes rule:

θ  FT

=

arg  ​  max

θ

⁡  log

⁡  p

​

(

θ  |

𝒟  SSL

,

𝒟  FT

)

≈

arg  ​  max

θ

⁡  log

⁡  p

​

(

𝒟  FT

|  θ

)

−

1  2

​

‖

θ  −

θ  SSL

‖

H  ​

(

θ  SSL

)

2

.

subscript  𝜃

FT

subscript

arg  max

𝜃

𝑝

conditional  𝜃

subscript  𝒟

SSL

subscript  𝒟

FT

subscript

arg  max

𝜃

𝑝

conditional

subscript  𝒟

FT

𝜃

1  2

superscript

subscript

norm

𝜃

subscript  𝜃

SSL

𝐻

subscript  𝜃

SSL

2

\displaystyle\theta_{\text{FT}}=\operatorname*{arg\,max}_{\theta}\ \log p(\theta\;|\;\mathcal{D}_{\text{SSL}},\mathcal{D}_{\text{FT}})\approx\operatorname*{arg\,max}_{\theta}\ \log p(\mathcal{D}_{\text{FT}}\;|\;\theta)-\frac{1}{2}\|\theta-\theta_{\text{SSL}}\|_{H(\theta_{\text{SSL}})}^{2}.

(2)

Since the Hessian is quadratic in the number of model parameters, it is impractical to
store it for anything bes