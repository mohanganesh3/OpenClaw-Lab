[2012.08215] Generative model for reciprocity and community detection in networks

†

†  thanks:  Contributed equally.

Generative model for reciprocity and community detection in networks

Hadiseh Safdari

hadiseh.safdari@tuebingen.mpg.de

Max Planck Institute for Intelligent Systems, Cyber Valley, Tuebingen 72076, Germany

Martina Contisciani

martina.contisciani@tuebingen.mpg.de

Max Planck Institute for Intelligent Systems, Cyber Valley, Tuebingen 72076, Germany

Caterina De Bacco

caterina.debacco@tuebingen.mpg.de

Max Planck Institute for Intelligent Systems, Cyber Valley, Tuebingen 72076, Germany

Abstract

We present a probabilistic generative model and efficient algorithm to model reciprocity in directed networks. Unlike other methods that address this problem such as exponential random graphs, it assigns latent variables as community memberships to nodes and a reciprocity parameter to the whole network rather than fitting order statistics. It formalizes the assumption that a directed interaction is more likely to occur if an individual has already observed an interaction towards her. It provides a natural framework for relaxing the common assumption in network generative models of conditional independence between edges, and it can be used to perform inference tasks such as predicting the existence of an edge given the observation of an edge in the reverse direction. Inference is performed using an efficient expectation-maximization algorithm that exploits the sparsity of the network, leading to an efficient and scalable implementation. We illustrate these findings by analyzing synthetic and real data, including social networks, academic citations and the Erasmus student exchange program. Our method outperforms others in both predicting edges and generating networks that reflect the reciprocity values observed in real data, while at the same time inferring an underlying community structure. We provide an open-source implementation of the code online.

I  Introduction

Reciprocity in directed networks, i.e., the tendency of a pair of nodes to form mutual connections between each other
 [ wasserman1994social ] , is an important feature of many social relationships. Its impact ranges from affecting the development of exchange and power to determining the emergence of trust and solidarity  [ molm2010structure ,  nowak2005evolution ] . Behavior of this kind has also been found in many kinds of networks that reflect human and institutional interaction, e.g., the world wide web, online dating, interfirm contracts, journal citations and email communication  [ garlaschelli2005 ,  zhao2013user ,  wincent2010quality ,  li2019reciprocity ,  newman2002email ] .

Among the various network modeling approaches, that of probabilistic generative models enable us for a rigorous theoretical foundation within the framework of statistical inference, as well as a flexible incorporation of domain knowledge in the modeling assumptions. Here, we consider a latent variable model, a probabilistic approach that contains latent and observed variables. The latent variables encode hidden patterns in the data, such as community memberships, and determine the probability of ties between nodes. For instance, knowing which communities two nodes belong to helps determine the likelihood of their interaction.

While in some simple cases, community structure may explain the tendency toward reciprocation  [ holland1983stochastic ] , this mechanism may not be enough to capture more complex scenarios. Indeed, many generative models with community structure fail to reproduce the values of reciprocity observed in real networks, as we discuss in more details later. Conversely, several models aimed at capturing reciprocity do not account for community structure  [ holland1981exponential ,  park2004statistical ] . It is reasonable to expect that the mechanism regulating the existence of interactions can be influenced by both patterns of communities and reciprocity. In addition, communities are often interpretable objects and may correspond to functional unit, hence the value of including them in the model formulation.
Incorporating reciprocity as well as community structure into a coherent latent variable model comes with the main challenge of relaxing the conditional independence assumption between edges, a common assumption in generative models to ease mathematical derivations. In addition, this task requires properly capturing conditional probabilities, as we describe later.
Inspired by these insights, we propose a novel probabilistic latent variable approach to model networks that preserves the benefits of generative models, while capturing both community structure and reciprocity.

Models for reciprocity and latent community structure have largely been developed independently of one another, and only a handful of works have hinted at incorporating them into a unique framework.
For instance, Garlaschelli and Loffredo  [ garlaschelli2006 ]  point towards a possible relationship between their model for reciprocity and general hidden variable models.
Most notably, the pair-dependent stochastic block model of Holland et al.  [ holland1983stochastic ] , well explained also by Wasserman and Anderson  [ wasserman1987stochastic ] , holds assumptions similar to ours, in that it models jointly pairs of edges, which they call dyad vectors. While a seminal work, it is, nevertheless, limited to hard membership and unweighted networks; hence the likelihood function that they propose substantially differs from the likelihood represented by our model. One practical aspect of our choice for the likelihood is that parameters’ inference in our model is optimized to fully exploit the sparsity of the dataset and is scalable to large network sizes.

Reciprocity is often modeled by means of exponential random graphs  [ holland1981exponential ,  park2004statistical ,  robins2007introduction ,  squartini2013reciprocity ] , where it is treated as a measured network property that needs to be reproduced (often together with other network properties like the degree) by sampling networks using statistical mechanics principles, e.g., maximum entropy.
The approach presented in this work significantly differs from the previous studies in that we include latent variables, such as community membership, as a mechanism to determine edge formation. However, in the case of exponential random graphs, possible group structures are not given a priori as the latent parameters; instead, they can only be estimated a posteriori on the sampled networks. More broadly, our approach is that of generative models, which incorporate a priori community structure by means of latent variables, and these are inferred from the observed interactions  [ de2017community ,  ball2011efficient ] . However, in these generative models reciprocity is not explicitly included as a mechanism for tie formation, thus these models often fail to reproduce the observed reciprocity values of real networks.
Consequently, a generative method whose latent variables describe both reciprocity and community memberships is needed.

II  Relaxing the conditional independence assumption

A possible explanation for the practical deficiency of generative models with communities to reproduce observed reciprocity values is the common assumption of conditional independence between edges, which makes the problem both analytically and computationally more tractable. This assumption states that the likelihood of a directed tie between two nodes depends only on their community membership (and other possible model parameters), but not on the existence of the reciprocated edge. This might be too strict of an assumption to capture the feature of reciprocity, where it is reasonable to expect that the existence of an edge in one direction should also be conditioned on the existence of an edge in the opposite direction. For instance, if an author

i

𝑖

i

has cited another auth