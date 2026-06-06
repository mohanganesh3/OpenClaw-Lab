[2010.01118] Gaussian Process Molecular Property Prediction with FlowMO

Gaussian Process Molecular Property Prediction with FlowMO

Henry B. Moss

STOR-i Centre for Doctoral Training

Lancaster University, UK

h.moss@lancaster.ac.uk

&amp;Ryan-Rhys Griffiths ∗

Department of Physics

University of Cambridge, UK

rrg27@cam.ac.uk

Equal contribution

Abstract

We present FlowMO: an open-source Python library for molecular property prediction with Gaussian Processes. Built upon GPflow and RDKit, FlowMO enables the user to make predictions with well-calibrated uncertainty estimates, an output central to active learning and molecular design applications. Gaussian Processes are particularly attractive for modelling small molecular datasets, a characteristic of many real-world virtual screening campaigns where high-quality experimental data is scarce. Computational experiments across three small datasets demonstrate comparable predictive performance to deep learning methods but with superior uncertainty calibration.

1  Introduction

In the early stages of exploring a new class of drug molecules or molecular material, there is often only a small quantity of high-quality experimental data available  [Thawani et al.,  2020 , Griffiths et al.,  2018 ] . In contrast to the big data regime of established molecule classes, where much of the information regarding performance has already been acquired, it is the small data regime which holds most promise for techniques such as Bayesian optimization  [Gómez-Bombarelli et al.,  2018 , Korovina et al.,  2020 , Griffiths and Hernández-Lobato,  2020 , Tripp et al.,  2020 , Moss et al.,  2020a ]  and active learning  [Zhang and Lee,  2019 ]  to expedite the rate at which novel and performant molecules are discovered.

The current trend in molecular machine learning research has been to leverage Bayesian deep learning to provide the uncertainty estimates which underpin active learning and Bayesian optimization methods  [Ryu et al.,  2019 , Zhang and Lee,  2019 , Hwang et al.,  2020 , Scalia et al.,  2020 ] . Deep learning methods, however, are often not the model of choice for small datasets. In fact, leading experts on deep learning have expressed a preference for Gaussian Processes (GPs)  [Rasmussen,  2004 ]  for small datasets  [Bengio,  2011 ] . Furthermore, Bayesian deep learning methods rely on approximate inference in order to produce uncertainty estimates. In contrast, GPs admit exact inference at the expense of computationally prohibitive cost for very large datasets. In this paper, we compare calibration of the uncertainty estimates of GPs against a popular Bayesian deep learning method as well as against the recently-introduced attentive neural process  [Kim et al.,  2019 ] .

Applying GPs to molecules in non-trivial, as the vast majority of existing applications of GPs assume inputs of low and fixed dimension. This assumption is not satisfied for the popular molecular representations of fingerprints  [Rogers and Hahn,  2010 ]  and SMILES strings  [Weininger,  1988 ] , sparse high-dimensional bit vectors and strings of variable length, respectively. To build GPs over molecules, FlowMO provides GPU-supported implementations of the Tanimoto and string kernels, providing a user-friendly way to make probabilistic predictions, whilst avoiding the expensive architecture tuning and model optimization often required to find effective deep learning models.

Our primary contributions can be summarized as follows. (1) We propose a GP framework for molecular property prediction using Tanimoto and string kernels. (2) We provide an open-source Python implementation built upon GPflow [De G. Matthews et al.,  2017 ]  and RDKit  [Landrum,  2013 ] , addressing the absence of GP support in popular libraries such as DeepChem  [Ramsundar et al.,  2019 ] , DGL-LifeSci  [Wang et al.,  2020 ]  and ASAP  Cheng et al. [ 2020 ] . (3) We compare FlowMO with established baselines across three benchmark datasets with plans for a more extensive comparison of uncertainty calibration.

2  Molecular Representations

Figure 1 :

SMILES strings for structurally similar molecules have local differences (red) but common non-contiguous sub-sequences (black).

Figure 2 :

Fingerprints provide high-dimensional binary representations of molecules by encoding fragments whose size is dictated by a pre-specified bond radius.

To apply GPs to molecules, we need a meaningful way to represent molecules. FlowMO currently supports two popular representations: SMILES and ECFP fingerprints.

SMILES strings . The Simplified Molecular-Input Line-Entry System (SMILES)  [Weininger,  1988 ]  is a procedure for expressing molecules as character strings. Each SMILES string is assembled by traversing the molecule’s chemical graph. Two SMILES strings for structurally similar pyridine derivatives are presented in Figure

2  . The alphabet of SMILES strings contains letters representing aliphatic and aromatic atoms

(  B  ,  C  ,  N  ,  O  ,  S  ,  P  ,  F  ,

C  ​  l

,

B  ​  r

,  I  ,  b  ,  c  ,  n  ,  o  ,  s  ,  p  )

𝐵  𝐶  𝑁  𝑂  𝑆  𝑃  𝐹

𝐶  𝑙

𝐵  𝑟

𝐼  𝑏  𝑐  𝑛  𝑜  𝑠  𝑝

(B,C,N,O,S,P,F,Cl,Br,I,b,c,n,o,s,p)

, parentheses and integers, as well as additional ASCII symbols representing bonds and the presence of rings

(  −  ,  =  ,  #  ,  &amp;  ,  /  ,  ∖  ,  .  ,  %  )

(-,=,\#,\&amp;,/,\setminus,.,\%)

.

ECFP fingerprints . ECFP fingerprints are high-dimensional bit vector representations of molecules  [Christie et al.,  1993 ]  designed to search chemical databases and in the context of kernel-based classification have been used in conjunction with support vector machines  [Ralaivola et al.,  2005 ] . In this work, we consider extended-connectivity (ECFP) fingerprints  [Rogers and Hahn,  2010 ]  as generated by the RDKit library  [Landrum,  2013 ] . ECFP fingerprints are calculated with an iterative procedure. First, each atom is represented in terms of its local properties, for example its atomic mass and valency. These local representations are then updated based on the representations of its neighbors, iteratively building representations of all fragments containing as many atoms as the chosen bond radius of the fingerprint. Finally, duplicates structures are removed and the remaining representations are hashed into a unique bit vector (see Figure

2  ). For our experiments, we use 2048-bit ECFP fingerprints with a bond radius of 3.

3  Molecular Property Prediction with Gaussian Processes

Given a training set of (possibly noisy) experimentally-determined molecular properties and a kernel function

k

𝑘

k

measuring intermolecular similarity, Gaussian processes (See  Rasmussen [ 2004 ]  for a comprehensive introduction) provide tractable Gaussian predictive distributions for the properties of any out-of-sample target molecule. In FlowMO, we provide a kernel for each of our supported molecular representations (as introduced above): a Sub-sequence String Kernel (SSK) and a Tanimto Kernel (TK) suitable for measuring the similarity between SMILES strings of varying length and high-dimensional binary fingerprint vectors, respectively.

The computational complexity of training and inference for the GP are

O  ​

(

n  3

+

n  2

​  C

)

𝑂

superscript  𝑛  3

superscript  𝑛  2

𝐶

O(n^{3}+n^{2}C)

and

O  ​

(

n  ​  C

)

𝑂

𝑛  𝐶

O(nC)

, where

n

𝑛

n

is the number of training molecules and

C

𝐶

C

is the cost of a single kernel evaluation. Although there exist many methods that reduce the computational complexity of GPs (for example  Hensman et al. [ 2013 ] ), they are designed with smooth input spaces in mind and do not yet support molecular data. In practice, we found standard GPs to be adequate for the datasets considered in this paper, and furthermore they can be comfortably applied to larger data sets using FlowMO’s GPU support.

String Kernels . Sub-sequence String Kernels  [Lodhi et al.,  2002 , Cancedda et al.,  200