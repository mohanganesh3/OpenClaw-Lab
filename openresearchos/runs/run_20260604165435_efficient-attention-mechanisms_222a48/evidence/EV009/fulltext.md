[2110.04485] Application of quantum computing to a linear non-Gaussian acyclic model for novel medical knowledge discovery

Application of quantum computing to a linear non-Gaussian acyclic model for novel medical knowledge discovery

Hideaki Kawaguchi

Quantum Computing Center

Keio University
 3 - 14 - 1 Hiyoshi, Kohoku, Yokohama 223-8522, Japan

hikawaguchi@keio.jp

Abstract

Recently, with the digitalization of medicine, the utilization of real-world medical data collected from clinical sites has been attracting attention. In this study, quantum computing was applied to a linear non-Gaussian acyclic model to discover causal relationships from real-world medical data alone. Specifically, the independence measure of DirectLiNGAM, a causal discovery algorithm, was calculated using the quantum kernel and its accuracy on real-world medical data was verified. When DirectLiNGAM with the quantum kernel (qLiNGAM) was applied to real-world medical data, a case was confirmed in which the causal structure could be correctly estimated when the amount of data was small, which was not possible with existing methods. Furthermore, qLiNGAM was implemented on real quantum hardware in an experiment using IBMQ. It is suggested that qLiNGAM may be able to discover new medical knowledge and contribute to the solution of medical problems, even when only a small amount of data is available.

Introduction

The utilization of medical data, which is increasing with the digitalization of medicine, has been attracting attention

[ 1 ] . As a method of utilizing medical data so far, clinical trials such as randomized controlled trials have been conducted to establish scientific evidence. However, it has been reported that there are problems associated with clinical trials, including limited number of subjects owing to the strict selection/exclusion criteria, as well as time, cost, and ethical restrictions required for their implementation  [ 2 ] . By contrast, real-world medical data, which are secondary data collected from clinical environments, have been attracting attention as data that are rapidly increasing with the digitalization of medicine, for example, disease registries, electronic medical record data, and claim data containing details of medical procedures  [ 2 ,  3 ] . Real-world medical data can be collected from a wide range of patients with lower cost, time, and ethical constraints than those of clinical trials.

Currently, the main method for analyzing real-world medical data is “causal inference,” in which the direction of a certain causal relationship is determined and then the causal effect, which represents the strength of the causal relationship, is estimated. As long as the direction of the causal relationship is known in advance, the methods for estimating causal effects have continued to evolve significantly in recent years with the improvement in statistical methods  [ 4 ,  5 ,  6 ,  7 ,  8 ] . However, in the medical field where there are many uncertain cases, it is difficult to determine the direction of causality in advance in many situations. In these situations, “causal discovery,” which detects causal relationships from data, is important for the discovery of new knowledge  [ 9 ] . Particularly in the case of real-world medical data, the number of variables to be handled is so large that it is not realistic to exhaustively confirm causal relationships in advance, and thus, the importance of causal discovery methods is expected to grow stronger.

The causal discovery algorithm is a method for identifying causal graphs that represent causal relationships among variables by determining the direction of causal relationships according to the data, with some assumptions. In the conventional method, there is a limitation in that the structure of the causal graph cannot be uniquely identified  [ 10 ,  11 ] . The linear non-Gaussian acyclic model (LiNGAM) is a causal discovery method that has recently attracted attention  [ 12 ] . LiNGAM uniquely identifies the structure of causal graphs by making non-Gaussian assumptions about the data  [ 12 ] . More precisely, it assumes that the error variables are non-Gaussian and independent, that each variable has a linear relationship, and that the causal graph is non-cyclic. LiNGAM is an algorithm that uses the independence assumed for the error variables to identify the parameters that determine the causal graph structure. Although LiNGAM has the advantage of being able to uniquely identify the causal structure, it has been pointed out that it cannot properly estimate the causal structure depending on the independence measure used to estimate the parameters  [ 13 ] . Among the proposed independence measures, the method using the kernel method has been attracting attention  [ 13 ] . It is known that the kernel method can determine independence with high accuracy and can estimate the causal structure with high accuracy even when the error variable, which is assumed to be non-Gaussian, is actually close to a Gaussian distribution or when there are outliers in the data  [ 13 ] . In the literature  [ 13 ] , independence measures were designed using Gaussian kernels, but an improvement is to seek kernels that can construct independence measures with even better accuracy.

Recently, quantum kernels, which apply quantum computing to kernel methods, have been attracting attention  [ 14 ,  15 ,  16 ] . In conventional (or classical) computers, the state of a bit is either 0 or 1; however, quantum computers use quantum bits, or qubits, which can take superpositions of 0 and 1, and quantum mechanical principles such as quantum entanglement for information processing. Because qubits are sensitive to noise and the superposition state is broken after a certain period of time, a fault-tolerant quantum computer (FTQC) that can correct errors caused by noise is required; however, there are still many technical and essential problems to be solved for realizing FTQC. By contrast, a quantum computer called noisy intermediate-scale quantum (NISQ) computer, which does not have an error correction function, has been realized in the last few years  [ 17 ,  18 ] . For an experiment on quantum kernels using a NISQ computer, the work of applying quantum kernels to support vector machine frameworks is known  [ 15 ] . Kernel methods embed data in a high-dimensional feature space to facilitate analysis, whereas quantum kernels use quantum circuits to construct the kernel. With the use of the superposition state and of the information in the high-dimensional feature space, it is expected that quantum computers can efficiently construct functions that are difficult to represent with conventional computers.

Quantum computing and kernel methods can both be viewed as efficient ways to perform computations in the Hilbert space  [ 14 ] . This commonality is the focus of this study, and it is proposed that the causal discovery algorithm can be improved by using quantum kernels for the independence measure in the LiNGAM using kernel methods. To the best of my knowledge, this is the first study to apply quantum computing to LiNGAM. The causal discovery algorithm using quantum kernels is applied to real-world medical data and its performance is examined to determine whether it can contribute to the detection of new medical knowledge. This study shows the potential usefulness of utilizing quantum computing as a way to improve the performance of LiNGAM.

Method

Causal discovery algorithm

DirectLiNGAM

There are two main types of LiNGAM: ICA-LiNGAM  [ 12 ] , which uses independent component analysis, and DirectLiNGAM  [ 19 ] , which is based on regression analysis. On the basis of the findings of existing literature  [ 13 ] , DirectLiNGAM was more accurate for LiNGAM using a Gaussian kernel, and therefore, DirectLiNGAM was used as the basis for this study.

In DirectLiNGAM with a total of  p  variables, a single regression
analysis is performed  p -1