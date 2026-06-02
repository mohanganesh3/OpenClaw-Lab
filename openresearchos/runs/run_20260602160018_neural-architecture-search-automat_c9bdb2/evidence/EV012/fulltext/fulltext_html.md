[2110.06018] On the Security Risks of AutoML

On the Security Risks of AutoML

Ren Pang
 Pennsylvania State University

Zhaohan Xi
 Pennsylvania State University

Shouling Ji
 Zhejiang University

Xiapu Luo
 The Hong Kong Polytechnic University

Ting Wang
 Pennsylvania State University

Ren Pang †

Zhaohan Xi †

Shouling Ji ‡

Xiapu Luo ⋆

Ting Wang †

† Pennsylvania State University, {rbp5354, zxx5113, ting}@psu.edu

‡ Zhejiang University, sji@zju.edu.cn

⋆ Hong Kong Polytechnic University, csxluo@comp.polyu.edu.hk

Automation is good, so long as you know exactly where to put the machine.

– Eliyahu Goldratt

Abstract

Neural Architecture Search (NAS) represents an emerging machine learning (ML) paradigm that automatically searches for models tailored to given tasks, which greatly simplifies the development of ML systems and propels the trend of ML democratization. Yet, little is known about the potential security risks incurred by NAS, which is concerning given the increasing use of NAS-generated models in critical domains.

This work represents a solid initial step towards bridging the gap. Through an extensive empirical study of 10 popular NAS methods, we show that compared with their manually designed counterparts, NAS-generated models tend to suffer greater vulnerability to various malicious attacks ( e.g. , adversarial evasion, model poisoning, and functionality stealing). Further, with both empirical and analytical evidence, we provide possible explanations for such phenomena: given the prohibitive search space and training cost, most NAS methods favor models that converge fast at early training stages; this preference results in architectural properties associated with attack vulnerability ( e.g. , high loss smoothness and low gradient variance). Our findings not only reveal the relationships between model characteristics and attack vulnerability but also suggest the inherent connections underlying different attacks. Finally, we discuss potential remedies to mitigate such drawbacks, including increasing cell depth and suppressing skip connects, which lead to several promising research directions.

1  Introduction

Automated Machine Learning (AutoML) represents a new paradigm of applying ML techniques in real-world settings. For given tasks, AutoML automates the pipeline from raw data to deployable ML models, covering model design [ 18 ] , optimizer selection [ 37 ] , and
parameter tuning [ 1 ] . The use of AutoML greatly simplifies the development of ML systems and propels the trend of ML democratization. Many IT giants have unveiled their AutoML frameworks, such as Microsoft Azure AutoML, Google Cloud AutoML, and IBM Watson AutoAI.

Figure 1 :

Cell-based neural architecture search.

In this paper, we focus on one primary task of AutoML, Neural Architecture Search (NAS), which aims to find performant deep neural network (DNN) architectures  1

1  1 In the following, when the context is clear, we use the terms of “model” and “architecture” exchangeably.

tailored to given tasks. For instance, as illustrated in Figure

1  , cell-based NAS constructs a model by repeating the motif of a cell structure following a pre-specified template, wherein a cell is a topological combination of operations ( e.g. ,

3  ×  3

3  3

3\times 3

convolution). With respect to the given task, NAS optimizes both the topological structure and the operation assignment. It is shown that in many tasks, NAS finds models that remarkably outperform manually designed ones [ 46 ,  39 ,  35 ,  11 ] .

Yet, in contrast to the intensive research on improving the capabilities of NAS, its security implications are fairly unexplored. As ML systems are becoming the new targets for malicious attacks [ 6 ] , the lack of understanding about the potential risks of NAS is highly concerning, given its surging popularity in security-sensitive applications. Specifically,

RQ1 –  Does NAS introduce new weaknesses, compared with the conventional ML practice?

RQ2 –  If so, what are the possible root causes of such vulnerability?

RQ3 –  Further, how would ML practitioners mitigate such drawbacks in designing and operating NAS?

The answers to these key questions are crucial for the use of NAS in security-sensitive domains ( e.g. , cyber-security, finance, and healthcare).

Our work –  This work represents a solid initial step towards answering such questions.

A1 - First, through an extensive empirical study of 10 representative NAS methods, we show that compared with their manually designed counterparts, NAS-generated models tend to suffer greater vulnerability to various malicious manipulations such as adversarial evasion [ 42 ,  8 ] , model poisoning [ 5 ] , backdoor injection [ 23 ,  40 ] , functionality stealing [ 44 ] , and label-only membership inference [ 13 ] . The findings suggest that NAS is likely to incur larger attack surfaces, compared with the conventional ML practice.

A2 - Further, with both empirical and analytical evidence, we provide possible explanations for the above observations. Intuitively, due to the prohibitive search space and training cost, NAS tends to prematurely evaluate the quality of candidate models before their convergence. This practice favors models that converge fast at early training stages, resulting in architectural properties that facilitate various attacks ( e.g. , high loss smoothness and low gradient variance). Our analysis not only reveals the relationships between model characteristics and attack vulnerability but also suggests the inherent connections underlying different attacks.

A3 - Finally, we discuss potential remedies. Besides post-NAS mitigation ( e.g. , adversarial training [ 42 ] ), we explore in-NAS strategies that build attack robustness into the NAS process, such as increasing cell depth and suppressing skip connects. We show that while such strategies mitigate the vulnerability to a certain extent, they tend to incur non-trivial costs of search efficiency and model performance. We deem understanding the fundamental trade-off between model performance, attack robustness and search efficiency as an important topic for further investigation.

Contributions –  To our best knowledge, this work represents the first study on the potential risks incurred by NAS (and AutoML in general) and reveals its profound security implications. Our contributions are summarized as follows.

– We demonstrate that compared with conventional ML practice, NAS tends to introduce larger attack surfaces with respect to a variety of attacks, which raises severe concerns about the use of NAS in security-sensitive domains.

– We provide possible explanations for such vulnerability, which reveal the relationships between architectural properties ( i.e. , gradient smoothness and gradient variance) and attack vulnerability. Our analysis also hints at the inherent connections underlying different attacks.

– We discuss possible mitigation to improve the robustness of NAS-generated models under both in-situ and ex-situ settings. This discussion suggests the necessity of improving the current practice of designing and operating NAS, pointing to several research directions.

Roadmap –  The remainder of the paper proceeds as follows. §

2

introduces fundamental concepts and assumptions; §

3

conducts an empirical study comparing the vulnerability of NAS-generated and manually designed models; §

4

provides possible explanations for the observations; §

5

discusses potential mitigation and the limitations of this work; §

6

surveys relevant literature; the paper is concluded in §

7  .

2  Preliminaries

We first introduce a set of key concepts and assumptions.
Table

5

summarizes the important notations.

2.1  Neural Architecture Search

Deep neural networks (DNNs) represent a class of ML models to learn high-level abstractions of complex data. We assume a predictive setting, in which a DNN

f

\scaleobj

​  0.8  ​  θ

subscript  𝑓

\scaleobj

0.8  𝜃

{f