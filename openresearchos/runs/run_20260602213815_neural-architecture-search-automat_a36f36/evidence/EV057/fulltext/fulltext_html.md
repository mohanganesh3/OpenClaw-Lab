[2507.00589] Quantum Circuit Structure Optimization for Quantum Reinforcement Learning

Quantum Circuit Structure Optimization for Quantum Reinforcement Learning

Seok Bin Son and
Joongheon Kim

This research was supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-RS-2024-00436887) supervised by the IITP (Institute for Information &amp; Communications Technology Planning &amp; Evaluation); and also by IITP grant funded by MSIT (RS-2024-00439803, SW Star Lab).  (Corresponding author: Joongheon Kim)

Department of Electrical and Computer Engineering, Korea University, Seoul, Republic of Korea

E-mails:
 {lydiasb,joongheon}@korea.ac.kr

Abstract

Reinforcement learning (RL) enables agents to learn optimal policies through environmental interaction. However, RL suffers from reduced learning efficiency due to the curse of dimensionality in high-dimensional spaces. Quantum reinforcement learning (QRL) addresses this issue by leveraging superposition and entanglement in quantum computing, allowing efficient handling of high-dimensional problems with fewer resources. QRL combines quantum neural networks (QNNs) with RL, where the parameterized quantum circuit (PQC) acts as the core computational module. The PQC performs linear and nonlinear transformations through gate operations, similar to hidden layers in classical neural networks. Previous QRL studies, however, have used fixed PQC structures based on empirical intuition without verifying their optimality. This paper proposes a QRL-NAS algorithm that integrates quantum neural architecture search (QNAS) to optimize PQC structures within QRL. Experiments demonstrate that QRL-NAS achieves higher rewards than QRL with fixed circuits, validating its effectiveness and practical utility.

Index Terms:

Quantum Neural Architecture Search, Quantum Reinforcement Learning, Neural Architecture Search, Reinforcement Learning

I

Introduction

Reinforcement learning (RL) has achieved remarkable progress across various application domains based on classical neural networks (NN). NN-based RL has been successfully applied to game playing, robot control, autonomous driving, and satellite communication systems  [ 1 ,  2 ,  3 ] . It often surpasses human-level performance even in complex environments. However, existing RL faces fundamental limitations when handling high-dimensional state and action spaces  [ 4 ] . As state and action dimensions increase, the number of learnable parameters grows exponentially, resulting in the curse of dimensionality  [ 5 ] . This exponential growth significantly reduces learning convergence speed and computational efficiency. Moreover, data sparsity in high-dimensional spaces requires massive sample sizes to learn optimal policies  [ 5 ] , imposing substantial temporal and cost constraints on real-world systems.

To overcome these structural limitations, quantum reinforcement learning (QRL) has gained increasing attention  [ 6 ,  7 ,  8 ] . QRL utilizes quantum neural networks (QNNs) to exploit quantum computing properties such as superposition and entanglement  [ 6 ] . As illustrated in Fig.  1  , QNNs consist of three components: encoder, parameterized quantum circuit (PQC), and measurement [ 9 ] . The encoder converts classical inputs into quantum states for processing within the quantum circuit. The PQC performs linear and nonlinear transformations similar to hidden layers in classical NNs. The measurement stage converts quantum states into classical outputs for verification. By using QNNs, QRL addresses the curse of dimensionality and sample inefficiency inherent in NN-based RL. For example, QNNs can compressively represent high-dimensional data using a small number of qubits, reducing computational costs. Furthermore, entanglement enables QNNs to achieve high performance with fewer training samples  [ 8 ] . Due to these advantages, QRL is emerging as a powerful alternative in complex environments such as high-dimensional satellite communication networks, multi-UAV cooperative systems, and smart factories  [ 6 ,  7 ,  8 ] .

Figure 1:  The structure of quantum neural network.

However, important limitations remain unresolved in current QRL research. One critical issue concerns the optimality of the PQC structure, which plays a key role in determining QRL performance. Most existing QRL studies have relied on researchers’ empirical intuition or conventional circuit patterns when designing PQCs. Typically, combinations of a few

R  ​  X

,

R  ​  Y

RX,RY

, and

R  ​  Z

RZ

rotation gates with entanglement gates such as

C  ​  X

CX

are adopted. Repeatedly stacking single-type PQC blocks is also a common practice. Nevertheless, such passive and fixed designs do not reflect the complexity and variability of real-world environments. This design limitation inherently restricts the ability to ensure optimal circuit structures for specific problems. The PQC structure directly influences learning stability, convergence speed, and the final policy performance of agents. Furthermore, gate arrangement and parameter configuration determine the expressiveness and function approximation capability of quantum circuits. Inefficient structures also increase the accumulation of quantum noise. As a result, suboptimal PQC designs degrade learning performance and sharply raise computational costs.

To maximize the potential of QRL, recent studies have recognized the need for automatic PQC optimization beyond empirical design. Specifically, it is necessary to automatically search and select gate types, placements, and depths based on environmental characteristics and learning objectives. This enables the design of optimal circuits that balance expressiveness and computational efficiency. This paper proposes a methodology that introduces neural architecture search (NAS) into PQC design for QRL to address these limitations. NAS is a technique widely used in deep learning to automatically explore and optimize model structures based on data  [ 10 ,  11 ,  12 ] . It replaces manual design reliant on researcher intuition and significantly improves model performance and efficiency. The core idea is to apply NAS’s exploration capabilities to quantum circuit design problems. This enables interactive exploration of gate combinations and layout structures within PQCs in real time in RL environments. Ultimately, this approach derives optimal circuits that minimize unnecessary or redundant gate usage while maximizing computational efficiency and policy performance. By adopting this methodology, suboptimal PQC design issues in existing QRL can be addressed. This allows agents to achieve higher cumulative rewards and enhanced learning performance.

The main contributions of this paper are as follow,

•

This paper proposes an approach that integrates NAS into QRL to overcome the limitations of existing empirically designed QRL methods. By automatically exploring and optimizing PQC structures, this approach presents a new research direction for designing quantum circuits optimized for specific environments.

•

A comprehensive set of gate candidates was constructed, including single-qubit gates (i.e., U3, RX, RY, RZ) and two-qubit gates (i.e., CU3, SWAP, CX, CY, CZ), to maximize the expressiveness and versatility of NAS exploration. This enabled efficient exploration of diverse quantum gate combinations and layout structures.

•

Comparison of the proposed QRL-NAS framework with existing QRL using fixed PQC structures showed superior performance in policy cumulative rewards. These results empirically demonstrate the importance of PQC structure optimization in enhancing QRL performance and validate NAS as a practically effective approach in QRL design.

II

Preliminaries

II-A

Neural Architecture Search

Traditionally, NN structures were manually designed by machine learning experts and repeatedly tested to verify their performan