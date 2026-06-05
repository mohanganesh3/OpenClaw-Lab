[2202.01899] DeepQMLP: A Scalable Quantum-Classical Hybrid Deep Neural Network Architecture for Classification

DeepQMLP: A Scalable Quantum-Classical Hybrid Deep

Neural Network Architecture for Classification

Mahabubul Alam

Department of Electrical Engineering

Penn State University

mxa890@psu.edu

Swaroop Ghosh

Department of Electrical Engineering

Penn State University

szg212@psu.edu

Abstract

Quantum machine learning (QML) is promising for potential speedups and improvements in conventional machine learning (ML) tasks (e.g., classification/regression). The search for ideal QML models is an active research field. This includes identification of efficient classical-to-quantum data encoding scheme, construction of parametric quantum circuits (PQC) with optimal expressivity and entanglement capability, and efficient output decoding scheme to minimize the required number of measurements, to name a few. However, most of the empirical/numerical studies lack a clear path towards scalability. Any potential benefit observed in a simulated environment may diminish in practical applications due to the limitations of noisy quantum hardware (e.g., under decoherence, gate-errors, and crosstalk). We present a scalable quantum-classical hybrid deep neural network (DeepQMLP) architecture inspired by classical deep neural network architectures. In DeepQMLP, stacked shallow Quantum Neural Network (QNN) models mimic the hidden layers of a classical feed-forward multi-layer perceptron network. Each QNN layer produces a new and potentially rich representation of the input data for the next layer. This new representation can be tuned by the parameters of the circuit. Shallow QNN models experience less decoherence, gate errors, etc. which make them (and the network) more resilient to quantum noise. We present numerical studies on a variety of classification problems to show the trainability of DeepQMLP. We also show that DeepQMLP performs reasonably well on unseen data and exhibits greater resilience to noise over QNN models that use a deep quantum circuit. DeepQMLP provided up to 25.3% lower loss and 7.92% higher accuracy during inference under noise than QMLP.

I

Introduction

Quantum computing is one of the major transformative technologies. Although quantum computing is still in a nascent stage, the community is seeking computational advantages with quantum computers (i.e., quantum supremacy) for practical applications. Recently, Google claimed quantum supremacy with a 53-qubit quantum processor to complete a computational task in 200 seconds that might take 10K years  [ 1 ]  (later rectified to 2.5 days  [ 2 ] ) on the state-of-the-art supercomputers. This experiment was a significant milestone for quantum computing even though the computational task used for this experiment had no practical value.

Quantum machine learning (QML) is a promising application domain to archive quantum advantage with noisy quantum computers in the near term. Numerous QML models built upon parametric quantum circuits (PQC), also referred to as quantum neural networks (QNN), are already available in the literature  [ 3 ,  4 ,  5 ] . A PQC is a quantum circuit with tunable parameterized gates as shown in Fig.

1  (b) (w1,w2,… are the tunable parameters). A PQC may generate various output states based on the values of these parameters. QNN models are claimed to be more expressive compared to the classical neural networks  [ 6 ,  7 ,  8 ] . In other words, QNN models have a higher capability to approximate the desired functionality (e.g., classifying data samples) compared to the classical models of a similar scale (e.g., with the same number of tunable parameters/weights). Deep Neural Networks (DNN) have experienced huge success in machine learning (ML) in the past decade (essentially superseding most other models) because they are powerful function approximators. With an even higher ability to approximate functions, QNN holds great potential for the future.

A conventional QNN architecture is shown in Fig.

1  (b). In a typical QNN model, the input data is encoded into a quantum state using a suitable encoding scheme (e.g., angle encoding, amplitude encoding, etc.)  [ 9 ] . The encoding is followed by layers of PQC with tunable parameters (w1,w2,… in Fig.

1  (b)). These parameters are analogous to weights in a classical neural network. In the end, the output quantum state of the PQC is measured (/sampled) on the appropriate basis (e.g., the default Pauli-Z measurement basis in IBM quantum computers  [ 10 ] ). The sampling process is repeated many times with the same parameters. A cost function is derived from the measurements. A classical optimizer (e.g., gradient-descent) updates the parameter values to minimize the cost.

Figure 1:  Conventional Multi-Layer Perceptron (MLP) and Quantum Neural Network (QNN) architectures (a)-(b), alongside the proposed Quantum Multi-Layer Perceptrons (QMLP) and Deep Quantum Multi-Layer Perceptrons (DeepQMLP) architectures (c)-(d). While QMLP uses a deeper parameterized circuit to accommodate larger search space, DeepQMLP uses multiple shallow-depth circuits. Shallower circuits provide greater robustness against quantum noises (less accumulation of gate errors and decoherence over each circuits) to DeepQMLP over QMLP.

The choice of the PQC can have a significant impact on the performance (e.g., trainability) of a QNN model. For instance, deep PQC with lots of parameters may be desirable for learning but may experience vanishing gradient problems (also referred to as barren plateaus) making it harder for the gradient-based optimizers to navigate through the solution space  [ 11 ] . Moreover, quantum computers are plagued with various noise sources such as gate error, readout error, decoherence, and crosstalk  [ 12 ] . The output quantum state can be random (i.e., meaningless) if the noise accumulation is high. A large amount of noise can also induce a barren plateau in the QNN solution space  [ 13 ] .

Shallow-depth circuits are preferred for QNN to avoid the aforementioned issues  [ 11 ,  14 ] . However, shallow-depth circuits may often be unable to approximate complex functionality (similar to shallow classical neural networks with small number of parameters).

In this article, we propose two new quantum-classical hybrid deep neural network architectures: Quantum Multi-Layer Perceptrons (QMLP) and DeepQMLP to partially address the aforementioned issues. Both architectures are inspired by conventional Multi-Layer Perceptron (MLP) networks used in deep learning. In MLP, multiple layers of neurons are used to define and search through a solution space for a given ML task (Fig.

1  (a)). Neurons of successive layers are connected through trainable weights. The first and the last layers of MLP are commonly referred to as input and output layers. The internal layers are referred to as hidden layers. Typically, MLP models contain multiple hidden layers. In QMLP, the hidden layer of an MLP is mimicked by a QNN layer as shown in Fig.

1  (c). The QNN takes a quantum encoded representation of the classical data and produces an output representation (e.g., Pauli-Z expectation values of the qubits) which is fed to the classical output layer. The network can be trained with any conventional loss function. However, in this work, we only use cross-entropy loss. In DeepQMLP, multiple shallow-depth QNN models (two used in this work) are used as hidden layers of an MLP (Fig.

1  (d)). Each layer produces a new representation for the next layer. For example, the qubit expectation values of the first hidden layer in Fig.

1  (d) are used as the inputs to the second hidden layer.

QMLP uses a deep QNN alongside a classical dense output layer to exploit the higher expressive power of QNN. To accommodate a larger search space, the quantum hidden layer in QMLP needs a deep parameterized circuit. However, deep circuits are more error-prone. DeepQMLP addresses the is