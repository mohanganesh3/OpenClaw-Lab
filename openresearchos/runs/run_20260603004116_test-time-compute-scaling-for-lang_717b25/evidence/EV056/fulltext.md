[2005.09526] In-memory Implementation of On-chip Trainable and Scalable ANN for AI/ML Applications

In-memory Implementation of On-chip Trainable and Scalable ANN for AI/ML Applications

Abhash Kumar,
Jawar Singh,
Sai Manohar Beeraka,
and Bharat Gupta

A Kumar, S M Beeraka, and B Gupta are with the Department
of Electronics and Communication Engineering, National Institute of Technology, Patna, Bihar, INDIA. abhash2205@gmail.comJ Singh is with Electrical Engineering Department, Indian Institute of Technology Patna, Bihar, INDIA. dr.jawar@gmail.com

Abstract

Traditional von Neumann architecture based processors become inefficient in terms of energy and throughput as they involve separate processing and memory units, also known as  memory wall . The memory wall problem is further exacerbated when massive parallelism and frequent data movement are required between processing and memory units for real-time implementation of artificial neural network (ANN) that enables many intelligent applications. One of the most promising approach to address the memory wall problem is to carry out computations inside the memory core itself that enhances the memory bandwidth and energy efficiency for extensive computations. This paper presents an in-memory computing architecture for ANN enabling artificial intelligence (AI) and machine learning (ML) applications. The proposed architecture utilizes deep in-memory architecture based on standard six transistor (6T) static random access memory (SRAM) core for the implementation of a multi-layered perceptron. Our novel on-chip training and inference in-memory architecture reduces energy cost and enhances throughput by simultaneously accessing the multiple rows of SRAM array per precharge cycle and eliminating the frequent access of data. The proposed architecture realizes backpropagation which is the keystone during the network training using newly proposed different building blocks such as weight updation, analog multiplication, error calculation, signed analog to digital conversion, and other necessary signal control units. The proposed architecture was trained and tested on the IRIS dataset which exhibits

≈  46  ×

\approx 46\times

more energy efficient per MAC (multiply and accumulate) operation compared to earlier classifiers.

Index Terms:

In-memory computing, SRAM, artificial neural network, artificial intelligence, machine learning, classification.

I

Introduction

Artificial intelligence (AI) and machine learning (ML) algorithms are ubiquitous and integral part of contemporary computing devices, and significantly changing the way we live and interact with the world around us. Most of these computing systems are based on von Neumann architecture that involves separate processing and memory units where data need to be shuttled back and forth frequently between the processing and the memory units [ 1 ] . Therefore, significant amount of the energy and time are consumed during data movement rather than actual computing, and this problem further exacerbated due to massive parallelism and data centric tasks such as decision making, object recognition, speech and video processing. This calls for a radical departure from the orthodox von Neumann approach to an unorthodox non-von Neumann computing architectures to carry out computations within the memory core itself, referred as to in-memory computing. Recently, hardware implementation of AI/ML algorithms based on in-memory computing has attracted huge attention because of unmatched computing performance and energy efficiency. In-memory computation overcomes the problem of frequent data movement between processing and memory units in the traditional von Neumann architecture based processors by carrying out computations within the memory core using its periphery circuitry.

Artificial Neural Network (ANN) is one of the most widely used tool for AI and ML based applications due to its very good resemblance and mimicking properties of human brain. It is used in a wide variety of AI/ML applications because of its self-learning ability to produce output that is not just limited to the input provided to them. Most of these AI/ML algorithms are software based and poses energy and delay optimization challenges for real time hardware implementation. The major limitations for hardware based solutions are large number of interconnections, massive parallelism, and time consuming calculations that requires huge data for training the networks and related algorithms. So, in-memory computation for such networks is one of the most preferred and efficient way for hardware realization of such complex networks. Researchers have come up with different in-memory implementations of popular machine learning networks and algorithms such as Convolution Neural Networks (CNNs) [ 2 ,  3 ,  4 ] , Deep Neural Networks (DNNs) [ 5 ,  6 ]  and machine learning classifiers  [ 7 ,  8 ] . A large number of machine learning network architecture uses modified form of artificial neural networks such as in recurrent neural network (RNN), the fully connected layer of CNN, or in Deep Neural Networks (DNNs) which are the foundations of deep learning. There is a remarkable improvement in learning and predicting the complex pattern of a large data set which otherwise would have been very difficult even for the Graphic Processing Units (GPUs) that still require on-chip or off-chip memory access despite parallel computations for inference and training of these algorithms.

Most of the AI/ML algorithms require processing of a large data sets and the energy cost involved is mostly governed by the frequent access of memory [ 9 ] , especially for dense networks such as DNNs [ 10 ,  11 ] . In DianNao [ 11 ]  and Eyeriss [ 10 ] , the data reuse have been an efficient and highly effective solutions for saving energy, however, it still resulted in frequent on-chip memory access leading to 35% to 45% of total energy dissipation. Other techniques, such as reducing the precision of parameters to even 1-b during inference [ 12 ,  13 ,  14 ]  can address the energy and latency issues to a some extent but they will lead to accuracy trade-off. Further, implementation of architectures in digital domain using low power circuits have been employed such as power gating technique for speech recognition [ 5 ] , RAZOR [ 15 ]  for Internet of Things (IoT) applications [ 6 ] , and architectural designs such as dynamic-voltage accuracy frequency scalable CNN processor [ 3 ]  have been introduced earlier to reduce energy cost. They exploits the advantages of scalability and programmability of implementations in digital domain but they miss out the opportunities that lies in analog domain for implementation of AI/ML algorithms. This is due to the unique data flow of ANN during feedforward and backpropagation that can be exploited to design energy efficient and high throughput in-memory ANN architectures in analog domain.

Exploiting the opportunities available in analog domain for realizing in-memory AI/ML algorithms to reduce energy and delay cost, an in-memory inference processor based on  deep in-memory architecture  (DIMA) [ 16 ,  17 ]  have been presented earlier. The DIMA stores binary data in a column-major format as opposed to the row-major format employed in traditional Static Random Access Memory (SRAM) array organization. It reduces energy cost by simultaneously accessing multiple rows of the standard six-transistor (6T) SRAM cells per precharge cycle through the application of modulated pulse width signals to the word lines (WLs), and thus increases the throughput. Previously, DIMA is also used for AI/ML algorithms such as CNNs [ 2 ]  and its versatility was also established by mapping the ML algorithms for template matching [ 18 ] , and architectures of sparse distributed memory [ 19 ,  20 ] . A multi-functional in-memory inference processor [ 16 ]  based on DIMA have been presented earlier which achieves

53  ×

53\times

red