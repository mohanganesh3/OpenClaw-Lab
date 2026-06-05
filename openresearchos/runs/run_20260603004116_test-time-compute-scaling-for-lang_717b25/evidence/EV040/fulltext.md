[2501.18894] Nonlinear Inference Capacity of Fiber-Optical Extreme Learning Machines

Nonlinear Inference Capacity of Fiber-Optical Extreme Learning Machines

Sobhi Saeed

Leibniz-Institute of Photonic Technology, Albert-Einstein-Str. 9, 07745 Jena, Germany

Mehmet Müftüoğlu

Leibniz-Institute of Photonic Technology, Albert-Einstein-Str. 9, 07745 Jena, Germany

Glitta R. Cheeran

Leibniz-Institute of Photonic Technology, Albert-Einstein-Str. 9, 07745 Jena, Germany

Thomas Bocklitz

Leibniz-Institute of Photonic Technology, Albert-Einstein-Str. 9, 07745 Jena, Germany

Institute of Physical Chemistry, Friedrich Schiller University Jena, Helmholtzweg 4, 07743 Jena, Germany

Bennet Fischer

Leibniz-Institute of Photonic Technology, Albert-Einstein-Str. 9, 07745 Jena, Germany

Mario Chemnitz

Leibniz-Institute of Photonic Technology, Albert-Einstein-Str. 9, 07745 Jena, Germany

Institute of Applied Optics and Biophysics, Friedrich Schiller University Jena, Albert-Einstein-Str. 15, 07745 Jena, Germany

mario.chemnitz@leibniz-ipht.de

(February 5, 2025)

Abstract

The intrinsic complexity of nonlinear optical phenomena offers a fundamentally new resource to analog brain-inspired computing, with the potential to address the pressing energy requirements of artificial intelligence. We introduce and investigate the concept of nonlinear inference capacity in optical neuromorphic computing in highly nonlinear fiber-based optical Extreme Learning Machines. We demonstrate that this capacity scales with nonlinearity to the point where it surpasses the performance of a deep neural network model with five hidden layers on a scalable nonlinear classification benchmark. By comparing normal and anomalous dispersion fibers under various operating conditions and against digital classifiers, we observe a direct correlation between the system’s nonlinear dynamics and its classification performance. Our findings suggest that image recognition tasks, such as MNIST, are incomplete in showcasing deep computing capabilities in analog hardware. Our approach provides a framework for evaluating and comparing computational capabilities, particularly their ability to emulate deep networks, across different physical and digital platforms, paving the way for a more generalized set of benchmarks for unconventional, physics-inspired computing architectures.

Keywords:  Optical Neural Networks, Extreme Learning Machine, Supercontinuum Generation, Nonlinear Fiber Optics, Optical Soliton, Machine Learning

1  Introduction

The rapid advancement of artificial intelligence has sparked renewed interest in brain-inspired hardware, particularly optical implementations that promise energy-efficient solutions for AI acceleration and intelligent edge sensing. Traditional computing architectures face significant challenges when executing analog neural networks, resulting in substantial energy, water, and computational time requirements for operating large networks on conventional digital processors (i.e., CPUs, GPUs, TPUs).
 Optical approaches have garnered particular attention due to their intrinsic parallelism and scalability across multiple optical degrees of freedom, offering reduced energy consumption  [ 1 ] .
Unlike electronic solutions, a primary challenge in realizing competitive neuromorphic optical hardware lies in the all-optical implementation of nonlinearity to circumvent the electro-optical bottleneck toward deep all-optical architectures  [ 2 ] ,  [ 3 ] .
 Nonlinearity is crucial for emulating synaptic switching behavior in information networks and enabling advanced learning capabilities, including improved accuracy and generalization. In deep networks, this issue is often addressed through multiple layers, which offer higher nonlinear mapping capabilities of the model. The concept of "nonlinear mapping capability" refers to a system’s ability to transform input data into higher-dimensional spaces where previously inseparable patterns become linearly separable.
Moving to hardware, physical substrates can inherently provide complex nonlinear responses through their natural dynamics without the need for additional layers or computational resources.
In optics, taking advantage of the nonlinearity offered by light-matter interactions in the media, while actively studied  [ 4 ] , is widely assumed to be complicated and inefficient due to its high-power demands  [ 5 ] .
 Several recent studies suggest alternatives to realize nontrivial nonlinearity in optical systems. These include electronic feedback loops to electro-optic modulators  [ 6 ] ,  [ 7 ] , saturable absorption  [ 8 ]  complex active gain dynamics in mode-competitive cavities  [ 9 ] , and, most recently, repeated linear encoding through multiple scattering of input data  [ 10 ] ,  [ 11 ] .
 Neuromorphic computing with wave dynamics offers a promising approach to the elegant utilization of natural nonlinear dynamics in physical substrates. In optics, the concept aligns with Extreme Learning Machines (ELM) - a form of reservoir computing  [ 12 ] ,  [ 13 ]  without internal recurrence. Unlike other optical implementations, the learning machine comprises multiple virtual nodes that are coupled by the intrinsic feed-forward propagation dynamics of a single optical component, such as a fiber. This concept, recently proposed by Marcucci et al.  [ 14 ] , utilizes process-intrinsic, nonlinear mode interactions from natural waveguide dynamics as a computing resource. Experimental demonstrations include nonlinearly coupling spatial modes in multimode fibers  [ 15 ]  and spectral frequency generation in single-mode fibers  [ 16 ] ,  [ 17 ] ,  [ 18 ] .
 However, analog wave computers have a common difficulty: the systems are generally complex and cannot be easily mapped to practical computing models, a critical challenge lies in the diversity of existing approaches and the general inability to measure a system’s nonlinear mapping capabilities independent of specific tasks and to correlate them with learning abilities. This transformation is fundamental to solving complex tasks. Current approaches in the optical community empirically test new nonlinear systems on this capability through task-specific image recognition benchmarks. Here, it is common practice to compare, e.g., accuracies across system configurations to demonstrate improvements and to relate those improvements to the scaling in the learning behavior of deeper, hence more nonlinear neural networks computer models.
 In particular, the MNIST dataset is frequently used as a benchmark. However, it is not highly nonlinear, as a logistic regression can achieve approximately 92% accuracy  [ 19 ] . Similarly, linear Support Vector Machines (SVMs), which leverage kernel tricks to project input data into higher-dimensional spaces for linear separation, demonstrate comparable performance. Furthermore, performance improvements in big networks cannot be clearly attributed to stronger synaptic activation (i.e., deepness) yet might also result from increased connectivity as we will discuss further down in this paper. Task-specific benchmarks thus prove unsuitable for demonstrating deeper network activity. The question of nonlinearity’s relevance in neural networks also remains largely unexplored in computer science, with only limited investigations into non-binary nonlinear activation functions, such as a multi-step perceptron  [ 20 ]  or trainable spline activation functions  [ 21 ] .
 This work attempts to better understand and quantify the nonlinear inference capacity using a scalable, task-independent dataset and validate it using two different optical fiber processors. This investigation is particularly relevant for comparing computational capabilities across different physical substrates.
 We begin with illustrating the operation principle of a frequency-based ELM in a single optical fiber, as an arbitrary kernel machine, we refer to it from this chapter onward as fiber-optical ELM. We