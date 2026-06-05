[2407.10338] Long Sequence Decoder Network for Mobile Sensing

Long Sequence Decoder Network for Mobile Sensing

Jiazhong Mei ∗  and J. Nathan Kutz  ∗,†

∗

Department of Applied Mathematics, University of Washington, Seattle WA 98195

†

Department of Electrical and Computer Engineering, University of Washington, Seattle WA 98195

Abstract

The reconstruction and estimation of spatio-temporal patterns poses significant challenges when sensor measurements are limited.
The use of mobile sensors adds additional complexity due to the change in sensor locations over time.
In such cases, historical measurement and sensor information are useful for better performance, including models such as Kalman filters, recurrent neural networks (RNNs) or transformer models. However, many of these approaches often fail to efficiently handle long sequences of data in such scenarios and are sensitive to noise.
In this paper, we consider a model-free approach using the  structured state space sequence  (S4D) model as a deep learning layer in traditional sequence models to learn a better representation of historical sensor data. Specifically, it is integrated with a shallow decoder network for reconstruction of the high-dimensional state space.
We also introduce a novel initialization of the S4D model using a Butterworth filter design to reduce noise in the inputs.
Consequently, we construct a robust S4D (rS4D) model by appending the filtering S4D layer before the original S4D structure.
This robust variant enhances the capability to accurately reconstruct spatio-temporal patterns with noisy mobile sensor measurements in long sequence. Numerical experiments demonstrate that our model achieves better performance compared with previous approaches. Our results underscore the efficacy of leveraging state space models within the context of spatio-temporal data reconstruction and estimation using limited mobile sensor resources, particularly in terms of long-sequence dependency and robustness to noise.

1  Introduction

Sensor technologies are ubiquitous across scientific and engineering domains, revolutionizing the way we collect and analyze data. From static installations for environmental monitoring to the emergence of mobile sensors for applications in domains such as autonomous vehicles and wearable health trackers, these sensors play a pivotal role in modern data-driven systems  [ 4 ,  36 ,  23 ] . In many cases, measurements of the full state are impossible, impractical, or not even desired. More commonly, limited sensors are used to infer the full characteristic of the system of interest in high dimension from the measurements they collect. Thus the fundamental mathematical problem is to approximate the full state space from the limited collected data. We consider the problem of state estimation through time sequence measurements from limited mobile sensors by combining a  structured state space sequence  (S4D) model with a decoder network, further leveraging a novel initialization scheme and long temporal sequences to produce a robust model with improved performance in comparison with existing methods.

Mobile sensors are becoming more popular and ubiquitous in many applications, for example human biomechanics motion tracking, ocean dynamics monitoring buoys, drone monitoring, and weather balloons  [ 1 ,  24 ,  18 ] . The mobility of the sensors provide more flexibility and lower cost compared to installing fixed sensors  [ 22 ,  7 ,  19 ] .
However, unlike stationary sensors, state estimation from mobile sensors brings additional challenges. Traditional techniques, while effective for static sensor arrays, often fall short when applied to mobile sensors operating in dynamic environments.
These models typically employ linear or non-linear mappings from sensor measurements at the current time step to the full system state. For instance, leveraging the inherent low-rank features of the system, methodologies such as singular value decomposition (SVD), also referred to as proper orthogonal decomposition (POD), identify dominant modes of the system and construct a linear mapping from measurements to high-dimensional state space  [ 9 ,  35 ,  2 ,  4 ] . Similarly, dynamic mode decomposition (DMD) extracts linear modes for reconstruction while simultaneously capturing the temporal evolution of these modes in low-rank representation  [ 32 ,  17 ,  3 ] . More complex approaches like shallow decoder networks (SDN) learn nonlinear reconstructions between measurements and high-dimensional state spaces, exhibiting exceptional performance even with a minimal number of sensors  [ 8 ,  5 ,  26 ] . However, given that the location of each measurement collected by a mobile sensor varies over time, relying solely on such mappings proves inadequate. Considering the impracticality of learning separate models for each sensor location within a high-dimensional state space, there arises a necessity for a generalized model that incorporates sensor location information. Other approaches, such as the Kalman filter, incorporates historical values alongside current measurements  [ 29 ] .
By considering the time history of measurements, additional insights into the system dynamics are gained, enhancing reconstruction performance and robustness to noise. Notably, the Kalman filter naturally accommodates the mobile sensor scenario, where the measurement matrix can vary with the sensor trajectory over time  [ 19 ] .
Despite its adaptability, the Kalman filter is fundamentally a statistical model, necessitating prior knowledge of system dynamics or an approximation, as well as statistical priors regarding noise and disturbances for optimal performance.
Furthermore, the effect of the historical measurements has a compound decay in time depending on the observation noise covariance, lacking the flexibility and ability of memorization in the long run.

The recurrent neural network (RNN)  [ 25 ]  has emerged as a powerful tool for preserving information from past inputs in sequential data commonly seen in a variety of tasks such as speech recognition  [ 11 ] , machine translation  [ 30 ] , spatiotemporal predictive learning  [ 28 ,  33 ] , and much more  [ 27 ] .
By iteratively applying a series of learnable transformations to input sequences, RNNs adeptly capture temporal dependencies, allowing them to encode and interpret patterns spanning across time.
A notable approach applying RNN layers to sensing is the  SHallow REcurrent Decoder  (SHRED), which has shown promising performance in both stationary and mobile sensor scenarios  [ 34 ,  7 ] .
Unlike approaches such as Kalman filter whose performance relies heavily on an approximated model of the system dynamics, SHRED is model-free and directly reconstructs the full system from sensor measurement sequences.
SHRED leverages long short-term memory networks (LSTM)  [ 16 ] , a variant of RNN architecture, in conjunction with a fully-connected, shallow decoder to process time series of sensor measurements for effective reconstruction.
Chen et al.  [ 6 ]  used a similar deep learning approach combining a recurrent network and a reconstruction network.
Nevertheless, previous research has yet to address several key challenges inherent in mobile sensor reconstruction.

It has been shown that most conventional sequence models such as RNNs and transformers fail to scale to sequences with long time dependencies  [ 14 ,  31 ] .
They perform poorly on tasks such as byte-level text classification and retrieval, image classification on sequences of pixels, and finding valid paths connecting two points that are benchmarked by the Long Range Arena  [ 31 ] .
Long-range dependencies are also very common in a limited mobile sensor reconstruction problem to understand a complex system in a high dimension.
A mobile sensor would need to take frequent measurements over an extended time to capture the transient and dominant dynamical characteristics of a complex system, which