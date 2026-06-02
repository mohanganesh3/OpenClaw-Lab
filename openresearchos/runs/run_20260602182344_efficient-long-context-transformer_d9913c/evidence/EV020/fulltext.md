[2506.11892] Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices

Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices

Lu Zhang  1  , Sangarapillai Lambotharan  1  , Gan Zheng  2  , Guisheng Liao  3  , Basil AsSadhan  4  , Fabio Roli  5

1  Wolfson School of Mechanical, Electrical and Manufacturing Engineering, Loughborough University, Loughborough, UK

2

School of Engineering, University of Warwick, Coventry, CV4 7AL, UK

3  School of Electronic Engineering, Xidian University,
Xi’an 710071, People’s Republic of China

4  Department of Electrical Engineering, King Saud University, Riyadh, Saudi Arabia

5  Department of Informatics, Bioengineering,
Robotics, and Systems Engineering, University of Genova, 16145 Genoa, Italy

Abstract

Due to great success of transformers in many applications such as natural language processing and computer vision, transformers have been successfully applied in automatic modulation classification. We have shown that transformer-based radio signal classification is vulnerable to imperceptible and carefully crafted attacks called adversarial examples. Therefore, we propose a defense system against adversarial examples in transformer-based modulation classifications. Considering the need for computationally efficient architecture particularly for Internet of Things (IoT)-based applications or operation of devices in environment where power supply is limited, we propose a compact transformer for modulation classification. The advantages of robust training such as adversarial training in transformers may not be attainable in compact transformers. By demonstrating this, we propose a novel compact transformer that can enhance robustness in the presence of adversarial attacks. The new method is aimed at transferring the adversarial attention map from the robustly trained large transformer to a compact transformer. The proposed method outperforms the state-of-the-art techniques for the considered white-box scenarios including fast gradient method and projected gradient descent attacks. We have provided reasoning of the underlying working mechanisms and investigated the transferability of the adversarial examples between different architectures. The proposed method has the potential to protect the transformer from the transferability of adversarial examples.

Index Terms:

low-power IoT devices, transformer, adversarial examples, fast gradient method, projected gradient descent algorithm, adversarial training, adversarial attention map

I

Introduction

The Internet of Things (IoT) and mobile networks are evolving rapidly to fulfil the need for ultra reliable and low latency performance, seamless connectivity, mobility, and intelligence  [  1  ,

2  ,

3  ,

4  ] . It is estimated that over 50 billion devices are wirelessly connected to the internet, which can sense their surroundings and offer high-quality services. The explosive growth of IoT devices demands efficient management of already scarce radio spectrum, which is very challenging particularly in a non-cooperative communication environment. As a result, classifying modulation types at the receiver under non-cooperative communication conditions becomes a critical task. Automatic modulation classification (AMC) is proposed which plays a key role in wireless spectrum monitoring by performing modulation classifications possibly without prior knowledge of the received signals or channel parameters  [  5  ,

6  ] . It also plays an important role in wireless spectrum anomaly detection, transmitter identification and radio environment awareness, consequently improving radio spectrum usage and the context aware intelligent decision making in autonomous wireless spectrum monitoring.

Traditionally, AMC was mainly achieved by carefully extracted features (such as higher order cyclic moments) by experts and certain classification criteria  [  7  ,

8  ,

9  ,

10  ,

11  ] . These existing feature-based methods are easy to implement in practice, however, hand-crafted features and hard-coding criteria for AMC make scaling to new modulation types challenging. Recently, due to the superior performance of deep learning, many researchers have resorted to various deep neural network (DNN) architectures for AMC  [  12  ,

13  ,

14  ,

15  ,

16  ,

17  ,

18  ,

19  ] . For example, a convolutional neural network (CNN) was used for AMC in  [  12  ] . Later convolutional long short-term deep neural networks (CLDNN), long short-term memory neural networks (LSTM), and deep residual networks (ResNet) were proposed to improve the classification performance  [  16  ] . A complex CNN was proposed in  [  17  ]  for the identification of signal spectrum information. A spatio-temporal hybrid deep neural network was proposed in  [  18  ]  for AMC which is based on multi-channels and multi-function blocks. Furthermore, to reduce the communication overhead, the authors of  [  19  ]  proposed an innovative learning framework which is based on the combination of decentralized learning and ensemble learning. With the great success of transformer in the computer vision area  [  20  ,

21  ,

22  ] , the work in  [  23  ]  has successfully applied transformers in AMC which shows considerable performance improvement compared to the state-of-the-art techniques.

Despite its superior performance of DNN, several recent research works have pointed out that DNNs are vulnerable to adversarial examples, which are imperceptible and deliberately crafted modifications to the input that result in misclassifications  [  24  ] . Adversarial examples have been proven to be effective in terms of hindering operation of several machine learning applications, such as face recognition  [  25  ] , object detection  [  26  ] , semantic segmentation  [  27  ] , natural language processing  [  28  ] , and malware detection  [  29  ] . Notably, adversarial examples using fast gradient methods (FGM) have been shown to reduce the classification accuracy in AMC  [  30  ,

31  ] . Recently, we have shown a transformer-based AMC is also vulnerable to adversarial examples using a projected gradient descent (PGD) method  [  32  ] .

In practice, AMC can be applied to both military and civilian scenarios. In a networked battlefield, important information may be shared using radio signals by the units of each adversary (opponent transmitter and receiver as indicated in Figure

1  ). The allied forces (playing the role of eavesdropper in this scenario) can employ AMC to determine the modulation used to eavesdrop information transmitted between the adversary units (opponents). To deter the allied forces from eavesdropping messages, small perturbations (adversarial perturbations) can be applied to the communication signals by the opponents such that the modulation discovery executed by the allied forces eventually fails. The modulation can still be discovered by the allied forces, however in this case, an AMC system which is robust against adversarial attacks should be applied as proposed in this paper. To the best of our knowledge, this work is the first to propose a defense for the transformer-based AMC in the literature. The proposed defense will be used to protect the allied forces from the adversarial perturbations such that the allied forces could successfully discover the modulation. While the transformers offer superior performance, it comes with a price in terms of large model size and computational complexity, which may limit reaping its benefits in applications that rely on low-power sensors and IoT networks  [  33  ,

34  ,

35  ,

36  ]  with possibly a low memory size. Therefore we propose a novel compact transformer-based defense for low-power IoT devices based on distillation of knowledge through the adversarial attention map, which is a critical element for the transformer and the details are given in Section III.