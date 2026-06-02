[2102.05348] Regional Attention with Architecture-Rebuilt 3D Network for RGB-D Gesture Recognition

Regional Attention with Architecture-Rebuilt 3D Network
 for RGB-D Gesture Recognition

Benjia Zhou  1  ,
Yunan Li

3,4  ,
Jun Wan

2,5

Benjia Zhou and Yunan Li contribute equally to this paper.Corresponding author;
This work was done by Benjia Zhou who visited the lab of Center for Biometrics and Security Research, NLPR, CASIA.

Abstract

Human gesture recognition has drawn much attention in the area of computer vision. However, the performance of gesture recognition is always influenced by some gesture-irrelevant factors like the background and the clothes of performers. Therefore, focusing on the regions of hand/arm is important to the gesture recognition. Meanwhile, a more adaptive architecture-searched network structure can also perform better than the block-fixed ones like ResNet since it increases the diversity of features in different stages of the network better.
In this paper, we propose a regional attention with architecture-rebuilt 3D network (RAAR3DNet) for gesture recognition.
We replace the fixed Inception modules with the automatically rebuilt structure through the network via Neural Architecture Search (NAS), owing to the different shape and representation ability of features in the early, middle, and late stage of the network.
It enables the network to capture different levels of feature representations at different layers more adaptively. Meanwhile, we also design a stackable regional attention module called Dynamic-Static Attention (DSA), which derives a Gaussian guidance heatmap and dynamic motion map to highlight the hand/arm regions and the motion information in the spatial and temporal domains, respectively. Extensive experiments on two recent large-scale RGB-D gesture datasets validate the effectiveness of the proposed method and show it outperforms state-of-the-art methods.
The codes of our method are available at: https://github.com/zhoubenjia/RAAR3DNet.

Introduction

Gesture is produced as part of deliberate action and signs, involving the motion of the up body, especially the arms, hands, and fingers. Video-based classification makes an essential component in gesture recognition. It has been applied to many human-centred tasks, such as apparent personality analysis  (Li et al.  2020 ) , sign language recognition  (Cui, Liu, and Zhang  2019 )  and human-computer interaction (HCI)  (Wang et al.  2016b ) . The handcrafted features  (Wan et al.  2013 ; Wan, Guo, and Li  2015 )  are always used for gesture recognition in the early years. The powerful feature representation ability of deep learning also promotes the application of neural networks in the field of gesture recognition  (Li et al.  2016 ; Miao et al.  2017 ; Simonyan and Zisserman  2014 ; Narayana, Beveridge, and Draper  2018 ) .

For most of the deep learning-based gesture recognition methods, some popular networks like ResNet  (He et al.  2016 ) , SENet  (Hu, Shen, and Sun  2018 )  and Inflated 3D Network (I3D)  (Carreira and Zisserman  2017 )  are usually employed as the backbone for gesture recognition. Although these networks have achieved great success in many tasks, it still worth pointing that the same modules are shared from shallow to deep layers in these networks. Even the modules in networks like I3D that employ multi-branch structure to improve the width and diversity are fixed and all the same through the network. However, features in the early stage and late stage are quite different. Features in the early stage are low-level features, which show the visual texture in each frame, whereas the high-level features in the late stage are abstracted and more related to the class of gestures. Therefore, it is not suitable to use the same structure to learn different features, and then we need to make the network more adaptive and automatically determine what the shape is for different parts of it.

Figure 1:  The pipeline of the proposed method. (a) the structure of the entire RAAR3DNet. (b) the inner structure of automatically rebuilt cell1, cell2, and cell3.
Each cell is composed of two input nodes, four intermediate nodes and one output node. The output node is obtained by combing features from intermediate nodes with some reduction operation ( e . g ., concatenation), which are marked as the dashed lines in (b).
We take the I3D network as the backbone, and utilize NAS to automatically rebuild the structure of Inception Modules in it. The reconstructed network shows different structure to fit multi-scale features.
Cell1 and cell2, which are in the early and middle stage of the network, tend to employ convolution kernels with small receptive fields to capture the low-level texture features more easily, whereas cell3s at the end of the network perform dilated convolution operations to capture the more abstract and semantically high-level features.

Meanwhile, one of the most significant challenge hindering the improvement of recognition accuracy is the influence of gesture-irrelevant factors, such as backgrounds, different clothes of performers, and so on. The various textures and appearances could mislead the network to learn inconsequential or less important features. For dynamic gesture recognition from a video sequence, we believe it is vital to focus on gesture movements, such as hands, arms or elbows of the performers. Many researchers notice that it is critical to make the network focusing on the gesture regions both spatially and temporally. Modules such as hand detector  (Wang et al.  2016c ; Liu et al.  2017 ) , and additional modalities of data like optical flow  (Li et al.  2017 )  or saliency  (Li et al.  2018 )  are widely used via combining with the raw RGB (and depth) data to design different algorithms. However, most of them require extra offline operations ( e . g ., hand detection, optical flow calculation) in advance. It would increase time complexity because of using hand detector network in the testing stage. Therefore, it may be more reasonable if the attention maps of gesture regions are learned along with the task of gesture recognition in the same network.

Inspired by the above discussions, we propose a regional attention with architecture-rebuilt 3D network for dynamic gesture recognition based on RGB-D data, which is illustrated in Fig.  1  . We take the I3D  1

1  1 We still utilize a two-stream configuration - with one I3D network trained on RGB inputs, and another on depth inputs.

network as the backbone and employ the theory of NAS to find the optimal combination of different operations in each module of the network.
To make the network focus on the gesture regions, we propose a regional attention module DSA, which includes a static attention sub-module (SAtt) and dynamic attention sub-module (DAtt). For static attention, we learn a heatmap of hands or body for each frame with the supervision of the Gaussian map of skeleton keypoints. It indicates where the hands/arms are and highlights these regions.
For dynamic attention, we present a fast approximate rank pooling algorithm to learn the accumulated dynamic images, which reduces the time complexity a lot when compared with the traditional rank pooling techniques  (Bilen et al.  2016 ,  2017 )  and thus can give a real-time dynamic image computation.
Then with the DSA structure applied, the network can pay attention to the gesture regions spatiotemporally.
Our contributions can be summarized as three-fold:

(1) We replace the structure-fixed modules in the general network with automatically reconstructed cells via NAS. The cells in the early, middle, and late stages of the network can have different structures and learn the low-level and high-level features more adaptively.

(2) We propose a stackable attention structure, called DSA, to generate attention map in both spatial and temporal space. DSA consists of the SAtt and DAtt sub-modules. SAtt highlights the hands/ar