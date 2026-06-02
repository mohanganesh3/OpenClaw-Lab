[2502.18555] Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision.

Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision.

Erick da Silva Farias  1  , Eduardo Palhares Júnior

2

Abstract

The automatic detection of human conflicts through videos is a crucial area in computer vision, with significant applications in monitoring and public safety policies. However, the scarcity of public datasets and the complexity of human interactions make this task challenging. This study investigates the integration of advanced deep learning techniques, including Attention Mechanism, Convolutional Neural Networks (CNNs), and Bidirectional Long Short-Term Memory (BiLSTM), to improve the detection of violent behaviors in videos. The research explores how the use of the attention mechanism can help focus on the most relevant parts of the video, enhancing the accuracy and robustness of the model. The experiments indicate that the combination of CNNs with BiLSTM and the attention mechanism provides a promising solution for conflict monitoring, offering insights into the effectiveness of different strategies. This work opens new possibilities for the development of automated surveillance systems that can operate more efficiently in real-time detection of violent events.

1  Introduction

Violence is a complex phenomenon that permeates the history of humanity, manifesting itself in different ways and in different contexts. Since the beginning of civilization, violence has been present in wars, territorial conflicts, and power disputes. Over time, new manifestations emerged, such as domestic violence, urban crime, and terrorist attacks. Violence manifests itself in seemingly trivial situations, such as fights in bars or traffic conflicts. These episodes reflect social tensions, accumulated frustrations, and, often, the lack of adequate conflict resolution mechanisms. The culture of aggression and the normalization of violence in social relationships can intensify these situations, creating a cycle that is difficult to break.

In many contexts, social inequality, poverty, and marginalization also fuel violence, creating an environment conducive to organized crime and urban violence. Thus, violence in today’s world is a multifaceted phenomenon that requires a critical and multidisciplinary approach to understand it and, above all, combat it. Analysis of its historical, social, and cultural roots is fundamental to developing effective prevention and intervention strategies.

Surveillance cameras are widely used in commercial establishments, homes, industries, schools, and public places. These cameras are intended to assist agents who monitor the location, however, this type of conventional monitoring is not very effective when hundreds of cameras are deployed because of human involvement, because identifying incidents using conventional cameras becomes an inefficient task.

An efficient way to identify incidents via a surveillance camera would be through computer vision, because images from the CCTV system can be linked to a trained deep learning model to make inferences about incidents related to violence between humans in real time. This approach to using computer vision is relevant as it will eliminate the cost of surveillance by humans. But for this to work, it is necessary to carry out tests, collect images to train the model, compare deep learning models, and other adjustment processes to refine the human conflict detection system.

With respect to data collection, it is important that the data set has a significant volume, with variance in class data and good resolution. According to  [ Dashdamirov 2024 ] , for effective algorithm training, the collection and labeling of a vast volume of data is essential. Although there are public sets of videos available, there is still a significant need to expand the amount of this data. Furthermore, aspects such as video resolution, frame frequency, lighting conditions, and camera angles vary greatly. These differences complicate the development of models that are both robust and capable of generalizing appropriately.

The use of Deep Learning in the context of human conflict monitoring is relatively new, because the data available publicly has a small volume and has low quality in the video frames.
 [ Dashdamirov 2024 ]  evaluates deep learning techniques in detecting violence in videos, highlighting that increasing the dataset from 500 to 1,600 videos improves the average accuracy of the models by 6%. It demonstrates the importance of large data sets and transfer learning for more effective surveillance systems.

[ Datta et al. 2002 ]  analyzed the trajectory of movements and orientation of body limbs to detect violent behavior.  [ Nguyen et al. 2005 ]  introduced a hierarchical hidden Markov model (HHMM), showing that it can be useful for recognizing aggressive attitudes, especially through a standard HHMM approach aimed at identifying violence.

[ Kim and Grauman 2009 ]  combined probabilistic Principal Component Analysis (PCA), used to identify flow patterns in local areas, with Markov Random Fields (MRF), which help maintain global model coherence. On the other hand,  [ Mahadevan et al. 2010 ]  argued that optical flow-based representations are not suitable for detecting unusual changes in both appearance and motion. They proposed a technique that identifies violent scenes by evaluating elements such as the presence of blood, flames, intensity of movement and sound volume.

2  Methodology

In this chapter the methodology will be presented. In

2.1  , computer vision was discussed. In section

2.2

Deep Learning and Neural Networks were covered, LSTM and BiLSTM in the subtopics and in session

2.3

about the Attention Mechanism.

2.1  Computer Vision

Computer vision is an area of artificial intelligence (AI) that deals with developing methods that allow computers to acquire, process and interpret visual information from the real world, with the aim of making decisions or providing recommendations  [ Szeliski 2010 ] .

The main challenges of computer vision in videos involve the need to identify and classify objects and actions in dynamic environments, such as recognizing human behavior patterns or detecting specific events, such as conflicts, aggressions or complex interactions.

2.1.1  Neural Network Models in Computer Vision

Convolutional neural network (CNN) models are widely used in computer vision due to their ability to extract hierarchical spatial features from images and videos. CNNs operate by applying filters (or convolutions) to the image to extract local features such as edges, textures and shapes. These models are efficient for tasks such as object detection, scene recognition and action identification in videos  [ LeCun et al. 2015 ] .

For video analysis, CNNs are often combined with temporal models such as Long Short-Term Memory (LSTM) networks in order to capture dynamic information over time, effectively integrating spatial and temporal learning  [ Simonyan and Zisserman 2014 ] .

2.2  Deep Learning and Neural Networks

Deep learning is a subfield of artificial intelligence that relies on deep neural networks to perform complex recognition, classification, and prediction tasks. These networks are composed of multiple layers of processing, allowing them to learn hierarchical representations of data such as images, text and temporal sequences.

2.2.1  Convolutional Neural Networks (CNNs)

Convolutional neural networks (CNNs) are a class of deep neural networks that have been widely used in computer vision tasks due to their ability to learn efficient representations of visual data. Figure

1

shows a diagram that represents the architecture of a CNN. They are composed of convolutional layers, pooling layers, and fully conne