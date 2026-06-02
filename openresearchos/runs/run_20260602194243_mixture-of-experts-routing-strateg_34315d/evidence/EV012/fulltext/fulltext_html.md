[2501.12281] MoGERNN: An Inductive Traffic Predictor for Unobserved Locations in Dynamic Sensing Networks

MoGERNN: An Inductive Traffic Predictor for Unobserved Locations in Dynamic Sensing Networks

Qishen Zhou

Yifan Zhang

Michail A. Makridis

Anastasios Kouvelas

Yibing Wang

Simon Hu

simonhu@zju.edu.cn

Abstract

Given a partially observed road network, how can we predict the traffic state of interested unobserved locations? Traffic prediction is crucial for advanced traffic management systems, with deep learning approaches showing exceptional performance. However, most existing approaches assume sensors are deployed at all locations of interest, which is impractical due to financial constraints. Furthermore, these methods typically require costly retraining when sensor configurations change. Compared to the conventional traffic predictor, an effective traffic predictor for unobserved locations in a dynamically changing sensor network can not only predict the state of observed and unobserved locations robustly but also provide richer information for the traffic monitoring system and save the cost of sensor deployment. However, the absence of data from unobserved locations and the dynamically changing sensor networks present challenges. We propose MoGERNN, an inductive spatio-temporal graph representation model, to address these issues. Inspired by the Mixture of Experts (MoE) approach in Large Language Models, we introduce a Mixture of Graph Expert (MoGE) block to model complex spatial dependencies through multiple graph message aggregators and a sparse gating network. This block estimates initial states for unobserved locations, which are then processed by a GRU-based Encoder-Decoder that integrates a graph message aggregator to capture spatio-temporal dependencies and predict future states. Experiments on two real-world datasets show MoGERNN consistently outperforms baseline methods for both observed and unobserved locations. MoGERNN can accurately predict congestion evolution even in areas without sensors, offering valuable information for traffic management. Moreover, MoGERNN is adaptable to dynamic sensing networks, maintaining competitive performance even compared to its retrained counterpart. Tests performed with different numbers of available sensors confirm its consistent superiority, and ablation studies validate the effectiveness of its key modules.

keywords:

spatio-temporal extrapolation , traffic state estimation , traffic prediction , kriging , inductive graph representation learning , mixture of experts

{highlights}

A novel inductive spatio-temporal graph representation model (MoGERNN) is proposed, providing accurate speed predictions at both unobserved and observed locations.

The proposed MoGERNN works well under dynamically changed sensor network without the need for retraining, enhancing the robustness and reducing the training cost.

Experiments demonstrate MoGERNN outperforms four baseline methods in terms of prediction performance and adaptability of the dynamic sensing network on two real-world datasets.

MoGERNN is able to accurately predict congestion evolution, even at locations where no sensors are installed.

Tests performed with different numbers of available sensors, demonstrating the consistent superiority of MoGERNN.

\affiliation

[a]organization=Institute of Intelligent Transportation Systems, College of Civil Engineering
and Architecture, Zhejiang University,
city=Hangzhou,
country=China
 \affiliation [b]organization=Traffic Engineering group, Institute for Transport Planning and Systems, ETH Zurich,
city=Zurich,
country =Switzerland
 \affiliation [c]organization=ZJU-UIUC Institute, Zhejiang University,
city=Haining,
country=China

1  Introduction

Large-scale traffic forecasting plays a pivotal role in advanced traffic management systems and has gained considerable attention over recent decades. Among the existing research, deep learning-based approaches dominate and achieve extraordinary performance  (Shaygan et al.,  2022 ; Yin et al.,  2022 ) . Despite the success of deep learning made in traffic prediction, two critical issues have received limited attention. First, prediction of the state for unobserved locations: nearly all existing studies presuppose the presence of sensors at the all locations of interest. However, due to financial constraints, it is impractical to deploy and maintain an adequate number of sensors across all areas of interest. Second, dynamically changing sensor networks: most research relies on static sensor networks for model design, operating under the assumption that the sensor network configuration remains the same during both the training and application phases. Nevertheless, given the inevitable damage to sensors and the installation of new ones, it is important for the model to accommodate dynamic sensor networks, which saves computational resources and enhances efficiency by eliminating the need for retraining. These two issues are summarized as a problem of Forecasting Unobserved Nodes Under a Dynamic Sensing network (FUNDS), illustrated in

Fig. ˜ 1  . For the sake of the subsequent presentation, unobserved locations are considered as being equipped with virtual sensors. Compared to the conventional traffic predictor, an effective predictor for FUNDS can not only predict the state of observed and unobserved locations robustly but also provides richer information for traffic monitoring system and save the cost of sensor deployment.

Figure 1 :

An illustration of forecasting unobserved nodes under a dynamic sensing network: (a) an example of sensor deployment on the map, with some interested but not observed locations; (b) graph typologies for dynamic sensor configuration.

To address these challenges, especially in spatial extrapolation under dynamic sensing, Graph Neural Networks (GNNs) have emerged as an effective solution. The sensors, including physical and virtual ones, are usually modeled as the nodes in the graph, while the spatial relationship of sensors is modeled as the edges in the graph. For one thing, the mechanism of message passing between nodes in GNNs enables inferring the state of unobserved nodes given the observed ones. For another, most well-trained GNNs are inductive which can be applied to a new graph structure directly  (Hamilton et al.,  2017 ) . For example,  Appleby et al. ( 2020 ) ,  Wu et al. ( 2021a )  and  Mei et al. ( 2023 )  demonstrate the prospect of graph convolution network and diffusion graph convolution network for spatial extrapolation by means of well-designed training procedure. However, these three methods do not fully exploit temporal dynamics in the input data. As a solution,  Zheng et al. ( 2023 )  introduces the Gated Recurrent Unit (GRU) to capture temporal patterns in the data, thereby greatly improving the accuracy of the model.
Furthermore, the GNN-based methods mentioned above  (Appleby et al.,  2020 ; Wu et al.,  2021a ; Mei et al.,  2023 ; Zheng et al.,  2023 )  are all built on a predefined (weighted) adjacency matrix, which is limited for complex spatial dependencies modeling. In response, Graph Attention mechanism (GAT) that adaptively assigns different weights to neighboring nodes based on their node embeddings is introduced  (Roth and Liebig,  2022 ) . However, it is important to note that in the context of FUNDS, the data for unobserved nodes, i.e., virtual sensors, is completely missing. In this case, the node-feature driven dynamic neighbor weights may lead to circular dependencies, therefore resulting in inaccurate node-wise information aggregation. In addition to GAT, another alternative to capture complex spatial pattern by using multiple aggregators for neighbors message together has been proposed by  Wu et al. ( 2021b ) , enhancing the generalization capability and accuracy of the model. However, it poses a new challenge to effectively integrate multiple aggregators according to varying graph struct