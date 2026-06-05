[2006.04043] SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds

SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds

Qingdong He,
Zhengning Wang,
Hao Zeng,
Yi Zeng,
Yijun Liu
 University of Electronic Science and Technology of China

heqingdong@alu.uestc.edu.cn,zhengning.wang@uestc.edu.cn

{haozeng,zengyii,yijunliu}@std.uestc.edu.cn

Corresponding Author

Abstract

Accurate 3D object detection from point clouds has become a crucial component in autonomous driving. However, the volumetric representations and the projection methods in previous works fail to establish the relationships between the local point sets. In this paper, we propose Sparse Voxel-Graph Attention Network (SVGA-Net), a novel end-to-end trainable network which mainly contains voxel-graph module and sparse-to-dense regression module to achieve comparable 3D detection tasks from raw LIDAR data. Specifically, SVGA-Net constructs the local complete graph within each divided 3D spherical voxel and global KNN graph through all voxels. The local and global graphs serve as the attention mechanism to enhance the extracted features. In addition, the novel sparse-to-dense regression module enhances the 3D box estimation accuracy through feature maps aggregation at different levels. Experiments on KITTI detection benchmark demonstrate the efficiency of extending the graph representation to 3D object detection and the proposed SVGA-Net can achieve decent detection accuracy.

1  Introduction

With the widespread popularity of LIDAR sensors in autonomous driving

[ 4 ]  and augmented reality  [ 17 ] , 3D object detection from point clouds has become a mainstream research direction. Compared to RGB images from video cameras, point clouds could provide accurate depth and geometric information

[ 37 ]  which can be used not only to locate the object, but also to describe the shape of the object

[ 38 ] . However, the properties of unordered, sparsity and relevance of point clouds make it a challenging task to utilize point clouds for 3D object detection directly.

In recent years, several pioneering approaches have been proposed to tackle these challenges for 3D object detection on point clouds. The main ideas for processing point clouds data are to project point clouds to different views [ 28 ,  2 ,  9 ,  14 ,  34 ]  or divide the point clouds into equally spaced voxels [ 12 ,  39 ,  33 ] . Then convolutional neural networks and mature 2D objection detection frameworks  [ 23 ,  22 ]  are applied to extract features. However, because projection alone cannot capture the object’s geometric information well, many methods [ 2 ,  31 ,  18 ,  29 ]  have to combine RGB images in the designed network. While the methods using only voxelization do not make good use of the properties of the point clouds and bring a huge computational burden [ 15 ]  as resolution increases. Apart from converting point clouds into other formats, some works  [ 26 ,  36 ]  take Pointnets  [ 19 ,  20 ]  as backbone to process point clouds directly. Although Pointnets build a hierarchical network and use a symmetric function to maintain permutation invariance, they fail to construct the neighbour relationships between the grouped point sets  [ 30 ] .

Considering the properties of point clouds, we should notice the superiority of graphs in dealing with the irregular data. In fact, in the domain of point clouds for segmentation and classification tasks, the method of processing with graphs has been deeply studied by many works  [ 21 ,  1 ,  10 ,  24 ,  30 ] . However, few researches have used graphs to make 3D object detection from point clouds. To our knowledge, Point-GNN [ 27 ]  may be the first to prove the potential of using the graph neural network as a new approach for 3D object detection. Point-GNN introduces auto-registration mechanism to reduce translation variance and designs box merging and scoring operation to combine detection results from multiple vertices accurately. However, similar to ShapeContextNet  [ 32 ]  and Pointnet++  [ 20 ] , the relationship between point sets is not well established in the feature extraction process and a large number of matrix operations will bring heavy calculation burden and memory cost.

In this paper, we propose the sparse voxel-graph attention network (SVGA-Net) for 3D object detection. SVGA-Net is an end-to-end trainable network which takes raw point clouds as input as outputs the category and bounding boxes information of the object. Specifically, SVGA-Net mainly consists of voxel-graph network module and sparse-to-dense regression module. Instead of normalized rectangle voxels, we divide the point clouds into 3D spherical space with a fixed radius. The voxel-graph network aims to construct local complete graph for each voxel and global KNN graph for all voxels. The local and global serve as the attention mechanism that can provide a parameter supervision factor for the feature vector of each point. In this way, the local aggregated features can be combined with the global point-wise features. Then we design the sparse-to-dense regression module to predict the category and 3D bounding box by processing the features at different scales. Evaluations on KITTI benchmark demonstrate that our proposed method can achieve comparable results with the state-of-the-art approaches.

Our key contributions can be summarized as follows:

•

We propose a new end-to-end trainable 3D object detection network from point clouds which uses graph representations without converting to other formats.

•

We design a voxel-graph network, which constructs the local complete graph within each spherical voxel and the global KNN graph through all voxels to learn the discriminative feature representation simultaneously.

•

We propose a novel 3D boxes estimation method that aggregates features at different scales to achieve higher 3D localization accuracy.

•

Our proposed SVGA-Net achieves decent experimental results with the state-of-the-art methods on the challenging KITTI 3D detection dataset.

2  Related Work

2.1  Projection-based methods for point clouds

To align with RGB images, series of works process point clouds through projection

[ 2 ,  9 ,  13 ] . Among them, MV3D  [ 2 ]  projects point clouds to bird view and trains a Region Proposal Network (RPN) to generate positive proposals. It extracts features from LiDAR bird view, LIDAR front view and RGB image, for every proposal to generate refined 3D bounding boxes. AVOD  [ 9 ]  improves MV3D by fusing image and bird view features and merges features from multiple views in the RPN phase to generate positive proposals. Note that accurate geometric information may be lost in the high-level layers with this scheme.

2.2  Volumetric methods for point clouds

Another typical method for processing point clouds is voxelization. VoxelNet  [ 39 ]  is the first network to process point clouds with voxelization, which use stacked VFE layers to extract features tensors. Following it, a large number of methods  [ 16 ,  33 ,  25 ,  3 ]  divide the 3D space into regular grids and group the points in a grid as a whole. However, they often need to stack heavy 3D CNN layers to realize geometric pose inference which bring large computation.

2.3  Pointnet-based methods for point clouds

To process point clouds directly, PointNet  [ 19 ]  and PonintNet++  [ 20 ]  are the two groundbreaking works to design parallel MLPs to extract features from the raw irregular data, which improve the accuracy greatly. Taking them as backbone, many works  [ 26 ,  18 ,  11 ,  36 ,  35 ]  begin to design different feature extractors to achieve better performance. Although Pointnets are effective to abstract features, they still suffer feature loss between the local and global point sets.

2.4  Graph-based methods for point clouds

Constructing graphs to learn the order-invariant representation of the irregular poi