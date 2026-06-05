[2208.00281] Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding

1

1  institutetext:  Tsinghua University

1

1  email:  wenh19@mails.tsinghua.edu.cn, liuyzchina@gmail.com

2

2  institutetext:  Huawei Technologies

2

2  email:  {huangjingwei6,duanbo5}@huawei.com

3

3  institutetext:  Shanghai Qi Zhi Institute

3

3  email:  ericyi0124@gmail.com

Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding

Hao Wen  ∗

11

Yunze Liu  ∗

11

Jingwei Huang

22

Bo Duan

22

Li Yi

1133

Abstract

This paper proposes a 4D backbone for long-term point cloud video understanding. A typical way to capture spatial-temporal context is using 4Dconv or transformer without hierarchy. However, those methods are neither effective nor efficient enough due to camera motion, scene changes, sampling patterns, and complexity of 4D data. To address those issues, we leverage the primitive plane as mid-level representation to capture the long-term spatial-temporal context in 4D point cloud videos, and propose a novel hierarchical backbone named Point Primitive Transformer(PPTr), which is mainly composed of intra-primitive point transformers and primitive transformers. Extensive experiments show that PPTr outperforms the previous state of the arts on different tasks.

Keywords:  Transformer; Primitive; Long-term Point Cloud Video

†

†  footnotetext:

∗  Equal contribution. Author ordering determined by coin flip.

1  Introduction

Point cloud videos are ubiquitous in robots and AR systems that act as a window into our dynamically changing 3D world. Being able to record movements in the physical space, point cloud sequences play a key role in comprehending environmental changes and supporting interactions with the world, which can be hardly described by 2D images or static 3D point clouds. Therefore, an intelligent agent must process such a form of data precisely to better model the real world, adapt to environmental changes, and interact with them.

Despite its importance, processing point cloud sequences is a quite challenging task for machines that are largely determined by two aspects: effectiveness and efficiency. Effectiveness refers to the ability to capture long-term spatial-temporal structures. Due to camera motion, scene changes, occlusion changes, and sampling patterns, points between different frames are unstructured and inconsistent, making it difficult to effectively integrate different frames into the underlying spatio-temporal structure. Efficiency refers to how to efficiently process long point cloud videos with limited computing resources. The complexity and dimension of 4D data can easily cause memory and computation explosions. Both challenges grow dramatically as the length of the video increases.

One typical way to tackle the dynamics of point clouds videos is treating the point cloud video as a 4D volume  [ 6 ] , which applies 4D convolution directly after voxelization. It is computationally prohibitive when processing large scenes and long videos. Compared with transformer-based 4D backbones, pure convolution is less effective at capturing long-term spatio-temporal context. However, the existing transformer-based 4D backbone(P4Transformer  [ 10 ] ) also fails to solve the above challenges. The entire point cloud video still needs to be loaded into memory during the training process, which severely limits the length of the point cloud video (for example, a 24GB graphics card can only handle a synthia4D  [ 34 ]  point cloud video of 3 frames). Additionally, even though flat transformers may be able to capture long-term context theoretically, they are difficult to optimize as point numbers increase and usually do not provide much gain in dense prediction tasks, such as 4D semantic segmentation.

Based on the challenges described above, we have several key observations. First, considering the large variety of points, distance point cloud frames should not be extracted at the point level, as this is neither efficient nor effective. Second, a middle-level abstraction representing the underlying geometry spatially and temporally can be better suited for context modeling, which will not only alleviate the need to process raw points for better efficiency but also allow for easier association across frames for a more effective spatial-temporal structure. After revisiting the geometry processing literature, we choose primitive plane as a mid-level representation, which describes the underlying planar structures in a scene and tends to be much more stable across frames.

Figure 1:  Architecture of Primitive Point Transformer. On the lower level, PPTr extracts short-term spatial-temporal features through an intra-primitive point transformer for a short video clip around the frame of interest. On the upper level, PPTr extracts long-term spatial-temporal features through a primitive transformer.

In this paper, we leverage primitive planes to develop an efficient and effective 4D backbone named Point Primitive Transformer(PPTr). As primitive planes induce a natural scene-primitive-point hierarchy in space, we also design PPTr as a hierarchical transformer operating on two different levels as shown in Figure

1  . On the lower level, PPTr extracts short-term spatial-temporal features through an intra-primitive point transformer for a short video clip around the frame of interest. Primitive planes are used to restrict the spatial support of attention maps in a point-level transformer. Such geometry-aware locality inductive bias is not only beneficial for the optimization of the transformer but also very effective for extracting descriptive and temporally stable geometric features. On the upper level, PPTr extracts long-term spatial-temporal features through a primitive transformer. We allow very efficient consideration of a long sequence by fitting primitives and computing the primitive features in a pre-processing stage. Through the primitive transformer, we could better associate primitives from different frames and effectively integrate long-term context to the frame of interest.

We evaluate our Point Primitive Transformer(PPTr) on several tasks, such as 3D action recognition on MSR-Action  [ 25 ]  and 4D semantic segmentation on Synthia4D  [ 34 ]  and HOI4D  [ 29 ] . we demonstrate significant improvements over previous method(

+

1.33  %

percent  1.33

+1.33\%

mIoU on synthia4D,

+

6.28  %

percent  6.28

+6.28\%

mIoU on HOI4D and

+

1.39  %

percent  1.39

+1.39\%

accuracy on MSR-Action).

The contributions of this paper are fourfold:

•

First, we leverage the primitive plane to capture the long-term spatial-temporal context in 4D point cloud videos and propose a novel backbone named Point Primitive Transformer(PPTr).

•

Second, we propose an intra-primitive point transformer for extracting spatially descriptive and temporally stable  short-term  geometric features.

•

Third, we propose a primitive transformer to capture  long-term  spatial-temporal features efficiently.

•

Fourth, extensive experiments on three datasets show that the proposed Point Primitive Transformer is more effective and efficient than previous state-of-the-art 4D backbones.

2  Related Work

Deep learning on Point Cloud Video Processing. 
Different from grid-based RGB video, point cloud video exhibits irregularities and lacks order along the spatial dimension where points emerge inconsistently across time. One approach to deal with that is voxilization. For instance,

[ 6 ]  extends temporal dimension to 3D sparse convolution  [ 15 ]  to extract spatial temporal features on 4D occupancy grids. 3DV  [ 41 ]  proposes a 3D motion representation to encode 3D motion information via temporal rank pooling  [ 12 ] . Another approach is to perform directly on point sets. MeteorNet  [ 28 ]  adopts PointNet++  [ 32 ]  to aggregate information from neighbors, while point-track is needed to merge points. PSTNet  [ 11 ]  firstly decomposes