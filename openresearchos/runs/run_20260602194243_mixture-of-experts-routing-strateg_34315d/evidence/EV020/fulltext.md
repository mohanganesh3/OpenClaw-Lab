[2312.09925] CNC-Net: Self-Supervised Learning for CNC Machining Operations

CNC-Net: Self-Supervised Learning for CNC Machining Operations

Mohsen Yavartanoo  1

Sangmin Hong  2

†

†  footnotemark:

Reyhaneh Neshatavar  1

†

†  footnotemark:

Kyoung Mu Lee  1,2

1  Dept. of ECE &amp; ASRI,

2  IPAI, Seoul National University, Seoul, Korea

{myavartanoo,mchiash2,reyhanehneshat,kyoungmu}@snu.ac.kr

equal contribution

Abstract

CNC manufacturing is a process that employs computer numerical control (CNC) machines to govern the movements of various industrial tools and machinery, encompassing equipment ranging from grinders and lathes to mills and CNC routers.
However, the reliance on manual CNC programming has become a bottleneck, and the requirement for expert knowledge can result in significant costs.
Therefore, we introduce a pioneering approach named CNC-Net, representing the use of deep neural networks (DNNs) to simulate CNC machines and grasp intricate operations when supplied with raw materials.
CNC-Net constitutes a self-supervised framework that exclusively takes an input 3D model and subsequently generates the essential operation parameters required by the CNC machine to construct the object.
Our method has the potential to transformative automation in manufacturing by offering a cost-effective alternative to the high costs of manual CNC programming while maintaining exceptional precision in 3D object production.
Our experiments underscore the effectiveness of our CNC-Net in constructing the desired 3D objects through the utilization of CNC operations.
Notably, it excels in preserving finer local details, exhibiting a marked enhancement in precision compared to the state-of-the-art 3D CAD reconstruction approaches.

1  Introduction

Manufacturing processes have undergone remarkable transformations over the past decades, driven by automation and the advancement of computational techniques.
A domain that has witnessed substantial innovation is Computer Numerical Control (CNC) machining, a pivotal pillar of modern manufacturing.
CNC machines have revolutionized manufacturing by producing complex products with better precision, efficiency, and robustness  [  1  ]  in diverse industries, from aerospace to medical devices.
Despite their numerous advantages, CNC machines still grapple with certain limitations, particularly in manual programming and adaptability.
Traditional CNC programming requires intricate sets of instructions crafted by Computer-Aided Manufacturing (CAM) software that guide machine tools, including mills and drills, to produce the intended object.
However, despite its effectiveness, this process introduces bottlenecks due to its labor-intensive nature and reliance on expert knowledge.
Furthermore, adapting CNC machines to new tasks typically involves extensive reprogramming, hindering their agility and responsiveness in dynamic manufacturing environments.
Incorporating deep learning techniques into CNC machining offers a transformative solution to address these challenges.
In particular, several recent studies use deep neural networks (DNNs) to explore 3D objects using Constructive Solid Geometry (CSG)  [  19  ]  operations, employing both a set of simple  [  34  ,

31  ]  and more complex  [  6  ,

36  ,

37  ]  primitives.
Therefore, the ability of DNNs to learn complex patterns from data makes them an ideal candidate for revolutionizing CNC manufacturing, which can pave the way for automation, adaptive programming, and efficient utilization of CNC machines.
However, the intricate search space for operations on complex objects involving NP-hard problems presents a challenge in labeling optimal solutions as ground truth (GT).
Consequently, lacking a dataset with such a GT as supervision poses challenges in training a DNN model.

To mitigate these challenges, we propose CNC-Net, a DNN-based approach designed to simulate generic CNC machines in a self-supervised manner.
Our approach can construct target objects without relying on the GT labels ( i.e .  , a set of sequential operations).
CNC-Net is structured to incrementally learn the production of target 3D shapes, thereby determining the subsequent set of operations by implicitly modeling milling and drilling operations.
This capability enables CNC-Net to generate the necessary machining steps effectively.
At each operational step, the tools are represented as cylindric primitives, and the CNC-Net determines the radius of the tool and identifies the path coordinates for the subsequent milling or drilling action.
To enhance the carving capabilities of a CNC machine, we introduced a feature that enables the machine to rotate the workpiece along the

X

𝑋

X

and

Y

𝑌

Y

axes.
This functionality is commonly found in advanced CNC setups.
In this scenario, guided solely by the target shape and lacking prior information or labeled operations, CNC-Net models 3D shapes at each step, involving subtracting operations, represented as the union of cylindric primitives, from the outcomes of preceding steps.
This enables CNC-Net to learn the essential operations to reconstruct the target shapes accurately.

In our experiments, we provide the competitive performance of CNC-Net in reproducing 3D shapes compared to state-of-the-art (SOTA) CAD reconstruction methods.
To validate the effectiveness of our approach, we conduct experiments on both industrial objects from the ABC dataset  [  16  ]  and more intricate objects obtained from the ShapeNet dataset  [  3  ] .
Our main contributions are threefold:

•

We introduce CNC-Net, a pioneer self-supervised and DNN-based approach for simulating CNC machines.

•

CNC-Net learns to automatically find the sequential operations required for sculpting a 3D shape and exhibits capability akin to expert human labor without the need for labels or any prior information.

•

The experiments demonstrate that our self-supervised CNC-Net method can precisely reproduce target objects and outperform SOTA methods in terms of 3D reconstruction performance based on volume-based metrics.

(a)

Target Shape.

(b)

Input Material.

(c)

Tool.

(d)

Path.

(e)

Milling.

(f)

Drilling.

(g)

Rotation.

Figure 1 :

Overview of a generic CNC machine features.

2  Related Works

This section covers previous studies related to our method, divided into two categories: reverse engineering of 3D shapes and machine learning for CNC machines.

Reverse engineering 3d shapes.

Reverse engineering a 3D shape refers to understanding the features and structure of the original object and learning how it is constructed.
With the development of deep learning,
several approaches have been proposed to investigate how a 3D shape is assembled.
In recent years, there has been an exploration of the use of simple geometric primitives to approximate a 3D shape with a pre-defined set of cubes  [  33  ,

40  ,

27  ] , ellipsoids  [  10  ] .
More recent studies improve the representation ability and surface reconstruction by introducing more flexible and deformable primitives  [  6  ,

36  ,

28  ,

13  ] .
These works represent a shape as a union of primitives using constructive solid geometry (CSG)  [  19  ] , which relies on Boolean operations applied to the primitives  [  9  ] .
On the other hand, there exist various methods  [  31  ,

8  ,

5  ]  that assemble primitives using a sequence of modeling operations through reinforcement learning (RL).
These methods aim to match a target shape in a reverse engineering manner.
Furthermore, recent supervised primitive networks  [  22  ,

32  ]  have been designed to detect and fit primitives within point clouds, which initially identify primitive types.
Subsequently, they estimate their parameters or integrate spline patches, incorporating differentiable metric-learning segmentation.
Additionally, CSGNet  [  31  ]  is a neural network approach to form a CSG program from a given shape, and InverseCSG  [  8  ]  solves it as a program synthesis