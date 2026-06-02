[2402.16915] More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning

More Than Routing: Joint GPS and Route Modeling for Refine Trajectory Representation Learning

Zhipeng Ma

0009-0008-1485-0766

Southwest Jiaotong University  Chengdu  China

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

mazhipeng1024@my.swjtu.edu.cn

,

Zheyan Tu

0000-0003-0839-4262

McGill University  Montreal  Canada

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

zheyan.tu@mail.mcgill.ca

,

Xinhai Chen

0009-0001-9924-2397

Southwest Jiaotong University  Chengdu  China

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

chenxinhaier@gmail.com

,

Yan Zhang

0000-0003-2142-5094

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

zhangyan@air.tsinghua.edu.cn

,

Deguo Xia

0000-0003-3366-2230

Baidu Inc.  Beijing  China

xiadeguo@baidu.com

,

Guyue Zhou

0000-0002-3894-9858

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

zhouguyue@air.tsinghua.edu.cn

,

Yilun Chen

0000-0003-0618-3621

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

chenyilun@air.tsinghua.edu.cn

,

Yu Zheng

0000-0003-2537-4685

JD iCity, JD Technology  Beijing  China

JD Intelligent Cities Research  Beijing  China

msyuzheng@outlook.com

and

Jiangtao Gong

0000-0002-4310-1894

Institute for AI Industry Research (AIR), Tsinghua University  Beijing  China

gongjiangtao2@gmail.com

Abstract.

Trajectory representation learning plays a pivotal role in supporting various downstream tasks.
Traditional methods in order to filter the noise in GPS trajectories tend to focus on routing-based methods used to simplify the trajectories.
However, this approach ignores the motion details contained in the GPS data, limiting the representation capability of trajectory representation learning.
To fill this gap, we propose a novel representation learning framework that  J oint  G PS and  R oute  M odelling based on self-supervised technology, namely  JGRM .
We consider GPS trajectory and route as the two modes of a single movement observation and fuse information through inter-modal information interaction.
Specifically, we develop two encoders, each tailored to capture representations of route and GPS trajectories respectively. The representations from the two modalities are fed into a shared transformer for inter-modal information interaction.
Eventually, we design three self-supervised tasks to train the model.
We validate the effectiveness of the proposed method on two real datasets based on extensive experiments. The experimental results demonstrate that JGRM outperforms existing methods in both road segment representation and trajectory representation tasks. Our source code is available at  https://anonymous.4open.science/r/JGRM-DAD6/ .

Trajectory representation learning, Spatio-temporal data mining, self-supervised learning

†

†  conference:  Make sure to enter the correct
conference title from your rights confirmation emai; MAY 13–17,
2024; Singapore

1.  Introduction

With the development of location-based services including map services and location-based social networks, the generation and analysis of trajectory data have become pervasive, providing valuable insights into the mobility of various entities, such as individuals, vehicles and animals. These trajectory data contain rich spatial and temporal information that can be applied to urban planning  (Bao et al . ,  2017 ; He et al . ,  2020 ) , urban emergency management  (Zhu et al . ,  2021 ; Ji et al . ,  2022 ) , infectious disease prevention and control  (Alessandretti,  2022 ; Feng et al . ,  2020 ) , and intelligent logistics systems  (Ruan et al . ,  2022 ; Feng et al . ,  2023 ; Lyu et al . ,  2023 ) . However, to exploit the full potential of these data, the development of effective trajectory representation methods has emerged as a critical topic. Trajectory representation learning focuses on transforming raw trajectory data into meaningful and compact representations that can be used for a variety of tasks, such as travel time estimation  (Wang et al . ,  2018a ) , trajectory classification  (Liang et al . ,  2022 )  and Top-k similar trajectory query  (Yao et al . ,  2019 ) .

Early studies on learning trajectory representations were based on sequential models designed for a specific downstream task and trained using the specific task loss  (Yao et al . ,  2017a ; Wang et al . ,  2018b ; Liu et al . ,  2019 ) . These representations are not generalized and tend to crash on other tasks. To solve this problem, seq2seq-based methods have been proposed, which are trained by reconstructive loss  (Yao et al . ,  2017b ; Li et al . ,  2018 ; Fang et al . ,  2021 )  to make generalized representations. After that, due to redundancy and noise in the GPS trajectory, the method using route trajectory instead of raw GPS trajectory became mainstream. These methods introduce many NLP techniques, including Word2Vec and BERT, due to the similarity between route trajectories and natural language sentences  (Chen et al . ,  2021 ; Yang et al . ,  2023 ) . Recently, with the rise of graph neural networks, researchers have begun to focus on the spatial relationships between road segments. Therefore, some two-step methods  (Fu and Lee,  2020 ; Fang et al . ,  2022 )  have been proposed, which first model the spatial relationships between road segments using the topology of the road network, and then use the updated road segments for temporal modeling using the sequence model. On this basis, a multitude of self-supervised training methods have been designed in order to train trajectory representation models in a task-free manner  (Yang et al . ,  2021a ; Mao et al . ,  2022 ; Jiang et al . ,  2023 ) .

However, these methods simply treat road segments as conceptual entities (similar to words in natural language), ignoring the fact that a road segment is a real geographic entity that can interact with objects that pass through it. For example, when a road segment is congested, the movement pattern of passing vehicles is different than when the road is clear. So, different types of roads and different traffic states can really affect mobility. To this end, we believe that modeling road segments as geographic entities can effectively improve trajectory representation. Fortunately, the raw GPS points can serve as localized observations of the geographic entity. However, while the GPS trajectory contains richer information, it also contains a large amount of redundancy and noise and is not effective at capturing high-level transfer patterns. An intuitive idea is to combine the GPS view and the route view together to represent the trajectory more comprehensively.

Figure 1.  Route Modeling v.s. Fusion Modeling.

As shown in Figure

1  (a), a road segment in the route trajectory, can only be modeled through preceding and succeeding road segments and lack of direct self-observation. In contrast, road segments in GPS trajectories offer much richer sampling information allowing for a fine-grained representation of road segment entities. Moreover, the context of road segments in the route trajectory can further refine the road representations. In fact, GPS trajectory and route trajectory simultaneously describe different perspectives of the same movement behavior and can complement each other. The GPS trajectory describes the movement details of the object, which can reflect the interaction of the object with the geospatial space as it moves, and can better model the road segment entities. However, GPS trajectories are inherently noisy and redundant, which can degrade performance when modeling sequences. Route trajectory describes the travel semantics of an object, has a robust state transfer record, and can reflect travel intentions and preferences. What’s