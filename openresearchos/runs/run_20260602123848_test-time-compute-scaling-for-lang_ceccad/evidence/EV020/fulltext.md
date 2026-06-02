[2503.03854] Vision-Language Models Struggle to Align Entities across Modalities

Vision-Language Models Struggle to Align Entities across Modalities

Iñigo Alonso 1  ,
 Ander Salaberria 2  ,
 Gorka Azkune 2  ,
 Jeremy Barnes 2  ,
 Oier Lopez de Lacalle 2  ,

1 Institute for Language, Cognition and Computation, University of Edinburgh

2 HiTZ Center - Ixa, University of the Basque Country UPV/EHU,

Correspondence:

ialonso@ed.ad.uk

Abstract

Cross-modal entity linking refers to the ability to align entities and their attributes across different modalities. While cross-modal entity linking is a fundamental skill needed for real-world applications such as multimodal code generation, fake news detection, or scene understanding, it has not been thoroughly studied in the literature. In this paper, we introduce a new task and benchmark to address this gap. Our benchmark, MATE, consists of 5.5k evaluation instances featuring visual scenes aligned with their textual representations. To evaluate cross-modal entity linking performance, we design a question-answering task that involves retrieving one attribute of an object in one modality based on a unique attribute of that object in another modality. We evaluate state-of-the-art Vision-Language Models (VLMs) and humans on this task, and find that VLMs struggle significantly compared to humans, particularly as the number of objects in the scene increases. Our analysis also shows that, while chain-of-thought prompting can improve VLM performance, models remain far from achieving human-level proficiency. These findings highlight the need for further research in cross-modal entity linking and show that MATE is a strong benchmark to support that progress.

Vision-Language Models Struggle to Align Entities across Modalities

Iñigo Alonso 1 ,
Ander Salaberria 2 ,
Gorka Azkune 2 ,
Jeremy Barnes 2 ,
Oier Lopez de Lacalle 2 ,

1 Institute for Language, Cognition and Computation, University of Edinburgh

2 HiTZ Center - Ixa, University of the Basque Country UPV/EHU,

Correspondence:

ialonso@ed.ad.uk

1  Introduction

Several real-world applications demand the ability to perform cross-modal entity linking, i.e., being able to align entities and attributes across modalities. In autonomous driving, for example, a single image of a scene may contain multiple entities, such as pedestrians and other vehicles. Additionally, textual or structured data about these entities, provided by smart devices or other cars, can include information like speed or future trajectory. While some attributes, such as vehicle color or shape, are shared between visual and textual representations, others, like speed, exist only in the text. To navigate effectively, the car must link vehicles in the image to their corresponding textual data, creating a unified representation of the scene. The same is true for other tasks such as multimodal code generation

Mu et al.  ; Li et al. ( 2024b ) , multimodal fake news detection  Jing et al. ( 2023 ); Ma et al. ( 2024 )  and multimodal scene understanding  Su et al. ( 2024 ); Li et al. ( 2024a ) .

Figure 1:  Example of two MATE questions: (1)  Image-to-text : The model must identify the object in the image based on a visual attribute (red) and retrieve its name from the text ("Object_0"). (2)  Text-to-image : The model must locate the object in the text using its name and determine its color, which is only present in the image. Both tasks require linking entities across modalities, but in opposite directions.

We can state, thus, that cross-modal entity linking is a basic ability needed to enable further applications of multimodal artificial intelligence systems. However, to the best of our knowledge, no exhaustive and targeted studies can be found in the literature. To fill that gap, in this paper we analyze the capabilities of current Vision-Language Models (VLM) for cross-modal entity linking. Specifically, we build a new multimodal question-answering benchmark called MATE that contains synthetic images of 3D scenes aligned with textual representations of those scenes (Figure

1  ). We use synthetic images to control for all the variables of the problem and to easily generate different experiments. The task we use to evaluate the ability of models to perform cross-modal entity linking is shown in Figure

1  . Given a pointer attribute which is unique in one of the modalities (e.g., the red color in the first question), we ask for a target attribute of that object which is shown in the other modality (e.g., the name of the referred entity). As the modalities we consider are visual and textual, we have image-to-text (question 1 of Figure

1  ) and text-to-image tasks (question 2), depending on where the pointer and target attributes exist. MATE is specifically designed for evaluating cross-modal entity linking and alignment, a core skill for multimodal tasks.

We evaluate and analyze open- and closed-weight VLMs on MATE and perform further human evaluations. As a result of our experiments, we find that:

1) VLMs and humans show very different behaviors for cross-modal entity linking:

While humans achieve almost perfect performance, current VLMs fail to consistently align attributes across modalities out-of-the-box. Furthermore, VLMs are generally worse for the image-to-text variant; humans show a balanced performance. Finally, the performance of VLMs is heavily influenced by the number of objects in the scene and the target attribute requested, whereas human performance is stable across different configurations.

2) The real challenge of our task resides in the cross-modal setting:

We show that VLMs can proficiently solve the task of entity-attribute linking when only one modality is considered. However, VLM performance decreases significantly as the number of attributes required to link entities across both modalities increases.

3) MATE is an effective benchmark for evaluating cross-modal entity linking in VLMs:

Our results indicate that MATE is a useful resource for evaluating the cross-modal alignment capabilities of current models. While human performance remains consistently high, VLMs suffer a significant drop as scene complexity increases, highlighting the challenges that cross-modal entity linking presents for current models.

Our dataset, evaluation results, and code are publicly available.  1

1  1 Our dataset, evaluation results, and code are publicly available at  https://github.com/AlonsoApp/MATE

2  Related Work

Several multimodal tasks in the literature are related to the one we propose, often grouped under the generic term of  visual grounding . For example, Referring Expression Comprehension (REC)  Kazemzadeh et al. ( 2014 )  requires identifying the image region described by a textual mention, typically referring to objects or physical entities along with their attributes and relations to other objects. Similarly, the Situated and Interactive Multimodal Conversations dataset (SIMMC)  Moon et al. ( 2020 ); Kottur et al. ( 2021 )  introduces a multimodal dialogue task where a system assists users in a virtual shopping scenario. To complete the task, the system must link visual objects to their textual metadata and search for relevant information. While both tasks share the challenge of aligning visual and textual content, our task extends this further by requiring explicit cross-referencing of object attributes across modalities. In particular, SIMMC avoids this challenge in its shared task by providing gold object IDs, eliminating the need for linking from raw multimodal inputs.

Multimodal Entity Linking (MEL)  Gan et al. ( 2021 ); Adjali et al. ( 2020 ); Song et al. ( 2024 )  is a related task where mentions in multiple modalities are disambiguated by linking them to the corresponding named entities in a knowledge base such as Wikipedia. While previous research  Yao et al. ( 2024 )  has primarily focused on scenarios where the image provides supporting visual information for a