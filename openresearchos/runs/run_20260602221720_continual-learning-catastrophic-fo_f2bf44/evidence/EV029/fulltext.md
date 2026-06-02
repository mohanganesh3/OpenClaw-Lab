[2412.11074] Adapter-Enhanced Semantic Prompting for Continual Learning

Adapter-Enhanced Semantic Prompting for Continual Learning

Baocai Yin  1

Ji Zhao  1

Huajie Jiang  1

Corresponding authors

Ningning Hou  2

Yongli Hu  1

Amin Beheshti  2

Ming-Hsuan Yang  3

Yuankai Qi  2

1  School of Information Science and Technology, Beijing University of Technology,China

2  School of Computing, Macquarie University,Australia

3  Electrical Engineering and Computer Science, University of California at Merced, USA

Abstract

Continual learning (CL) enables models to adapt to evolving data streams.
A major challenge of continual learning is catastrophic forgetting, where new knowledge overwrites previously acquired knowledge.
Prompt-based CL methods receive increasing attention due to the avoidance of computing-heavy replay of old data.
In this paper, we introduce a new lightweight framework for continual learning called Adapter-Enhanced Semantic Prompting (AESP).
It stands out from the existing prompt-based CL framework in three aspects.
First, we propose to utilize a  (semantic prompt, learnable visual prompt)  pair to represent task-specific knowledge instead of just visual prompts as in existing works.
The semantic prompt is a representative summary of each task, obtained via a large language model,  e.g ., BERT, using class labels as input. It provides extra rich information to visual prompts.
Second, the semantic prompt is also used as input for our Adapter-enhanced ViT. Both the newly introduced semantic prompt and adapters enable better adaptation of frozen ViTs for new tasks.
Third, we design a strong Query-Key matching method for selecting an appropriate task prompt pair to improve final prediction accuracy.
Extensive experiments across three continual learning datasets demonstrate that the proposed framework achieves favorable performance compared to several state-of-the-art approaches.

1  Introduction

Continual learning (CL)  [  2  ,

4  ,

25  ,

46  ,

56  ,

47  ]  aims to enable models to continuously learn from evolving data streams and adapt to new environments.
Different from traditional machine learning models  [  21  ,

30  ]  that are trained on fixed datasets and become static after training, CL enables models to accumulate existing knowledge and learn new information with new-task data and existing models, which avoids retaining all old task data and retrain the model  [  25  ,

15  ] .
CL has received increasing attention in recent years due to its promising capability to deal with challenges like data privacy and memory resource limitations  [  29  ,

34  ,

39  ] .

However, CL suffers from catastrophic forgetting  [  31  ,

13  ,

41  ] , which is caused by the overwriting of old knowledge when learning new tasks, resulting in a significant decline in performance for previous tasks.
To address this issue, some approaches  [  19  ,

45  ,

3  ]  propose to retain a few representative samples from the old tasks for knowledge replay. When the next task comes, the retained samples are replayed with the new task data.
Such approaches have demonstrated their effectiveness in mitigating the model’s forgetting.
However, retaining samples may impose a memory burden and introduce privacy leakage risks.
Besides retaining samples,  [  25  ,

5  ]  propose to add new branches to the network for each new task, which increases the model size and results in slower inference speed and lower efficiency.

Figure 1 :

Comparison between previous prompt-based approaches and our framework. Previous approaches mainly use visual prompts to update the image features. In contrast, our method introduces semantic prompts (S-Prompts) to enrich the semantic information and embeds a fine-tunable adapter structure to effectively learn adaptive image features.

In recent years, prompt-based methods  [  49  ,

48  ,

44  ,

35  ,

23  ]  have been proposed for CL thanks to their ability to leverage the knowledge of pre-trained Visual Transformer (ViT) with fewer parameters and higher efficiency.
As is shown in Figure

1

(a), these methods learn visual prompts to efficiently adapt the pre-trained ViT model to the CL tasks.
However, they rely solely on visual information, which may cause the model to struggle in acquiring knowledge with strong generalization capabilities, especially when the training data is limited and lacks sufficient diversity.
By contrast, class semantic information extracted by large language models is more generalized and adaptable across tasks.
Models like BERT  [  10  ]  and CLIP  [  38  ]  align samples of the same class with the corresponding semantic information associated with their class label, thus creating a more generalized feature space. This alignment enables the model to maintain robustness when encountering unseen samples of the same class, as the model learns to focus on the shared semantic features, rather than relying solely on specific visual features.

In this work, we propose a novel Adapter-Enhanced Semantic Prompting (AESP) framework, which introduces semantic information to enhance the generalization of the pre-trained ViT model for more adaptive visual feature learning.
However, semantic and visual information differ significantly in terms of representation and feature space. Relying solely on the attention modules in a pre-trained ViT may not effectively fuse these two types of information.
To address this, we introduce small, learnable adapter modules into the ViT model. As shown in Figure

1

(b), adapters are integrated into each layer of the ViT to help the model better accommodate semantic information and facilitate the effective interaction between visual and semantic features.
Therefore, the model not only combines multimodal features more effectively but also enables the learned knowledge to exhibit better generalization ability, leading to improved adaptability in the new coming tasks.

To accurately select the task-specific prompt for feature adaptation, we propose a new prompt-key matching method that combines four matching strategies to ensure that the most relevant prompts are selected from the prompt pool.
Extensive experiments demonstrate that our proposed AESP framework can effectively improve the model’s performance across various incremental tasks.

The main contributions are summarized as follows:

•

We propose a novel Adapter-Enhanced Semantic Prompting framework for continual learning, which integrates multimodality prompts and adapters to facilitate adaptive feature learning.

•

We propose to utilize semantic prompts to improve the generalization ability of visual features and design a cosine contrast loss for effective optimization.

•

We integrate multiple prompt-key matching strategies to improve the accuracy of prompt selection.

•

Extensive experiments on three continual learning datasets demonstrate that our method achieves favorable performance compared to several state-of-the-art approaches.

2  Related Work

Continual Learning.

Continual Learning (CL) is an advanced area within machine learning that allows models to continuously learn new information while preserving previously acquired knowledge  [  4  ,

25  ,

54  ,

56  ,

47  ] .
However, the process of acquiring new knowledge can overwrite earlier learning, leading to the issue of catastrophic forgetting  [  31  ,

13  ,

41  ] .
To address this, CL techniques implement various strategies to balance retaining old knowledge and integrating new information.
One such approach involves regularization methods, which safeguard critical parameters by imposing constraints that prevent significant changes that might erase previously learned knowledge  [  33  ,

1  ,

50  ] .
Another approach called knowledge distillation, is frequently used to facilitate a smooth transition between older and newer models  [  11  ,

26  ,

27  ] .
Parameter isolation strategies  [  40  ,

51  ,

53  ]  protect existing knowledge by freezing ce