[2602.06862] Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing

Parameters as Experts: Adapting Vision Models with Dynamic

Parameter Routing

Meng Lou

Stanley Yu

Yizhou Yu

Abstract

Adapting pre-trained vision models using parameter-efficient fine-tuning (PEFT) remains challenging, as it aims to achieve performance comparable to full fine-tuning using a minimal number of trainable parameters. When applied to complex dense prediction tasks, existing methods exhibit limitations, including input-agnostic modeling and redundant cross-layer representations. To this end, we propose AdaRoute, a new adapter-style method featuring a simple mixture-of-experts (MoE) architecture. Specifically, we introduce shared expert centers, where each expert is a trainable parameter matrix. During a feedforward pass, each AdaRoute module in the network dynamically generates weight matrices tailored for the current module via a simple dynamic parameter routing mechanism, which selectively aggregates parameter matrices in the corresponding expert center. Dynamic weight matrices in AdaRoute modules facilitate low-rank adaptation in an input-dependent manner, thus generating more customized and powerful feature representations.
Moreover, since AdaRoute modules across multiple network layers share the same expert center, they improve feature diversity by promoting implicit cross-layer feature interaction.
Extensive experiments demonstrate the superiority of AdaRoute on diverse vision tasks, including semantic segmentation, object detection and instance segmentation, and panoptic segmentation.
Code will be released at:  https://bit.ly/3NZcr0H .

Dense Predictions, Parameter-efficient Fine-tuning, Dynamic Parameter Routing, Shared Expert Centers

1  Introduction

Parameter-efficient fine-tuning (PEFT)

( han2024parameter )  aims to update or embed only a small number of parameters into a pre-trained model while performing comparably to full fine-tuning. This approach has been widely adopted in both natural language processing (NLP) and computer vision. For instance, prompt-based tuning in NLP tasks  ( liu2023pre )  has inspired many PEFT methods in vision. A representative work is VPT  ( jia2022visual ) , which inserts a set of learnable tokens into the input sequence of Vision Transformers (ViTs)  ( dosovitskiy2020image ;  liu2021swin ) , achieving task adaptation with minimal additional parameters. Although prompt-based tuning methods demonstrate promising performance on classification tasks, there still exists a considerable performance gap between these methods and full fine-tuning on more complex vision tasks, such as dense predictions. On the other hand, adapter-based PEFT methods  ( houlsby2019parameter )  have also attracted considerable attention. A well-known example is LoRA  ( hu2022lora ) , which learns low-rank adapters to achieve efficient fine-tuning of large language models (LLMs). In the same spirit, AdaptFormer  ( chen2022adaptformer )  introduces a lightweight MLP module to adapt ViTs, representing an early attempt to utilize adapters in visual recognition. Subsequently, Mona  ( yin2025Mona )  further integrates multi-scale depthwise convolutions into the adapter module to enhance its spatial modeling capacity for dense predictions. Although existing adapter-based methods have achieved promising results in diverse vision tasks, two key challenges remain unresolved:

Figure 1 :

(a)  Classical adapter-based PEFT methods (e.g., Mona  ( yin2025Mona ) ).  (b)  Our proposed AdaRoute. Normalization layers and residual connections are omitted for simplicity.  (c)  The first and second rows show ERF and CKA visualizations for various fine-tuned models, respectively. Specifically, Swin-L model pre-trained on ImageNet-21K is used as the backbone network, which is fine-tuned on the COCO2017 using various fine-tuning methods and the Mask R-CNN framework. Quantitative results are listed in Table

2  .

Representation Deficiency . As shown in Figure

1

(a), each adapter is responsible for task-specific model adaptations using input-agnostic low-rank modeling. For complex tasks such as dense predictions, it is inherently challenging to learn task-specific transformations that work universally well for all possible inputs. The fact that such adapters cannot support input-dependent adaptations to account for input variations limits the feature representation capacity of the resulting model. To empirically verify this, we use the effective receptive field (ERF)  ( luo2016understanding )  to visualize a model’s representation capacity. Specifically, we fine-tune the Swin-L model  ( liu2021swin )  pre-trained on ImageNet-21K  ( deng2009imagenet )  on the COCO2017 dataset  ( lin2014microsoft )  using the Mask R-CNN framework  ( he2017mask ) . As shown in the first row of Figure

1

(c), previous representative PEFT methods, including AdaptFormer and Mona, exhibit smaller ERFs compared to full fine-tuning. This phenomenon indicates that these methods weaken the model’s ability to capture complex spatial dependencies required for dense prediction tasks.

Feature Redundancy . Adapters embedded in different network layers and their parameters are isolated from each other, and such a lack of cross-layer interaction may lead to redundant representations. To illustrate this limitation, we perform a centered kernel alignment (CKA) analysis  ( kornblith2019similarity )  for different methods. As shown in the second row of Figure

1

(c), the patterns learned by different layers in both AdaptFormer and Mona exhibit a higher similarity compared to those under full fine-tuning, which means that different layers capture redundant information.

To address these limitations, we propose a new adapter-based PEFT method dubbed AdaRoute. As illustrated in Figure

1

(b), AdaRoute is built upon a simple mixture-of-experts (MoE) architecture. Specifically, we construct a large shared expert center comprising a collection of trainable parameter matrices, each having the same size as the corresponding weight matrix in a standard adapter. Each AdaRoute module in the network dynamically generates weight matrices tailored for the current module via a dynamic parameter routing mechanism, which selectively aggregates parameter matrices in this shared expert center. This routing mechanism is analogous to the gating mechanism in MoE that selects appropriate experts for a given input, and the trainable parameter matrices are treated as experts in this work. Although our design is simple, it offers two advantages that are absent in previous works:

First, dynamic weight matrices in AdaRoute modules facilitate low-rank adaptation in an input-dependent manner, thus generating more customized and powerful feature representations. As evidenced in Figure

1

(c), the ERF of our model is larger than those of other PEFT methods and comparable to that of full fine-tuning. Such a large ERF enables our model to capture long-range dependencies more easily, which is crucial in dense predictions  ( xie2021segformer ;  fu2025segman ;  lou2025overlock ) .

Second, since the same expert center is shared among AdaRoute modules in multiple network layers, an implicit cross-layer feature interaction can be developed to diversify the information flow, thus reducing feature redundancy  ( huang2017densely ;  kim2024densenets ;  lou2024sparx ) . As evidenced in Figure

1

(c), the feature diversity of our fine-tuned model is better than that of other PEFT methods and very close to that of full fine-tuning. This means that our method can extract more representative features from complex scenes for dense predictions.

We have evaluated our method on a wide range of visual recognition tasks, including semantic segmentation, object detection and instance segmentation, panoptic segmentation, and image classification. Extensive experiments in Section

4

demonstrate that our method has achieved superior pe