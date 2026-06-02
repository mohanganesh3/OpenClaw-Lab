[2508.06024] EC2MoE: Adaptive End-Cloud Pipeline Collaboration Enabling Scalable Mixture-of-Experts Inference

EC  2  MoE : Adaptive End-Cloud Pipeline Collaboration Enabling Scalable Mixture-of-Experts Inference

Zheming Yang 1 , Yunqing Hu 1,3 , Sheng Sun 1 , Wen Ji 1,2

Abstract

The Mixture-of-Experts (MoE) paradigm has emerged as a promising solution to scale up model capacity while maintaining inference efficiency. However, deploying MoE models across heterogeneous end-cloud environments poses new challenges in expert scheduling, communication overhead, and resource heterogeneity. In this paper, we propose  EC  2  MoE , an adaptive framework for scalable MoE inference via end-cloud pipeline collaboration. First, we design a hardware-aware lightweight group gate network that enhances expert selection and computational efficiency. By incorporating a hardware-aware local expert selection mechanism, the system adaptively filters candidate experts based on real-time device profiles. A lightweight group gate module then integrates local and global gating outputs to achieve high-quality expert routing with minimal overhead. Second, we develop a pipeline optimization mechanism based on end-cloud collaboration to accelerate MoE inference. This includes an encoder-decoder structure based on low-rank compression, which reduces transmission and computation costs. And a route-aware heuristic pipeline scheduling algorithm that dynamically allocates inference stages across devices according to workload and network topology. Extensive experiments show that  EC  2  MoE  can increase throughput by 2.2

×  \times

to 5.1

×  \times

and reduce end-to-end latency by 53% to 67% while maintaining high accuracy compared to state-of-the-art methods. It also maintains good scalability under dynamic load and network environments.

Introduction

In recent years, the demand for large-scale deep learning models has grown dramatically  (Ge et al.  2023 ; Shen et al.  2024a ; Menghani  2023 ) , driven by the rapid advancement of AI applications in areas such as natural language understanding, computer vision, and multi-modal reasoning  (Liang et al.  2024 ) . To meet this demand, researchers have explored various model scaling strategies  (Hwang et al.  2023 ; Chen et al.  2024 ; Yin et al.  2024 ) . Among them, the Mixture-of-Experts (MoE) architecture has emerged as a particularly promising solution  (Zhou et al.  2022 ) . By selectively activating a sparse subset of expert networks during inference, MoE models enable substantial increases in parameter count—often reaching hundreds of billions—without incurring a proportional increase in computational cost  (Chen et al.  2022 ; Riquelme et al.  2021 ) . This sparsity-aware computation makes MoE architectures well-suited for balancing inference efficiency and model expressiveness  (Szatkowski et al.  2024 ; Liu et al.  2025 ) , enabling state-of-the-art performance across a range of complex AI tasks.

Despite these advantages, efficiently deploying MoE models in real environments remains challenging  (Liu, Wang, and Wu  2025 ) . Unlike conventional single models that can be easily compressed or quantized for end deployment, MoE models consist of multiple dynamically invoked expert models, whose activation patterns vary per input and require efficient gating and routing mechanisms  (Rajbhandari et al.  2022 ) . This variability introduces new difficulties in system-level resource scheduling, model placement, and expert communication  (Gale et al.  2023 ; Cao et al.  2025 ) . On one hand, resource-constrained end devices struggle to support the intensive computational demands of high-capacity MoE backbones, especially when multiple expert paths are involved  (Shen et al.  2024b ; Jin et al.  2025 ) . This results in degraded inference accuracy or throughput if only local computation is used  (Li et al.  2025 ) . On the other hand, relying solely on the cloud introduces latency overhead and may lead to underutilization of local resources  (Deshpande et al.  2024 ) . Moreover, cloud-only execution becomes highly sensitive to network fluctuations  (Yu et al.  2024 ) , making it difficult to guarantee consistent performance in latency-critical applications. These issues are further exacerbated by fluctuating network conditions, device heterogeneity, and dynamic workload patterns commonly encountered in real-world scenarios.

To address these challenges, we propose  EC  2  MoE , a novel adaptive framework that enables scalable and efficient MoE inference through end-cloud pipeline collaboration in heterogeneous distributed environments.  EC  2  MoE  integrates hardware-aware expert selection with coordinated execution across end and cloud, effectively mitigating resource constraints, network variability, and communication overhead. By jointly optimizing expert routing and inference scheduling, our framework achieves high throughput, low latency, and robust scalability under dynamic workloads. Our main contributions are as follows:

•

We propose a hardware-aware lightweight group gate network that efficiently adapts MoE expert selection to heterogeneous end-cloud environments. By incorporating a local expert selection mechanism based on device hardware characteristics and designing a lightweight group gate network module, it significantly reduces inference latency and routing overhead while maintaining expert selection quality.

•

We develop a collaborative end-cloud pipeline optimization mechanism tailored for scalable MoE inference. It integrates a low-rank compression-based encoder-decoder to reduce transmission costs and a route-aware heuristic pipeline scheduler that dynamically maps inference sub-tasks across end and cloud based on workload and communication patterns, maximizing overall throughput.

•

We evaluate the performance of the framework and compare it with mainstream baseline methods. Experimental results show that  EC  2  MoE  can increase throughput by 2.2

×  \times

to 5.1

×  \times

and reduce end-to-end latency by 53% to 67%, and without sacrificing accuracy.

Related Works

Cloud-based MoE inference optimization

Cloud-based MoE inference optimization has been widely studied due to the abundant computing resources and scalability of cloud infrastructures  (Hwang et al.  2024 ; Hu et al.  2025 ) . Early works primarily focused on efficient expert routing and load balancing to minimize the communication and computation overhead during inference. For example, GShard  (Lepikhin et al.  2020 )  and Switch Transformer  (Fedus, Zoph, and Shazeer  2022 )  introduced sparse expert activation and simplified gating mechanisms to enable large-scale MoE training and inference in cloud environments. These methods significantly improved model scalability while controlling inference costs by activating only a subset of experts per input. Subsequent research further explored expert placement and communication optimization. Tutel  (Hwang et al.  2023 )  and EfficientMoE  (Zeng et al.  2025 )  implemented expert parallelism strategies and expert sharding to minimize cross-device communication during MoE inference, enhancing throughput and reducing latency. In addition, model parallelism frameworks such as DeepSpeedMoE  (Dai et al.  2024 )  and Fsmoe  (Pan et al.  2025 )  provided system-level optimizations for cloud-based deployment by improving the scheduling of expert computation and network transfer. These works, however, often assume homogeneity and high bandwidth availability in cloud clusters, which may not generalize well to hybrid or end scenarios.

End-based MoE inference optimization

.
Deploying MoE models on resource-constrained end devices faces limitations in terms of memory and computing power. To address this, previous studies have proposed various lightweight and dynamic management methods to improve inference efficiency. EdgeMoE  (Yi et al.  2025 )  and D 2 MoE  (Wang et al.  2025 )  achieve expert sele