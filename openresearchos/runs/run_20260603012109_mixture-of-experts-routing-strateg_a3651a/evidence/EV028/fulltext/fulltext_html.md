[2511.11628] Mixture-of-Schedulers: An Adaptive Scheduling Agent as a Learned Router for Expert Policies

Mixture-of-Schedulers: An Adaptive Scheduling Agent as a Learned Router for Expert Policies

Xinbo Wang

xinbowang@zju.edu.cn

0009-0006-0010-2623

,

Shian Jia

csjsa@zju.edu.cn

0009-0003-6759-5889

Zhejiang University  Hangzhou  Zhejiang  China

,

Ziyang Huang

Zhejiang University  Hangzhou  Zhejiang  China

3220105926@zju.edu.cn

0009-0000-1006-4209

,

Jing Cao

HangZhou City University  Hangzhou  Zhejiang  China

jingcao@hzcu.edu.cn

and

Mingli Song

Zhejiang University  Hangzhou  Zhejiang  China

brooksong@zju.edu.cn

0000-0003-2621-6048

Abstract.

Modern operating system schedulers employ a single, static policy, which struggles to deliver optimal performance across the diverse and dynamic workloads of contemporary systems. This “one-policy-fits-all” approach leads to significant compromises in fairness, throughput, and latency, particularly with the rise of heterogeneous hardware and varied application architectures.

This paper proposes a new paradigm: dynamically selecting the optimal policy from a portfolio of specialized schedulers rather than designing a single, monolithic one. We present the Adaptive Scheduling Agent (ASA), a lightweight framework that intelligently matches workloads to the most suitable “expert” scheduling policy at runtime. ASA’s core is a novel, low-overhead offline/online approach. First, an offline process trains a universal, hardware-agnostic machine learning model to recognize abstract workload patterns from system behaviors. Second, at runtime, ASA continually processes the model’s predictions using a time-weighted probability voting algorithm to identify the workload, then makes a scheduling decision by consulting a pre-configured, machine-specific mapping table to switch to the optimal scheduler via Linux’s sched_ext framework. This decoupled architecture allows ASA to adapt to new hardware platforms rapidly without expensive retraining of the core recognition model.

Our evaluation, based on a novel benchmark focused on user-experience metrics, demonstrates that ASA consistently outperforms the default Linux scheduler (EEVDF), achieving superior results in 86.4% of test scenarios. Furthermore, ASA’s selections are near-optimal, ranking among the top three schedulers in 78.6% of all scenarios. This validates our approach as a practical path toward more intelligent, adaptive, and responsive operating system schedulers.

†

†  copyright:  none

1.  Introduction

Figure 1 .

Core Philosophy of ASA: A Paradigm Shift from Designing a Single Optimal Policy to Dynamically Selecting the Best Policy from a Portfolio of Experts.

The design of operating system schedulers has always been an exercise in managing fundamental trade-offs, primarily between fairness, throughput, and latency. For decades, scheduler development has advanced by refining heuristics to find a suitable balance for the hardware and software of the era. However, the complexity of modern systems has amplified these trade-offs to a critical point. The concurrent rise of heterogeneous hardware (e.g., P/E-cores, NUMA, and diverse cache hierarchies) and highly diverse software architectures (e.g., microservices, interactive applications, batch processing) means that a single, fixed set of scheduling rules represents a significant compromise. The evolution of the default Linux scheduler from the O(1) scheduler  (Love,  2005 ) , through the Completely Fair Scheduler (CFS)  (Molnár,  2007 ) , to the most recent EEVDF  (LWN.net,  n.d. ) , while increasingly sophisticated, illustrates that even these advanced designs, engineered for general-purpose robustness, often leave significant performance potential untapped in specific, critical scenarios. The core challenge has thus shifted from merely  balancing  these goals to  dynamically adapting  which goal to prioritize from moment to moment.

In recent years, the research community has pursued this goal of dynamic adaptation through several technical routes. One prominent direction has been the development of  scenario-specific static policies

(Liu and Layland,  1973 ; Delimitrou and Kozyrakis,  2013 ; Kaffes et al.,  2019 ) , which essentially pre-select a fixed trade-off, such as prioritizing throughput for datacenter workloads. This approach delivers excellent performance in stable, well-defined environments but lacks the agility required for dynamic, general-purpose systems. Another significant line of work has investigated  AI-assisted dynamic policies

(Downey,  1997 ; Delimitrou and Kozyrakis,  2014 ; Mao et al.,  2019 ) , which seek to learn the optimal trade-offs online. While this aligns well with the need for adaptability, the associated costs in terms of runtime overhead, training stability, and model interpretability present practical barriers to widespread adoption  (Sanabria et al.,  2022 ; Chen et al.,  2018 ) . This suggests that while the goal of dynamic adaptation is widely recognized as correct, the mechanisms for achieving it warrant re-evaluation.

Inspired by the work of  sched_ext

(Vernet,  2024 ) , which provides a series of expert schedulers (e.g., LAVD (Galin and Hedblom,  2025 )  for latency-sensitive workloads), we argue that the key to resolving this dilemma lies in a paradigm shift: from “designing the optimal policy” to “dynamically selecting the optimal policy”. As illustrated in Figure

1  , instead of pursuing a single, monolithic scheduler, we propose integrating a portfolio of simpler, specialized schedulers and using an intelligent agent to select the most suitable one at runtime. Based on this philosophy, this research designs and implements the  Adaptive Scheduling Agent  (ASA), a lightweight framework built on the Linux kernel’s extensible scheduler interface,  sched_ext , which endows the operating system with online awareness and decision-making capabilities.

At its core, ASA operates as an intelligent agent following a classic cycle of  perception ,  decision , and  action . In the  perception  module, it continuously monitors runtime metrics to understand the system’s state. The  decision  module then uses a trained model to recognize the active workload pattern and consults a pre-calibrated scheduler mapping table to determine the optimal scheduling strategy. Finally, the  action  module executes this decision by dynamically switching to the most suitable scheduler via the sched_ext framework.

The intelligence behind ASA’s decisions is built through a comprehensive three-stage offline preparation process. This process begins with  Prototype Learning  to create a baseline model and scheduler performance map. It is followed by  Dynamic Overhead Calibration , which simulates scheduler switch and system monitor cost to make ASA take its own operational overhead into account. Finally,  Generalization Model Training  allows a fully operational ASA to refine its models and mappings in a live environment, ensuring high accuracy and adaptability to diverse hardware.

Our work decomposes the complex scheduling decision problem into two more manageable sub-problems: workload pattern recognition and policy matching. This not only avoids the immense challenges of designing a “one-policy-fits-all” scheduler but also, by introducing machine learning, endows the scheduling system with unprecedented flexibility and adaptability.

The main contributions of this research are as follows:

•

We design and implement  Adaptive Scheduling Agent (ASA) , a lightweight, adaptive scheduling agent framework built on eBPF and sched_ext, which intelligently routes workloads to expert scheduling policies.

•

We propose a  Time-Weighted Probability Voting Algorithm  for workload recognition that uses a voting mechanism with exponential decay to ensure stable and responsive pattern identification, mitigating the impact of transient system noise.

•

We introduce a three-stage  Offli