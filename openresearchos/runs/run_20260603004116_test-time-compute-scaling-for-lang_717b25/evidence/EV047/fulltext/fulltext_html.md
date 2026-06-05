[2506.10420] Multi-dimensional Autoscaling of Processing Services: A Comparison of Agent-based Methods

\useunder

\ul

\AtBeginShipout

1

1  institutetext:  Distributed Systems Group, TU Wien, Vienna, Austria.

2

2  institutetext:  DISL, ICREA | Universitat Pompeu Fabra, Barcelona, Spain
 Corresponding author contact:

2

2  email:  b.sedlak@dsg.tuwien.ac.at

Multi-dimensional Autoscaling of Processing Services: A Comparison of Agent-based Methods

Boris Sedlak

11**

0009-0001-2365-8265

Alireza Furutanpey

11

0000-0001-5621-7899

Zihang Wang

11

0009-0009-8194-4698

Víctor Casamayor Pujol

22

0000-0003-2830-8368

Schahram Dustdar

1122

0000-0001-6872-8821

Abstract

Edge computing breaks with traditional autoscaling due to strict resource constraints, thus, motivating more flexible scaling behaviors using multiple elasticity dimensions. This work introduces an agent-based autoscaling framework that dynamically adjusts both hardware resources and internal service configurations to maximize requirements fulfillment in constrained environments. We compare four types of scaling agents: Active Inference, Deep Q Network, Analysis of Structural Knowledge, and Deep Active Inference, using two real-world processing services running in parallel: YOLOv8 for visual recognition and OpenCV for QR code detection. Results show all agents achieve acceptable SLO performance with varying convergence patterns. While the Deep Q Network benefits from pre-training, the structural analysis converges quickly, and the deep active inference agent combines theoretical foundations with practical scalability advantages. Our findings provide evidence for the viability of multi-dimensional agent-based autoscaling for edge environments and encourage future work in this research direction.

Keywords:

Internet of Things
Stream Processing
Active Inference
Autoscaling
Markov Decision Processes
Reinforcement Learning

1  Introduction

The rise of Edge Computing and the Computing Continuum (CC) addresses the limitations of traditional Cloud infrastructures  [ 4 ] . By bringing computation closer to users and data sources (e.g., IoT devices) these paradigms significantly reduce network latency, critical for applications that demand near real-time responses, such as autonomous driving, e-health, and virtual reality. A common use case, as depicted in Figure

1  , could be to detect entities in a video stream (e.g., humans) or tracking objects that have a QR code attached. By running these inference services locally, the overall network congestion is also mitigated by minimizing long-distance data transfers.

However, Edge and CC environments introduce new challenges  [ 1 ] : They rely on resource-constrained computing hardware, and thus break with traditional Cloud-based autoscaling. In Cloud systems, autoscaling mechanisms elastically respond to increased user demand by allocating more resources to a service or replicating it. This is infeasible at the Edge or in the CC, where computing resources are strictly limited  [ 9 ] . Especially when resources are scarce, applications require a more flexible scaling behavior that uses a wider range of adaptations – hence, operating in multiple  elasticity dimensions

[ 3 ] .
On the one hand, this protects the service execution and promises higher requirements fulfillment – captured through a set of Service Level Objectives (SLOs). On the other hand, this increases the complexity for choosing optimal scaling actions. What is needed, hence, are lightweight multi-dimensional scaling mechanisms that optimize the service execution without obstructing existing workloads.

To fill this gap, we propose an agent-based autoscaling approach tailored for Edge and CC systems, which adjusts processing services in multiple elasticity dimensions. Our approach employs decentralized local agents that (1) observe the service execution and their SLO fulfillment without centralized control; thus, we can monitor the resource allocation per service or the application throughput. If SLOs are violated, the agents attempt to restore the desired state by (2) adjusting the service execution; the exact scaling policy is learned by the agent according to environmental feedback. Notably, our approach allows scaling policies tailored to the individual services, where one service could, for example, scale down its machine learning (ML) model, while another service claim the remaining resources. This allows building composite and customizable scaling policies, which go further than existing approaches  [ 13 ] .

To show the viability of our approach, we implement four different versions of our general agent and compare their performance in a processing environment, where the agent needs to dynamically scale two processing services on an Edge device. In particular, we compare an Active Inference agent (AIF), a Deep Active Inference agent (DACI), a Deep Q-Network agent (DQN), and an agent using a numerical solver – called Analysis of Structural Knowledge (ASK). During our experiments, a scaling agent manages two physically executed services: one for video stream inference (Figure

1(a)  ) using the well-known Yolov8 model  [ 15 ] , and another for QR code reading (Figure

1(b)  ), implemented with OpenCV2  [ 11 ] .

On the short term, our work provides well-needed baselines for comparing the performance of different agent-based approaches – particularly needed for emerging solutions. On the long term, our evaluation environment is extensible and allows incorporating other agents.

(a)

CV (Yolov8)

(b)

QR (OpenCV)

Figure 1 :

Demo output of the results produced by the two processing services

We summarize our contributions as:

1.

Introducing an agent-based autoscaling approach within a  multi-dimensional elasticity space  that dynamically maximize SLO fulfillment in IoT and Edge environments by adjusting hardware and service configurations

2.

Evaluating on  real-world applications  of four distinct scaling agent architectures (Active Inference, Deep Active Inference, Deep Q-Network, and Analysis of Structural Knowledge) for real-time service orchestration.

3.

Providing a benchmarking environment for future autonomous research and demonstrating viability through agent-based resource allocation for parallel processing services (YOLOv8 and OpenCV) on constrained hardware.

2  Preliminaries

In the following, we provide a formal description of the problem, as well as how researchers have addressed it so far with agent-based methods – including AIF.

2.1  Problem Definition

As depicted in Figure

2  , multiple services are executed within one Edge device – sharing the device’s processing resources between them. The execution of the individual services is affected by the amount of resources allocated (e.g., CPU, or RAM), and the configuration of the service-internal parameters (e.g., input quality, or model size). Considering that both sets of parameters can be  elastically  adjusted during runtime  [ 3 ] , we summarize them as elastic configurations.

The allocated resources and the service configuration influence the degree to which the service outcome satisfies the client requirements (i.e., the SLOs). However, and this is the core of the problem, it is not a priori known what will be the resulting SLO fulfillment for a specific configuration.
Hence, the problem boils down to adjusting the elastic configurations in such a way, that the SLO fulfillment is maximized.
While the number of service configurations and resource allocations is limited, it is not easily possible to brute-force the problem by exhaustively searching the solution space. The reason is that actions taken on the environment require a considerable amount of time to show effect. For instance, orchestration tools like Docker and Kubernetes usually consider a cooldown period of several minutes after taking an action.

2.1.1  Formal Definition

More formally, the problem domain and the physical process