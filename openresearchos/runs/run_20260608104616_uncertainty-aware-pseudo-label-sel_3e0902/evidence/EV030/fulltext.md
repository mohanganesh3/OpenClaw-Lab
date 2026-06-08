<!-- page 1 -->
1
Towards Fast Personalized Semi-Supervised
Federated Learning in Edge Networks:
Algorithm Design and Theoretical Guarantee
Shuai Wang, Yanqing Xu, Yanli Yuan, and Tony Q. S. Quek
Abstract
Recent years have witnessed a huge demand for artificial intelligence and machine learning applications
in wireless edge networks to assist individuals with real-time services. Owing to the practical setting and
privacy preservation of federated learning (FL), it is a suitable and appealing distributed learning paradigm
to deploy these applications at the network edge. Despite the many successful efforts made to apply FL to
wireless edge networks, the adopted algorithms mostly follow the same spirit as FedAvg, thereby heavily
suffering from the practical challenges of label deficiency and device heterogeneity. These challenges
not only decelerate the model training in FL but also downgrade the application performance. In this
paper, we focus on the algorithm design and address these challenges by investigating the personalized
semi-supervised FL problem and proposing an effective algorithm, namely FedCPSL. In particular, the
techniques of pseudo-labeling, and interpolation-based model personalization are judiciously combined
to provide a new problem formulation for personalized semi-supervised FL. The proposed FedCPSL
algorithm adopts novel strategies, including adaptive client variance reduction, local momentum, and
normalized global aggregation, to combat the challenge of device heterogeneity and boost algorithm
convergence. Moreover, the convergence property of FedCPSL is thoroughly analyzed and shows that
FedCPSL is resilient to both statistical and system heterogeneity, obtaining a sublinear convergence rate.
Experimental results on image classification tasks are also presented to demonstrate that the proposed
approach outperforms its counterparts in terms of both convergence speed and application performance.
Index Terms
Shuai Wang and Yanli Yuan are with the Information Systems Technology and Design, Singapore University of Technology
and Design, Singapore 487372 (e-mail: shuai wang@sutd.edu.sg, yanliyuan@bit.edu.cn).
Tony Q. S. Quek is with the Information Systems Technology and Design, Singapore University of Technology and Design,
Singapore 487372, and also with the Department of Electronic Engineering, Kyung Hee University, Yongin 17104, South Korea
(e-mail: tonyquek@sutd.edu.sg).
Yanqing Xu is with the Shenzhen Research Institute of Big Data and School of Science and Engineering, The Chinese
University of Hong Kong, Shenzhen 518172, China (e-mail: xuyanqing@cuhk.edu.cn).
arXiv:2306.04155v1  [cs.DC]  7 Jun 2023


<!-- page 2 -->
2
Wireless edge networks, heterogeneous edge devices, federated learning, semi-supervised learning,
personalized federated learning
I. INTRODUCTION
The proliferation of distributed sensitive data at the network edge and the increasing computational
capability of network edge devices have motivated the research on deploying artificial intelligence (AI)
and machine learning (ML) applications in wireless edge networks to provide diverse customized and
heterogeneous mobile services for customers [1]. This promising direction has attracted significant interest
from researchers who pursue high-quality wireless AI services in a distributed edge network [2], [3]. Such
a network usually consists of a central server which needs to collaborate with a massive number of edge
devices to train a ML model for application scenarios where privacy protection and resource utilization
are critical. Motivated by this, federated learning (FL), as an emerging distributed learning paradigm, has
become increasingly appealing in wireless edge networks in recent years as a result of its similar and
practical configuration to edge computing, as well as its inherent nature for privacy preservation [4], [5].
In particular, FL runs under the orchestration of the central server without the need of knowing the raw
private data of each edge device.
Different from the conventional centralized learning, FL requires frequent model exchanges between
the central server and edge devices via wireless links. Therefore, it is important to investigate the impacts
of wireless aspects on the performance of the FL algorithms. In view of this, recent research on FL in
wireless networks can be categorized into two main directions. The first direction focuses on the wireless
resource allocation, such as bandwidth allocation, power control and computation resource allocation,
during the FL process to improve the algorithm convergence [6]–[11]. For example, [6] investigated the
edge device scheduling problem to improve the convergence rate of the FL algorithms under the constraint
of the limited radio resources of each device. [7] studied the joint learning and radio resource allocation
design to minimize the cost function. [9] investigated the hybrid beamforming design via FL, and [10],
[11] proposed to use FL to estimate the wireless channel of massive MIMO systems. However, these
works are generally based on simple FL frameworks. Among them, federated averaging (FedAvg) [12] is
the most frequently adopted FL algorithm because of its simple-yet-good empirical performance. Roughly,
it follows a computation and aggregation protocol where the central server updates the global model
based on the local ML models received from and computed by edge devices, and then broadcast it to
the devices. The process continues in a round-by-round fashion until convergence. The other important
direction focuses on advanced FL algorithm designs [13]–[17]. For example, [13] proposed the dynamic


<!-- page 3 -->
3
batch sizes assisted FL (DBFL) algorithm, which has been shown to converge faster than the FedAvg
algorithm. [14] proposed a novel multi-task personalized FL framework which also outperformed FedAvg.
Despite the successful progress of FL, it still faces many practical challenges. Especially when applied
to wireless edge networks, there are two main challenges that cannot be well addressed by traditional
simple FL algorithms, e.g., FedAvg. Let us elaborate on the challenges in detail and explain why the
adopted FedAvg algorithm fails to handle them. Firstly, the majority of data in edge devices is typically
unlabeled, and each edge device only processes low-quality, inadequately labeled data. This realistic
phenomenon arises from a severe shortage of experts to perform correct labeling, as well as the huge cost
of labeling the massive data [18] stored in these edge devices. For instance, in a smart healthcare system,
there is a scarcity in the amount of labeled data in each medical institute, and it is daunting for doctors
to label them manually. However, FedAvg and its variants are incapable of addressing this issue because
they assume that all data in edge devices are properly labeled. It is acknowledged that these algorithms
perform poorly with merely a small amount of labeled data. This naturally necessitates the urgent demand
for algorithmic development for FL with a deficiency of labeled data, namely semi-supervised FL (SSFL),
where both labeled and unlabeled data are exploited for model training [16], [17].
The second challenge is the presence of massive heterogeneous edge devices. The high degree of device
heterogeneity decelerates the model training in FL and downgrades the application performance that the
FL algorithms can finally reach. In particular, these heterogeneous edge devices have both statistical and
system heterogeneity.
Statistical heterogenity: In wireless edge networks, the local datasets owned by edge devices are usually
unbalanced and non-i.i.d. (non-independent and identically distributed) across these devices [19], [20].
It has been shown that FedAvg is inefficient in dealing with the non-i.i.d. data [21]. The presence of
unbalanced and non-i.i.d. data could corrupt the algorithm convergence speed and even lead to model
divergence [22]. On the other hand, as the conventional FL requires all edge devices to agree on a
global model, the resulting global model could perform arbitrarily poorly once applied to the local
dataset of each device because of the non-i.i.d. data [23]. A large body of literature has overcomed
this issue by adopting the technique of model personalization which aims to achieve a device-specific
or personalized model to fit the dataset of each device for good generalization performance [23]–[25].
This motivates the need for personalized FL in wireless edge networks.
System heterogeneity:
In addition to non-i.i.d. data, edge devices may have varied system constraints
including different hardwares (CPU, GPU, and memory size), network configurations (wireless
channels, bandwidth, and transmission power), battery states, etc. Indeed, these heterogeneous
characteristics can cause unpredictable behaviors of edge devices as well as the straggler problem in


<!-- page 4 -->
4
FL [26]–[29]. Specifically, different computational capabilities and the unavailability of edge devices
can adversely impact the algorithm convergence. Besides, the delay caused by the straggler (the
slowest edge device) can significantly slow down the model training because FL requires waiting for
all edge devices to communicate with the server in each round.
The statistical heterogeneity and system heterogeneity can be widely seen in practical wireless edge
communication systems. Considering a distributed multiple-input multiple-output (MIMO) system at the
network edge, there are many distributed access points (APs) to serve multiple users. Recently, the FL
algorithms have been applied to such network for signal processing (SP) tasks, e.g, channel estimation
(CE) and beamforming, to overcome the disadvantages of high communication and computation costs of
tradition SP algorithms [10], [11]. Take the uplink CE problem for example, each AP is viewed as an
edge device and the dataset of each edge device is the received pilot signals from its associated users.
There is a central sever who coordinates the CE of all APs to approach the centralized performance
based on the FL algorithms. However, applying these algorithms to such a system is faced with the
aforementioned challenges. For example, i) since each AP may serve different users, the datasets (received
pilot signals) of the APs could be unbalanced and non-i.i.d.; ii) the APs may be equipped with different
system hardwares, and thus have varied computational capabilities; iii) the wireless channels between the
APs to the central sever are dramatically different due to the small-scale fading and shadowing.
It is worth-noting that the interplay of these two challenges could exacerbate their respective negative
effects. For example, the statistical heterogeneity among edge devices increases in the presence of
unlabeled data since the unlabeled data in each device may contain most of the same patterns as those of
other devices [18]. As a result, using merely labeled data for training often leads to not only algorithm
inefficiency but also a severe performance degradation of FL algorithms. Due to the limitations of FedAvg
in addressing the two main challenges, our work focuses on designing an advanced FL algorithm to
overcome these challenges in wireless edge networks.
A. Related works
We notice that, while many successful efforts have been made to address the aforementioned challenges
in wireless edge networks, most of them merely take into account one challenge and might not well
address it. Few have simultaneously considered these two issues and offered effective solutions.
Recent FL works [18], [30]–[32] attempted to address the challenge of label deficiency by studying
the SSFL problem. They mostly relied on the technique of pseudo-labeling and performance-promoting
regularization, where the unlabeled data samples are respectively assigned artificial labels, and both
labeled and unlabeled data are then considered together with specific regularizations to improve the


<!-- page 5 -->
5
training performance. The two works [33], [34] respectively applied similar strategies to the applications
of human activity recognition and travel model identification while accounting for the resource constraints
in wireless edge networks. Nevertheless, these works adopted or followed the same spirit as FedAvg for
model training, and thus, as mentioned previously, suffer from the adverse effect of device heterogeneity.
To address the challenge of device heterogeneity, researchers have proposed methods such as FedProx
[35], SCAFFOLD [21], and FedDyn [36], which penalized model updates with appropriate regularizations
in order to stabilize the algorithm convergence. However, they either fail to address the challenge or
generalize badly for many edge devices’ local datasets because of the lack of personalization. To deal
with the personalized FL problem, recent works proposed algorithms such as PerFedAvg [24], FedAMP
[25], Ditto [37], pFedMe [38] and APFL [23]. In particular, the authors of APFL [23] adopted model
interpolation to provide a device-specific model for each edge device through a convex mixture of a
local model and the global model. On the other hand, it is noticed that, instead of model personalization,
some works [39]–[42] mitigated this issue with the distributionally robust FL (DRFL). Unlike the former,
DRFL aims to pursue a global model which performs well over the worst-case combination of all local
data distributions, and thus its generalization performance will be resilient to the non-i.i.d. data. However,
the preceding works all relied solely on labeled data for model training, thereby performing poorly in
practical scenarios where label deficiency is prevalent.
More recently, a few works [43]–[45] have emerged to investigate the problem of personalized SSFL.
Specifically, the works [43] [44] focused on the application of human activity recognition and utilized
active learning and label propagation for data labeling and transfer learning for personalization. The
authors in [45] proposed to generate a personalized autoencoder using a hyper-network over the labeled
and unlabeled data for multi-sensory classification. Unfortunately, these works still did not address the
challenge of device heterogeneity as the adopted algorithms again are similar to FedAvg and thus affected
by heterogeneous edge devices. Besides, none of these works theoretically analyzed the convergence of
their proposed algorithms.
B. Contribution
In this paper, we study the personalized semi-supervised FL (PSSFL) problem in wireless edge networks,
aiming at proposing an efficient algorithm with both high-quality application performance and solid
theoretical guarantee. In particular, we propose an algorithm named FedCPSL, which leverages novel
strategies to address the aforementioned two challenges and outperforms its existing counterparts in terms
of both convergence speed and application performance. The contributions can be expounded from the
following aspects:


<!-- page 6 -->
6
1) Problem formulation: We propose a general problem formulation for the PSSFL problem in wireless
edge networks which combines the techniques of pseudo-labeling, regularization, and model interpolation
to model the PSSFL problem in a novel way. Unlike APFL [23], we introduce a new localized model for
each device and consider the mixture of the outputs of the localized model and the global model as that
of the personalized model, which enjoys better theoretical justification. To the best of our knowledge, this
is the first work considering both the interpolation-based model personalization and semi-supervised FL.
The resulting problem formulation is novel for modeling the PSSFL problem.
2) Algorithm design: We propose the FedCPSL algorithm, which ensures fast and simultaneous
algorithm convergence with respect to both the global and localized model parameters by addressing
the challenge of device heterogeneity. Specifically, the global and localized model parameters are
simultaneously trained. FedCPSL adopts the strategy of adaptive client variance reduction to combat the
statistical heterogeneity among edge devices. For system heterogeneity, it allows edge devices to perform
heterogeneous local updates across rounds based on their system constraints and leverages the strategy of
global normalized aggregation to alleviate the resulting solution bias. Besides, the momentum approach
is introduced into the updates of both the global and localized model parameters to further boost the
algorithm convergence.
3) Convergence analysis : We establish the convergence property of the FedCPSL algorithm. In
contrast to the existing analysis of FL algorithms with one block of variables, we justify the convergence
of both the global and localized model parameters in the presence of the above novel strategies. In
particular, under mild assumptions, they all have a sublinear convergence rate for nonconvex objectives,
which matches the state-of-the-art lower bounds. More importantly, the theoretical analysis shows that the
convergences of both the global and localized model parameters are resilient to the challenge of device
heterogeneity, which causes the speedup of FedCPSL over its counterparts. Such convergence results for
the PSSFL problem are new and have never appeared in the literature.
4) Experiments: We evaluate the performance of the proposed FedCPSL algorithm by applying it to
the image classification tasks on the MNIST and CIFAR-10 datasets. The experimental results demonstrate
that FedCPSL outperforms benchmark algorithms under different experimental settings. It not only enjoys
a faster convergence but also achieves a better application performance.
Synopsis: Section II introduces the proposed system model and problem formulation of PSSFL in
wireless edge networks. Section III presents the proposed FedCPSL algorithm to solve the formulated
problem, and the associated theoretical analysis is given in Section IV. The experiment results are presented
in Section V. Section VI concludes this paper.


<!-- page 7 -->
7
II. PROBLEM FORMULATION
Consider a wireless network where a central server coordinates N edge devices to learn a ML model.
Without loss of generality, we focus on the classic data classification task, and denote Di as the local dataset
that collects the data generated or sensed by edge device i. In this work, we consider a semi-supervised
setting where the local dataset Di consists of a set of labeled data Li and a set of unlabeled data Ui, i.e.,
Di = Li ∪Ui, Li = {(xi,k, yi,k)}ni
k=1, Ui = {ui,k}mi
k=1,
(1)
where (xi,k, yi,k) ∈RS × RC is the k-th input and output pair in Li with S denoting the dimension
of the input data and C signifying the number of data classes; ui,k is the k-th input of Ui, and ni and
mi are the sizes of the labeled and unlabeled data at edge device i, respectively. Since the generated
or sensed data by edge devices are biased to the applications and environment, the datasets have the
following characteristics: i) the local datasets, Di, i ∈[N], could be non-i.i.d. and their sizes could be
unbalanced; ii) the datasets may follow different data distributions; iii) the number of labeled data could
be much smaller than the unlabeled data, i.e., ni < mi. These edge devices i ∈[N] may have varied
system constraints including different hardwares (CPU, GPU, and memory size), network configurations
(wireless channels, bandwidth, and transmission power), battery states, etc.
Our objective is to learn an accurate and personalized ML model for each edge device by utilizing their
local datasets Di. To this end, one needs to address the two challenges, i.e., the label deficiency and device
heterogeneity. Since the traditional FL algorithms, e.g., FedAvg, cannot deal with these two challenges
well, we are interested in developing an advanced FL algorithm by considering the challenges into the
problem formulation and the algorithm design, respectively. Specifically, to handle the challenge of label
deficiency, we consider to use semi-supervised learning and interpolation-based model personalization to
formulate the learning problem. Meanwhile, to deal with the adverse impacts of device heterogeneity on
the algorithm convergence, a sophisticated and effective FL algorithm is developed (see Sec. III).
To fully exploit the unlabeled data Ui, ∀i ∈[N], the pseudo-labeling strategy is applied. In this strategy,
each unlabeled data sample is initially assigned an artificial label, called pseudo label. These pseudo labels
of unlabeled data are then optimized and used during the training process to improve the application
performance of the ML model [30]. Specifically, let h(θ; ·) represent the ML model with model parameter
θ, and introduce νi = [νi,1, νi,2, . . . , νi,mi] with νi,k ∈RC being a probability vector as the pseudo labels
for the unlabeled data Ui. Then, we define
xi = [xi,1, xi,2, . . . , xi,ni], yi = [yi,1, yi,2, . . . , yi,ni], ui = [ui,1, ui,2, . . . , ui,mi],
(2)


<!-- page 8 -->
8
and consider to train a global model θ by minimizing a joint cost function as follows.
min
θ,νi
N
X
i=1
ωifi(θ, νi), s.t. νi ∈Vi,
(3)
where Vi = {νi|νi,k ≥0, 1⊤νi,k = 1, ∀k ∈[mi]} collects all the possible pseudo labels, and
fi(θ, νi) ≜l(h(θ; xi), yi) + αpl(h(θ; ui), νi) + αrR(h(θ; ui), νi)
(4)
is the local cost function for each edge device i; ωi is the weight associated with edge device i satisfying
PN
i=1 ωi = 1; for example, ωi = ni+mi
n+m
with n and m representing the total number of labeled and
unlabeled data of all edge devices, respectively; l is the loss function for the ML model h w.r.t. a set
of data samples. One can see that the local cost fi(θ, νi) penalizes the output of the model h(θ, ·)
for being dissimilar to the target labels with respect to both the labeled and unlabeled datasets. The
model h(θ, :) could be any ML classifier, such as decision tree, SVM, and deep neural network, which
accepts data inputs and then respectively outputs their classes or class probability distributions. The
regularizer, R(h(θ; ui), νi), on the model parameter and pseudo labels can be well-defined to avoid
incorrect estimation of pseudo labels and thus benefits the learning process from unlabeled data.
As mentioned previously, model interpolation is an effective way to combat the negative effects caused
by non-i.i.d. data by convexly combining the global model and local model [23]. This motivates us to
design a personalized model under the semi-supervised setting for each edge device with the help of a
localized model and the global model. Specifically, we assume that, for any input x, the output of the
personalized model h(θi,p; ·) can be represented as a convex combination of the outputs of a localized
model h(θi,lc; ·) and the global model h(θ; ·), i.e.,
h(θi,p; x) = βih(θi,lc; x) + (1 −βi)h(θ; x), ∀x,
(5)
where βi ∈[0, 1] is a predefined mixture coefficient. Then, the problem of personalized semi-supervised
FL is to find a localized model which makes (5) be the minimizer of the local cost function. To be
concrete, one can write the problem as
min
θi,lc,i∈[N]
N
X
i=1
ωiFi(θi,lc, θ, νi),
(6a)
s.t. (θ, {νi}N
i=1) = arg
min
θ′,ν′
i∈Vi
N
X
i=1
ωifi(θ′, ν′
i),
(6b)
where
Fi(θlc
i , θ, νi) ≜l(βih(θi,lc) + (1 −βi)h(θ), yi) + αpl(βih(θi,lc) + (1 −βi)h(θ), νi)
+ αrR(βih(θi,lc) + (1 −βi)h(θ), νi),
(6c)


<!-- page 9 -->
9
and fi(·, ·) is defined in (4). Here we use h(θ) to represent h(θ; ·) for ease of presentation. It is worth-
noting that, in contrast to the traditional model interpolation, we construct the personalized model by
taking a convex combination of the global model and the localized model in (6c), rather than combining
their parameters. This particularly aligns with the theoretical justification of the model interpolation
technique in [23, Theorem 1 and Corollary 3].
One notable feature of the problem formulation (6) is that the optimization process for the localized
models can be completely decoupled from the training of the global model. This is because problem (6)
has a form of two-phase optimization, with global model training and model personalization as two key
ingredients. To be specific, problem (6) can be solved using a two-phase algorithm where in the first
phase, the global model parameter θ and pseudo labels {νi}N
i=1 are trained by solving the subproblem
(6a), and then, with θ and {νi}N
i=1 as inputs, each edge device i solves (6a) in the second phase to obtain
a localized model parameter θi,lc. In the first phase, since problem (6a) is a SSFL problem, it is natural
to follow the model averaging (MA) strategy [22] adopted in the FL algorithms to solve problem (6a)
[46]. The second phase is responsible for model personalization, and lastly, the output of a new input
data x of edge device i is generated by following (5).
Nevertheless, problem (6) is much more challenging to solve than that in the literature [12], [35], [47]–
[49] as it combines the tasks of global model training, pseudo label prediction, and model personalization
into a unified formulation which is possibly non-convex and non-smooth, and involves three different
blocks of optimization variables. The aforementioned two-phase optimization algorithm is not preferable
for solving problem (6) in wireless edge networks because of its low efficiency, the lack of considering
the statistical and system heterogeneity, and its incapability of handle them. For example, to achieve
desirable performance, the global model and the pseudo labels should be optimized for high accuracy and
then fed back to edge devices for the training of the localized models. This process demands a significant
amount of training time. Therefore, it needs careful considerations on how to effectively design efficient
and high-performance algorithms for problem (6).
Remark 1 It is indicated in [23, Theorem 1 and Corollary 3] that, by carefully choosing the mixture
coefficient βi, ∀i ∈[N], the joint prediction model in (5) with an optimal solution (θ⋆
i,lc, θ⋆) to problem
(6) generalizes well on the associated local data distribution. Specifically, the value of βi controls the
contribution of the global model to model personalization. The setting of βi = 1 suggests no FL while
βi = 0 implies no model personalization. Thus, the choice of βi for edge device i should match the
dissimilarity (the degree of non-i.i.d. data) between its own data distribution and the average of all edge
devices’ data distributions. When the dissimilarity is large, a large βi is preferred to incorporate more of


<!-- page 10 -->
10
the localized model into the personalized model of edge device i. On the contrary, a small βi is suggested
such that the personalized model of edge device i benefits more from the global model.
III. PROPOSED FEDCPSL ALGORITHM
In this section, we develop an efficient FL algorithm for problem (6), termed FedCPSL. The proposed
FL algorithm allows simultaneous training of all the optimization variables to spead up the learning
process, and overcomes the challenge of device heterogeneity in wireless edge networks with advanced
algorithm strategies. Furthermore, an intuitive explanation on the advantage of FedCPSL in dealing with
the challenge is presented.
A. Algorithm design
Instead of using the two-phase optimization, we aim to optimize the global model and the localized
models simultaneously, such that one can obtain desirable personalized models when an accurate global
model is reached. Notice that the decoupling structure of (6) makes it possible to update the global model
and pseudo labels similarly to the existing FL algorithms while parallelly optimizing the localized models
with low complexity. In particular, since problem (6) involves three kinds of variables (model parameters),
i.e., θ, νi, and θi,lc, we consider leveraging the strategies of alternating minimization (AM) and MA in
FL to sequentially update these model parameters round by round until they converge.
As per the above idea, one can design an iterative algorithm as follows. For round r = 1, 2, . . ., the server
randomly samples a small, fixed-size subset of edge devices (denoted by Ar with size |Ar| = m ≪N) and
broadcasts the global model parameter θr to them. Each of the selected edge devices i ∈Ar sequentially
obtains approximate solutions to the following subproblems of (6):
νr+1
i
= arg min
νi∈Vi fi(θr, νi),
(7a)
θr+1
i
= arg min
θi
fi(θi, νr+1
i
),
(7b)
θr+1
i,lc = arg min
θi,lc Fi(θi,lc, θr+1
i
, νr+1
i
).
(7c)
Then, the server collects and takes certain average of θr+1
i
, ∀i ∈Ar to produce θr+1 for the next round
of updates. In practice, (7a) may have a closed-form solution; otherwise, it will be approximated by
one-step gradient descent (GD). Besides, it is sufficient to employ the local SGD [47] to approximate the
solutions of (7b) and (7c). In particular, each edge device i ∈Ar performs Qr
i ≥1 consecutive steps of
SGD with respect to θi and θi,lc, i.e., for t = 0, . . . , Qr
i −1,
θr,t+1
i
= θr,t
i
−ηgi(θr,t
i , νr+1
i
),
(8a)


<!-- page 11 -->
11
θr,t+1
i,lc
= θr,t
i,lc −ηcGi(θr,t
i,lc, θr,t
i , νr+1
i
),
(8b)
where η > 0 and ηc > 0 are two step sizes; gi(θi, νi) denote the SGD of fi(·, ·) over a mini-batch
S = Bl ∪Bu which consists Sl samples i.i.d. drawn from Li and Su samples i.i.d. drawn from Ui;
Gi(·, νr+1
i
) is the SGD of Fi over S;
θr,0
i
= θr, θr,0
i,lc = θr−1,Qr
i −1
i,lc
;
(9)
Then, θr+1
i
and θr+1
i,lc are set as θr+1
i
= θr,Qr
i
i
, θr+1
i,lc = θr,Qr
i
i,lc ,∀i ∈Ar.
In contrast to [18], we leverage the scheme of partial client participation (PCP) in which only a small
set of edge devices are sampled to perform the local update and communicate with the server in each
round. It is practically preferred because federated training of a ML model demands huge communication
resources and reliable connections, which are not affordable for wireless edge networks where bandwidth
is limited and communication links are usually unreliable. In particular, radio resource limited wireless
edge networks cannot support the participation of all edge devices in every communication round, which
leads to network congestion if full participation is required. The scheme of PCP can greatly alleviate
such problem by adjusting the size of participants according to the available bandwith. Besides, the poor
link quality in wireless edge networks may raise the problem of transmission outage [50] because part of
edge devices experience deep channel fading and thus unsuccessful transmission of the model parameters
to the server. PCP also can mitigate this issue through modeling the randomly offline edge devices. It is
worth-noting that, the algorithm convergence could be further accelerated by PCP if advanced device
selection and frequency bandwidth allocation schemes are incorporated [6]–[8].
On the other hand, as mentioned, the edge devices in wireless edge networks may have heterogeneous
system constraints including computational capabilities and network configurations. The heterogeneity in
the computational speeds results in large variations in the number of local update steps performed by
the participating devices. This would cause large model training delay if the participating devices are
restricted to perform the same amount of local computations in each round [35]. To resolve this issue, we
allow time-varying heterogeneous local updates (HLU) in which each active edge device i performs Qr
i
local update steps at round r. It should be remarked that the scheme of time-varying HLU across rounds
is highly desired in wireless edge computing scenarios where low-latency ML services are demanded.
The benefit of such scheme lies in that each device can flexibility decide the amount of its local update
steps according to its current system state in each round to reduce the training delay. For example, at one
round, the mobile devices with full battery and good wireless connections can contribute more to the
training process than those in poor system conditions.


<!-- page 12 -->
12
Nevertheless, the above update rules (8a)(8b)(9) have the following disadvantages: 1) The scheme of
local SGD usually leads to relatively slow algorithm convergence; 2) the non-i.i.d. data can cause the
client drift problem [21]; 3) using time-varying HLU can result in the issue of objective inconsistency [51].
To address these issues and boost the efficiency, three specific strategies are introduced in the proposed
algorithm to deal with the above three issues, respectively. The details of the adopted strategies are stated
as follows:
• Accelerated local SGD with momentum. To speed up the algorithm convergence, the momentum
technique is applied to the local updates (8a) and (8b). Specifically, in the momentum-based SGD,
two local momentum buffers are introduced for each edge device and initialized as zero at the
beginning of each round. In each round, the local momentum buffers dynamically accumulate the
historical gradients in the current round, and are then used for updating (8a) and (8b) respectively.
By taking the gradients in all the previous steps into consideration, the momentum-based SGD can
speed up the algorithm convergence, which is verified by the numerical results in Section V.
• Adaptive client variance reduction. As the non-i.i.d. data causes high gradient variance among
edge devices, which leads to the client drift problem and slows down the convergence of the global
model parameter θ, the technique of client variance reduction is desired. In parallel, the introduction
of momentum and HLU presents new challenges that traditional variance reduction schemes cannot
address. To remedy this issue, we adopt an approach which follows similar spirits as client variance
reduction but is adapted to the momentum-based local SGD and HLU. In particular, two kinds of
control variables (one for the server and the other one for edge devices) are introduced to correct the
local update directions. Unlike traditional client variance reduction schemes [21], we update these
control variables in a novel way that not only retains the capability of client variance reduction but
also accommodates the challenges brought by these two schemes. Therefore, it can benefit the stable
and fast convergence of the proposed algorithm; See more details in Section III-B.
• Normalized global aggregation. Using time-varying HLU may result in the issue of objective
inconsistency, which downgrades the application performance [51]. Following the same spirit as
FedNova in [51], we employ the strategy of normalized global aggregation to account for (time-
varying) HLU. However, unlike FedNova, we utilize a flexible step-size ηg for global aggregation,
which not only benefits the theoretical analysis but also leads to faster algorithm convergence [21].
B. Algorithm implementation and explanation
With the above three strategies, the details of the proposed FedCPSL algorithm are summarized in
Algorithm 1. During the process of Algorithm 1, the server maintains two variables: the global model


<!-- page 13 -->
13
Algorithm 1 Proposed FedCPSL Algorithm
Input: initial values of θ0
i = θ, c0 = c0
i = 0, ∀i ∈[N], η > 0, ηv > 0.
for round r = 0 to R −1 do
Server side: select a set of edge devices Ar (with size |Ar| = m) by uniform sampling without replacement, and broadcast
θr and cr to all edge devices in Ar.
edge device side:
for edge device i = 1 to N in parallel do
if edge device i /∈Ar then
Set θr+1
i
= θr, θr+1
i,lc = θr
i,lc.
else
Compute νr+1
i
via (7a) or
νr+1
i
= νr
i −ηv∇νifi(θr, νr
i ).
(10)
Set θr,0
i
= θr, θr,0
i,loc = θ
r−1,Qr−1
i
i,loc
, µr,0
i
= 0, µr,0
i,lc = 0.
for t = 0 to Qr
i −1 do
µr,t+1
i
= γµr,t
i
+ gi(θr,t
i , νr+1
i
) −cr
i + cr,
(11)
θr,t+1
i
= θr,t
i
−ηµr,t+1
i
,
(12)
µr,t+1
i,lc
= γµr,t
i,lc + Gi(θr,t
i,lc, θr,t
i , νr+1
i
),
(13)
θr,t+1
i,lc
= θr,t
i,lc −ηcµr,t+1
i,lc
.
(14)
end for
Set θr+1
i
= θ
r,Qr
i
i
, θr+1
i,lc = θ
r,Qr
i
i,lc , and compute
eQr
i =
Qr
i
1 −γ −γ(1 −γQr
i )
(1 −γ)2
,
(15)
cr+1
i
= cr
i −
 
cr + θr+1
i
−θr
η eQr
i
 
.
(16)
Send θr+1
i
−θr and eQr
i to the server.
end if
end for
Server side: Compute
θr+1 =θr + ηg N
m
X
i∈Ar
ωi θr+1
i
−θr
eQr
i
,
(17)
cr+1 =cr −
X
i∈Ar
ωi
 
cr + θr+1
i
−θr
η eQr
i
 
,
(18)
end for
parameter θ and the server control variable c, while each edge device i maintains the localized model
parameter θi,lc, the local copy of global model parameter θi, the client control variable ci, the pseudo label
νi, and the local momentum buffers µi and µi,lc. In each communication round r, the server randomly
selects a set of active edge devices Ar by uniform sampling without replacement, and broadcasts θr and


<!-- page 14 -->
14
cr to the devices in Ar. Given θr as the initial point, each active edge device i ∈Ar computes νr+1
i
via
(7a). If (7a) cannot be solved analytically, one step of GD will be adopted, i.e.,
νr+1
i
= νr
i −ηv∇νifi(θr, νr
i ).
(19)
One feature of the proposed FedCPSL algorithm is that each edge device i ∈Ar updates θi using the
local SGD with momentum for fast convergence. In each local step t, it employs a local momentum buffer
µi which is initialized as zero at the beginning of each round and updated by geometrically accumulating
all past variance-reduced SG’s in the current round as
µr,t+1
i
=γµr,t
i
+ gi(θr,t
i , νr+1
i
) −cr
i + cr
=
t
X
k=0
γt−k(gi(θr,t
i , νr+1
i
) −cr
i + cr),
(20)
where γ ∈[0, 1) is the momentum factor. It then updates θi by applying one step of SGD with µr,t+1
i
,
i.e., (12). Note that, similar to traditional client variance reduction schemes [21], the proposed FedCPSL
algorithm incorporates c−ci to the local update of θi to perform client variance reduction. However, owing
to the introduction of momentum µi and time-varying local update steps Qr
i , the updates of the control
variables, ci and c, are respectively adjusted by (16) and (18) to accommodate these changes. Analogously,
a local momentum buffer µi,lc is adopted for the update of the localized model θi,lc. Specifically, it is
initialized as zero at the beginning of round r, updated by (13) and then used to update θi,lc via (14).
After that, these active edge devices upload the local updates θr+1
i
−θr and eQr
i to the server, which will
be used for global aggregation through (17). Lastly, the next round starts with the newly generated θr+1
and cr+1.
Let us see how c and ci’s help to reduce the gradient variance among edge devices and correct the
corresponding local update directions. In round r, if edge device i is active, one can derive from (11) and
(12) that
θr+1
i
−θr = −η eQr
i
Qr
i −1
X
t=0
(br
i )t
eQr
i
(gi(θr,t
i , νr+1
i
) + cr −cr
i ),
(21)
where br
i = [1 −γQr
i , 1 −γQr
i −1, . . . , 1 −γ]/(1 −γ) and eQr
i = ∥br
i ∥1. As a result, cr+1
i
can be rewritten
as
cr+1
i
= cr
i −
 
cr −
Qr
i −1
X
t=0
(br
i )t
eQr
i
(gi(θr,t
i , νr+1
i
) + cr −cr
i )
 
=
Qr
i −1
X
t=0
(br
i )t
eQr
i
gi(θr,t
i , νr+1
i
).
(22)


<!-- page 15 -->
15
Intriguingly, the above results (21)(22) suggest that the update of θi generalizes the variance-reduction
techniques used in traditional client variance reduction schemes where the local update is performed along
an estimated global gradient direction considering the data of all edge devices. To be concrete, suppose
ρ = 1, γ = 0 , Qr
i = Q, νr
i = ν0
i , ∀i, r, then we obtain µr,t
i
= 0 and
θr,t+1
i
= θr,t
i
−η
 
gi(θr,t
i , νr+1
i
) −cr
i +
N
X
i=1
ωicr
i
 
,
(23)
cr+1
i
= 1
Q
Q−1
X
t=0
gi(θr,t
i , νr+1
i
).
(24)
Clearly, these resulting update rules of θi and ci are identical to that of traditional client variance reduction
schemes. Thus, the update rules of θi, ci and c in Algorithm 1 can be seen as an extension of client
variance reduction to the local SGD with momentum in the presence of heterogeneous edge devices.
It is worth noting that, the server control variable c is updated only relying on the local updates
θr+1
i
−θr and eQr
i from active edge devices. In contrast, traditional client variance reduction schemes [21]
usually require transmitting back and forth both the local update and the server control variable. Thus,
(18) enables a significant reduction of communication cost as the uplink channel is typically much slower
than the downlink and the downlink cost is usually negligible compared to the uplink cost. Furthermore,
to remedy the objective inconsistency, we utilize the technique of the local update normalization in the
stage of global aggregation where the local update θr+1
i
−θr is normalized before aggregation.
IV. CONVERGENCE ANALYSIS
Before proceeding, we make the following assumptions, which are commonly adopted in the FL
literature, e.g., [6], [7], [23].
Assumption 1 (Lower bounded) All local cost functions fi(·, ·) in problem (6) are non-convex and
lower bounded, i.e.,
fi(θi, νi) ≥f > −∞, ∀θi, νi ∈Vi.
(25)
Assumption 2 (Convexity) The cost function l(·, ·) in problem (6) is bi-convex and fi(·, νi) is strongly
convex with modulus µ, i.e, ∀νi, ν′
i ∈Vi,
fi(θi, νi) ≥fi(θi, ν′
i) + ⟨∇θfi(θi, ν′
i), νi −ν′
i⟩+ µ
2 ∥νi −ν′
i∥2.
(26)
Let Li(zi, νi) ≜l(zi, yi) + αpl(zi, νi) + αrR(θ, νi), then both Li(·, νi) and Li(zi, ·) are strongly convex
with modulus µ.


<!-- page 16 -->
16
Assumption 3 (Smoothness) Each local cost function fi(·, ·) in problem (6) is L-smooth with respect
to θi and vi. i.e, ∀(θi, νi), (θ′
i, ν′
i) ∈Rn × Vi,
∥∇fi(θi, νi) −∇fi(θ′
i, ν′
i)∥≤L∥(θi, νi) −(θ′
i, ν′
i)∥.
(27)
Meanwhile, the local cost function Fi(·, ·, ·) is LF -smooth. The function Li(·, ·) (resp. h(·)) is eL-smooth
(resp. Lh-smooth).
Assumption 4 (Bounded variance) For a data samples di uniformly sampled at random from Li or Ui,
the resulting stochastic gradients (SG) for problem (6) are unbiased and have bounded variances, i.e.,
E[∇θfi(θi, νi; di)] = ∇θfi(θi, νi),
(28)
E[∥∇θfi(θi, νi; di) −∇θfi(θi, νi)∥2] ≤σ2,
(29)
E[∇θi,lcFi(θi,lc, θi, νi; di)] = ∇θi,lcFi(θi,lc, θi, νi),
(30)
E[∥∇θi,lcFi(θi,lc, θi, νi; di) −∇θi,lcFi(θi,lc, θi, νi)∥2] ≤σ2,
(31)
where σ > 0 is a constant.
Then, gi(θi, νi) is unbiased with variance bounded by σ2
S where S = Sl+Su. Analogously, Gi(θi,lc, θi, νi)
is also unbiased with variance bounded by σ2
S .
Assumption 5 (Model discrepancy) For any two data inputs x1, x2, the distance between their model
outputs is bounded, i.e.,
∥h(x1) −h(x2)∥≤Γ.
(32)
In practice, Assumption 5 is mild as it is known that the output of h(x) usually lies in a bounded space.
For example, for a classification task, h(x) ∈[0, 1]n, and thus Γ ≤2.
Then, we start to build the convergence properties of the proposed FedCPSL algorithm. It demands the
convergence of the global model parameter θ, the pseudo labels ν, and the localized model parameter
θi,lc to a solution of problem (6). To achieve this, we first establish the convergence of the sequence
{(θr, νr)} generated by FedCPSL to a stationary point of problem (6b). Then, we build the convergence
of {(θr
i,lc, νr)} to a stationary point of problem (6).
We define the following term as the optimality gap between (θr, νr) and a stationary solution of
problem (6b).
G(θr, νr) ≜∥∇θf(θr, νr)∥2 + 31L
64
N
X
i=1
ωi∥νr+1
i
−νr
i ∥2.
(33)


<!-- page 17 -->
17
Note that if G(θr, νr) = 0, then (θr, νr) is a stationary solution of problem (6b). The following theorem
establishes the convergence of {(θr, νr)}.
Theorem 1 Let Ar(|Ar| = m < N) be a set of edge devices obtained by uniform sampling without
replacement in round r. In Algorithm 1, suppose νi is updated via (10), the stepsize ηv ≤
1
4L, the stepsize
η satisfy
η ≤min
 
m
48ηgNL,
1
8LQ,
3ηgN
100mLQ
2 ,
m
32ηgNL
 
1 + 2N
m
 −1
2  
,
(34)
where Q = max
i,r
eQr
i . Then, under Assumption 1, 2, 3 and 4, the generated sequence {(θr, νr)} of FedCPSL
satisfies
1
R
R−1
X
r=0
E[G(θr, νr)] ≤4(P 0 −f)
ηηgR
+
 259
64 + m
N + 5ηmLQ
2
24ηgN
 12ηηgNLσ2
mS
.
(35)
where P 0 is independent of R, σ, S, and it is defined in (48).
Proof: Unlike the existing works [18] [21], we consider a non-convex SSFL problem with the
presence of heterogeneous edge devices. To reduce the negative effects of non-i.i.d. data and time-varying
HLU and accelerate the algorithm convergence, we adopt schemes of local SGD with momentum and
novel client variance reduction, which makes Theorem 1 much more challenging to prove. Following
the analysis framework in [21], we attempt to build a potential function which descends as (θi, θ, νi)
proceeds in Algorithm 1 by analyzing the one-round progress of the cost function f. To this end, we
develop some new techniques to overcome the challenges brought by the novel update of θi in Algorithm
1. Details are presented in Appendix A.
One can see from Theorem 1 that the convergence of the global model parameter θ is resilient to
heterogeneous edge devices. In particular, it converges with much less influence caused by non-i.i.d.
data compared to FedAvg [12] and allows time-varying HLU. The following corollary demonstrates the
sublinear convergence rate of θ and νi with a carefully chosen constant step size η. The convergence rate
matches the known lower bounds for nonconvex FL with client variance reduction [21]. However, with
the inclusion of local momentum, the convergence of θ and νi is provably faster in practice. This point
will be further examined via numerical experiments in Section V.
Corollary 1 Let η = O
 
√m
√
ηgNR
 
, then under the same setting as Theorem 1, we obtain a sublinear
convergence rate of the sequence {(θr, νr
i )}, i.e. O
 
√
N
√
mηgR
 
.


<!-- page 18 -->
18
Next, let us move to the convergence of the localized model parameter θi,lc. To build the conver-
gence condition, we employ the term PN
i=1 ωi∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2 as the optimality gap. When
PN
i=1 ωi∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2 = 0, θr
i,lc, i ∈[N] is a stationary point of problem (6).
Theorem 2 Consider the same setting of ρ and the update of νi as Theorem 1, and suppose the stepsizes
η, ηc and βi, i ∈[N] satisfy
η ≤min
 
m
48ηgNL,
m
48ηgND0
,
1
8LQ,
3ηgN
100mLQ
2 ,
m
32ηgNL
 
1 + 2N
m
 −1
2  
,
(36)
ηc ≤min
 
1
2QLF
, eηNL2
QmL2
F
, eηN
Qm
 
,
(37)
D0 ≤11L
8
+ 11m
2N ,
(38)
where D0 ≜1
4
PN
i=1 ωi(1 −βi)(4βieLLhΓ + 5L), Q = min
i,r
eQr
i , and Q are defined in Theorem 1. Then,
under Assumption 1, 2, 3 and 4, the generated sequence {θr
i,c} of FedCPSL satisfies
1
R
R−1
X
r=0
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2] ≤
3N
ηcQmR(f(θ0, ν0
i ) −f) +
11N
2ηcQmR(P 0 −f)
+ 3ηcLF (ηcLF + 1)Qσ2
2S
|
{z
}
≜(a)
+ 3η2η2
gN2σ2
2ηcQm2S
|
{z
}
≜(b)
+
 259
64 + m
N + 5ηmLQ
2
24ηgN
 66η2η2
gN2Lσ2
ηcQm2S
|
{z
}
≜(c)
+ 3eL2Γ2
2ηcQL
N
X
i=1
ωiβ2
i
|
{z
}
≜(d)
+ 3NΓ2
ηcQm
N
X
i=1
ωiβi(1 −βi)(βi(eL −µ) + eL)
|
{z
}
≜(e)
,
(39)
where P 0 is independent of R, σ, S, and it is defined in (48).
Proof: In contrast to APFL [23], we study the convergence of the localized model parameter
θi,lc instead of the mixed model βih(θi,lc) + (1 −βi)h(θ). The challenge arises from the fact that the
optimization of θi,lc partially incorporates the global model h(θ), which brings difficulties in characterizing
the evolution of θi,lc as the algorithm proceeds. To establish the convergence of θi,lc, we follow the
analysis framework of local SGD within each edge device i, while studying the contribution of the
dynamic θ and νi to the update of θi,lc. To be specific, we analyze the one-round progress of Fi by
considering not only θi,lc but also the effects of θ and νi from the proof of Theorem 1. The bound of
the optimality gap is then derived. More details can be found in Appendix F .
Theorem 2 shows the convergence of θi,lc with respect to four kinds of error terms. The term (a) in the
RHS of (39) is caused by SGD in the update of θi,lc while (b)(c) are due to SGD in the update of θ. Both
of them are obviously 0 if σ = 0 or can vanish to 0 in a sublinear manner with a carefully chosen step


<!-- page 19 -->
19
size η and mini-batch size S. The term (d) (resp. (e)) in the RHS of (39) stems from the incorporation
of νi (resp. θ) into the update of (39), which implies that the dynamic θ and νi can deteriorate the
convergence of θi,lc, even though they gradually converge. According to Corollary 2, we obtain that the
sequence {θr
i,lc} sublinearly converges towards a ball with the center being a stationary point and radius
being (d) + (e). It is worth-noting that, when βi →0, i ∈[N], the terms (d)(e) will approximate 0, and
βi →1, i ∈[N] also means that (e) →0. This implies that, with proper choices of ηc and βi, both (d)
and (e) are relatively small and {θr
i,lc} converges within a neighborhood of the solutions.
Corollary 2 Let η = O
 
√m
√
ηgNR
 
and S = O(
√
R), then under the same setting as Theorem 2, the
following convergence result of the sequence {θr
i,lc} holds.
1
R
R−1
X
r=0
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2]
≤O
  N
mR + σ2
√
R
 
+ (d) + (e),
(40)
where (d) and (e) are defined in Theorem 2.
From Theorem 2 and Corollary 2, we have some important observations and insights which are
summarized in Remark 2, Remark 3 and Remark 4.
Remark 2 (Robustness to heterogeneous edge devices) As seen from Theorem 2, unlike [23], the
convergence of θi,lc does not suffer from the non-i.i.d. data. This is understandable as the effect of
non-i.i.d. data on θi,lc originates from the update of θ where we, however, employ client variance reduction
to remedy that issue. In addition, we adopts the technique of momentum to boost the convergence of θi,lc.
Thus, it is believed to be faster than existing algorithms using merely local SGD without client variance
reduction and momentum [23].
Remark 3 (Trade-off between accuracy and efficiency) One can notice that the terms (d) and (e) in the
RHS of (39) exist because of the simultaneous training of θ, νi and θi,lc. They will disappear if θ and νi
are constant in the optimization of θi,lc. However, the joint optimization of θ, νi and θi,lc in Algorithm 1
would greatly improve the algorithm efficiency as we can obtain these desirable parameters simultaneously.
Thus, there is a trade-off between convergence performance and algorithm efficiency. Fortunately, this can
be easily resolved by simply running a few more rounds for updating θi,lc when θ and νi have negligible
changes. By this way, the proposed FedPCSL can offer both high efficiency and model accuracy.
Remark 4 The above theoretical results are all based on one-step GD update rule in (10) for νi, i ∈[N].
Similar theoretical guarantees can also be easily obtained when the exact solution of (7a) is pursued to


<!-- page 20 -->
20
update νi, i ∈[N] at each round. In particular, following the same spirit in [18], we can utilize the strong
convexity of fi(θ, ·) and Fi(θi,lc, θ, ·) to establish similar descent lemmas with respect to νi to obtain
Lemma 1 in the proof of Theorem 1. Due to space limitations, we omit this and refer [18] for more
details.
Remark 5 The choice of βi for each edge device plays an important rule in the generalization performance
of FedCPSL, and a particular setup of it would complement FedCPSL. Nevertheless, the task of effectively
estimating the optimal βi, i ∈[N] is non-trivial. By [23, Theorem 1 and equation (3)], the optimal
value of βi depends on numerous factors which are remarkably difficult to compute or estimate. Direct
incorporation of existing techniques such as the similarity scores [52], [53] between local data distributions
of any two devices would yield incorrect estimations. The strategy of regarding βi as an optimization
variable and updating it periodically also cannot be straightforwardly applied because of the complexity
of problem (6). It will be an interesting direction to study the choices of βi with theoretical justification
in the future.
V. EXPERIMENT RESULTS
In this section, we will examine the performance of the proposed FedCPSL algorithm against the
state-of-the-art algorithms on real datasets.
A. Experiment setup
Cost function fi and Fi: We consider the problem formulation in [18] for the local cost function fi
in problem (6b) with
l(h(θ; xi), yi) = −1
ni
ni
X
k=1
⟨yi,k, log(h(θ; xi,k)),
(41)
l(h(θ; ui), νi) = −1
mi
mi
X
k=1
⟨νi,k, log(h(θ; ui,k)),
(42)
R(h(θ; ui), νi) = 1
mi
mi
X
k=1
KL(νi,k, d) + 1
mi
mi
X
k=1
KL(h(θ; ui,k), d),
(43)
where KL(·, ·) is the Kullback-Leibler divergence; d ≜1
C with 1 being all-one vector. The cost function
Fi is obtained by replacing h(θ; ·)) in fi with βih(θi,lc; ·) + (1 −βi)h(θ; ·).
Datasets and models h: The popular MNIST [54] and CIFAR-10 [55] datasets are considered for
evaluation both of which are widely used in previous FL works. Specifically, the MNIST dataset contains
60K training images and 10K test ones while the CIFAR-10 dataset has 50K training images of handwritten
digits and 10K test ones. We respectively adopt a CNN model for the CIFAR-10 image classification task


<!-- page 21 -->
21
and a fully-connection neural network model for the MNIST image classification task. The former is
similar to that in [12] which consists of two convolutional layers and two fully connected layers while
the latter is the same as that in [36].
We simulate the PSSFL process by distributing the training samples of each dataset to N = 20 edge
devices in the fashion of Non-IID. To obtain the non-IID distributed data, we follow the heterogeneous
data partition method in [12] where each edge device is allocated data samples of only a few class labels.
In particular, in [12], the training data is sorted by the label and then partitioned into shards. Each edge
device is assigned a small and fixed number of shards uniformly at random. This leads to highly non-i.i.d.
datasets among edge devices. The number of class labels in each edge device determines the non-i.i.d.
degree. Note that the method in [12] is, as far as we know, one of the most widely used data distribution
method in the FL literature, and is acknowledged to be sufficient for performance evaluation. Then, we
randomly extract 20% of data samples from each edge device as its test data for performance evaluation.
Similar to [18], in each edge device, we vary a ratio ϵ (0 ≤ϵ ≤1) of unlabeled data to divide the rest
80% data samples into the unlabeled data Ui and the labeled data Li. ϵ = 0.5 means that half of the
training data of each edge device have no labels. By default, ϵ is set as 0.9 in the experiment to simulate
the real scenario where each edge device only has a few labeled data samples.
Baseline algorithms for comparison: We evaluate the proposed FedCPSL algorithm by comparing
it with the following four baseline FL algorithms. For all these methods, we report test accuracy as
a function of communication rounds. The reported results are the average of the performances of the
corresponding models over all edge devices’ test data.
• FedSHVR: The algorithm proposed in [18] for semi-supervised FL but which only supports full
participation.
• FedSHVRP: The algorithm implemented by adapting FedSHVR to the case of PCP and using the
technique of client variance reduction in [21].
• APFL1: The algorithm proposed in [23] for personalized FL by adopting model interpolation, but
which only leverages the labeled data for training.
• APFSL: The algorithm implemented by adapting APFL to our proposed formulation so as to handle
the semi-supervised FL with both labeled and unlabeled data.
Parameter setting: For all algorithms under test, the mini-batch size Bl = 32 for labeled data and
Bu = 32 for unlabeled data. In each communication round, we only sample 10% or 20% of the total edge
1For fair comparison, we modify the original APFL by considering the mixture of the global model and the localized model
as the personalized model for each edge device.


<!-- page 22 -->
22
devices (|Ar| = 2 or 4) and all active edge devices perform E = 2 local epochs. To evaluate FedCPSL in
the presence of heterogeneous edge devices, we also consider (time-varying) HLU in which the number
of local update steps of each participating edge device is randomly sampled from a fixed range in each
communication round. In particular, if HLU is considered, we choose the number of local epochs at
random from [1, 5] at each communication round. The learning rate η is set to be 0.005 (resp. 0.002)
for the MNIST (resp. CIFAR-10) dataset if HLU is not applied, and it is set to be 0.002 (resp. 0.001)
for the MNIST (resp. CIFAR-10) dataset otherwise. Besides, the ηlc of FedCPSL is set as 2η and ηg is
the average of the number of local steps of the local edge devices. We adopt (7a) to update the pseudo
labels νi for each round as (7a) has a closed-form solution [18]. Finally, all algorithms stop when 100
communication rounds are achieved.
B. The effect of model personalization
First, we examine the performance of the proposed FedCPSL algorithm when different ratios of the
global model are considered in the personalized model. We conducted experiments with different choices
of βi on both the MNIST and CIFAR-10 datasets. The numerical results are depicted in Fig. 1. Note
that no local momentum is used, i.e., γ = 0. As seen, the personalized model significantly improve the
test accuracy of FedCPSL over the global model when faced with non-i.i.d. data. In particular, when
m = 0.1N, smaller values of βi (using more of the global model as the personalized model) yield lower
test accuracy and exhibit unstable convergence curves. Increasing βi can diminish the negative impacts of
partial participation on the personalized models. This demonstrates that larger values of βi are preferred
for highly non-i.i.d. datasets, and it is necessary to adopt model personalization instead of merely using
the global model. With βi = 0.75 for all edge devices, the proposed FedCPSL algorithm significantly
outperforms the case where βi = 0.25 in terms of both convergence speed and test accuracy. In addition,
one can also see that βi = 1 may not be the optimal choice as the test accuracy can be improved with
the assistance of the global model. Similar results can be observed when m = 0.2N on both the MNIST
and CIFAR-10 datasets.
C. The effect of local momentum
We further examine the effect of local momentum on the performance of the proposed FedCPSL
algorithm by considering different values of γ. Fig. 2 displays the corresponding test accuracy of FedCPSL
versus the number of communication rounds on the MNIST dataset. One can see from both Fig. 2(a)
and Fig. 2(b) that a non-zero γ can speed up the convergence of FedCPSL and FedCPSL with γ = 0.8
can quickly reach a higher test accuracy with fewer communication rounds than almost all other values


<!-- page 23 -->
23
(a) MNIST, m = 0.1N
0
0
20
40
60
80
100
Round r
0.75
0.80
0.85
0.90
0.95
1.00
Test Accuracy
i = 0.25
i = 0.5
i = 0.75
i = 1
(b) MNIST, m = 0.2N
0
20
40
60
80
100
Round r
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Test Accuracy
i = 0.25
i = 0.5
i = 0.75
i = 1
(c) CIFAR-10, m = 0.1N
0
20
40
60
80
100
Round r
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Test Accuracy
i = 0.25
i = 0.5
i = 0.75
i = 1
(d) CIFAR-10, m = 0.2N
Fig. 1. Test accuracy versus number of rounds of FedCPSL with different values of βi and m. Note that γ is set as 0 for all
cases.
of γ under test. Such results corroborate our thoughts that local momentum could speed up the algorithm
convergence and improve the performance of FedCPSL.
D. Performance comparison
In this subsection, we compare our proposed FedCPSL algorithm against the four aforementioned
baseline algorithms on both the MNIST and CIFAR-10 datasets. Based on above results, we choose
βi = 0.75, i ∈[N] and γ = 0.8 for FedCPSL. The mix ratios used in APFL and APFSL are also set as


**[Table p23.1]**
|  | 1.00 |
| --- | --- |
| (a) MNIST, m 0.8 0.7 Accuracy 0.6 0.5 Test 0.4 0.3 | 0.95 Accuracy 0.90 Test 0.85 0.80 0.75 0 0 20 40 R = 0.1N (b) MNIST 0.8 0.7 Accuracy i = 0.25 0.6 = 0.5 i i = 0.75 0.5 Test = 1 i 0.4 0.3 |
| 0.2 0 20 40 Rou (c) CIFAR-10, | 0.2 60 80 100 0 20 40 nd r R m = 0.1N (d) CIFAR-1 |

[CAPTION] Fig. 1. Test accuracy versus number of rounds of FedCPSL with different values of βi and m. Note that γ is set as 0 for all


<!-- page 24 -->
24
0
20
40
60
80
100
Round r
0.86
0.88
0.90
0.92
0.94
0.96
0.98
1.00
Test Accuracy
= 0
= 0.2
= 0.4
= 0.6
= 0.8
(a) MNIST, m = 0.1N
0
0
20
40
60
80
100
Round r
0.86
0.88
0.90
0.92
0.94
0.96
0.98
1.00
Test Accuracy
= 0
= 0.2
= 0.4
= 0.6
= 0.8
(b) MNIST, m = 0.2N
Fig. 2. Test accuracy versus number of rounds of FedCPSL with different values of γ and m.
0
20
40
60
80
100
Round r
0.75
0.80
0.85
0.90
0.95
1.00
Test Accuracy
FedSHVRP, m = 0.1
FedSHVRP, m = 0.3
FedSHVRP, m = 0.5
FedSHVR
FedCPSL, m = 0.1
(a) MNIST
0
20
40
60
80
100
Round r
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Test Accuracy
FedSHVRP, m = 0.1
FedSHVRP, m = 0.3
FedSHVRP, m = 0.5
FedSHVR
FedCPSL, m = 0.1
(b) CIFAR-10
Fig. 3. Test accuracy versus number of rounds of FedCPSL, FedSHVRP and FedSHVR.
0.75. Fig. 3 displays the comparison results of FedCPSL against FedSHVRP and FedSHVR while Fig. 4
displays the comparison results of FedCPSL against APFL and APSFL.
First of all, one can see from Fig. 3 that on both datasets the proposed FedCPSL algorithm with
only 10% of all edge devices being active performs significantly better than FedSHVRP with all values
of m under test. Although, with increased m, FedSHVRP obtains improved convergence speed and


**[Table p24.1]**
| 1.00 1.00 0.98 0.98 0.96 0.96 Accuracy Accuracy 0.94 0.94 0.92 0.92 Test Test = 0 0.90 0.90 = 0.2 = 0.4 0.88 0.88 = 0.6 0.86 = 0.8 0.86 0 20 40 60 80 1000 0 20 40 Round r Ro (a) MNIST, m = 0.1N (b) MNIST, t accuracy versus number of rounds of FedCPSL with different values of γ a |  |
| --- | --- |
| 1.00 0.95 Accuracy 0.90 Test 0.85 F | 0.8 0.7 0.6 Accuracy 0.5 0.4 Test 0.3 edSHVRP, m = 0.1 |
| F 0.80 F F F 0.75 0 20 40 Rou (a) MN t accuracy versus number of 3 displays the compari he comparison results all, one can see from of all edge devices be er test. Although, wit | edSHVRP, m = 0.3 0.2 edSHVRP, m = 0.5 edSHVR 0.1 edCPSL, m = 0.1 0.0 60 80 100 0 20 40 nd r Ro IST (b) CI rounds of FedCPSL, FedSHVRP and FedSHVR. son results of FedCPSL against FedSHV of FedCPSL against APFL and APSFL. Fig. 3 that on both datasets the propo ing active performs significantly better th h increased m, FedSHVRP obtains imp |
|  |  |

[CAPTION] Fig. 2. Test accuracy versus number of rounds of FedCPSL with different values of γ and m.

[CAPTION] Fig. 3. Test accuracy versus number of rounds of FedCPSL, FedSHVRP and FedSHVR.


<!-- page 25 -->
25
0
20
40
60
80
100
Round r
0.70
0.75
0.80
0.85
0.90
0.95
1.00
Test Accuracy
APFL
APSFL
FedCPSL
(a) MNIST, m = 0.1N
0
0
20
40
60
80
100
Round r
0.70
0.75
0.80
0.85
0.90
0.95
1.00
Test Accuracy
APFL
APSFL
FedCPSL
(b) MNIST, m = 0.2N
0
20
40
60
80
100
Round r
0.3
0.4
0.5
0.6
0.7
0.8
Test Accuracy
APFL
APSFL
FedCPSL
(c) CIFAR-10, m = 0.1N
0
20
40
60
80
100
Round r
0.3
0.4
0.5
0.6
0.7
0.8
Test Accuracy
APFL
APSFL
FedCPSL
(d) CIFAR-10, m = 0.2N
Fig. 4. Test accuracy versus number of rounds of FedCPSL, APFL and APSFL.
test performance, it still has a big performance gap with our proposed FedCPSL algorithm. Note that
FedSHVR corresponds to FedSHVRP with m = N. The proposed FedCPSL algorithm is also quite
effective in reducing the communication cost because even a much smaller value of m (m = 0.1N) can
yield better performance than FedSHVRP. Since both FedSHVRP and FedSHVR employ the global model
as the personalized model, this again demonstrates the necessity and superiority of model personalization,
especially in the presence of heterogeneous edge devices.
Then, as shown in Fig. 4, the proposed FedCPSL algorithm greatly outperforms APFL and APSFL on
both the MNIST and CIFAR-10 datasets. On one hand, it can be seen from Fig. 4(a) and Fig. 4(c) that


**[Table p25.1]**
|  |  | 1.00 1.00 |  |
| --- | --- | --- | --- |
|  | Test Accuracy Test Accuracy | 0.95 0.90 0.85 0.80 0.75 0.70 0 20 40 Ro (a) MNIST, 0.8 0.7 0.6 0.5 0.4 | 0.95 0.90 Accuracy 0.85 Test 0.80 APFL APSFL 0.75 FedCPSL 0.70 60 80 100 0 0 20 und r m = 0.1N (b) MN 0.8 0.7 Accuracy 0.6 Test 0.5 APFL 0.4 APSFL FedCPSL |
|  | 0.3 0 20 40 Ro (c) |  | 0.3 60 80 100 0 20 und r m = 0.1N (d) |

[CAPTION] Fig. 4. Test accuracy versus number of rounds of FedCPSL, APFL and APSFL.


<!-- page 26 -->
26
(a) MNIST, m = 0.1N
(b) CIFAR-10, m = 0.1N
Fig. 5. Test accuracy versus number of rounds of FedCPSL, FedSHVRP, APFL and APSFL under HLU.
the proposed FedCPSL algorithm and APSFL yield a much more stable convergence and perform better
than APFL. This demonstrates the advantage of semi-supervised FL which leverages both labeled and
unlabeled data. Similar results can be observed in Fig. 4(b) and Fig. 4(d). On the other hand, we also
see that the proposed FedCPSL algorithm usually converges much faster to a high test accuracy than the
others. In particular, as seen in Fig. 4(a) (resp. Fig. 4(c)), FedCPSL takes about 30 (resp. 21) rounds to
achieve 95% test accuracy, whereas the others need more than 80 rounds to obtain such performance. In
summary, the number of communication rounds needed to reach a high accuracy is reduced by up to 60%.
We further examine these algorithms under the scenario of HLU. Fig. 5 depicts the convergence
performance of FedCPSL against the baseline algorithms when m = 0.1N and HLU is applied. We can
observe from Fig. 5 that, when HLU is considered, all algorithms under test would have a performance
degradation and their convergence curves are less stable than that without HLU. However, using HLU
has a much less influence on the convergence performance of our proposed FedCPSL and FedCPSL
still performs much better than FedSHVRP, APFL and APFSL in terms of both convergence speed and
application performance. This demonstrates the superior robustness of our proposed FedCPSL algorithm
to HLU over the others. Table I shows the detailed test accuracy achieved by all the five algorithms under
test. One can observe that the proposed FedCPSL algorithm performs the best and achieves the highest
test accuracy for most cases. Moreover, there exists a considerable performance gap between FedCPSL
and the others. One can also notice that, under HLU, the APFL algorithm performs slightly better than

[CAPTION] Fig. 5. Test accuracy versus number of rounds of FedCPSL, FedSHVRP, APFL and APSFL under HLU.


<!-- page 27 -->
27
TABLE I
TEST ACCURACY (%) OF THE CONSIDERED FIVE ALGORITHMS ON THE TWO DATASETS WITH TWO VALUES OF m.
Dataset
MNIST
(m = 0.1N)
MNIST
(m = 0.2N)
CIFAR-10
(m = 0.1N)
CIFAR-10
(m = 0.2N)
FedSHVRP
93.8
85.6
35.4
46.8
FedSHVR
94.3
94.3
39.8
39.8
APFL
94.9
96.8
69.1
77.6
APSFL
95.1
96.7
74.9
79.5
FedCPSL
98.2
98.8
80.7
80.8
FedSHVRP (HLU)
91.8
91.8
37.8
38.3
FedSHVR (HLU)
73.0
73.0
31.2
31.2
APFL (HLU)
88.9
95.3
74.3
81.1
APSFL (HLU)
91.0
95.1
72.5
78.7
FedCPSL (HLU)
97.5
98.5
79.5
80.3
FedCPSL in terms of test accuracy on the CIFAR-10 data with m = 0.2N. But FedCPSL converges
much faster and it achieves that test accuracy using much less communication rounds.
VI. CONCLUSION
To address the challenges of label deficiency and device heterogeneity in wireless edge networks, we
have investigated the PSSFL problem in this paper and proposed an efficient and effective algorithm,
FedCPSL. Specifically, we rely on the technique of pseudo-labeling and interpolation based model
personalization to model the PSSFL problem. The proposed FedCPSL algorithm incorporates the strategies
of adaptive client variance reduction and normalized global aggregation to mitigate the adverse effects of
heterogeneity on algorithm convergence. It also adopts the technique of local momentum to further boost
the algorithm convergence. Theoretically, we have built the convergence property of FedCPSL, which
shows that FedCPSL is resilient to the challenge of device heterogeneity and can converge in a sublinear
manner. The experimental results have also demonstrated the superiority of FedCPSL over the existing
methods with respect to both convergence speed and application performance.
APPENDIX A
PROOF OF THEOREM 1
Before delving into the proof of Theorem 1, we define the virtual sequence {(eθr
i , eνr
i ,ecr
i )} by assuming
that all edge devices are active at round r, i.e., ∀i, 0 ≤t ≤Qr
i −1,
eνr+1
i
= arg min
νi∈Vi fi(θr, νi),
(44a)


**[Table p27.1]**
| Dataset | MNIST | MNIST | CIFAR-10 | CIFAR-10 |
| --- | --- | --- | --- | --- |
|  | (m = 0.1N) | (m = 0.2N) | (m = 0.1N) | (m = 0.2N) |
| FedSHVRP | 93.8 | 85.6 | 35.4 | 46.8 |
| FedSHVR | 94.3 | 94.3 | 39.8 | 39.8 |
| APFL | 94.9 | 96.8 | 69.1 | 77.6 |
| APSFL | 95.1 | 96.7 | 74.9 | 79.5 |
| FedCPSL | 98.2 | 98.8 | 80.7 | 80.8 |
| FedSHVRP (HLU) | 91.8 | 91.8 | 37.8 | 38.3 |
| FedSHVR (HLU) | 73.0 | 73.0 | 31.2 | 31.2 |
| APFL (HLU) | 88.9 | 95.3 | 74.3 | 81.1 |
| APSFL (HLU) | 91.0 | 95.1 | 72.5 | 78.7 |
| FedCPSL (HLU) | 97.5 | 98.5 | 79.5 | 80.3 |


<!-- page 28 -->
28
eθr,0
i
= θr, eµr,0
i
= 0,
(44b)
eµr,t+1
i
= γ eµr,t
i
+ gi(eθr,t
i , eνr+1
i
) −cr
i + cr,
(44c)
eθr,t+1
i
= eθr,t
i
−ηeµr,t+1
i
,
(44d)
eθr+1
i
= eθr,Qr
i
i
,
(44e)
ecr+1
i
= cr
i −
 
cr +
eθr+1
i
−θr
η eQr
i
 
.
(44f)
The following additional terms are introduced for ease of analysis.
Ψr ≜
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i
−θr∥2
 
,
(45)
Ξr ≜
N
X
i=1
ωiE[∥∇θfi(θr, νr
i ) −cr
i ∥2],
(46)
Φr ≜
N
X
i=1
ωiE[∥eνr+1
i
−νr
i ∥2].
(47)
We also build the following potential function
P r = E[f(θr, νr)] + 24eη2N2L
m2
Ξr.
(48)
where eη ≜ηηg. Then, we proceed to the proof of Theorem 1 with the help of the following three lemmas,
while the detailed proofs of the lemmas refer to the supplementary materials.
Lemma 1 For any round r, it holds that
E[f(θr+1, νr+1)] −E[f(θr, νr)] ≤−
  eη
2 −4eη2NL
m
 
E[∥∇θf(θr, νr)∥2] +
  eηL2
2
+ 4eη2NL3
m
 
Ψr
−
  µ
2 + 1
ηv
−3L
2
 m
N −eηL2
2
−4eη2NL3
m
 
Φr + 8eη2NL
m
Ξr + eη2σ2NL
mS
.
(49)
Lemma 2 For any round r, if eη ≤
m
32NL(1 + 2N
m )−1
2 , it holds that
Ξr+1 −Ξr ≤
 257
256 + m
2N
 mL2
N Ψr −m
2N
 63
64 + m
N
 
Ξr + mL2
256N Φr
+
m
256N E[∥∇θf(θr, νr)∥2] +
 
1 + m
2N
 mσ2
NS +
mσ2
1024NS .
(50)
Lemma 3 Let Q = max
i,r
eQr
i , for any round, if η ≤
1
√
5LQ, it holds that
Ψr ≤5η2Q
2E[∥∇θf(θr, νr)∥2] + 5η2Q
2σ2
4S
+ 5η2Q
2L2Φr + 10η2Q
2Ξr,
(51)
In particular, combining the results (49) and (50) yields
P r+1 −P r ≤−
  eη
2 −131eη2NL
32m
 
E[∥∇θf(θr, νr)∥2] +
  eηL2
2
+ 4eη2NL3
m
+
 257
32 + 4m
N
 3eη2NL3
m
 
Ψr
−
  µ
2 + 1
ηv
−3L
2
 m
N −eηL2
2
−131eη2NL3
32m
 
Φr −
 61
16 + 12m
N
  eη2NL
m
Ξr + 131eη2NLσ2
128mS


<!-- page 29 -->
29
+
 
1 + m
2N
 24eη2NLσ2
mS
.
(52)
We multiply (51) by 2eηL2 and then combine it with (52) to obtain
P r+1−P r
≤−C1E[∥∇θf(θr, νr)∥2]−C2Ψr−C3Ξr −C4Φr +
 259
64 + m
N
 12eη2NLσ2
mS
+ 5η2eηQ
2L2σ2
2S
.
(53)
where
C1 ≜eη
2 −131eη2NL
32m
−10eηη2L2Q
2,
(54)
C2 ≜3eηL2
2
−4eη2NL3
m
−
 257
32 + 4m
N
 3eη2NL3
m
,
(55)
C3 ≜
 61
16 + 12m
N
  eη2NL
m
−20eηη2L2Q
2,
(56)
C4 ≜
 µ
2 + 1
ηv
−L
2
 m
N −eηL2
2
−131eη2NL3
32m
−10eηη2L4Q
2.
(57)
In order to get the result, we require the following four conditions
C1 ≥eη
4, C2 ≥0, C3 ≥0, C4 ≥0.
(58)
Then, we respectively show that these conditions hold.
Condition I: C1 ≥eη
4: As eη ≤
m
48NL, it suffices to have
eη
4 −131eη2NL
32m
−10eηη2L2Q
2 ≥eη
4
 
1 −
131
8 × 48 −40η2L2Q
2 
(59)
≥eη
4
  253
8 × 48 −40η2L2Q
2 
(60)
≥eη
4
 40
64 −40η2L2Q2
 
≥0,
(61)
which is equivalent to η ≤
1
8LQ. Thus, the condition C1 ≥eη
4 is true.
Condition II: C2 ≥0: Similarly, we have
C2 = 3eηL2
2
 
1 −8eηNL
3m
−
 257
16 + 4m
N
  eηNL
m
 
(62)
≥3eηL2
2
 
1 −23eηNL
m
 
(63)
≥3eηL2
4
,
(64)
where (64) follows because eη ≤
m
48NL.
Condition III: C3 ≥0: It suffices to have
 61
16 + 12m
N
  eη2NL
m
−20eηη2L2Q
2 ≥2eηηL
 9ηgN
5m
−10ηLQ
2 
+ 12eη2L ≥3eη2LN
m
+ 12eη2L,
(65)
where (65) follows because η ≤
3ηgN
100mLQ
2 . Thus, the condition C3 ≥0 is also true.


<!-- page 30 -->
30
Condition IV: C4 ≥0: We have
C4 =
 µ
2 + 1
ηv
−3L
2
 m
N −eηL2
2
−131eη2NL3
32m
−10eηη2L4Q
2
(66)
≥mL
2N −eηL2
2
−eηL2
4
+ µm
2N
(67)
≥31mL
64N + µm
2N ,
(68)
where the first inequality holds by (61) and the fact that ηv ≤
1
2L, the second inequality follows because
eη ≤
m
48NL. Therefore, we have from (53) that
P r+1 −P r ≤−eη
4E[∥∇θf(θr, νr)∥2] +
 259
64 + m
N + 5ηmLQ
2
24ηgN
 12eη2NLσ2
mS
,
(69)
which implies that
E[∥∇θf(θr, νr)∥2] ≤4(P r −P r+1)
eη
−31mL
64N Φr +
 259
64 + m
N + 5ηmLQ
2
24ηgN
 12eηNLσ2
mS
.
(70)
Summing (70) up from r = 0 to R −1 and dividing it by R yields
1
R
R−1
X
r=0
 
E[∥∇θf(θr, νr)∥2] + 31L
64
N
X
i=1
ωiE[∥νr+1
i
−νr
i ∥2]
 
≤4(P 0 −f)
eηR
+
 259
64 + m
N + 5ηmLQ
2
24ηgN
 12eηNLσ2
mS
.
(71)
This completes the proof.
■
APPENDIX B
POOF OF LEMMA 1
By Assumption 2, we have
E[f(θr, νr) −f(θr, νr+1)]
= E
  N
X
i=1
ωiIi
Ar(fi(θr, νr
i ) −fi(θr, eνr+1
i
))
 
(72)
≥E
  N
X
i=1
ωiIi
Ar
 
⟨∇νifi(θr, eνr+1
i
), νr
i −eνr+1
i
⟩+ µ
2 ∥eνr+1
i
−νr
i ∥2
  
(73)
= E
  N
X
i=1
ωiIi
Ar
 
⟨∇νifi(θr, eνr+1
i
)−∇νifi(θr, νr
i ), νr
i −eνr+1
i
⟩+ ⟨∇νifi(θr, νr
i ), νr
i −eνr+1
i
⟩+ µ
2 ∥eνr+1
i
−νr
i ∥2
  
(74)
≥
 µ
2 + 1
ηv
−L
 m
N
N
X
i=1
ωiE[∥eνr+1
i
−νr
i ∥2],
(75)
where Ii
Ar is the indicator denoting whether i ∈Ar or not; (73) follows by Assumption 2;(75) holds
because Prob(i ∈Ar) = m
N , (10) and the Cauchy-Schwarz inequality. Then, by Assumption 3, fi(θ, νi)
is L-smooth with respect to θ, and thus we obtain
E[f(θr+1, νr+1)] −E[f(θr, νr+1)]


<!-- page 31 -->
31
=
N
X
i=1
ωi(E[fi(θr+1, νr+1
i
)] −E[fi(θr, νr+1
i
)])
≤
N
X
i=1
ωi
 
E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩] + L
2 E[∥θr+1 −θr∥2]
 
=
N
X
i=1
ωi
 
E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩] + L
2 E[∥θr+1 −θr∥2] + E[⟨∇θfi(θr, νr+1
i
) −∇θfi(θr, νr
i ), θr+1 −θr⟩]
 
≤
N
X
i=1
ωi
 
E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩] + LE[∥θr+1 −θr∥2] + 1
2LE[∥∇θfi(θr, νr+1
i
) −∇θfi(θr, νr
i )∥2]
 
(76)
≤E[⟨∇θf(θr, νr), θr+1 −θr⟩] + L
2
N
X
i=1
ωiE[∥νr+1
i
−νr
i ∥2] + LE[∥θr+1 −θr∥2]
= mL
2N Φr + E[⟨∇θf(θr, νr), θr+1 −θr⟩] + LE[∥θr+1 −θr∥2],
(77)
where (76) holds thanks to the Jensen’s Inequality; (77) follows because Prob(i ∈Ar) = m
N . The last
two terms in the RHS of (75) can be further bounded using the following lemma.
Lemma 4 For any round r, it holds that
E[θr −θr+1] = eη
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θfi(eθr,t
i , eνr+1
i
)
 
,
(78)
E[∥θr+1 −θr∥2] ≤4eη2NL2
m
Ψr + 8eη2N
m
Ξr + 4eη2N
m
E[∥∇θf(θr, νr)∥2] + 4eη2NL2
m
Φr + eη2σ2N
mS
.
(79)
In particular, applying Lemma S.1 to (77) gives
E[⟨∇θf(θr, νr), θr+1 −θr⟩] = −eηE
  
∇θf(θr, νr),
N
X
i=1
ωi
Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θfi(eθr,t
i , eνr+1
i
)
  
= eη
2E
 ∇θf(θr, νr) −
N
X
i=1
ωi
Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θfi(eθr,t
i , eνr+1
i
)

2 
−eη
2E
 
N
X
i=1
ωi
Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θfi(eθr,t
i , eνr+1
i
)

2 
−eη
2E[∥∇θf(θr, νr)∥2]
(80)
≤−eη
2E[∥∇θf(θr, νr)∥2] + eηL2
2 Ψr + eηL2
2 Φr.
(81)
where (80) follows because ⟨v1, v2⟩= 1
2∥v1∥2 + 1
2∥v2∥2 −1
2∥v1 −v2∥2, ∀v1, v2 ∈Rn; (81) holds by
the convexity of ∥· ∥2 and Assumption 3. Substituting (81) and (79) into (75) yields
E[f(θr+1, νr+1)] −E[f(θr, νr+1)] ≤−
 eη
2 −4eη2NL
m
 
E[∥∇θf(θr, νr)∥2] +
 eηL2
2
+ 4eη2NL3
m
 
Ψr
+ 8eη2NL
m
Ξr +
  eηL2
2
+ mL
2N + 4eη2NL3
m
 
Φr + eη2σ2NL
mS
.
(82)
Lastly, combining (75) and (77) gives rise to
E[f(θr+1, νr+1)] −E[f(θr, νr)] ≤−
  eη
2 −4eη2NL
m
 
E[∥∇θf(θr, νr)∥2] +
  eηL2
2
+ 4eη2NL3
m
 
Ψr


<!-- page 32 -->
32
−
  µ
2 + 1
ηv
−3L
2
 m
N −eηL2
2
−4eη2NL3
m
 
Φr + 8eη2NL
m
Ξr + eη2σ2NL
mS
.
(83)
■
C PROOF OF LEMMA 2
First, we have
E[∥∇θfi(θr+1, νr+1
i
) −cr+1
i
∥2] = E[∥∇θfi(θr+1, νr+1
i
) −∇θfi(θr, νr+1
i
) + ∇θfi(θr, νr+1
i
) −cr+1
i
∥2]
≤
 
1 + 1
ϵ
 
L2E[∥θr+1 −θr∥2] + (1 + ϵ)E[∥∇θfi(θr, νr+1
i
) −cr+1
i
∥2]
(84)
=
 
1 + 1
ϵ
 
L2E[∥θr+1 −θr∥2] + (1 + ϵ)m
N E[∥∇θfi(θr, eνr+1
i
) −ecr+1
i
∥2]
+ (1 + ϵ)
 
1 −m
N
 
E[∥∇θfi(θr, νr
i ) −cr
i ∥2],
(85)
where (84) holds by the Jensen’s Inequality; (85) follows because Prob(i ∈Ar) = m
N . Then, we proceed
to bound E[∥∇θfi(θr, eνr+1
i
) −ecr+1
i
∥2]. By the definition of ecr+1
i
, we have
E[∥∇θfi(θr, eνr+1
i
) −ecr+1
i
∥2] = E
 ∇θfi(θr, eνr+1
i
) −
Qr
i −1
X
t=0
(br
i )t
eQr
i
gi(eθr,t
i , eνr+1
i
)

2 
≤E
 ∇θfi(θr, eνr+1
i
) −
Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θfi(eθr,t
i , eνr+1
i
)

2 
+ E
 
Qr
i −1
X
t=0
(br
i )t
eQr
i
(∇θfi(eθr,t
i , eνr+1
i
) −gi(eθr,t
i , eνr+1
i
))

2 
(86)
≤E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇θfi(θr, eνr+1
i
) −∇θfi(eθr,t
i , eνr+1
i
)∥2
 
+ σ2
S
(87)
≤L2E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr −eθr,t
i ∥2
 
+ σ2
S ,
(88)
where (86) follows because Var[ν] = E[ν2]−(E[ν])2, if ν is a random variable; (88) holds by Assumption
3, Assumption 4 and the independence of SGD over t. Substituting (88) into (85) yields
E[∥∇θfi(θr+1, νr+1
i
) −cr+1
i
∥2]
≤
 
1 + 1
ϵ
 
L2E[∥θr+1 −θr∥2] + (1 + ϵ)mσ2
NS
+(1+ϵ)
 
1−m
N
 
E[∥∇θfi(θr, νr
i )−cr
i ∥2]
+ (1 + ϵ)mL2
N E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr −eθr,t
i ∥2
 
.
(89)
Let us pick ϵ =
m
2N and then (89) becomes
E[∥∇θfi(θr+1, νr+1
i
) −cr+1
i
∥2] ≤
 
1 + 2N
m
 
L2E[∥θr+1 −θr∥2] +
 
1 + m
2N
 mσ2
NS
+
 
1 −m
2N
 
1 + m
N
  
E[∥∇θfi(θr, νr
i ) −cr
i ∥2]


<!-- page 33 -->
33
+
 
1 + m
2N
 mL2
N E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr −eθr,t
i ∥2
 
,
(90)
where (90) follows because
(1 + ϵ)
 
1 −m
N
 
= 1 −m
2N
 
1 + m
N
 
,
(91)
1 + ϵ = 1 + m
2N , 1 + 1
ϵ = 1 + 2N
m .
(92)
Taking the average over the two sides of (90) with respect to all edge devices yields
Ξr+1 −Ξr ≤
 
1 + 2N
m
 
L2E[∥θr+1 −θr∥2] +
 
1 + m
2N
 mσ2
NS
−m
2N
 
1 + m
N
 
E[∥∇θfi(θr, νr
i ) −cr
i ∥2] +
 
1 + m
2N
 mL2
N Ψr.
(93)
Lastly, substituting the results of Lemma S.1 into (93) yields
Ξr+1 −Ξr ≤
  
1 + 2N
m
 4eη2NL4
m
+
 
1 + m
2N
 mL2
N
 
Ψr −
  m
2N
 
1 + m
N
 
−
 
1 + 2N
m
 8eη2NL2
m
 
Ξr
+
 
1 + 2N
m
 4eη2NL4
m
Φr +
 
1 + 2N
m
  eη2σ2NL2
mS
+
 
1 + 2N
m
 4eη2NL2
m
E[∥∇θf(θr, νr)∥2]
+
 
1 + m
2N
 mσ2
NS .
(94)
Since eη ≤
m
32NL(1 + 2N
m )−1
2 , we have from (94) that
Ξr+1 −Ξr ≤
 257
256 + m
2N
 mL2
N Ψr −m
2N
 63
64 + m
N
 
Ξr + mL2
256N Φr
+
m
256N E[∥∇θf(θr, νr)∥2] +
 
1 + m
2N
 mσ2
NS +
mσ2
1024NS .
(95)
■
D PROOF OF LEMMA 3
By the definition of eθr,t
i , we have
E[∥eθr,t
i −θr∥2] = η2E
 
t−1
X
k=0
(br
i )Qr
i −t+k(gi(eθr,k
i
, eνr+1
i
) + cr −cr
i )

2 
(96)
= η2E
 
t−1
X
k=0
(br
i )Qr
i −t+k(∇θfi(eθr,k
i
, eνr+1
i
) + cr −cr
i )

2 
+ η2E
 
t−1
X
k=0
(br
i )Qr
i −t+k(gi(eθr,k
i
, eνr+1
i
) −∇θfi(eθr,k
i
, eνr+1
i
))

2 
(97)
≤η2E
  t−1
X
k=0
(br
i )Qr
i −t+k
Qr
i −1
X
t=0
(br
i )t∥∇θfi(eθr,t
i , eνr+1
i
)+cr−cr
i ∥2
 
+ η2σ2
S
E
  t−1
X
k=0
((br
i )Qr
i −t+k)2
 
,
(98)


<!-- page 34 -->
34
where (96) holds because
eθr,t
i
= eθr,t−1
i
−η
t−1
X
k=0
γt−1−k(gi(eθr,k
i , eνr+1
i
) + cr −cr
i )
= −η
t−1
X
s=0
s
X
k=0
γs−k(gi(eθr,k
i , eνr+1
i
) + cr −cr
i )
= −η
t−1
X
k=0
t−1
X
s≥k
γs−k(gi(eθr,k
i , eνr+1
i
) + cr −cr
i )
= −η
t−1
X
k=0
1 −γt−k
1 −γ
(gi(eθr,k
i , eνr+1
i
) + cr −cr
i )
= −η
t−1
X
k=0
(br
i )Qr
i −t+k(gi(eθr,k
i , eνr+1
i
) + cr −cr
i );
(97) follows because Var[ν] = E[ν2]−(E[ν])2, if ν is a random variable; (98) holds due to the independence
of SGD over t and Assumption 4. Furthermore, note that
Qr
i −1
X
t=0
(br
i )t
eQr
i
t−1
X
k=0
((br
i )Qr
i −t+k)2 ≤
Qr
i −1
X
t=0
(br
i )t
eQr
i
Qr
i −2
X
k=0
((br
i )k+1)2
=
Qr
i −2
X
k=0
((br
i )k+1)2
= ∥br
i ∥2 −((br
i )0)2 ≤∥br
i ∥2,
(99)
Qr
i −1
X
t=0
(br
i )t
eQr
i
t−1
X
k=0
(br
i )Qr
i −t+k ≤
Qr
i −1
X
t=0
(br
i )t
eQr
i
Qr
i −2
X
k=0
(br
i )k+1
=
Qr
i −2
X
k=0
(br
i )k+1
= eQr
i −(br
i )0
≤eQr
i .
(100)
Thus, taking average over the two sides of (98) with respect to t gives rise to
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr−eθr,t
i ∥2
 
≤(η eQr
i )2E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇θfi(eθr,t
i , eνr+1
i
) + cr −cr
i ∥2
 
+ η2σ2∥br
i ∥2
S
(101)
≤(η eQr
i )2
 
4L2E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr −eθr,t
i ∥2
 
+ 4L2E[∥eνr+1
i
−νr
i ∥2] + 4Ξr
+ 4E[∥∇θfi(θr, νr
i ) −cr
i ∥2] + 4E[∥∇θf(θr, νr)∥2]
 
+ η2σ2∥br
i ∥2
S
(102)


<!-- page 35 -->
35
≤(η eQr
i L)2E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr −eθr,t
i ∥2
 
+ 4(η eQr
i )2Ξr + 4(η eQr
i L)2E[∥eνr+1
i
−νr
i ∥2]
+ η2σ2∥br
i ∥2
S
+ 4(η eQr
i )2E[∥∇θf(θr, νr)∥2] + 4(η eQr
i )2E[∥∇θfi(θr, νr
i ) −cr
i ∥2],
(103)
where (101) follows by (99) and (100); (102) holds by (116). Rearranging the two sides of (101) yields
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr−eθr,t
i ∥2
 
≤
1
1 −(η eQr
i L)2
 
4(η eQr
i )2E[∥∇θf(θr, νr)∥2] + 4(η eQr
i L)2E[∥eνr+1
i
−νr
i ∥2] + 4(η eQr
i )2Ξr
+ 4(η eQr
i )2E[∥∇θfi(θr, νr
i ) −cr
i ∥2] + η2σ2∥br
i ∥2
S
 
(104)
≤5(η eQr
i )2E[∥∇θf(θr, νr)∥2] + 5(η eQr
i )2Ξr + 5η2σ2∥br
i ∥2
4S
+ 5(η eQr
i L)2E[∥eνr+1
i
−νr
i ∥2]
+ 5(η eQr
i )2E[∥∇θfi(θr, νr
i ) −cr
i ∥2],
(105)
where (105) follows because (η eQr
i L)2 ≤1
5. Lastly, taking the average over the two sides of (105) with
respect to all edge devices yields
Ψr ≤5η2Q
2E[∥∇θf(θr, νr)∥2] + 5η2Q
2σ2
4S
+ 5η2Q
2L2Φr + 10η2Q
2Ξr.
(106)
where Q = max
i,r
eQr
i .
■
E PROOF OF LEMMA 4
According to the update of θ0 and θi, we have
E[θr −θr+1] = N
mηgE
  N
X
i=1
ωiIi
Ar θr −eθr+1
i
eQr
i
 
= ηg
N
X
i=1
ωiE
 θr −eθr+1
i
eQr
i
 
(107)
= eη
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
(gi(eθr,t
i , eνr+1
i
) + cr −cr
i )
 
= eη
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
(∇θfi(eθr,t
i , eνr+1
i
) + cr −cr
i )
 
(108)
= eη
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θfi(eθr,t
i , eνr+1
i
)
 
,
(109)
where eη ≜ηηg; (108) follows because Prob(i ∈Ar) = m
N ; (109) follows because cr = PN
i=1 ωicr
i . Next,
we have
E[∥θr+1 −θr∥2]
=E
 ηg
N
X
i=1
ωiIi
Ar N
m
θr −eθr+1
i
eQr
i

2 


<!-- page 36 -->
36
=E
 eη
N
X
i=1
ωiIi
Ar N
m
Qr
i −1
X
t=0
(br
i )t
eQr
i
(gi(eθr,t
i , eνr+1
i
) + cr −cr
i )

2 
=eη2E
 
N
X
i=1
ωiIi
Ar N
m
Qr
i −1
X
t=0
(br
i )t
eQr
i
(gi(eθr,t
i , eνr+1
i
) + cr −cr
i )

2 
=eη2E
 
N
X
i=1
ωiIi
Ar N
m
Qr
i −1
X
t=0
(br
i )t
eQr
i
(∇θfi(eθr,t
i , eνr+1
i
) + cr −cr
i )

2 
+ eη2E
 
N
X
i=1
ωiIi
Ar N
m
Qr
i −1
X
t=0
(br
i )t
eQr
i
(gi(eθr,t
i , eνr+1
i
) −∇θfi(eθr,t
i , eνr+1
i
))

2 
(110)
=eη2E
 
N
X
i=1
ωiIi
Ar N
m
Qr
i −1
X
t=0
(br
i )t
eQr
i
(∇θfi(eθr,t
i , eνr+1
i
) + cr −cr
i )

2 
+ eη2E
  N
X
i=1
 
ωiIi
Ar N
m
 2 Qr
i −1
X
t=0
 (br
i )t
eQr
i
 2
∥gi(eθr,t
i , eνr+1
i
) −∇θfi(eθr,t
i , eνr+1
i
)∥2
 
(111)
≤eη2N
m
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇fi(eθr,t
i , eνr+1
i
)+cr−cr
i ∥2
 
+ eη2σ2N
mS
N
X
i=1
ω2
i E
  Qr
i −1
X
t=0
 (br
i )t
eQr
i
 2 
(112)
≤eη2N
m
N
X
i=1
ωiE
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇fi(eθr,t
i , eνr+1
i
) + cr −cr
i ∥2
 
+ eη2σ2N
mS
,
(113)
where (110) follows because Var[ν] = E[ν2] −(E[ν])2, if ν is a random variable; (111) holds due to the
independence of SGD over i, t; (112) follows by Assumption 4; (113) follows because
N
X
i=1
ω2
i E
  Qr
i −1
X
t=0
 (br
i )t
eQr
i
 2 
≤1.
(114)
Furthermore, note that
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇fi(eθr,t
i , eνr+1
i
) + cr −cr
i ∥2
 
= E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇θfi(eθr,t
i , eνr+1
i
) −∇θfi(θr, νr
i ) + cr −∇θf(θr, νr) + ∇θf(θr, νr) −(cr
i −∇θfi(θr, νr
i ))∥2
 
≤E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
 
4L2∥θr−eθr,t
i ∥2+4L2∥eνr+1
i
−νr
i ∥2+4Ξr+4∥∇θfi(θr, νr
i )−λr
i ∥2+4∥∇θf(θr, νr)∥2
  
(115)
= 4L2E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥θr−eθr,t
i ∥2
 
+4L2E[∥eνr+1
i
−νr
i ∥2]+4Ξr+4E[∥∇θfi(θr, νr
i )−cr
i ∥2]+4E[∥∇θf(θr, νr)∥2],
(116)
where (115) follows by Assumption 3. As a result, we have from (113) that
E[∥θr+1 −θr∥2] ≤4eη2NL2
m
Ψr + 8eη2N
m
Ξr + 4eη2N
m
E[∥∇θf(θr, νr)∥2] + 4eη2NL2
m
Φr + eη2σ2N
mS
.
(117)
■


<!-- page 37 -->
37
F PROOF OF THEOREM 2
Similar to the poof of Theorem 1, we start with defining the virtual sequences {eθr
i,c} with by assuming
that all edge devices are active at round r, i.e., ∀i, 0 ≤t ≤Qr
i −1,
eθr,0
i,lc = θr
i,lc, eµr,0
i,lc = 0,
(118)
eµr,t+1
i,lc
= γµr,t
i,lc + Gi(eθr,t
i,lc, eθr,t
i , eνr+1
i
),
(119)
eθr,t+1
i,lc
= eθr,t
i,lc −ηeµr,t+1
i,lc
,
(120)
eθr+1
i,lc = eθr,Qr
i
i
.
(121)
Then we proceed with the help of the following Lemma.
Lemma 5 For any round r, if ηcQLF ≤1
2, it holds that
E[Fi(θr+1
i,lc , θr+1, νr+1
i
)] −E[Fi(θr
i,lc, θr, νr
i )]
≤−ηc eQr
i m
2N
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2]+ η2
cLF (ηcLF + 1)( eQr
i )2mσ2
2NS
+ ηc eQr
i mL2
F
2N
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i −θr∥2
 
+ (1 −βi)E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩] + (1 −βi)(4βieLLhΓ + 5L)
8
E[∥θr+1 −θr∥2] + β2
i (1 −βi)(eL −µ)Γ2
+ βi(1 −βi)eLΓ2 + meL2β2
i Γ2
2NL
−
  1
ηv
+ µ
2 −3L
2 −2(1 −βi)mL
N
 
E[∥eνr+1
i
−νi∥2].
(122)
Rearranging the two sides of (122) and then taking its average over all edge devices yields
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2]
≤
2N
ηcQm
N
X
i=1
ωi(E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr+1
i,lc , θr+1, νr+1
i
)]) + ηcLF (ηcLF + 1)Qσ2
S
+ L2
F Ψr
+
2N
ηcQmE[⟨∇θf(θr, νr), θr+1 −θr⟩] + E[∥θr+1 −θr∥2]
N
X
i=1
ωi
(1 −βi)(4βieLLhΓ + 5L)N
4ηc eQr
i m
+ Γ2
N
X
i=1
ωi
2Nβi(1 −βi)(βi(eL −µ) + eL)
ηc eQr
i m
+
eL2Γ2
ηcQL
N
X
i=1
ωiβ2
i −
  1
ηv
+ µ
2 −3L
2 −2(1 −β)mL
N
  2N
ηcQmΦr,
(123)
where β = min
i
βi. Substituting (81) and (79) into (123) yields
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2]
≤
2N
ηcQm
N
X
i=1
ωi(E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr+1
i,lc , θr+1, νr+1
i
)]) + ηcLF (ηcLF + 1)Qσ2
S
+
  eηNL2
ηcQm + L2
F + 4eη2N 2L2D0
ηcQm2
 
Ψr −
eηN
ηcQm
 
1 −4eηND0
m
 
E[∥∇θf(θr, νr)∥2]


<!-- page 38 -->
38
+ Γ2
N
X
i=1
ωi
2Nβi(1 −βi)(βi(eL −µ) + eL)
ηc eQr
i m
+
eL2Γ2
ηcQL
N
X
i=1
ωiβ2
i
−
  1
ηv
+ µ
2 −3L
2 −2(1 −βi)mL
N
−eηL2
2
−2eη2NL2D0
m
  2N
ηcQmΦr + 8eη2N 2D0
ηcQm2 Ξr + eη2N 2σ2
ηcQm2S ,
(124)
where D0 ≜PN
i=1 ωi
(1−βi)(4βi eLLhΓ+5L)
4
. Furthermore, we conclude from Theorem 1 that
P r+1 −P r ≤−eη
4E[∥∇θf(θr, νr)∥2] −3eηL2
4
Ψr −
 3eη2LN
m
+ 12eη2L
 
Ξr
−31mL
64N Φr +
 259
64 + m
N + 5ηmLQ
2
24ηgN
 12eη2NLσ2
mS
.
(125)
Combining (124) and (125) gives rise to
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2]
≤2N
ηcQm
N
X
i=1
ωi(E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr+1
i,lc , θr+1, νr+1
i
)])
+
4N
ηcQm
 
1 −4eηND0
m
 
(P r −P r+1) −D1Ψr −D2Ξr −D3Φr
+ ηcLF (ηcLF + 1)Qσ2
S
+ Γ2
N
X
i=1
ωi
2Nβi(1 −βi)(βi(eL −µ) + eL)
ηc eQr
i m
+
eL2Γ2
ηcQL
N
X
i=1
ωiβ2
i + eη2N 2σ2
ηcQm2S +
 259
64 + m
N + 5ηmLQ
2
24ηgN
 48eη2N 2Lσ2
ηcQm2S
 
1 −4eηND0
m
 
,
(126)
where
D1 ≜3eηNL2
ηcQm
 
1 −4eηND0
m
 
−eηNL2
ηcQm −L2
F −4eη2N 2L2D0
ηcQm2
,
(127)
D2 ≜4eη2NL
ηcQm
 3N
m + 12
  
1 −4eηND0
m
 
−8eη2N 2D0
ηcQm2 ,
(128)
D3 ≜
  1
ηv
+ µ
2 + 31mL
32N
 
1 −4eηND0
m
 
−3L
2 −2(1 −β)mL
N
−eηL2
2
−2eη2NL2D0
m
  2N
ηcQm.
(129)
We then proceed to prove that D1, D2 and D3 all are nonnegative. First, as eη ≤
m
48ND0 , it suffices to
have
D1 ≥11eηNL2
4ηcQm −eηNL2
ηcQm −L2
F −
eηNL2
12ηcQm
= 5eηNL2
3ηcQm −L2
F
≥2L2
F
3 ,
(130)
where (130) follows because ηc ≤
eηNL2
QmL2
F . Second, for D2, it suffices to have
D2 ≥11eη2NL
3ηcQm
 3N
m + 12
 
−8eη2N 2D0
ηcQm2


<!-- page 39 -->
39
= eη2N 2
ηcQm2
 
11L + 44m
N
−8D0
 
≥0,
(131)
where (131) follows because 11L + 44m
N −8D0 ≥0. Lastly, for D3, it suffices to have
D3 ≥
  1
ηv
+ µ
2 + 11 × 31mL
12 × 32N −7L
2 −eηL2
2
−eηL2
24
  2N
ηcQm
≥
  1
ηv
+ µ
2 −7L
2 −13eηL2
24
  2N
ηcQm
≥
  1
ηv
+ µ
2 −7L
2 −13eηL2
24
  2N
ηcQm
≥
 L
2 + µ
2 −L
60
  2N
ηcQm
≥46L2 + 48µL,
(132)
where (132) follows because ηv ≤
1
4L, ηc ≤eηN
Qm and eη ≤
m
48NL. Therefore, we have from (126) that
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2]
≤2N
ηcQm
N
X
i=1
ωi(E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr+1
i,lc , θr+1, νr+1
i
)]) +
11N
3ηcQm(P r −P r+1)
−46L2Φr + ηcLF (ηcLF + 1)Qσ2
S
+
eL2Γ2
ηcQL
N
X
i=1
ωiβ2
i + eη2N 2σ2
ηcQm2S
+ Γ2
N
X
i=1
ωi
2Nβi(1 −βi)(βi(eL −µ) + eL)
ηc eQr
i m
+
 259
64 + m
N + 5ηmLQ
2
24ηgN
 44eη2N 2Lσ2
ηcQm2S
,
(133)
where (133) follows because eη ≤
m
48ND0 . Furthermore, note that
E[∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2] ≤3L2E[∥eνr+1
i
−νr
i ∥2] + 3
2E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2].
(134)
As a result, we obtain
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2]
≤3N
ηcQm
N
X
i=1
ωi(E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr+1
i,lc , θr+1, νr+1
i
)])
+
11N
2ηcQm(P r −P r+1) + 3ηcLF (ηcLF + 1)Qσ2
2S
+ 3eL2Γ2
2ηcQL
N
X
i=1
ωiβ2
i
+ 3eη2N 2σ2
2ηcQm2S +
3Γ2
ηcQm
N
X
i=1
ωiβi(1 −βi)(βi(eL −µ) + eL)
+
 259
64 + m
N + 5ηmLQ
2
24ηgN
 66eη2N 2Lσ2
ηcQm2S
.
(135)


<!-- page 40 -->
40
Summing (135) up from r = 0 to R −1 and then dividing it by R yields
1
R
R−1
X
r=0
N
X
i=1
ωiE[∥∇θi,lcFi(θr
i,lc, θr, νr
i )∥2]
≤
3N
ηcQmR(f(θ0, ν0
i ) −f) +
11N
2ηcQmR(P 0 −f) + 3ηcLF (ηcLF + 1)Qσ2
2S
+ 3eL2Γ2
2ηcQL
N
X
i=1
ωiβ2
i
+ 3eη2N 2σ2
2ηcQm2S + 3NΓ2
ηcQm
N
X
i=1
ωiβi(1 −βi)(βi(eL −µ) + eL) +
 259
64 + m
N + 5ηmLQ
2
24ηgN
 66eη2N 2Lσ2
ηcQm2S
.
(136)
■
G PROOF OF LEMMA 5
First, since Fi(θi,lc, θi, νi) is strongly convex with modulus µ, we have
E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr
i,lc, θr, νr+1
i
)]
=m
N (E[Fi(θr
i,lc, θr, νr
i )] −E[Fi(θr
i,lc, θr, eνr+1
i
)])
≥m
N E[⟨∇νFi(θr
i,lc, θr, eνr+1
i
), νi −eνr+1
i
⟩] + µm
2N E[∥eνr+1
i
−νr
i ∥2].
(137)
Note that
∇νFi(θr
i,lc, θr, eνr+1
i
) = ∇νLi(βih(θr
i,lc) + (1 −βi)h(θr), eνr+1
i
),
∇νfi(θr, eνr+1
i
) = ∇νLi(h(θr), eνr+1
i
),
(138)
As a result, we have
E[⟨∇νFi(θr
i,lc, θr, eνr+1
i
), νi −eνr+1
i
⟩]
= E[⟨∇νFi(θr
i,lc, θr, eνr+1
i
) −∇νFi(θr
i,lc, θr, νr
i ), νi −eνr+1
i
⟩]
+ E[⟨∇νFi(θr
i,lc, θr, νr
i ) −∇νfi(θr, νr
i ), νi −eνr+1
i
⟩] + E[⟨∇νfi(θr, νr
i ), νi −eνr+1
i
⟩]
≥−E[∥∇νFi(θr
i,lc, θr, eνr+1
i
) −∇νFi(θr
i,lc, θr, νr
i )∥∥eνr+1
i
−νi∥]
−E[∥∇νFi(θr
i,lc, θr, νr
i ) −∇νfi(θr, νr
i )∥∥eνr+1
i
−νi∥] + E[⟨∇νfi(θr, νr
i ), νi −eνr+1
i
⟩]
(139)
≥
  1
ηv
−L
 
E[∥eνr+1
i
−νi∥2] −eLβiE[∥h(θr
i,lc) −h(θr
i )∥∥eνr+1
i
−νr
i ∥]
(140)
≥
  1
ηv
−3L
2
 
E[∥eνr+1
i
−νi∥2] −
eL2β2
i Γ2
2L
,
(141)
where (139) follows by the Cauchy-Schwarz inequality; (139) follows by Assumption 3; (141) holds by
the Young’s inequality. Substituting (141) into (138) yields
E[Fi(θr
i,lc, θr, νr+1
i
)] −E[Fi(θr
i,lc, θr, νr
i )] ≤−
  1
ηv
+ µ
2 −3L
2
 m
N E[∥eνr+1
i
−νi∥2] + meL2β2
i Γ2
2NL
.
(142)
Second, according to Assumption 3, we have
E[Fi(θr+1
i,lc , θr+1, νr+1
i
)] −E[Fi(θr
i,lc, θr, νr+1
i
)]


<!-- page 41 -->
41
≤E[⟨∇θi,lcFi(θr
i,lc, θr, νr+1
i
), θr+1
i,lc −θr
i,lc⟩] + LF
2 E[∥θr+1
i,lc −θr
i,lc∥2]
+ E[⟨∇θFi(θr
i,lc, θr, νr+1
i
), θr+1 −θr⟩] + LF
2 E[∥θr+1 −θr∥2]
=m
N E[⟨∇θi,lcFi(θr
i,lc, θr, eνr+1
i
), eθr+1
i,lc −θr
i,lc⟩] + E[⟨∇θFi(θr
i,lc, θr, νr+1
i
), θr+1 −θr⟩]
+ mLF
2N E[∥eθr+1
i,lc −θr
i,lc∥2] + LF
2 E[∥θr+1 −θr∥2].
(143)
We proceed to bound the terms in the RHS of (143) with the following Lemma.
Lemma 6 For any round r, it holds that
E[∥eθr+1
i,lc −θr
i,lc∥2] ≤η2
c( eQr
i )2E
 
Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ η2
c( eQr
i )2σ2
S
,
(144)
E[⟨∇θi,lcFi(θr
i,lc, θr, eνr+1
i
), eθr+1
i,lc −θr
i,lc⟩] ≤−ηc eQr
i
2
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] + η3
cL2
F ( eQr
i )2σ2
2S
−ηc eQr
i (1 −η2
c( eQr
i )2L2
F )
2
E
 
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ ηc eQr
i L2
F
2
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i
−θr∥2
 
,
(145)
E[⟨∇θFi(θr
i,lc, θr, νr+1
i
), θr+1 −θr⟩] ≤β2
i (1 −βi)(eL −µ)Γ2 + (1 −βi)E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩]
+ 2(1 −βi)mL
N
E[∥eνr+1
i
−νr
i ∥2] + βi(1 −βi)eLΓ2
+ (1 −βi)(4βieLLhΓ + L)
8
E[∥θr+1 −θr∥2].
(146)
Applying Lemma 6 to (143) yields
E[Fi(θr+1
i,lc , θr+1, νr+1
i
)] −E[Fi(θr
i,lc, θr, νr+1
i
)]
≤−ηc eQr
i m
2N
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] + η2
cLF (ηcLF + 1)( eQr
i )2mσ2
2NS
−ηc eQr
i m(1 −ηc eQr
i LF −η2
c( eQr
i )2L2
F )
2N
E
 
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ ηc eQr
i mL2
F
2N
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i
−θr∥2
 
+ 2(1 −βi)mL
N
E[∥eνr+1
i
−νr
i ∥2]
+ (1 −βi)E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩] + (1 −βi)(4βieLLhΓ + L) + 4LF
8
E[∥θr+1 −θr∥2]
+ β2
i (1 −βi)(eL −µ)Γ2 + βi(1 −βi)eLΓ2
≤−ηc eQr
i m
2N
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] + η2
cLF (ηcLF + 1)( eQr
i )2mσ2
2NS
+ ηc eQr
i mL2
F
2N
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i
−θr∥2
 
+ 2(1 −βi)mL
N
E[∥eνr+1
i
−νr
i ∥2]


<!-- page 42 -->
42
+ (1 −βi)E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩] + (1 −βi)(4βieLLhΓ + 5L)
8
E[∥θr+1 −θr∥2]
+ β2
i (1 −βi)(eL −µ)Γ2 + βi(1 −βi)eLΓ2,
(147)
where (147) follows because ηcQLF ≤1
2, and LF ≤(1 −βi)L. We combine (142) and (147) to obtain
E[Fi(θr+1
i,lc , θr+1, νr+1
i
)] −E[Fi(θr
i,lc, θr, νr
i )]
≤−ηc eQr
i m
2N
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] + η2
cLF (ηcLF + 1)( eQr
i )2mσ2
2NS
+ ηc eQr
i mL2
F
2N
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i
−θr∥2
 
+ (1 −βi)E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩]
+ (1 −βi)(4βieLLhΓ + 5L)
8
E[∥θr+1 −θr∥2] + β2
i (1 −βi)(eL −µ)Γ2 + βi(1 −βi)eLΓ2 + meL2β2
i Γ2
2NL
−
  1
ηv
+ µ
2 −3L
2 −2(1 −βi)mL
N
 
E[∥eνr+1
i
−νi∥2].
(148)
■
H PROOF OF LEMMA 6
First, by the definition of eθr+1
i,lc , we have
E[eθr+1
i,lc −θr
i,lc] = −ηc eQr
i E
  Qr
i −1
X
t−0
(br
i )t
eQr
i
Gi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)
 
(149)
= −ηc eQr
i E
  Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)
 
,
(150)
where (149) holds by following the spirit as (21). Then, we get
E[⟨∇θi,lcFi(θr
i,lc, θr, eνr+1
i
), eθr+1
i,lc −θr
i,lc⟩]
= −ηc eQr
i E
  
∇θi,lcFi(θr
i,lc, θr, eνr+1
i
),
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)
  
= −ηc eQr
i
2
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] −ηc eQr
i
2
E
 
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ ηc eQr
i
2
E
 ∇θi,lcFi(θr
i,lc, θr, eνr+1
i
) −
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
(151)
≤−ηc eQr
i
2
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] −ηc eQr
i
2
E
 
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ ηc eQr
i
2
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
) −∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)∥2
 
(152)
≤−ηc eQr
i
2
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] −ηc eQr
i
2
E
 
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 


<!-- page 43 -->
43
+ ηcL2
F eQr
i
2
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
(∥eθr,t
i,lc −θr
i,lc∥2 + ∥eθr,t
i
−θr∥2)
 
,
(153)
where (151) follows because ⟨v1, v2⟩= 1
2∥v1∥2 + 1
2∥v2∥2 −1
2∥v1 −v2∥2, ∀v1, v2 ∈Rn; (152) follows
by the convexity of ∥· ∥2; (153) holds by Assumption 3. Furthermore, note that
E[∥eθr+1
i,lc −θr
i,lc∥2] = η2
c( eQr
i )2E
 
Qr
i −1
X
t=0
(br
i )t
eQr
i
Gi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
≤η2
c( eQr
i )2E
 
Qr
i −1
X
t=0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ η2
c( eQr
i )2σ2
S
,
(154)
where (154) is obtained by following the same spirit as (113). Thus, we substitute (154) into (153) to
have
E[⟨∇θi,lcFi(θr
i,lc, θr, eνr+1
i
), eθr+1
i,lc −θr
i,lc⟩]
≤−ηc eQr
i
2
E[∥∇θi,lcFi(θr
i,lc, θr, eνr+1
i
)∥2] + η3
cL2
F ( eQr
i )2σ2
2S
−ηc eQr
i (1 −η2
c( eQr
i )2L2
F )
2
E
 
Qr
i −1
X
t−0
(br
i )t
eQr
i
∇θi,lcFi(eθr,t
i,lc, eθr,t
i , eνr+1
i
)

2 
+ ηcL2
F eQr
i
2
E
  Qr
i −1
X
t=0
(br
i )t
eQr
i
∥eθr,t
i
−θr∥2
 
.
(155)
Second, by the definition of Fi and Li, we have
∇θFi(θr
i,lc, θr, νr+1
i
) = (1 −βi)∇h(θr)∇Li(βih(θr
i,lc) + (1 −βi)h(θr), νr+1
i
),
(156)
∇θfi(θr, νr+1
i
) = ∇h(θr)∇Li(h(θr), νr+1
i
).
(157)
Then, we obtain
E[⟨∇θFi(θr
i,lc, θr, νr+1
i
), θr+1 −θr⟩]
=E[⟨∇θFi(θr
i,lc, θr, νr+1
i
) −(1 −βi)∇θfi(θr, νr+1
i
), θr+1 −θr⟩] + (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
=(1 −βi)E[⟨∇Li(βih(θr
i,lc) + (1 −βi)h(θr), νr+1
i
) −∇Li(h(θr, νr+1
i
), ∇h(θr)⊤(θr+1 −θr)⟩]
+ (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
=(1 −βi)E[⟨∇Li(zr
i , νr+1
i
) −∇Li(h(θr), νr+1
i
), ∇h(θr)⊤(θr+1 −θr)⟩]
+ (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
(158)
=(1 −βi)E[⟨∇Li(zr
i , νr+1
i
) −∇Li(h(θr), νr+1
i
), −(zr
i −h(θr)) + (zr
i −h(θr)) + ∇h(θr)⊤(θr+1 −θr)⟩]
+ (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
≤−(1 −βi)µE[∥zr
i −h(θr)∥2] + (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
+ (1 −βi)E[⟨∇Li(zr
i , νr+1
i
) −∇Li(h(θr), νr+1
i
), zr
i −h(θr) + ∇h(θr)⊤(θr+1 −θr)⟩]
(159)
≤−(1 −βi)µE[∥zr
i −h(θr)∥2] + (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]


<!-- page 44 -->
44
+ (1 −βi)eLE[∥zr
i −h(θr)∥∥zr
i −h(θr) + ∇h(θr)⊤(θr+1 −θr)∥],
(160)
where zr
i ≜βih(θr
i,lc) + (1 −βi)h(θr); (159) follows by the strong convexity of Li(·, νi), i.e.,
⟨∇Li(zr
i , νr+1
i
) −∇Li(h(θr), νr+1
i
), zr
i −h(θr)⟩≥µ∥zr
i −h(θr)∥2;
(161)
(160) follows by the Cauchy-Schwarz inequality and Assumption 3. We proceed to bound ∥zr
i −h(θr) +
∇h(θr)⊤(θr+1 −θr)∥by
∥zr
i −h(θr) + ∇h(θr)⊤(θr+1 −θr)∥
= ∥βi(h(θr
i,lc) −h(θr)) + ∇h(θr)⊤(θr+1 −θr)∥
≤βi∥h(θr
i,lc) −h(θr)∥+ ∥∇h(θr)⊤(θr+1 −θr)∥
≤βi∥h(θr
i,lc) −h(θr)∥+
h(θr) −h(θr+1) + Lh
2 ∥θr+1 −θr∥2

(162)
≤βi∥h(θr
i,lc) −h(θr)∥+ ∥h(θr) −h(θr+1)∥+ Lh
2 ∥θr+1 −θr∥2,
(163)
where (162) follows by Assumption 3. Substituting (163) into (160) gives rise to
E[⟨∇θFi(θr
i,lc, θr, νr+1
i
), θr+1 −θr⟩]
≤β2
i (1 −βi)(eL −µ)E[∥h(θi,lc) −h(θr)∥2] + (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
+ βi(1 −βi)eLE[∥h(θi,lc) −h(θr)∥∥h(θr) −h(θr+1)∥]
+ βi(1 −βi)eLLh
2
E[∥h(θi,lc) −h(θr)∥∥θr+1 −θr∥2]
≤β2
i (1 −βi)(eL −µ)E[∥h(θi,lc) −h(θr)∥2] + (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩]
+ βi(1 −βi)eLE[∥h(θi,lc) −h(θr)∥∥h(θr) −h(θr+1)∥]
+ βi(1 −βi)eLLh
2
E[∥h(θi,lc) −h(θr)∥∥θr+1 −θr∥2]
≤β2
i (1 −βi)(eL −µ)Γ2 + (1 −βi)E[⟨∇θfi(θr, νr+1
i
), θr+1 −θr⟩] + βi(1 −βi)eLΓ2
+ βi(1 −βi)eLLhΓ
2
E[∥θr+1 −θr∥2]
(164)
= β2
i (1−βi)(eL−µ)Γ2+(1−βi)E[⟨∇θfi(θr, νr
i ), θr+1−θr⟩]+(1−βi)E[⟨∇θfi(θr, νr+1
i
) −∇θfi(θr, νr
i ), θr+1−θr⟩]
+ βi(1 −βi)eLΓ2 + βi(1 −βi)eLLhΓ
2
E[∥θr+1 −θr∥2]
≤β2
i (1 −βi)(eL −µ)Γ2 + (1 −βi)E[⟨∇θfi(θr, νr
i ), θr+1 −θr⟩] + 2(1 −βi)mL
N
E[∥eνr+1
i
−νr
i ∥2]
+ βi(1 −βi)eLΓ2 + (1 −βi)(4βieLLhΓ + L)
8
E[∥θr+1 −θr∥2],
(165)
where (164) follows by Assumption 5; (165) holds by the Cauchy-Schwarz inequality, i,e., ⟨d1, d2⟩≤
2
L∥d1∥2 + L
8 ∥d2∥2, ∀d1, d2, and Prob(i ∈Ar) = m
N .
■


<!-- page 45 -->
45
REFERENCES
[1] F. Tariq, M. R. A. Khandaker, K.-K. Wong, M. A. Imran, M. Bennis, and M. Debbah, “A speculative study on 6G,” IEEE
Wireless Commun., vol. 27, no. 4, pp. 118–125, Aug. 2020.
[2] W. Tong and G. Y. Li, “Nine challenges in artificial intelligence and wireless communications for 6G,” IEEE Wireless
Commun., pp. 1–10, 10.1109/MWC.006.2100543 2022.
[3] K. B. Letaief, W. Chen, Y. Shi, J. Zhang, and Y.-J. A. Zhang, “The roadmap to 6G: AI empowered wireless networks,”
IEEE Commun. Mag., vol. 57, no. 8, pp. 84–90, Aug. 2019.
[4] D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, and H. Vincent Poor, “Federated learning for Internet of
things: A comprehensive survey,” IEEE Commun. Surv. Tutorials, vol. 23, no. 3, pp. 1622–1658, Apr. 2021.
[5] L. U. Khan, W. Saad, Z. Han, E. Hossain, and C. S. Hong, “Federated learning for Internet of things: Recent advances,
taxonomy, and open challenges,” IEEE Commun. Surv. Tutorials, vol. 23, no. 3, pp. 1759–1799, Jun. 2021.
[6] H. H. Yang, Z. Liu, T. Q. S. Quek, and H. V. Poor, “Scheduling policies for federated learning in wireless networks,” IEEE
Trans. Commun., vol. 68, no. 1, pp. 317–333, Jan. 2020.
[7] M. Chen, Z. Yang, W. Saad, C. Yin, H. V. Poor, and S. Cui, “A joint learning and communications framework for federated
learning over wireless networks,” IEEE Trans. Wireless Commun., vol. 20, no. 1, pp. 269–283, Jan. 2021.
[8] G. Zhu, Y. Wang, and K. Huang, “Broadband analog aggregation for low-latency federated edge learning,” IEEE Trans.
Wireless Commun., vol. 19, no. 1, pp. 491–506, Oct. 2020.
[9] A. M. Elbir and S. Coleri, “Federated learning for hybrid beamforming in mm-wave massive MIMO,” IEEE Commun.
Lett., vol. 24, no. 12, pp. 2795–2799, Dec. 2020.
[10] X. Zheng and V. Lau, “Federated online deep learning for csit and csir estimation of fdd multi-user massive mimo systems,”
IEEE Trans. Signal Process., vol. 70, pp. 2253–2266, Apr. 2022.
[11] A. M. Elbir and S. Coleri, “Federated learning for channel estimation in conventional and RIS-assisted massive MIMO,”
IEEE Trans. Wireless Commun., vol. 21, no. 6, pp. 4255–4268, Jun. 2022.
[12] X. Li, K. Huang, W. Yang, S. Wang, and Z. Zhang, “On the convergence of FedAvg on non-iid data,” in Proc. ICLR, Addis
Ababa, ETHIOPIA, Apr. 26 - May 1, 2020, pp. 1–11.
[13] D. Shi, L. Li, M. Wu, M. Shu, R. Yu, M. Pan, and Z. Han, “To talk or to work: Dynamic batch sizes assisted time efficient
federated learning over future mobile edge devices,” IEEE Trans. Wireless Commun., vol. 21, no. 12, pp. 11 038–11 050,
Dec. 2022.
[14] J. Mills, J. Hu, and G. Min, “Multi-task federated learning for personalised deep neural networks in edge computing,” IEEE
Trans. Parallel Distrib. Syst., vol. 33, no. 3, pp. 630–641, Mar. 2022.
[15] Z. Zhao, C. Feng, W. Hong, J. Jiang, C. Jia, T. Q. S. Quek, and M. Peng, “Federated learning with Non-IID data in wireless
networks,” IEEE Trans. Wireless Commun., vol. 21, no. 3, pp. 1927–1942, Mar. 2022.
[16] C. Zhang, Y. Zhu, C. Markos, S. Yu, and J. J. Yu, “Towards crowdsourced transportation mode identification: A semi-
supervised federated learning approach,” IEEE Internet Things J., pp. 1–1, 2021.
[17] Z. Zhang, S. Ma, Z. Yang, Z. Xiong, J. Kang, Y. Wu, K. Zhang, and D. Niyato, “Robust semi-supervised federated learning
for images automatic recognition in internet of drones,” IEEE Internet Things J., pp. 1–1, 2022.
[18] Z. Wang, X. Wang, R. Sun, and T.-H. Chang, “Federated semi-supervised learning with class distribution mismatch,” 2021.
[Online]. Available: https://arxiv.org/abs/2111.00010
[19] Q. Ma, Y. Xu, H. Xu, Z. Jiang, L. Huang, and H. Huang, “FedSA: A semi-asynchronous federated learning mechanism in
heterogeneous edge computing,” IEEE J. Sel. Areas Commun., vol. 39, no. 12, pp. 3654–3672, Otc. 2021.


<!-- page 46 -->
46
[20] Z. Zhao, C. Feng, W. Hong, J. Jiang, C. Jia, T. Q. S. Quek, and M. Peng, “Federated learning with Non-IID data in wireless
networks,” IEEE Trans. Wireless Commun., vol. 21, no. 3, pp. 1927–1942, Sep. 2022.
[21] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, “Scaffold: Stochastic controlled averaging for
federated learning,” in Proc. ICML, Jul. 13-18 2020, pp. 5132–5143.
[22] S. Wang and T.-H. Chang, “Federated matrix factorization: Algorithm design and application to data clustering,” IEEE
Trans. Signal Process., vol. 70, pp. 1625–1640, Feb. 2022.
[23] Y. Deng, M. M. Kamani, and M. Mahdavi, “Adaptive personalized federated learning,” arXiv preprint arXiv:2003.13461,
2020.
[24] A. Fallah, A. Mokhtari, and A. Ozdaglar, “Personalized federated learning with theoretical guarantees: A model-agnostic
meta-learning approach,” in Proc. NeuIPS, Vancouver, Canada, Dec. 6-12 2020, pp. 1–6.
[25] Y. Huang, L. Chu, Z. Zhou, L.Wang, J. Liu, J. Pei, and Y. Zhang, “Personalized cross-silo federated learning on non-iid
data,” in Proc. AAAI, Dec. 6-12 2021, pp. 7865–7873.
[26] Y. Wang, Y. Xu, Q. Shi, and T.-H. Chang, “Quantized federated learning under transmission delay and outage constraints,”
IEEE J. Sel. Areas Commun., vol. 40, no. 1, pp. 323–341, Jan. 2022.
[27] C. Xie, S. Koyejo, and I. Gupta, “Asynchronous federated optimization,” 2020. [Online]. Available: https:
//arxiv.org/abs/1903.03934
[28] S. U. Stich, “Local SGD converges fast and communicates little,” in Proc. ICLR, New Orleans, LA, USA, May 6 - May 9
2019, pp. 1–5.
[29] A. Reisizadeh, I. Tziotis, H. Hassani, A. Mokhtari, and R. Pedarsani, “Straggler-resilient federated learning: Leveraging the
interplay between statistical accuracy and system heterogeneity,” IEEE Journal on Selected Areas in Information Theory,
vol. 3, No. 2, pp. 197–205, June 2022.
[30] H. Lin, J. Lou, L. Xiong, and C. Shahabi, “SemiFed: Semi-supervised federated learning with consistency and pseudo-
labeling,” arXiv preprint arXiv:2108.09412, 2021.
[31] B. Zhang, Y. Wang, W. Hou, H. Wu, J. Wang, M. Okumura, and T. Shinozaki, “Flexmatch: Boosting semi-supervised
learning with curriculum pseudo labeling,” in Proc. NeuIPS, California, USA, Dec. 6-14 2021, pp. 1–12.
[32] W. Jeong, J. Yoon, E. Yang, and S. J. Hwang, “Federated semi-supervised learning with inter-client consistency & disjoint
learning,” in Proc. ICLR, May 3-7 2021, pp. 1–6.
[33] A. Albaseer, M. Abdallah, A. Al-Fuqaha, A. Erbad, and O. A. Dobre, “Semi-supervised federated learning over heterogeneous
wireless iot edge networks: Framework and algorithms,” 2022. [Online]. Available: https://doi.org/10.36227/techrxiv.19317632
[34] Y. Zhu, Y. Liu, J. J. Q. Yu, and X. Yuan, “Semi-supervised federated learning for travel mode identification from gps
trajectories,” IIEEE Trans. Intell. Transp. Syst., vol. 23, no. 3, pp. 2380–2391, 2022.
[35] T. Li, A. K. Sahu, M. Sanjabi, M. Zaheer, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,”
in Proc. MLSys, Austin, TX, USA, Mar. 2-4, 2020, pp. 1–12.
[36] D. A. E. Acar, Y. Zhao, R. Matas, M. Mattina, P. Whatmough, and V. Saligrama, “Federated learning based on dynamic
regularization,” in Proc. ICLR, May 3-7 2021, pp. 1–6.
[37] T. Li, S. Hu, A. Beirami, and V. Smith, “Ditto: Fair and robust federated learning through personalization,” in Proc. ICML,
Virtual Conference, Jul. 18-24 2021, pp. 1–10.
[38] C. T. Dinh, N. H. Tran, and T. D. Nguyen, “Personalized federated learning with moreau envelopes,” in Proc. NeuIPS,
Vancouver, Canada, Dec. 6-12 2020, pp. 1–12.
[39] Y. Deng, M. M. Kamani, and M. Mahdavi, “Distributionally robust federated averaging,” in Proc. NeuIPS, Vancouver,
Canada, Dec. 6-12 2020, p. 15111–15122.


<!-- page 47 -->
47
[40] B. Wu, Z. Liang, Y. Han, Y. Bian, P. Zhao, and J. Huang, “Drflm: Distributionally robust federated learning with inter-client
noise via local mixup,” arXiv preprint arXiv:2204.07742, 2022.
[41] M. Mohri, G. Sivek, and A. T. Suresh, “Agnostic federated learning,” in Proc. ICML, Long Beach, CA, USA, Jun. 9-15,
2019, pp. 4615–4625.
[42] M. Zecchin, M. Kountouris, and D. Gesbert, “Communication-efficient distributionally robust decentralized learning,” arXiv
preprint arXiv:2205.15614, 2022.
[43] C. Bettini, G. Civitarese, and R. Presotto, “Personalized semi-supervised federated learning for human activity recognition,”
arXiv preprint arXiv:2104.08094, 2021.
[44] H. Yu, Z. Chen, X. Zhang, X. Chen, F. Zhuang, H. Xiong, and X. Cheng, “FedHAR: Semi-supervised online learning for
personalized federated human activity recognition,” IEEE Trans. Mob. Comput., pp. 1–1, Dec. 2021.
[45] A. Tashakori, W. Zhang, Z. J. Wang, and P. Servati, “SemiPFL: Personalized semi-supervised federated learning framework
for edge intelligence,” arXiv preprint arXiv:2203.08176, 2022.
[46] S. Wang and T.-H. Chang, “Demystifying model averaging for communication-efficient federated matrix factorization,” in
Proc. IEEE ICASSP, Toronto, Ontario, Canada, June 6-11, 2021, pp. 1–5.
[47] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Areas, “Communication-efficient learning of deep networks
from decentralized data,” in Proc. ICML, Sydney, Australia, Aug. 6-11, 2017, pp. 1–10.
[48] M. Kamp, L. Adilova, J. Sicking, F. Huger, P. Schlicht, T. Wirtz, and S. Wrobel, “Efficient decentralized deep learning by
dynamic model averaging,” in Proc. ECML PKDD, Dublin, Ireland, Sept. 10-14, 2018, pp. 393–409.
[49] H. Yu, S. Yang, and S. Zhu, “Parallel restarted SGD with faster convergence and less communication: Demystifying why
model averaging works,” in Proc. AAAI, Honolulu, Hawaii, USA, Jan. 27-Feb. 1, 2019, pp. 5693–5700.
[50] Y. Wang, Y. Xu, Q. Shi, and T.-H. Chang, “Quantized federated learning under transmission delay and outage constraints,”
IEEE Journal on Selected Areas in Communications, vol. 40, pp. 323 – 341, Jan. 2022.
[51] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “Tackling the objective inconsistency problem in heterogeneous
federated optimization,” arXiv preprint arXiv:2007.07481, 2020.
[52] F. Sattler, K.-R. M¨uller, and W. Samek, “Clustered federated learning: Model-agnostic distributed multitask optimization
under privacy constraints,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, pp. 3710–3722, 08 2020.
[53] M. Mestoukirdi, M. Zecchin, D. Gesbert, Q. Li, and N. Gresset, “User-centric federated learning,” in WCDI 2021,
GLOBECOM Workshop on Wireless Communication for Distributed Intelligence, Madrid, Spain, 2021.
[54] Y. LeCun, C. Cortes, and C. Burges, “The mnist database.” [Online]. Available: http://yann.lecun.com/exdb/mnist/
[55] A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” Master’s thesis, Department of
Computer Science, University of Toronto, 2009.