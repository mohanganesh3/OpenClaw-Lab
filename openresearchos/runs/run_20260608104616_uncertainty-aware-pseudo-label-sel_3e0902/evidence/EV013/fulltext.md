<!-- page 1 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for
Label-Limited Dynamic Graph
Shengtao Zhang∗
zhangst@stu.xjtu.edu.cn
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Haokai Zhang∗
zhanghaokai@stu.xjtu.edu.cn
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Shiqi Lou†
shiqilou01@gmail.com
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Zicheng Wang†
cicadidae@stu.xjtu.edu.cn
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Zinan Zeng
2194214554@stu.xjtu.edu.cn
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Yilin Wang
13148035071xjtu@stu.xjtu.edu.cn
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Minnan Luo‡
minnluo@xjtu.edu.cn
Xi’an Jiaotong University
Xi’an, Shaanxi, China
Abstract
Dynamic node classification is critical for modeling evolving sys-
tems like financial transactions and academic collaborations. In
such systems, dynamically capturing node information changes
is critical for dynamic node classification, which usually requires
all labels at every timestamp. However, it is difficult to collect all
dynamic labels in real-world scenarios due to high annotation costs
and label uncertainty (e.g., ambiguous or delayed labels in fraud
detection). In contrast, final timestamp labels are easier to obtain
as they rely on complete temporal patterns and are usually main-
tained as a unique label for each user in many open platforms,
without tracking the history data. To bridge this gap, we propose
PTCL (Pseudo-label Temporal Curriculum Learning), a pioneer-
ing method addressing label-limited dynamic node classification
where only final labels are available. PTCL introduces: (1) a tem-
poral decoupling architecture separating the backbone (learning
time-aware representations) and decoder (strictly aligned with final
labels), which generate pseudo-labels, and (2) a Temporal Curricu-
lum Learning strategy that prioritizes pseudo-labels closer to the
final timestamp by assigning them higher weights using an expo-
nentially decaying function. We contribute a new academic dataset
(CoOAG), capturing long-range research interest in dynamic graph.
Experiments across real-world scenarios demonstrate PTCL’s con-
sistent superiority over other methods adapted to this task. Beyond
methodology, we propose a unified framework FLiD (Framework
for Label-Limited Dynamic Node Classification), consisting of a
complete preparation workflow, training pipeline, and evaluation
standards, and supporting various models and datasets. The code
can be found at https://github.com/3205914485/FLiD.
∗Both authors contributed equally to this research.
†Both authors contributed equally to this research.
‡Corresponding author: Minnan Luo, School of Computer Science and Technology,
Xi’an Jiaotong University, Xi’an 710049, China.
Keywords
Dynamic Graph, Node classification, Pseudo-label, Curriculum
Learning
1
Introduction
Graph-structured data has become increasingly prevalent in various
domains, ranging from social networks [14, 40, 65] and biological
systems [34, 73] to financial transactions [24, 33] and academic
collaborations [23, 70]. Graphs provide a natural way to represent
relationships and interactions between entities, making them a
powerful tool for modeling complex systems [5]. In particular, node
classification, a fundamental task in graph analysis, aims to assign
labels to nodes based on their features and the structure of the graph
[4, 47, 59]. This task has been extensively studied in static graphs,
where the graph structure and node features remain unchanged
over time.
Nowadays, many real-world graphs are inherently dynamic, with
nodes and edges evolving over time [29, 48, 61]. For instance, in
academic collaboration networks, authors’ research interests may
shift as they publish new papers, leading to changes in their labels
(e.g., from "computer vision" to "natural language processing") [25].
Similarly, in financial transaction networks, users’ behavior may
change over time, potentially transitioning from "normal" to "fraud-
ulent" activities [24]. These scenarios highlight the importance of
dynamic label node classification, where the goal is to classify nodes
in a dynamic graph whose labels may change over time.
Ideally, we can get an experienced classifier through training
with a highly dynamically changing trajectory. However, the chal-
lenge in dynamic graph node classification arises from the difficulty
of collecting dynamic node labels due to high annotation costs
(e.g., requiring continuous monitoring and manual effort) and label
uncertainty (e.g., ambiguous or delayed labels in fraud detection).
Although some dynamic datasets exist, they often provide weak
dynamic labels that rarely change [29], making it difficult to cap-
ture nodes’ dynamic evolution effectively. While collecting labels
at every timestamp is challenging, obtaining a single label at the
arXiv:2504.17641v2  [cs.LG]  25 Apr 2025


<!-- page 2 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Trovato et al.
NAME Jim
Status:
Unknown
Account
t0
t1
tN
TimeLine
NAME Jim
Status:
Fraud
Account
NAME Jim
Status:
Unknown
Account
Figure 1: A present of a financial system. The graph rep-
resents a dynamic financial system where nodes represent
entities such as users, payment cards, and financial institu-
tions, while edges represent transactional relationships. Over
time, user behavior is tracked through a sequence of transac-
tions, and some users’ labels(account status) may eventually
be identified as fraudulent.
final timestamp is relatively straightforward. As shown in Figure 1,
in a financial transaction network, it may be easier to determine
whether a user is fraudulent at the end of a given period rather than
monitoring their behavior at each time step [24]. And many open
platforms, such as OAG [51, 55, 67, 68], maintain a single final label
for each user, like assigning each author a fixed research interest
without tracking changes over time.
This observation aligns with the core challenge of our task, label-
limited dynamic node classification. The objective is to classify
nodes in dynamic graphs with limited label information, partic-
ularly focusing on the final timestamp labels, as these labels in-
herently capture the comprehensive status of nodes over time. To
effectively leverage the dynamic information from previous times-
tamps, which lack labels, it is crucial to develop a robust model to
represent and utilize these historical data.
One naive solution is to use the final timestamp labels as a sub-
stitute for the node’s label at previous timestamps. Specifically, we
can replicate the final labels across all previous timestamps, treating
the node’s label as static up to the final timestamp. However, this
approach fails to capture the true temporal evolution of node labels,
causing significant approximation errors.
The challenge associated with dynamic label collection naturally
lead us to consider semi-supervised learning (SSL) [7, 31, 56, 64,
71, 72] as a potential solution. A common SSL approach, pseudo-
labeling [26, 32], can directly generate temporary labels for unla-
beled data based on model predictions, and has been widely used
in graph learning [35–37, 54]. In the context of our task, pseudo-
labeling can be employed to generate labels of earlier timestamps,
enabling the model to capture the underlying structure and dynam-
ics of the graph while iteratively refining its predictions.
Building on the foundation of SSL and pseudo-labeling, we pro-
pose a novel method PTCL (Pseudo-label Temporal Curriculum
Learning). Our model includes a backbone and a decoder. The
backbone is designed to model the dynamic graph data, while the
decoder predicts node labels from backbone embeddings [48, 66].
To align the decoder with final timestamp labels while leveraging
earlier timestamps for capturing node label evolution, we draw
inspiration from prior works on the variational EM (Expectation
Maximization) framework [10, 39, 45, 46, 69], which separates the
optimization of model components to iteratively refine predictions.
Specifically, we train the decoder exclusively on the final timestamp
labels, to ensure the consistency with the only available labels. Once
the decoder is trained, it generates pseudo-labels for all timestamps,
which serve as approximations of the true labels at earlier time
steps. These pseudo-labels, along with the final timestamp labels,
are then used to train the backbone via a weighted loss function, al-
lowing the backbone to learn embeddings that capture the temporal
evolution of node labels.
To address the potential issue of low-quality pseudo-labels dur-
ing the initial stages of EM iteration, inspired by Curriculum
Learning [2, 53, 58], we design a Temporal Curriculum Learning
method to compute temporal weights for pseudo-labels. Based on
the easy-to-hard rule, we assign higher weights to pseudo-labels
near the final timestamps in early training, as they are easier to
predict. As training progresses, we gradually shift focus to earlier
timestamps, guiding the model toward more challenging examples.
Extensive experiments show that PTCL consistently outperforms
other methods adapted to this task across multiple datasets, validat-
ing its effectiveness in capturing the temporal evolution of nodes.
Additionally, we conduct a series of additional studies to verify the
contribution of each design of PTCL.
To sum up, our contributions are as follows:
• Pioneering Study: To the best of our knowledge, this is a pio-
neering study to systematically investigate the problem of label-
limited dynamic node classification. We formalize the task, iden-
tify its unique challenges, and propose a comprehensive method
to address them.
• Novel Method: We introduce a new method PTCL, which cap-
tures the dynamic nature of nodes with limited labels, advancing
dynamic graph study by constructing highly varying history
information. Various experiments demonstrate the effectiveness
of PTCL and the necessity of each design.
• New Dataset: We contribute a new dataset, CoOAG, which is
derived from academic collaboration networks and designed
for dynamic graph learning. It captures the dynamic nature of
research interests, providing a rich testbed for evaluating PTCL.
• Unified Framework: We propose a unified framework, FLiD
(Framework for Label-Limited Dynamic Node Classification),
for our task, which includes a complete preparation workflow,
a training pipeline, and evaluation protocols. Our framework
supports various backbones and datasets, offering a flexible and
extensible solution.
2
Related Work
2.1
Dynamic Node Classification
Dynamic graph learning is typically divided into discrete-time and
continuous-time approaches. Discrete-time methods segment dy-
namic graphs into snapshot sequences [12, 49], whereas continuous-
time methods capture fine-grained temporal interactions for better
real-world alignment [48, 57]. While link prediction dominates
existing research in downstream tasks [44, 66], dynamic node clas-
sification remains underexplored despite its practical significance.

[CAPTION] Figure 1: A present of a financial system. The graph rep-


<!-- page 3 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Existing works have explored various approaches to dynamic
node classification: JODIE [29] predicts user state changes in user-
item graphs, DynPPE [18] classifies nodes in large-scale dynamic
graphs using a discrete-time method, and OTGNet [13] handles
dynamically expanding class sets. However, these methods overlook
the underlying node dynamics and assume full dynamic labels,
which rarely hold in real-world scenarios. We address the critical yet
understudied problem of label-limited dynamic node classification,
where only final timestamp annotations are available. To capture
node dynamics effectively, we propose PTCL, a continuous-time
method leveraging pseudo-labels with Curriculum Learning.
2.2
Pseudo-Labeling
Pseudo-labeling [32] has emerged as a prominent and widely uti-
lized technique in semi-supervised learning, primarily aimed at
predicting labels for unlabeled data. This approach functions as a
form of entropy minimization, which encourages the reduction of
data embedding density near decision boundaries, thereby facilitat-
ing the formation of more robust boundaries in low-density regions
of the embedding space [8, 17, 43]. Its has been applied across
various domains [26], including computer vision [32, 62], graph
learning [35–37, 54], knowledge distillation [21], and adversarial
training [38, 60].
However, the efficiency of pseudo-labeling highly relies on the
accuracy of the pseudo-labels. Incorrect pseudo-labels can severely
impair model performance by introducing noise and misleading
patterns into the learning process [43]. A natural strategy to address
this issue is Curriculum Learning [2], which involves initially train-
ing the model on "easier" examples of a class or concept, followed by
progressively more challenging ones. This ensures that the model
first assigns correct pseudo-labels to the easier examples from the
unlabeled set, thereby reliably expanding the labeled dataset. Build-
ing on this foundation, numerous studies have proposed various
metrics to identify easier examples and more reliable pseudo-labels
[26]. For instance, Cascante-Bonilla et al. [6], He et al. [19], and Sun
et al. [54] employ a confidence score threshold, while Pei et al. [43]
and Song et al. [52] utilize the entropy of softmax trajectory over
previous epochs.
In the context of our research, existing methods overlook the
temporal dynamics inherent in the evolving nature of graphs. In
contrast, our proposed Temporal Curriculum Learning explicitly in-
tegrates temporal considerations, allowing the model to adapt more
effectively to shifts in data distribution and decision boundaries as
learning progresses.
2.3
Variational EM Framework
The variational EM framework [10, 39] is a widely used frame-
work for parameter estimation in probabilistic models with latent
variables. In the classical EM algorithm, the goal is to maximize
the likelihood of observed data by iteratively refining model pa-
rameters through alternating E-steps (expectation computation)
and M-steps (parameter maximization). GMNN [46] applies EM
for semi-supervised static graphs and GLEM [69] combines GNNs
with language models. Our contribution lies not in framework
innovation but in adapting this established paradigm to address
label-limited dynamic node classification.
3
Problem Formulation
A dynamic graph with dynamic labels can be mathematically
represented as a sequence of chronologically ordered events: G =
{𝑥(𝑡𝑖)} = {(𝑢𝑖, 𝑣𝑖,𝑡𝑖)}, where 0 ≤𝑡1 ≤𝑡2 ≤· · · . Each event 𝑥(𝑡𝑖)
describes an interaction between a source node 𝑢𝑖∈V and a
destination node 𝑣𝑖∈V at time 𝑡𝑖. And 𝑦𝑡𝑖𝑢𝑖,𝑦𝑡𝑖𝑣𝑖∈Y are their
respective labels at 𝑡𝑖. V denotes the set of all nodes, and Y is
the class set of all nodes. For each node 𝑢∈V, T𝑢= {𝑡𝑖| 𝑢=
𝑢𝑖or 𝑢= 𝑣𝑖in 𝑥(𝑡𝑖) ∈G} is the set of all timestamps at which 𝑢
participates in any event in G. The last occurrence time𝑇𝑢= max T𝑢
is the most recent timestamp in T𝑢. We further define Y𝐹= {𝑦𝑇𝑢
𝑢
|
𝑢∈V}, the set of ground-truth labels at the final timestamps,
and Y𝐸= {𝑦𝑡𝑢| 𝑢∈V,𝑡∈T𝑢\ {𝑇𝑢}}, the set of labels at non-
last timestamps of every node. In most cases, |Y𝐸| ≫|Y𝐹|. In
our research scenario, Y𝐹are known, whereas Y𝐸are considered
unknown due to data collection constraints. Following the training-
evaluation paradigm, we determine a boundary time 𝑇𝐵to separate
the training and evaluation datasets. The final timestamp label set
Y𝐹is then divided into two subsets: Y𝐹,𝐵= {𝑦𝑇𝑢
𝑢
| 𝑢∈V,𝑇𝑢≤𝑇𝐵}
consisting of labels for nodes whose final timestamps are before
𝑇𝐵, and Y𝐹,𝐴= {𝑦𝑇𝑢
𝑢
| 𝑢∈V,𝑇𝑢> 𝑇𝐵} containing labels for nodes
whose final timestamps is after 𝑇𝐵. Similarly, Y𝐸is partitioned into
Y𝐸,𝐵= {𝑦𝑡𝑢| 𝑢∈V,𝑡∈T𝑢\ {𝑇𝑢},𝑡≤𝑇𝐵} and Y𝐸,𝐴= {𝑦𝑡𝑢| 𝑢∈
V,𝑡∈T𝑢\ {𝑇𝑢},𝑡> 𝑇𝐵}, representing labels whose corresponding
timestamps are before and after 𝑇𝐵, respectively.
Specifically, when all timestamps 𝑡𝑖are identical or omitted,
the graph degenerates into a static graph [22, 28], where each
node 𝑢∈V is associated with a single label 𝑦𝑢. Then the problem
degrades to a static node classification task. Alternatively, if labels are
available for all nodes at all timestamps, i.e., Y𝐸is entirely known,
the problem becomes a fully supervised dynamic node classification
task, which has been extensively studied in prior research [9, 48,
61, 66].
The dynamic graph backbone generates node embeddings h𝑡𝑢
for each node 𝑢at each timestamp 𝑡∈T𝑢. The backbone takes the
node features as input n𝑢∈R𝑑𝑁and edge features e𝑡𝑢,𝑣∈R𝑑𝐸. If
the graph is non-attributed, we assume n𝑢= 0 and e𝑡𝑢,𝑣= 0 for all
nodes and edges, respectively.
Given a dynamic graph G with dynamic labels and |Y𝐸| ≫|Y𝐹|,
our task label-limited dynamic node classification aims to
maximize log𝑝(Y𝐹,𝐵|G). Specifically, our goal is to learn a model
that can finally accurately predict Y𝐹.
4
Methodology
In this section, we present our proposed method for label-limited dy-
namic label learning. As shown in Figure 2, our approach combines
the variational EM framework with a novel Temporal Curriculum
Learning strategy to effectively leverage both final timestamp labels
and pseudo-labels in a dynamic graph setting. The EM framework
alternates between two steps: the E-step, where the decoder is op-
timized to predict node labels using final timestamp labels, and
the M-step, where the dynamic graph backbone is trained using
both final timestamp labels and pseudo-labels generated by the
decoder. To further enhance the quality of pseudo-labels, we intro-
duce Temporal Curriculum Learning, which dynamically adjusts
the weight of pseudo-labels based on their temporal distance to the


<!-- page 4 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Trovato et al.
  
 Backbone θ 
u
t1
...
tj-1
tj
Tu
label
label
label
label
×0.1
t2
label
×0.15
×0.99
×1.0
output
input
×1.0
label pseudo-labels
label
final timestamp labels
M-Step
E-Step
×1.0 temporal weights
target
Decoder φ
❄
...
Node Embedding
...
Update
timeline
🔥
Decoder φ
🔥
...
...
1. Temporal sample
2. Temporal aggregate
3. Temporal propagate
Dynamic Graph 
Link Prediction Warmup
cascade🔥trainable
❄frozen
Figure 2: Overview of our proposed method. PTCL consists of a Variational EM process with a dynamic graph backbone and
a decoder. During the warmup phase, the dynamic graph backbone is trained on a link prediction task, where the dynamic
graph structure serves as the target. After warmup, in each M-step, the backbone receives final timestamp labels, pseudo-labels,
and the dynamic graph structure as input, while the decoder is trained in the E-step to refine pseudo-labels. Additionally, the
Temporal Curriculum Learning strategy prioritizes pseudo-labels based on their temporal proximity to the final timestamp
labels, ensuring higher-quality training.
final timestamp labels. This ensures that the backbone is trained
with high-quality supervision signals.
4.1
Variational EM Framework
Following previous work [46, 69], we adopt the variational EM
framework [10, 39] to maximize log𝑝(Y𝐹,𝐵|G).
4.1.1
Evidence Lower Bound (ELBO). To handle the unknown labels
Y𝐸,𝐵, instead of directly optimizing log𝑝𝜃(Y𝐹,𝐵|G), we maximize
the evidence lower bound (ELBO) of the log-likelihood function:
log𝑝(Y𝐹,𝐵|G) ≥E𝑞𝜙(Y𝐸,𝐵| G) [log𝑝𝜃(Y𝐹,𝐵, Y𝐸,𝐵|G)
−log𝑞𝜙(Y𝐸,𝐵|G)],
(1)
where 𝑝𝜃(Y𝐹,𝐵, Y𝐸,𝐵|G) is the joint distribution of observed and
unknown labels, modeled by the dynamic graph backbone with pa-
rameters 𝜃. 𝑞𝜙(Y𝐸,𝐵|G) is the variational distribution that approx-
imates the true posterior distribution 𝑝𝜃(Y𝐸,𝐵|Y𝐹,𝐵, G), modeled
by the decoder with parameters 𝜙.
To facilitate optimization, we follow mean field assumption [16],
which yields the following factorization:
𝑞𝜙(Y𝐸,𝐵|G) =
Ö
𝑢∈V
Ö
𝑡∈T𝑢\{𝑇𝑢}
𝑡≤𝑇𝐵
𝑞𝜙(𝑦𝑡
𝑢|h𝑡
𝑢),
(2)
where 𝑞𝜙(𝑦𝑡𝑢|h𝑡𝑢) is the label distribution predicted by the decoder.
It means the decoder maps the node embeddings h𝑡𝑢generated by
the backbone to label predictions ˆ𝑦𝑡𝑢.
The ELBO can be optimized by alternating between the E-step
and the M-step.
4.1.2
Backbone Warmup. Since the initial parameters of the EM
algorithm are crucial for its performance [11, 30], we first warm up
the backbone by training it on a link prediction task, which predicts
the existence of edges between nodes based on their embeddings.
Given a dynamic graph G = {(𝑢𝑖, 𝑣𝑖,𝑡𝑖)}, the link prediction loss is
defined as:
L𝑙𝑝= −
∑︁
(𝑢𝑖,𝑣𝑖,𝑡𝑖)∈G
log𝜎(MLP(h𝑡𝑖𝑢𝑖∥h𝑡𝑖𝑣𝑖))
−
∑︁
(𝑢𝑗,𝑣𝑗,𝑡𝑗)∉G
log
 
1 −𝜎(MLP(h𝑡𝑗
𝑢𝑗∥h𝑡𝑗
𝑣𝑗))
 
.
(3)
where 𝜎(·) is the sigmoid function, ∥means concatenation. The first
term encourages the model to predict existing edges correctly, while
the second term penalizes the model for predicting non-existent
edges.
By minimizing L𝑙𝑝, the backbone learns to capture the struc-
tural and temporal dependencies in the graph, providing a strong
initialization for the EM algorithm.
4.1.3
E-step. In the E-step, we use the wake-sleep algorithm [20],
following Zhao et al. [69]. We fix the dynamic graph backbone 𝜃
and optimize the decoder 𝜙to minimize the KL divergence between
the true posterior distribution 𝑝𝜃(Y𝐸,𝐵|G, Y𝐹,𝐵) and the variational

[CAPTION] Figure 2: Overview of our proposed method. PTCL consists of a Variational EM process with a dynamic graph backbone and


<!-- page 5 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
distribution 𝑞𝜙(Y𝐸,𝐵|G). The objective function for the decoder is:
ˆO𝜙=
∑︁
𝑢∈V
∑︁
𝑡∈T𝑢\{𝑇𝑢}
𝑡≤𝑇𝐵
E𝑝𝜃(𝑦𝑡𝑢| G,Y𝐹,𝐵) [log𝑞𝜙(𝑦𝑡
𝑢|h𝑡
𝑢)],
(4)
Following Zhao et al. [69], we use the pseudo-labels ˆY𝐸,𝐵gener-
ated by the decoder to approximate the distribution 𝑝𝜃(𝑦𝑡𝑢|G, Y𝐹,𝐵):
𝑝𝜃(𝑦𝑡
𝑢|G, Y𝐹,𝐵) ≈𝑝𝜃(𝑦𝑡
𝑢|G, Y𝐹,𝐵, ˆY𝐸,𝐵\ ˆY𝑢
𝐸,𝐵),
(5)
where ˆY𝑢
𝐸,𝐵= { ˆ𝑦𝑡𝑢| 𝑡∈T𝑢\{𝑇𝑢},𝑡≤𝑇𝐵} is the set of pseudo-labels
of node 𝑢. Then the objective function of the decoder changes to:
ˆO𝜙= 𝛼
∑︁
𝑢∈V
∑︁
𝑡∈T𝑢\{𝑇𝑢}
𝑡≤𝑇𝐵
E𝑝𝜃(𝑦𝑡𝑢| G,Y𝐹,𝐵) [log𝑞𝜙(𝑦𝑡
𝑢|h𝑡
𝑢)]
+(1 −𝛼)
∑︁
𝑢∈V
𝑇𝑢≤𝑇𝐵
log𝑞𝜙(𝑦𝑇𝑢
𝑢|h𝑇𝑢
𝑢),
(6)
where 𝛼is a hyperparameter that balances the weight of pseudo-
labels and final timestamp labels.
But in practice, as shown in Section 5.2.2, we find that setting 𝛼
to 0 yields the best performance, which means we train the decoder
only with final timestamp labels. Therefore, the final objective
function for the decoder is:
O𝜙=
∑︁
𝑢∈V
𝑇𝑢≤𝑇𝐵
log𝑞𝜙(𝑦𝑇𝑢
𝑢|h𝑇𝑢
𝑢),
(7)
4.1.4
M-step. In the M-step, following the previous work [46, 69],
we aim to maximize the following pseudo-likelihood [3]. Specifi-
cally, we fix the decoder 𝜙and optimize the dynamic graph back-
bone 𝜃using both the final timestamp labels Y𝐹,𝐵and the pseudo-
labels ˆY𝐸,𝐵generated in the E-step. The objective is to maximize
the pseudo-likelihood:
ˆO𝜃= 𝛽
∑︁
𝑢∈V
∑︁
𝑡∈T𝑢\{𝑇𝑢}
𝑡≤𝑇𝐵
log𝑝𝜃(𝑦𝑡
𝑢|G, Y𝐹,𝐵, ˆY𝐸,𝐵\ ˆY𝑢
𝐸,𝐵)
+(1 −𝛽)
∑︁
𝑢∈V
𝑇𝑢≤𝑇𝐵
log𝑝𝜃(𝑦𝑇𝑢
𝑢|G, Y𝐹,𝐵\ {𝑦𝑇𝑢
𝑢}, ˆY𝐸,𝐵),
(8)
where 𝛽is a hyperparameter that balances the weight of pseudo-
labels and final timestamp labels. Note that the backbone can only
generate embeddings, so we cascade the backbone and decoder,
fixing the decoder 𝜙to use final timestamp labels and pseudo-labels
generated by the decoder to train the backbone.
4.2
Temporal Curriculum Learning
In the M-step, the backbone is trained using both final timestamp
labels Y𝐹,𝐵and pseudo-labels ˆY𝐸,𝐵generated by the decoder. How-
ever, the quality of pseudo-labels can vary significantly, as nodes’
labels may change over time. To ensure that the backbone is trained
with high-quality pseudo-labels, we propose a Temporal Curricu-
lum Learning strategy, which dynamically adjusts the weight
of pseudo-labels based on their temporal proximity to the final
timestamp labels and 𝜏(EM iteration counter).
4.2.1
Motivation. Temporal Curriculum Learning is motivated by
the inherent continuity in node label evolution: a node’s label 𝑦𝑡𝑢
at timestamp 𝑡typically aligns more closely with its final label
𝑦𝑇𝑢
𝑢when 𝑡nears 𝑇𝑢, while earlier timestamps exhibit greater di-
vergence. This temporal proximity principle suggests that pseudo-
labels near 𝑇𝑢provide reliable supervision, whereas distant ones
introduce higher noise. Our strategy thus prioritizes predictions
near the final timestamp during the initial training iterations. As
optimization progresses, the model gradually incorporates earlier
pseudo-labels, leveraging their improving quality to capture com-
prehensive temporal evolution patterns.
4.2.2
Pseudo-label Temporal Weighting. To implement Temporal
Curriculum Learning, we introduce a weighting mechanism for
pseudo-labels based on their temporal order relative to the final
timestamp 𝑇𝑢. Specifically, for each node 𝑢∈V, timestamp 𝑡∈T𝑢
in 𝜏-th iteration, we define a weight 𝑤𝑡,𝜏
𝑢
as follows:
𝑤𝑡,𝜏
𝑢
= 𝑓TW(𝑡,𝑢,𝜏,𝑇𝑢,𝛾) =
(
1,
if 𝑑𝑡𝑢≤𝜏,
exp  −𝛾· (𝑑𝑡𝑢−𝜏)  ,
if 𝑑𝑡𝑢> 𝜏,
(9)
𝑑𝑡
𝑢= |{𝑡′ ∈T𝑢| 𝑡′ > 𝑡}|
(10)
where 𝑑𝑡𝑢is the discrete temporal distance between of times-
tamp 𝑡and 𝑇𝑢in T𝑢, representing its position in the sequence of
timestamps for node 𝑢. For example, if T𝑢= {𝑡1,𝑡2,𝑡3,𝑇𝑢} and
𝑡1 < 𝑡2 < 𝑡3 < 𝑇𝑢, then 𝑑𝑡1
𝑢= 3, 𝑑𝑡2
𝑢= 2, 𝑑𝑡3
𝑢= 1 and 𝑑𝑇𝑢
𝑢
= 0. It
serves as a threshold that determines whether a timestamp is con-
sidered "close" to the final timestamp 𝑇𝑢. 𝛾> 0 is a hyperparameter
that controls the rate of Temporal Curriculum Learning decay.
The weight 𝑤𝑡,𝜏
𝑢
dynamically adjusts the importance of pseudo-
labels during training. If 𝑑𝑡𝑢≤𝜏, the timestamp 𝑡is considered
close to the final timestamp 𝑇𝑢, and the pseudo-label is assigned a
weight of 1, indicating high confidence in its quality. And if 𝑑𝑡𝑢> 𝜏,
the timestamp 𝑡is considered far from 𝑇𝑢, and the pseudo-label
weight decays exponentially with the distance 𝜏−𝑑𝑡𝑢, reducing its
influence on the training process.
This mechanism ensures that the backbone is trained with higher-
quality pseudo-labels during the early stages of training, while
gradually reducing the impact of lower-quality pseudo-labels as
training progresses. As 𝜏increases over iterations, the model is able
to learn from pseudo-labels at earlier timestamps, capturing the
temporal evolution of node labels more effectively.
With the pseudo-label temporal weights 𝑤𝑡,𝜏
𝑢, the objective func-
tion for the M-step is modified as follows:
O𝜃= 𝛽
∑︁
𝑢∈V
∑︁
𝑡∈T𝑢\{𝑇𝑢}
𝑡≤𝑇𝐵
𝑤𝑡,𝜏
𝑢log𝑝𝜃(𝑦𝑡
𝑢|G, Y𝐹,𝐵, ˆY𝐸,𝐵\ ˆY𝑢
𝐸,𝐵)
+(1 −𝛽)
∑︁
𝑢∈V
𝑇𝑢≤𝑇𝐵
log𝑝𝜃(𝑦𝑇𝑢
𝑢|G, Y𝐹,𝐵\ {𝑦𝑇𝑢
𝑢}, ˆY𝐸,𝐵).
(11)
By incorporating temporal weight 𝑤𝑡,𝜏
𝑢, the backbone is trained to
prioritize high-quality pseudo-labels.
4.3
Learning and Optimization
First, we use a link prediction task to warm up the backbone. Then
we proceed with the variational EM algorithm, which alternates


<!-- page 6 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Trovato et al.
between the E-step and the M-step. Finally, we use the decoder to
predict Y𝐹,𝐴. The complete procedure is summarized in Appendix.
ref algorithm.
4.4
FLiD: A novel framework
We introduce FLiD, a novel code framework specifically designed
for our task, where only final timestamp labels are available. Unlike
existing dynamic graph frameworks such as DyGLib [66] and TGL
[57], FLiD provides comprehensive support through several key
features. It includes new data preprocessing modules for handling
raw datasets, and a flexible training pipeline adaptable to various
methods, while also supporting different dynamic graph backbones
and pseudo-label enhancement strategies. Additionally, FLiD incor-
porates specific evaluation protocols to ensure fair comparisons
across methods. We use FLiD to implement PTCL, and all experi-
ments in this study are conducted using FLiD, which is designed
to be highly extensible, enabling seamless integration of future
methods for this task.
5
Experiments
Our experiments are designed to address the following key research
questions (RQs):
RQ1: How does PTCL perform compared to other baselines when
evaluated on the final timestamp labels?
RQ2: Does pseudo-labels generated by PTCL improve perfor-
mance by capturing the dynamic information of nodes?
RQ3: Does the proposed Temporal Curriculum Learning strategy
improve performance?
RQ4: Is PTCL stable and computationally efficient?
5.1
Experiment Settings
5.1.1
Datasets. We evaluate PTCL in various real-world scenes.
Specifically, we use three existing datasets: Wikipedia [29], Reddit
[29] and a sub-graph of Dgraph [24] (Dsub for short). Moreover,
based on the OAG dataset [51, 55, 67, 68], we propose a new dataset
CoOAG. Statistics and more details of datasets can be found in
Appendix.B. For Wikipedia, Reddit, and Dsub (imbalanced binary
classification tasks), we use AUC as the evaluation metric [66],
excluding background nodes in Dsub. For CoOAG (multi-class clas-
sification tasks), we use accuracy (ACC) as the metric. To align with
real-world scenarios where only the final labels are available for
evaluation, we design a new dataset split and evaluation protocol.
Specifically, we split the datasets based on the final timestamp and
evaluate model performance only on the final timestamp labels
(Y𝐹,𝐴). By statistically analyzing the labels at the final timestamp,
we partition the nodes into training, validation, and test sets with
a ratio of 7:1.5:1.5 [66]. All experiments are conducted with five
random seeds, and metrics are reported as the average performance.
Implementation details and hyperparameter settings can be found
in Appendix.D and Appendix.E.
5.1.2
Baselines. We compare PTCL with several different methods
that can be adapted to our task. And specifically, we use five dif-
ferent dynamic backbones as backbones: TGAT [61], GraphMixer
[9], TCL [57], TGN [48], and DyGFormer [66], and apply a simple
MLP as the decoder. More backbone details are in Appendix.C. As
   B
   D
FTL
copy
DL
CFT
DLS
PL
update
NPL
FTL
+
PTCL-2D
Ours
output
input
PL pseudo-labels
FTL final timestamp labels
target
cascade
DL dynamic labels
update
FTL
+
PL
update
NE
SEM
NE node embeddings
🔥
   B
   D
🔥
   B
   D
🔥
   B
   D
🔥
❄
   D
FTL
+
 B
 D
NE
update
 D
PL
🔥
🔥
❄
FTL
+
 B
 D1
NE
update
 D2
PL
🔥
🔥
🔥
🔥
🔥
🔥
🔥
🔥trainable ❄
frozen
Figure 3: Architecture of different baselines and PTCL. "B"
stands for backbone, and "D" stands for decoder.
shown in Figure 3, the baselines are designed to cover a range of
approaches:
• CFT (Copy-Final Timestamp labels): A naive baseline that
simply copies the final timestamp labels (Y𝐹,𝐵) to earlier times-
tamps as approximations of dynamic labels (Y𝐸,𝐵) for training.
• DLS (Dynamic Label Supervision): A baseline that performs
supervised training directly using the dynamic labels provided
by the dataset (Wikipedia, Reddit), where available [48, 66].
• NPL (Naive Pseudo-Labels): A variant of PTCL that uses pseudo-
labels but jointly optimizes the backbone and decoder, without
EM optimization.
• PTCL-2D (PTCL with 2 Decoders): A variant of PTCL that uses
two decoders: one decoder is trained exclusively on the final
timestamp labels (E-step), generating pseudo-labels, while the
other decoder is jointly optimized with the backbone on weighted
pseudo-labels and final timestamp labels (M-step). The final em-
beddings are provided by the backbone for the E-step training.
• SEM (Standard EM): A variant of PTCL where both the E-step
and M-step are trained on the weighted loss of pseudo-labels and
final timestamp labels [69], while other components remain the
same as PTCL. In this way, E-step uses Eq. (6) as the objective
function instead of Eq. (7).
5.2
RQ1: Main Results
To evaluate the effectiveness of PTCL, we conduct experiments by
replacing different backbones and comparing various baselines.
The results in Table 1 show that PTCL improves the performance of
nearly all the backbones with respect to all datasets.
5.2.1
Superiority of Pseudo-Labels. PTCL consistently outperforms
the CFT and DLS baselines with significant improvements (ranging
from a minimum of 0.99% to a maximum of 11.23% in AUC/ACC),
demonstrating the effectiveness of using pseudo-labels. Compared
to simply copying final timestamp labels (CFT), our approach dy-
namically learns pseudo-labels, effectively mitigating the issue of
incorrect information propagation caused by direct label copying.
Notably, PTCL even surpasses supervised training with dynamic
labels (DLS), suggesting that the pseudo-labels generated by PTCL
better reflect the underlying node information changes than the
original dynamic labels. We provide a detailed analysis of this phe-
nomenon in Section 5.3.
5.2.2
Necessity of Separate Optimization. Compared to the NPL
baseline, which jointly optimizes the backbone and decoder, PTCL
achieves significantly better performance with an average improve-
ment of 2.74% in AUC/ACC. This highlights the importance of

[CAPTION] Figure 3: Architecture of different baselines and PTCL. "B"


<!-- page 7 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 1: Performance comparison across datasets (Wikipedia, Reddit, Dsub, CoOAG). We run all experiments with five random
seeds to ensure a consistent evaluation and report the average performance as well as the standard deviation in parentheses.
Bold indicates the best performance, underline the second best. Dsub and CoOAG datasets can not apply the DLS method due
to a lack of dynamic labels. TGN runs out of memory on Dsub due to its high space cost.
Backbone
Method
Wikipedia
Reddit
Dsub
CoOAG
AUC
AUC
AUC
ACC
TGAT
CFT
77.43(±3.01)
82.68(±0.06)
62.32(±1.27)
86.28(±0.18)
DLS
79.56(±2.55)
78.66(±0.04)
–
-
NPL
78.52(±2.28)
80.72(±2.65)
61.71(±2.21)
87.67(±0.61)
PTCL-2D
78.20(±6.83)
85.84(±6.09)
60.76(±2.91)
88.37(±0.16)
SEM
81.09(±3.62)
86.27(±5.99)
64.34(±0.99)
88.38(±0.38)
Ours
85.52(±3.29)
87.31(±6.50)
65.07(±1.57)
89.05(±0.63)
TCL
CFT
76.27(±4.68)
84.48(±5.53)
62.60(±1.26)
86.12(±1.11)
DLS
80.55(±1.93)
82.85(±0.38)
–
–
NPL
77.71(±5.66)
84.20(±3.86)
63.59(±3.09)
87.59(±0.26)
PTCL-2D
76.68(±2.86)
86.98(±3.34)
60.64(±1.41)
87.94(±0.30)
SEM
81.02(±2.82)
87.56(±1.56)
65.11(±1.26)
87.90(±0.18)
Ours
82.27(±4.62)
89.41(±3.32)
66.80(±2.45)
88.24(±0.26)
TGN
CFT
80.68(±2.02)
89.69(±2.07)
OOM
83.65(±0.63)
DLS
78.48(±1.60)
80.92(±4.99)
–
–
NPL
87.58(±2.14)
84.22(±2.48)
OOM
86.13(±0.31)
PTCL-2D
86.59(±3.01)
86.76(±3.89)
OOM
86.23(±0.66)
SEM
86.34(±2.66)
82.61(±3.07)
OOM
86.07(±0.66)
Ours
87.97(±2.90)
84.32(±2.07)
OOM
86.71(±0.66)
GraphMixer
CFT
76.60(±2.00)
66.11(±6.04)
62.78(±1.90)
85.63(±0.14)
DLS
80.70(±4.00)
61.97(±7.36)
–
–
NPL
80.86(±1.62)
71.72(±6.48)
67.14(±1.68)
86.98(±0.61)
PTCL-2D
81.41(±4.25)
66.86(±11.14)
62.33(±1.35)
87.51(±0.51)
SEM
83.33(±1.45)
68.65(±3.70)
69.23(±1.92)
88.07(±0.30)
Ours
84.09(±0.95)
71.93(±7.94)
69.76(±1.54)
88.26(±0.38)
DyGFormer
CFT
64.76(±9.21)
67.14(±8.04)
68.48(±1.46)
85.27(±0.83)
DLS
71.95(±2.29)
64.63(±4.90)
–
–
NPL
73.85(±5.44)
67.44(±3.47)
70.31(±1.11)
86.16(±0.38)
PTCL-2D
66.48(±6.76)
71.14(±6.27)
69.11(±2.96)
86.04(±0.30)
SEM
70.91(±8.80)
71.59(±4.51)
69.75(±2.47)
86.07(±0.66)
Ours
74.85(±3.07)
75.86(±8.04)
72.39(±1.91)
86.26(±0.27)
separate optimization, which ensures that the decoder remains con-
sistent with the final timestamp labels while allowing the backbone
to effectively learn from pseudo-labels at earlier timestamps. Addi-
tionally, PTCL outperforms SEM, demonstrating that training the
decoder exclusively on final timestamp labels (𝛼= 0, as described
in Section 4.1.3) leads to better results.
5.2.3
Advantages of PTCL over PTCL-2D. PTCL not only achieves
better performance but also requires fewer computational resources
compared to the PTCL-2D baseline. This indicates that using a single
decoder helps maintain consistency in the feature space between
the backbone and decoder, leading to more stable training and
higher performance.
Table 2: AUC comparison of different backbones using Dy-
namic Label Supervised Learning (DLS) and Pseudo-Label
Supervised Learning (PLS) on Wikipedia Dataset.
TGAT
TCL
TGN
GraphMixer
DyGFormer
DLS
79.56
80.55
78.48
80.7
71.95
PLS
81.11
82.19
79.32
81.02
72.42
5.3
RQ2: Pseudo-Label Analysis
In this section, we evaluate the effectiveness of our pseudo-labels
and their ability to capture dynamic patterns through two experi-
ments.


**[Table p7.1]**
| Backbone | Method | Wikipedia Reddit Dsub CoOAG |
| --- | --- | --- |
|  |  | AUC AUC AUC ACC |
| TGAT | CFT DLS NPL PTCL-2D SEM Ours | 77.43(±3.01) 82.68(±0.06) 62.32(±1.27) 86.28(±0.18) 79.56(±2.55) 78.66(±0.04) – - 78.52(±2.28) 80.72(±2.65) 61.71(±2.21) 87.67(±0.61) 78.20(±6.83) 85.84(±6.09) 60.76(±2.91) 88.37(±0.16) 81.09(±3.62) 86.27(±5.99) 64.34(±0.99) 88.38(±0.38) 85.52(±3.29) 87.31(±6.50) 65.07(±1.57) 89.05(±0.63) |
| TCL | CFT DLS NPL PTCL-2D SEM Ours | 76.27(±4.68) 84.48(±5.53) 62.60(±1.26) 86.12(±1.11) 80.55(±1.93) 82.85(±0.38) – – 77.71(±5.66) 84.20(±3.86) 63.59(±3.09) 87.59(±0.26) 76.68(±2.86) 86.98(±3.34) 60.64(±1.41) 87.94(±0.30) 81.02(±2.82) 87.56(±1.56) 65.11(±1.26) 87.90(±0.18) 82.27(±4.62) 89.41(±3.32) 66.80(±2.45) 88.24(±0.26) |
| TGN | CFT DLS NPL PTCL-2D SEM Ours | 80.68(±2.02) 89.69(±2.07) OOM 83.65(±0.63) 78.48(±1.60) 80.92(±4.99) – – 87.58(±2.14) 84.22(±2.48) OOM 86.13(±0.31) 86.59(±3.01) 86.76(±3.89) OOM 86.23(±0.66) 86.34(±2.66) 82.61(±3.07) OOM 86.07(±0.66) 87.97(±2.90) 84.32(±2.07) OOM 86.71(±0.66) |
| GraphMixer | CFT DLS NPL PTCL-2D SEM Ours | 76.60(±2.00) 66.11(±6.04) 62.78(±1.90) 85.63(±0.14) 80.70(±4.00) 61.97(±7.36) – – 80.86(±1.62) 71.72(±6.48) 67.14(±1.68) 86.98(±0.61) 81.41(±4.25) 66.86(±11.14) 62.33(±1.35) 87.51(±0.51) 83.33(±1.45) 68.65(±3.70) 69.23(±1.92) 88.07(±0.30) 84.09(±0.95) 71.93(±7.94) 69.76(±1.54) 88.26(±0.38) |
| DyGFormer | CFT DLS NPL PTCL-2D SEM Ours | 64.76(±9.21) 67.14(±8.04) 68.48(±1.46) 85.27(±0.83) 71.95(±2.29) 64.63(±4.90) – – 73.85(±5.44) 67.44(±3.47) 70.31(±1.11) 86.16(±0.38) 66.48(±6.76) 71.14(±6.27) 69.11(±2.96) 86.04(±0.30) 70.91(±8.80) 71.59(±4.51) 69.75(±2.47) 86.07(±0.66) 74.85(±3.07) 75.86(±8.04) 72.39(±1.91) 86.26(±0.27) |


**[Table p7.2]**
|  | TGAT TCL TGN GraphMixer DyGFormer |
| --- | --- |
| DLS | 79.56 80.55 78.48 80.7 71.95 |
| PLS | 81.11 82.19 79.32 81.02 72.42 |

[CAPTION] Table 1: Performance comparison across datasets (Wikipedia, Reddit, Dsub, CoOAG). We run all experiments with five random

[CAPTION] Table 2: AUC comparison of different backbones using Dy-


<!-- page 8 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Trovato et al.
0.0
0.2
0.4
0.6
0.8
1.0
Consistency Value
10
1
100
101
102
Frequency (Log Scale)
KDE curve of Consistency
Figure 4: Histogram of pseudo-labels consistency.
5.3.1
Pseudo-label Supervision Study. To analyze the effectiveness
of our pseudo-labels, we conduct the following experiment: We
train models from scratch using our generated pseudo-labels (from
our trained model) as full supervision labels and compare the results
with DLS. As shown in Table 2, models trained with our pseudo-
labels consistently outperform those using original dynamic labels
with an average improvement of 0.96% in AUC, demonstrating our
pseudo-labels better capture latent evolutionary patterns of nodes.
5.3.2
Label Consistency Analysis. To further investigate the tem-
poral changes of labels, we analyze the labels’ consistency on
Wikipedia’s positive samples. Consistency is quantified as follows:
ˆ𝑁𝑢′ = max
n
𝑘∈N+ |𝑦𝑡𝑖
𝑢′ = 𝑦𝑇𝑢′
𝑢′ ,
∀𝑖∈{|T𝑢′ | −𝑘, . . . , |T𝑢′ | −1}
o
(12)
𝐶𝑢′ =
ˆ𝑁𝑢′
|T𝑢′ | −1
(13)
where 𝑢′ ∈Vneg, Vneg = {𝑢′|𝑢′ ∈V,𝑦𝑇𝑢′
𝑢′ = 1}. In Wikipedia, the
dynamic label consistency of all negative samples is zero(𝐶𝑢′ ≡0),
which means that negative-class labels abruptly emerge at the final
timestamp (Y𝐹), neglecting gradual behavioral transitions. And CFT
method enforces strict label continuity by replicating Y𝐹,𝐵(𝐶𝑢′ ≡1),
suffering from severe feature misalignment. In contrast, as shown
in Figure 4, pseudo-labels generated by PTCL with TGAT backbone
exhibit varying consistency values, propagating smoothly across
timestamps in Y𝐸,𝐵, striking an optimal balance between temporal
coherence and feature alignment.
5.4
RQ3: Temporal Curriculum Learning
Analysis
To comprehensively evaluate our Temporal Curriculum Learning
design, we conduct a comparison experiment against the naive solu-
tion, which uses all the generated pseudo-labels, and two commonly
used baseline strategies to choose more reliable pseudo-labels in
Curriculum Learning, as introduced in Section 2.2:
• Confidence Score Threshold (CST) [6, 19, 54]: This method filters
pseudo-labels based on their confidence scores, improving the
overall quality of the labels.
• Entropy of Softmax Trajectory (EST) [43, 52]: This method filters
pseudo-labels using the entropy of the softmax trajectory, which
Table 3: AUC comparison of our Temporal Curriculum Learn-
ing with two commonly used strategies to choose more re-
liable pseudo-labels in Curriculum Learning and a naive
strategy which uses all the pseudo-labels. CST stands for
Confidence Score Threshold, and EST stands for Entropy
of Softmax Trajectory. Bold indicates the best performance,
underline the second best.
Backbone
Dataset
Naive
CST
EST
Ours
TGAT
Wikipedia
82.38
79.48
81.89
85.52
Dsub
64.12
62.78
63.80
65.07
TCL
Wikipedia
81.77
78.06
80.56
82.27
Dsub
65.29
63.68
64.57
66.80
TGN
Wikipedia
86.33
83.63
86.88
87.97
Dsub
OOM
OOM
OOM
OOM
GraphMixer
Wikipedia
83.34
81.59
81.13
84.09
Dsub
68.08
68.79
67.55
69.76
DyGFormer
Wikipedia
69.42
67.15
68.39
74.85
Dsub
72.30
70.86
69.21
72.39
1
2
3
4
5
6
7
8
9
10
 (EM Iteration Counter)
40
50
60
70
80
90
AUC (%)
Models
TGAT
DyGFormer
TGN
TCL
GraphMixer
Figure 5: Convergence curves for 5 backbones. Star markers
(★) denote peak performance; circled points(•)indicate sur-
passing baselines. Dashed lines show baseline AUC.
is an accumulated distribution that summarizes the model’s dis-
agreement across training rounds.
As shown in Table 3, it’s obvious that our Temporal Curriculum
Learning strategy consistently achieves the highest AUC scores
across all backbones and datasets, indicating the effectiveness of
our design. The Naive strategy performs significantly worse than
PTCL, with an average AUC drop of 1.74%, highlighting the noise
introduced by relying on the pseudo-labels. Similarly, CST and
EST, which rely on static filtering mechanisms, mostly underper-
form both PTCL and the Naive strategy due to their inability to
adapt to the dynamic reliability of pseudo-labels. This comparison
demonstrates that our Temporal Curriculum Learning design, by
effectively utilizing temporal information, is better suited for the
task.


**[Table p8.1]**
|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  | KDE curve of |  | Consiste |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
| 0 | 0. | 2 | 0. Co | 4 nsiste | 0. ncy Va | 6 lue | 0. | 8 |


**[Table p8.2]**
| Backbone | Dataset | Naive CST EST Ours |
| --- | --- | --- |
| TGAT | Wikipedia Dsub | 82.38 79.48 81.89 85.52 64.12 62.78 63.80 65.07 |
| TCL | Wikipedia Dsub | 81.77 78.06 80.56 82.27 65.29 63.68 64.57 66.80 |
| TGN | Wikipedia Dsub | 86.33 83.63 86.88 87.97 OOM OOM OOM OOM |
| GraphMixer | Wikipedia Dsub | 83.34 81.59 81.13 84.09 68.08 68.79 67.55 69.76 |
| DyGFormer | Wikipedia Dsub | 69.42 67.15 68.39 74.85 72.30 70.86 69.21 72.39 |


**[Table p8.3]**
|  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | Models TGAT DyGForm | er |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | TGN TCL GraphMi | xer |
|  |  |  |  |  |  |  |  |  |  |  |  |

[CAPTION] Figure 4: Histogram of pseudo-labels consistency.

[CAPTION] Table 3: AUC comparison of our Temporal Curriculum Learn-

[CAPTION] Figure 5: Convergence curves for 5 backbones. Star markers


<!-- page 9 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
5.5
RQ4: Convergence and Efficiency Analysis
We validate the convergence behavior and computational efficiency
of PTCL through iterative training analysis on the Wikipedia dataset
with 5 backbones. The compared baseline is the CFT, which only
utilizes the Y𝐹. As shown in Figure 5, PTCL achieves rapid con-
vergence across all backbones on Wikipedia. TGAT surpasses its
baseline immediately at the first iteration, while others require 2–4
iterations. All models reach peak performance within 6–10 itera-
tions, with DyGFormer achieving the largest gain (+16.1 % AUC).
Despite iterative training, each EM iteration incurs 0.8× – 1.2×
time overhead compared to baseline training, making the total cost
practical. This confirms PTCL achieves robust convergence across
diverse backbones while maintaining computational practicality.
6
Conclusion
In this work, we bridge the gap between highly dynamically chang-
ing graphs and limited accessible labels. Specifically, we propose
PTCL, an extensible method for dynamic node classification. By gen-
erating temporally-weighted pseudo-labels and optimizing through
the variational EM framework, our method achieves significant im-
provements (up to 11.23% AUC/ACC gain) over baselines in various
real-world scenes, including financial and academic networks, etc.,
which subsequently validates the importance of tracking dynamic
historical changes as well as the effectiveness of the variational EM
framework and Temporal Curriculum Learning. The introduced
CoOAG dataset and FLiD framework enable a practical evaluation
of temporal label propagation. Notably, PTCL also can be applied
to other dynamic graph tasks due to its high dynamic nature cap-
turing capabilities. To sum up, this work establishes foundational
techniques for learning evolving node behaviors under realistic an-
notation constraints, laying a solid foundation for future dynamic
graph studies.
References
[1] 2024. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671 (2024).
[2] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009.
Curriculum learning. In International Conference on Machine Learning. https:
//api.semanticscholar.org/CorpusID:873046
[3] Julian Besag. 1975. Statistical Analysis of Non-Lattice Data. The Statistician 24
(1975), 179–195. https://api.semanticscholar.org/CorpusID:116757950
[4] Smriti Bhagat, Graham Cormode, and S Muthukrishnan. 2011. Node classification
in social networks. Social network data analytics (2011), 115–148.
[5] Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Van-
dergheynst. 2017. Geometric deep learning: going beyond euclidean data. IEEE
Signal Processing Magazine 34, 4 (2017), 18–42.
[6] Paola Cascante-Bonilla, Fuwen Tan, Yanjun Qi, and Vicente Ordonez. 2020. Cur-
riculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning.
In AAAI Conference on Artificial Intelligence. https://api.semanticscholar.org/
CorpusID:228096598
[7] Olivier Chapelle, Bernhard Schölkopf, and Alexander Zien. 2006. Introduction to
semi-supervised learning. (2006).
[8] Olivier Chapelle and Alexander Zien. 2005. Semi-Supervised Classification by
Low Density Separation. In International Conference on Artificial Intelligence and
Statistics. https://api.semanticscholar.org/CorpusID:14283441
[9] Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hang-
hang Tong, and Mehrdad Mahdavi. 2023. Do we really need complicated model
architectures for temporal networks? arXiv preprint arXiv:2302.11636 (2023).
[10] Arthur P. Dempster, Nan M. Laird, and Donald B. Rubin. 1977. Maximum likeli-
hood from incomplete data via the EM - algorithm plus discussions on the paper.
https://api.semanticscholar.org/CorpusID:4193919
[11] Jennifer G Dy and Carla E Brodley. 2004. Feature selection for unsupervised
learning. Journal of machine learning research 5, Aug (2004), 845–889.
[12] Yucai Fan, Yuhang Yao, and Carlee Joe-Wong. 2021. Gcn-se: Attention as ex-
plainability for node classification in dynamic graphs. In 2021 IEEE International
Conference on Data Mining (ICDM). IEEE, 1060–1065.
[13] Kaituo Feng, Changsheng Li, Xiaolu Zhang, and Jun Zhou. 2023. Towards open
temporal graph neural networks. arXiv preprint arXiv:2303.15015 (2023).
[14] Shangbin Feng, Zhaoxuan Tan, Herun Wan, Ningnan Wang, Zilong Chen, Binchi
Zhang, Qinghua Zheng, Wenqian Zhang, Zhenyu Lei, Shujie Yang, et al. 2022.
Twibot-22: Towards graph-based twitter bot detection.
Advances in Neural
Information Processing Systems 35 (2022), 35254–35269.
[15] Matthias Fey and Jan E. Lenssen. 2019. Fast Graph Representation Learning with
PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and
Manifolds.
[16] Lise Getoor, Nir Friedman, Daphne Koller, and Ben Taskar. 2001. Learning Proba-
bilistic Models of Relational Structure. In International Conference on Machine
Learning. https://api.semanticscholar.org/CorpusID:10551607
[17] Yves Grandvalet and Yoshua Bengio. 2004. Semi-supervised Learning by Entropy
Minimization. In Conférence francophone sur l’apprentissage automatique. https:
//api.semanticscholar.org/CorpusID:7890982
[18] Xingzhi Guo, Baojian Zhou, and Steven Skiena. 2021. Subset node representation
learning over large dynamic graphs. In Proceedings of the 27th ACM SIGKDD
Conference on Knowledge Discovery & Data Mining. 516–526.
[19] Haiyun He, Gholamali Aminian, Yuheng Bu, Miguel L. Rodrigues, and Vincent
Y. F. Tan. 2022. How Does Pseudo-Labeling Affect the Generalization Error of
the Semi-Supervised Gibbs Algorithm?. In International Conference on Artificial
Intelligence and Statistics. https://api.semanticscholar.org/CorpusID:252918807
[20] Geoffrey E. Hinton, Peter Dayan, Brendan J. Frey, and R M Neal. 1995. The
"wake-sleep" algorithm for unsupervised neural networks. Science 268 5214
(1995), 1158–61. https://api.semanticscholar.org/CorpusID:871473
[21] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015.
Distilling the
Knowledge in a Neural Network. ArXiv abs/1503.02531 (2015).
https://api.
semanticscholar.org/CorpusID:7200347
[22] Petter Holme and Jari Saramäki. 2012. Temporal networks. Physics reports 519, 3
(2012), 97–125.
[23] Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure
Leskovec. 2021. Ogb-lsc: A large-scale challenge for machine learning on graphs.
arXiv preprint arXiv:2103.09430 (2021).
[24] Xuanwen Huang, Yang Yang, Yang Wang, Chunping Wang, Zhisheng Zhang,
Jiarong Xu, Lei Chen, and Michalis Vazirgiannis. 2022. Dgraph: A large-scale
financial dataset for graph anomaly detection. Advances in Neural Information
Processing Systems 35 (2022), 22765–22777.
[25] Tao Jia, Dashun Wang, and Boleslaw K Szymanski. 2017. Quantifying patterns
of research-interest evolution. Nature Human Behaviour 1, 4 (2017), 0078.
[26] Patrick Kage, Jay C Rothenberger, Pavlos Andreadis, and Dimitrios I Diochnos.
2024.
A review of pseudo-labeling for computer vision.
arXiv preprint
arXiv:2408.07221 (2024).
[27] Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Opti-
mization. CoRR abs/1412.6980 (2014). https://api.semanticscholar.org/CorpusID:
6628106
[28] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph
convolutional networks. arXiv preprint arXiv:1609.02907 (2016).
[29] Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting dynamic em-
bedding trajectory in temporal interaction networks. In Proceedings of the 25th
ACM SIGKDD international conference on knowledge discovery & data mining.
1269–1278.
[30] Wojciech Kwedlo. 2015. A new random approach for initialization of the multiple
restart EM algorithm for Gaussian model-based clustering. Pattern Analysis and
Applications 18 (2015), 757–770.
[31] Semi-Supervised Learning. 2006. Semi-supervised learning. CSZ2006. html 5
(2006), 2.
[32] Dong-Hyun Lee et al. 2013.
Pseudo-label: The simple and efficient semi-
supervised learning method for deep neural networks. In Workshop on challenges
in representation learning, ICML, Vol. 3. Atlanta, 896.
[33] Jianfeng Li and Dexiang Yang. 2023. Research on Financial Fraud Detection
Models Integrating Multiple Relational Graphs. Systems 11, 11 (2023), 539.
[34] Michelle M Li, Kexin Huang, and Marinka Zitnik. 2022. Graph representation
learning in biomedicine and healthcare. Nature Biomedical Engineering 6, 12
(2022), 1353–1369.
[35] Qimai Li, Zhichao Han, and Xiao-Ming Wu. 2018. Deeper Insights into Graph
Convolutional Networks for Semi-Supervised Learning. In AAAI Conference on
Artificial Intelligence. https://api.semanticscholar.org/CorpusID:11118105
[36] Yayong Li, Jie Yin, and Ling Chen. 2022. Informative pseudo-labeling for graph
neural networks with few labels. Data Mining and Knowledge Discovery 37 (2022),
228–254. https://api.semanticscholar.org/CorpusID:246063495
[37] Zhun Li, ByungSoo Ko, and Ho-Jin Choi. 2018. Naive semi-supervised deep
learning using pseudo-label. Peer-to-Peer Networking and Applications 12 (2018),
1358 – 1368. https://api.semanticscholar.org/CorpusID:69529604
[38] Takeru Miyato, Shin ichi Maeda, Masanori Koyama, and Shin Ishii. 2017. Vir-
tual Adversarial Training: A Regularization Method for Supervised and Semi-
Supervised Learning. IEEE Transactions on Pattern Analysis and Machine Intelli-
gence 41 (2017), 1979–1993. https://api.semanticscholar.org/CorpusID:17504174


<!-- page 10 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Trovato et al.
[39] Radford M. Neal and Geoffrey E. Hinton. 1998. A View of the Em Algorithm that
Justifies Incremental, Sparse, and other Variants. In Learning in Graphical Models.
https://api.semanticscholar.org/CorpusID:17947141
[40] Mark EJ Newman, Duncan J Watts, and Steven H Strogatz. 2002. Random graph
models of social networks. Proceedings of the national academy of sciences 99,
suppl_1 (2002), 2566–2572.
[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019.
Pytorch: An imperative style, high-performance deep learning library. Advances
in neural information processing systems 32 (2019).
[42] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss,
Vincent Dubourg, et al. 2011. Scikit-learn: Machine learning in Python. the
Journal of machine Learning research 12 (2011), 2825–2830.
[43] Hongbin Pei, Yuheng Xiong, Pinghui Wang, Jing Tao, Jialun Liu, Huiqi Deng,
Jie Ma, and Xiaohong Guan. 2024. Memory Disagreement: A Pseudo-Labeling
Measure from Training Dynamics for Semi-supervised Graph Learning. Proceed-
ings of the ACM on Web Conference 2024 (2024). https://api.semanticscholar.org/
CorpusID:269708945
[44] Meng Qin and Dit-Yan Yeung. 2023. Temporal link prediction: A unified frame-
work, taxonomy, and review. Comput. Surveys 56, 4 (2023), 1–40.
[45] Meng Qu. 2024. Towards combining deep learning and statistical relational learning
for reasoning on graphs. Ph.D. Thesis. Université de Montréal, Montreal, Canada.
Advisor(s) Jian Tang. http://hdl.handle.net/1866/32584
[46] Meng Qu, Yoshua Bengio, and Jian Tang. 2019. GMNN: Graph Markov Neural Net-
works. ArXiv abs/1905.06214 (2019). https://api.semanticscholar.org/CorpusID:
155093091
[47] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. 2019. Dropedge:
Towards deep graph convolutional networks on node classification. arXiv preprint
arXiv:1907.10903 (2019).
[48] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico
Monti, and Michael Bronstein. 2020. Temporal graph networks for deep learning
on dynamic graphs. arXiv preprint arXiv:2006.10637 (2020).
[49] Aravind Sankar, Yanhong Wu, Liang Gou, Wei Zhang, and Hao Yang. 2020.
Dysat: Deep neural representation learning on dynamic graphs via self-attention
networks. In Proceedings of the 13th international conference on web search and
data mining. 519–527.
[50] Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan
Günnemann. 2019. Pitfalls of Graph Neural Network Evaluation. doi:10.48550/
arXiv.1811.05868 arXiv:1811.05868 [cs]
[51] Arnab Sinha, Zhihong Shen, Yang Song, Hao Ma, Darrin Eide, Bo-June (Paul) Hsu,
and Kuansan Wang. 2015. An Overview of Microsoft Academic Service (MAS)
and Applications. In Proceedings of the 24th International Conference on World
Wide Web (New York, NY, USA, 2015-05-18) (WWW ’15 Companion). Association
for Computing Machinery, 243–246. doi:10.1145/2740908.2742839
[52] Hwanjun Song, Minseok Kim, and Jae-Gil Lee. 2019. SELFIE: Refurbishing Un-
clean Samples for Robust Deep Learning. In International Conference on Machine
Learning. https://api.semanticscholar.org/CorpusID:174800904
[53] Petru Soviany, Radu Tudor Ionescu, Paolo Rota, and N. Sebe. 2021. Curriculum
Learning: A Survey. International Journal of Computer Vision 130 (2021), 1526 –
1565. https://api.semanticscholar.org/CorpusID:231709290
[54] Ke Sun, Zhanxing Zhu, and Zhouchen Lin. 2019. Multi-Stage Self-Supervised
Learning for Graph Convolutional Networks. ArXiv abs/1902.11038 (2019). https:
//api.semanticscholar.org/CorpusID:67855919
[55] Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. 2008. Arnet-
Miner: Extraction and Mining of Academic Social Networks. In Proceedings of
the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining (New York, NY, USA, 2008-08-24) (KDD ’08). Association for Computing
Machinery, 990–998. doi:10.1145/1401890.1402008
[56] Jesper E Van Engelen and Holger H Hoos. 2020. A survey on semi-supervised
learning. Machine learning 109, 2 (2020), 373–440.
[57] Lu Wang, Xiaofu Chang, Shuang Li, Yunfei Chu, Hui Li, Wei Zhang, Xiaofeng
He, Le Song, Jingren Zhou, and Hongxia Yang. 2021. TCL: Transformer-based
Dynamic Graph Modelling via Contrastive Learning. ArXiv abs/2105.07944 (2021).
https://api.semanticscholar.org/CorpusID:234741805
[58] Xin Wang, Yudong Chen, and Wenwu Zhu. 2021. A Survey on Curriculum
Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 44
(2021), 4555–4576. https://api.semanticscholar.org/CorpusID:232362223
[59] Shunxin Xiao, Shiping Wang, Yuanfei Dai, and Wenzhong Guo. 2022. Graph
neural networks in node classification: survey and evaluation. Machine Vision
and Applications 33, 1 (2022), 4.
[60] Qizhe Xie, Zihang Dai, Eduard H. Hovy, Minh-Thang Luong, and Quoc V. Le.
2019. Unsupervised Data Augmentation. ArXiv abs/1904.12848 (2019). https:
//api.semanticscholar.org/CorpusID:139102880
[61] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan.
2020. Inductive representation learning on temporal graphs. arXiv preprint
arXiv:2002.07962 (2020).
[62] Mengde Xu, Zheng Zhang, Han Hu, Jianfeng Wang, Lijuan Wang, Fangyun Wei,
Xiang Bai, and Zicheng Liu. 2021. End-to-end semi-supervised object detection
with soft teacher. In Proceedings of the IEEE/CVF international conference on
computer vision. 3060–3069.
[63] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang
Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu,
Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. arXiv
preprint arXiv:2412.15115 (2024).
[64] Xiangli Yang, Zixing Song, Irwin King, and Zenglin Xu. 2022. A survey on deep
semi-supervised learning. IEEE Transactions on Knowledge and Data Engineering
35, 9 (2022), 8934–8954.
[65] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton,
and Jure Leskovec. 2018. Graph convolutional neural networks for web-scale
recommender systems. In Proceedings of the 24th ACM SIGKDD international
conference on knowledge discovery & data mining. 974–983.
[66] Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. 2023. Towards better dynamic graph
learning: New architecture and unified library. Advances in Neural Information
Processing Systems 36 (2023), 67686–67700.
[67] Fanjin Zhang, Xiao Liu, Jie Tang, Yuxiao Dong, Peiran Yao, Jie Zhang, Xiaotao
Gu, Yan Wang, Evgeny Kharlamov, Bin Shao, Rui Li, and Kuansan Wang. 2023.
OAG: Linking Entities Across Large-Scale Heterogeneous Knowledge Graphs.
35, 9 (2023), 9225–9239. doi:10.1109/TKDE.2022.3222168
[68] Fanjin Zhang, Xiao Liu, Jie Tang, Yuxiao Dong, Peiran Yao, Jie Zhang, Xiaotao
Gu, Yan Wang, Bin Shao, Rui Li, and Kuansan Wang. 2019. OAG: Toward Linking
Large-scale Heterogeneous Entity Graphs. In Proceedings of the 25th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining (New York, NY,
USA, 2019-07-25) (KDD ’19). Association for Computing Machinery, 2585–2595.
doi:10.1145/3292500.3330785
[69] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian
Tang. 2022. Learning on Large-scale Text-attributed Graphs via Variational Infer-
ence. ArXiv abs/2210.14709 (2022). https://api.semanticscholar.org/CorpusID:
253117079
[70] Hongkuan Zhou, Da Zheng, Israt Nisa, Vasileios Ioannidis, Xiang Song, and
George Karypis. 2022. Tgl: A general framework for temporal gnn training on
billion-scale graphs. arXiv preprint arXiv:2203.14883 (2022).
[71] Tianyi Zhou, Shengjie Wang, and Jeff Bilmes. 2020.
Time-consistent self-
supervision for semi-supervised learning. In International conference on machine
learning. PMLR, 11523–11533.
[72] Xiaojin Jerry Zhu. 2005. Semi-supervised learning literature survey. (2005).
[73] Marinka Zitnik, Monica Agrawal, and Jure Leskovec. 2018. Modeling polyphar-
macy side effects with graph convolutional networks. Bioinformatics 34, 13 (2018),
i457–i466.


<!-- page 11 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A
Algorithm for Training PTCL
Algorithm 1 Optimization Algorithm
1: Input: A dynamic graph G, final timestamp labels Y𝐹,𝐵, and
hyperparameter 𝛽
2: Output: Predicted ˆY𝐹,𝐴.
3: 𝜃←arg min𝜃L𝑙𝑝
⊲Warm up the backbone
4: 𝜏←1
⊲Initialize iteration counter
5: repeat
6:
E-Step: Decoder Optimization
7:
𝜙←arg max𝜙O𝜙
⊲Update the decoder 𝑞𝜙
8:
ˆY𝐸,𝐵←arg max𝑞𝜙(Y𝐸,𝐵|G) ⊲Generate pseudo-labels
9:
M-Step: Backbone Optimization
10:
𝑤𝑡,𝜏
𝑢
←𝑓TW(𝑡,𝑢,𝜏,𝑇𝑢,𝛾) ⊲Compute temporal weights
11:
𝜃←arg max𝜃O𝜃
⊲Update backbone 𝑝𝜃
12:
𝜏←𝜏+ 1
⊲Update iteration counter
13: until Converged
14:
ˆY𝐹,𝐴←arg max𝑞𝜙(Y𝐹,𝐴|G)
⊲Final prediction
15: return ˆY𝐹,𝐴
B
Dataset Details
Due to the lack of widely studied datasets for the label-limited
dynamic node classification, we utilize three existing datasets
that align closely with our task and propose a new dataset CoOAG
specifically designed for this problem. The statistics of four datasets
are introduced in Table 4.
Table 4: Dataset Statistics
Wikipedia
Reddit
Dsub
CoOAG
Nodes
9, 227
10, 984
150, 000
9, 559
Edges
15, 7474
67, 2447
16, 8154
11, 4337
Duration
1 month
1 month
1 year
22 years
Total classes
2
2
2
5
Bipartite
!
!
%
%
Node Feat Dim
–
–
34
384
Edge Feat Dim
172
172
1
384
B.1
Previous datasets
We use three publicly available datasets and do the preprocessing
to adapt them to our task:
B.1.1
Dsub. Description: Dsub is a subgraph of the Dgraph [24]
dataset, which is a financial fraud detection dataset where nodes
represent users, and edges represent emergency contact relation-
ships between users. Node labels indicate whether a user is ul-
timately identified as fraudulent (failing to repay loans over an
extended period). Node features are derived from user metadata.
In addition to confirmed fraudulent and non-fraudulent labels, the
dataset includes background nodes that lack sufficient information
for labeling but are retained to maintain graph connectivity.
Table 5: Label Distributions of CoOAG
Label Distribution
ROB
2,845 (29.64%)
CV
1,700 (17.71%)
NLP
1,652 (17.21%)
AI/ML
1,971 (20.53%)
DM/WS
1,431 (14.91%)
Preprocessing: To facilitate efficient training, we extract a sub-
graph called Dsub using Breadth-First Search (BFS), ensuring that
the subgraph remains connected and preserves the original label
distribution.
B.1.2
Wikipedia. Description: Wikipedia [29] is a bipartite in-
teraction graph that records edits on Wikipedia pages over one
month. Nodes represent users and pages, and edges denote editing
behaviors with timestamps. Each edge is associated with a 172-
dimensional Linguistic Inquiry and Word Count (LIWC) feature.
The dataset includes dynamic labels indicating whether users are
temporarily banned from editing [66].
Task Adaptation: To simulate real-world scenarios where only
the final labels are available, we split the dynamic labels into Y𝐹,𝐵
(final labels) and Y𝐸,𝐵(unobserved labels). During training, only
Y𝐹,𝐵is used.
B.1.3
Reddit. Description: Reddit [29] is a bipartite graph that
records user posts under subreddits over one month. Nodes repre-
sent users and subreddits, and edges represent timestamped posting
requests. Each edge is associated with a 172-dimensional LIWC fea-
ture. The dataset includes dynamic labels indicating whether users
are banned from posting [66].
Task Adaptation: Similar to Wikipedia, we split the dynamic
labels into Y𝐹,𝐵and Y𝐸,𝐵, using only Y𝐹,𝐵for training to simulate
real-world conditions.
B.2
CoOAG
B.2.1
Description. : To advance research in this domain, we intro-
duce CoOAG, a novel dataset derived from the academic sphere,
inspired by the Coauthor CS and Coauthor Physics networks [50].
This dataset has undergone stringent quality control and temporal
consistency checks. Label distributions are detailed in Table 5.
The CoOAG dataset is constructed using the Microsoft Academic
Graph (MAG) portion from Open Academic Graph 2.1[51, 55, 67, 68],
with a focus on publications from leading AI conferences. The node
labels in CoOAG denote authors’ research interests, classified into
the following categories:
• CV (Computer Vision)
• NLP (Natural Language Processing)
• ROB (Robotics)
• DM/WS (Data Mining/Web Search)
• AI/ML (Other AI Fields)
B.2.2
Preprocessing. : We employ structured prompts with the
Qwen-Plus API [1, 63] to categorize research domains using pa-
per Fields of Study (FoS) and abstracts. The prompt template, as
illustrated in Listing B.2.2, encompasses:


**[Table p11.1]**
|  | Wikipedia Reddit Dsub CoOAG |
| --- | --- |
| Nodes Edges Duration Total classes Bipartite Node Feat Dim Edge Feat Dim | 9, 227 10, 984 150, 000 9, 559 15, 7474 67, 2447 16, 8154 11, 4337 1 month 1 month 1 year 22 years 2 2 2 5 ! ! % % – – 34 384 172 172 1 384 |

[CAPTION] Table 4: Dataset Statistics

[CAPTION] Table 5: Label Distributions of CoOAG


<!-- page 12 -->
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Trovato et al.
• Category definitions with canonical examples
• Strict output format constraints
• Weighted keyword matching logic
• Interactive classification examples
This approach achieves 98.3% ACC on 120 manually verified sam-
ples. Edge features are generated by concatenating paper metadata
and abstracts, encoded using the all-MiniLM-L12-v2 model. Node
features are computed as the average of all paper features for each
author. Conference submission deadlines determine edge times-
tamps. The classification workflow maintains temporal consistency
by processing papers in chronological order.
Research Domain Classification Prompt Template
Classify the author 's research domain into
↩→one of the following 5 categories
↩→based on the given field keywords
↩→and weights:
- 0: CV (Computer Vision)
- 1: NLP (Natural Language Processing)
- 2: ROB (Robotics)
- 3: DM/WS (Data Mining/Web Search)
- 4: AI/ML (Other AI Fields)
Input: Multiple field keywords with weights
Output requirements:
- **Format**: Directly return
↩→classification result (0-4)
- **Constraint**: Answer must be a single
↩→
digit without explanation
Examples:
- Input: "[ computer vision (0.53377)] [
↩→image filter (0.5337) ]"
- Output: 0
Input:
{fos_text}
C
Backbone Details
• TGAT [61] leverages a self-attention mechanism to simultane-
ously capture spatial and temporal dependencies. Initially, TGAT
combines the raw node feature n𝑢with a learnable time encoding
𝑧(𝑡), forming n𝑢(𝑡) = [n𝑢||𝑧(𝑡)], where 𝑧(𝑡) = cos(𝑡𝑤+ 𝑏). Sub-
sequently, self-attention is applied to generate the representation
of node 𝑢at time 𝑡0, denoted as h𝑡0𝑢= SAM(n𝑢(𝑡0), {n𝑣(𝑚𝑣) |
𝑣∈𝑁𝑡0 (𝑢)}). Here, 𝑁𝑡0 (𝑢) represents the set of neighbors of
node 𝑢at time 𝑡0, and 𝑚𝑣indicates the timestamp of the most
recent interaction involving node 𝑣. Finally, predictions for any
node pair (𝑢, 𝑣) at time 𝑡0 are obtained via MLP([h𝑡0𝑢||h𝑡0𝑣]).
• TCL [57] adopts a contrastive learning framework. To construct
interaction sequences for each node, TCL employs a breadth-
first search algorithm on the temporal dependency subgraph. A
graph transformer is then utilized to learn node representations
by jointly considering graph topology and temporal dynamics.
Additionally, TCL integrates a cross-attention mechanism to
model the interdependencies between interacting nodes.
• TGN [48] combines RNN-based and self-attention-based tech-
niques. TGN maintains a memory module to store and update
the state 𝑠𝑢(𝑡) of each node 𝑢, which serves as a compact repre-
sentation of 𝑢’s historical interactions. Given the memory up-
dater as mem, when an edge 𝑒𝑢𝑣(𝑡) connecting nodes 𝑢and 𝑣
is observed, the memory state of node 𝑢is updated as 𝑠𝑢(𝑡) =
mem(𝑠𝑢(𝑡−),𝑠𝑣(𝑡−)||e𝑡𝑢,𝑣), where 𝑠𝑢(𝑡−) denotes the memory
state of 𝑢just prior to time 𝑡, and e𝑡𝑢,𝑣represents the edge feature.
The memory updater mem is implemented using a recurrent
neural network (RNN). Node embeddings h𝑡𝑢are computed by
aggregating information from the 𝐿-hop temporal neighborhood
through self-attention.
• GraphMixer [9] introduces a simple yet effective MLP-based ar-
chitecture. Instead of relying on trainable time encodings, Graph-
Mixer utilizes a fixed time encoding function, which is integrated
into a link encoder based on MLP-Mixer to process temporal
links. A node encoder with neighbor mean-pooling is employed
to aggregate node features. Specifically, for each node 𝑢, Graph-
Mixer computes its embedding h𝑡𝑢by summarizing the features
of its neighbors within the temporal context.
• DyGFormer [66] employs a self-attention mechanism to model
dynamic graphs. For a given node 𝑢, DyGFormer retrieves the
features of its involved neighbors and edges to represent their
encodings. It incorporates a neighbor co-occurrence encoding
scheme, which captures the frequency of each neighbor’s ap-
pearance in the interaction sequences of both the source and
destination nodes, thereby explicitly exploring pairwise correla-
tions. Rather than operating at the interaction level, DyGFormer
divides the interaction sequences of each source or destination
node into multiple patches, which are then processed by a trans-
former to compute node embeddings h𝑡𝑢.
D
Implementation Details
We use PyTorch [41], scikit-learn [42], PyTorch Geometric [15],
DyGLib [66] library to implement our proposed framework FLiD.
We conduct experiments on two clusters: (1) 4×Tesla V100 (32GB
memory) using 16-core CPUs and 395GB RAM; (2) 8×2080Ti (11GB
memory) using 12-core CPUs and 396GB RAM.
E
Hyperparameters
We optimize all methods across all models using the Adam optimizer
[27], with cross-entropy loss as the objective function. Initially, we
warm up all backbones through link prediction tasks [29]. Subse-
quently, the entire models are trained for 100 epochs, employing an
early stopping strategy with a patience of 15. For consistency, we set
the learning rate to 0.0001 and the batch size to 200 across all meth-
ods and datasets. To ensure robustness and minimize deviations,
we conduct five independent runs for each method with random
seeds ranging from 0 to 4 and report the average performance [66].
E.1
Model Configurations
Here, we present the configurations for each model (Table 6): TGAT,
TGN, TCL, GraphMixer, and DyGFormer, all of which remain con-
sistent across datasets.


<!-- page 13 -->
PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 6: Model Configurations Comparison
Hyperparameter
TGAT
TGN
TCL
GraphMixer
DyGFormer
Time encoding dim
100
100
100
100
100
Output dim
172
172
172
172
172
Attention heads
2
2
2
–
2
Graph conv layers
2
1
–
–
–
Transformer layers
–
–
2
–
2
MLP-Mixer layers
–
–
–
2
–
Node memory dim
–
172
–
–
–
Depth encoding dim
–
–
172
–
–
Co-occurrence dim
–
–
–
–
50
Aligned encoding dim
–
–
–
–
50
Memory updater
–
GRU
–
–
–
Time gap 𝑇
–
–
–
2000
–
E.2
PTCL Hyperparameters
Here, we present the hyperparameters of PTCL (Table 7): 𝛽is a
hyperparameter that balances the weight of pseudo-labels and final
timestamp labels; 𝛾is a hyper-parameter that controls the rate of
Temporal Curriculum fdecay. Note that TGN runs out of memory
on Dsub due to its high space cost.
Table 7: Hyperparameters of PTCL
Model
Hyperparameters
Wikipedia
Reddit
Dsub
CoOAG
TGAT
𝛽
0.9
0.9
0.7
0.2
𝛾
0.8
0.1
0.2
0.9
TCL
𝛽
0.1
0.9
0.1
0.8
𝛾
0.6
0.9
0.5
0.6
TGN
𝛽
0.9
0.9
–
0.9
𝛾
0.05
0.01
–
0.1
GraphMixer
𝛽
0.5
0.5
0.5
0.5
𝛾
1.3
0.1
0.1
0.4
DyGFormer
𝛽
0.7
0.5
0.1
0.1
𝛾
0.01
0.01
0.5
0.1

[CAPTION] Table 6: Model Configurations Comparison

[CAPTION] Table 7: Hyperparameters of PTCL