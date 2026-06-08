<!-- page 1 -->
PromptEM: Prompt-tuning for Low-resource Generalized
Entity Matching
Pengfei Wang, Xiaocan Zeng, Lu Chen, Fan Ye, Yuren Mao, Junhao Zhu, Yunjun Gao
Zhejiang University, Hangzhou, China
{wangpf,zengxc,luchen,fan.ye,maoyuren,zhujunhao,gaoyj}@zju.edu.cn
ABSTRACT
Entity Matching (EM), which aims to identify whether two entity
records from two relational tables refer to the same real-world
entity, is one of the fundamental problems in data management.
Traditional EM assumes that two tables are homogeneous with the
aligned schema, while it is common that entity records of different
formats (e.g., relational, semi-structured, or textual types) involve
in practical scenarios. It is not practical to unify their schemas due
to the different formats. To support EM on format-different entity
records, Generalized Entity Matching (GEM) has been proposed
and gained much attention recently. To do GEM, existing methods
typically perform in a supervised learning way, which relies on
a large amount of high-quality labeled examples. However, the
labeling process is extremely labor-intensive, and frustrates the use
of GEM. Low-resource GEM, i.e., GEM that only requires a small
number of labeled examples, becomes an urgent need. To this end,
this paper, for the first time, focuses on the low-resource GEM and
proposes a novel low-resource GEM method, termed as PromptEM.
PromptEM has addressed three challenging issues (i.e., designing
GEM-specific prompt-tuning, improving pseudo-labels quality, and
running efficient self-training) in low-resource GEM. Extensive
experimental results on eight real benchmarks demonstrate the
superiority of PromptEM in terms of effectiveness and efficiency.
PVLDB Reference Format:
Pengfei Wang, Xiaocan Zeng, Lu Chen, Fan Ye, Yuren Mao, Junhao Zhu,
Yunjun Gao. PromptEM: Prompt-tuning for Low-resource Generalized
Entity Matching. PVLDB, 16(1): XXX-XXX, 2022.
doi:XX.XX/XXX.XX
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at
https://github.com/ZJU-DAILY/PromptEM.
1
INTRODUCTION
Entity Matching (EM) aims to identify whether two entity records
from two relational tables refer to the same real-world entity, which
is one of the fundamental and significant tasks in data management.
Most existing solutions [23, 30, 31] assume that two tables are ho-
mogeneous with the aligned schema. However, there is an urgent
need to generalize the entity matching problem for more practi-
cal scenarios. Taking paper matching as an example, as shown in
This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of
this license. For any use beyond those covered by this license, obtain permission by
emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights
licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 16, No. 1 ISSN 2150-8097.
doi:XX.XX/XXX.XX
title
venue
year
efficient similarity search and 
classification via rank aggregation
{
  "title": "efficient similarity search 
and classification via rank aggregation",
  "year": 2003,
  "authors": ["ronald fagin", "ravi 
kumar ", "d. sivakumar"]
}
 "...... On the practical side, it turns 
out to be extremely efficient, often 
exploring  no ......  Our experiments 
include two scenarios where nearest 
neighbors are typically employed‐‐‐
similarity search and classification 
problems ...... "
Semi‐Structured
Textual
Relational
authors
ronald fagin , 
ravi kumar , 
d. sivakumar
2003
International 
conference on 
management of data
Figure 1: An example of generalized entity matching.
Figure 1, paper metadata is usually stored in relational tables or
semi-structured JSON objects, while paper description (e.g., ab-
stract) is textual data. It is not practical to unify their schemas
since we need a potentially expensive schema matching in the pre-
processing step [28], which is even not applicable when matching
data of different formats (e.g., matching paper metadata stored in
tabular format with their textual abstracts). Therefore, traditional
EM is unable to support those practical scenarios like paper match-
ing [50]. Recently, TDmatch [1] first attempts to match structured
data and textual data while having extremely poor scalability (to be
analyzed via experiments in Section 5.4). To support more practical
application scenarios, following the previous study [50], we study
Generalized Entity Matching (GEM), which allows matching entity
records of relational, semi-structured, or textual types.
Existing methods designed for GEM typically perform in a su-
pervised learning way, which relies on a large amount of labeled
training data, and thus, is extremely labor-intensive. Recent stud-
ies [23, 30] have achieved considerable performance by leveraging
the power of pre-trained language models (LMs) and the fine-tuning
paradigm. Nonetheless, the fine-tuning paradigm still requires a
non-trivial amount of high-quality labeled examples (e.g., thou-
sands of labels for a typical GEM application [50]). TDmatch [1]
performs in an unsupervised learning way via graph creation and
random walk. However, two drawbacks restrict its real-world appli-
cations as confirmed via experiments in Section 5: (i) it has unstable
performance due to the absence of labeled samples; and (ii) random
walk is not scalable on large datasets [44], incurring enormous exe-
cution time and memory usage (e.g., more than 120 hours and 130
gigabytes for once training of the SEMI-REL). To satisfy real-life ap-
plications, a low-resource (i.e., using only a small number of labeled
examples) but an effective and efficient solution is required. To the
best of our knowledge, low-resource GEM remains unexplored.
To overcome the low-resource dilemma, semi-supervised learn-
ing techniques (e.g., self-training [46]) are good choice. Self-training
has recently been shown to obtain state-of-the-art performance
arXiv:2207.04802v2  [cs.DB]  16 Jul 2022


**[Table p1.1]**
| title | authors |  | venue | year |
| --- | --- | --- | --- | --- |
| efficient similarity search and classification via rank aggregation | ronald fagin , ravi kumar , d. sivakumar |  | International conference on management of data | 2003 |
| Relational { "...... On the practical side, it turns "title": "efficient similarity search out to be extremely efficient, often and classification via rank aggregation", exploring no ...... Our experiments "year": 2003, include two scenarios where nearest "authors": ["ronald fagin", "ravi neighbors are typically employed‐‐‐ kumar ", "d. sivakumar"] similarity search and classification } problems ...... " |  |  |  |  |
| { "title": "efficient similarity search and classification via rank aggregation", "year": 2003, "authors": ["ronald fagin", "ravi kumar ", "d. sivakumar"] } |  | "...... On the practical side, it turns out to be extremely efficient, often exploring no ...... Our experiments include two scenarios where nearest neighbors are typically employed‐‐‐ similarity search and classification problems ...... " |  |  |

[CAPTION] Figure 1: An example of generalized entity matching.


<!-- page 2 -->
for low-resource tasks like sequence generation [18] and speech
recognition [52]. In self-training, a teacher model is trained on some
labeled data, and is used to produce pseudo-labels on unlabeled
data. Furthermore, the original labeled data is augmented with the
pseudo-label data, and is employed to train a student model. Thus,
a large amount of unlabeled data can be utilized effectively and the
requirement of labeled data is correspondingly reduced. Although
self-training has achieved promising performance in a wide range
of applications, it has not been explored in GEM.
Motivated by the above considerations, we study the problem
of learning high-quality models for low-resource GEM by means
of self-training. Our goal is to develop an effective and efficient
low-resource GEM solution based on pre-trained LMs along with
leveraging self-training to boost performance, which is a challeng-
ing endeavor. The challenges are mainly three-folds:
Challenge I: How to tune pre-trained LMs for GEM better? Despite
the success of fine-tuning LMs for the matching problem, some re-
cent studies [16, 24, 25] find that there is a significant gap between
objective forms in pre-training and fine-tuning, which restricts
taking full advantage of knowledge in LMs. Pre-training is usually
formalized as a cloze-style task to predict target words (e.g., masked
language models). However, the approaches based on fine-tuning
add additional layers to do different objective forms (e.g., classifica-
tion and generation) as illustrated in Figure 2. To do GEM, existing
approaches treat it as a classification problem. This gap hinders the
transfer and adaptation of knowledge in LMs for GEM tasks.
Challenge II: How to select high-quality pseudo-labels? The quality
of pseudo-labels determines whether self-training can improve
performance. Thus, the pseudo-label selection strategy is extremely
important. A common strategy is using confidence to select pseudo-
labels. However, this strategy has some serious drawbacks [32, 37].
On the one hand, incorrect predictions can have high confidence
scores in poorly calibrated networks. On the other hand, if we only
aim at the pseudo-labels with high confidence produced by the
teacher, there is little to gain for the student model.
Challenge III: How to avoid expensive self-training? Traditional
self-training can be costly. To be more specific, the labeled data is
augmented by the pseudo-labels produced by the teacher model,
which may be beneficial to performance but result in a longer
training time. Intuitively, maybe not all training data contribute
to boosting the performance of the student model. Nevertheless,
how to quantify the importance of training data to avoid expensive
self-training is still challenging.
To tackle the above three challenges, we propose a low-resource
GEM solution PromptEM. Prompt-tuning is a new promising para-
digm in natural language processing, and is able to bridge the gap
of objective forms between pre-training and fine-tuning [24, 39]. To
address the gap between pre-training and fine-tuning (CI), we
cast GEM as a cloze-style task via designing the GEM-specific
prompt-tuning, which has the same objective form as pre-training.
Thus, we can stimulate the rich knowledge distributed in LMs
through prompt-tuning. To select high-quality pseudo-labels (CII),
we develop a lightweight uncertainty-aware self-training method
to boost performance. High-quality pseudo-labels are a prerequisite
for boosting performance of self-training. To this end, we employ re-
cent advances in Bayesian deep learning [12] to obtain uncertainty
estimates of the teacher model for pseudo-labeling and boosting
the self-training process. To avoid expensive self-training (CIII),
we prune useless training data dynamically using our proposed
MC-EL2N, making the self-training process more lightweight and
efficient. Our contributions are summarized as follows:
• Low-resource GEM. This is the first work formally studying the
problem of low-resource generalized entity matching. We articu-
late the importance of this problem in more practical scenarios.
• Prompt-tuning for GEM. We present PromptEM, a new GEM
solution based on prompt-tuning, which casts GEM as a cloze-
style task. To the best of our knowledge, PromptEM is the first
GEM (EM) solution that stimulates the rich knowledge distributed
in LMs via designing GEM-specific prompt-tuning.
• Generic Lightweight Self-training. To further improve the perfor-
mance in low-resource settings, we develop a generic lightweight
self-training method, which selects pseudo-labels using uncer-
tainty and makes self-training more lightweight and efficient by
dynamic data pruning.
• Extensive Experiments. We conduct comprehensive experimental
evaluation on GEM tasks compared against state-of-the-art ap-
proaches, using eight real-world datasets from various domains.
Extensive experimental results demonstrate the superiority of
our proposed PromptEM in terms of effectiveness and efficiency.
Outline. Section 2 presents the problem definition and overviews
preliminaries. We introduce prompt-tuning for GEM in Section 3.
We further improve the performance by lightweight self-training
in Section 4. Section 5 presents the experimental results. Finally,
we discuss related work in Section 6 and conclude in Section 7.
2
PRELIMINARIES
In this section, we first present the problem definition of generalized
entity matching (GEM). Next, we introduce the serializing method
for GEM, followed by an introduction of conventional vanilla fine-
tuning and prompt-based tuning with LMs.
2.1
Problem Formulation
Given two collections of data entries, entity matching (EM) is to
identify pairs of data entries that refer to the same real-world en-
tity. A classic EM workflow [20] has two main steps: blocking and
matching. The blocking [42] typically uses simple heuristics or deep
learning techniques to reduce the quadratic number of candidates.
The matching identifies whether each candidate pair is a real match
or not. In this paper, we focus on the matching. Formally, given
two tables 𝐸𝐴and 𝐸𝐵, we assign a binary label 𝑦∈{0, 1} for each
candidate pair (𝑒𝑎,𝑒𝑏) ∈𝐸𝐴× 𝐸𝐵. Here, 𝑦= 1 denotes a truly
matched pair, while 𝑦= 0 represents a mismatched pair.
To generalize the classic setting to more practical application
scenarios, Machamp [50] comes up with the new research problem,
generalized entity matching (GEM). GEM can support a variety of
matching tasks with practical applications.
Problem 1. Generalized Entity Matching (GEM). Given two struc-
tured, semi-structured, or unstructured entity tables 𝐸𝐴and 𝐸𝐵with
homogeneous or heterogeneous schema, GEM is to assign a binary
label 𝑦∈{0, 1} for each candidate (𝑒𝑎,𝑒𝑏) ∈𝐸𝐴× 𝐸𝐵.
2.2
Serializing
The matching problem can be effectively solved by formulating
it as a sequence classification task [13, 23, 50]. First, entity pairs
2


<!-- page 3 -->
Prompt‐tuning
They
are
[MASK]
[CLS]  serialize(e)   serialize(e')
Input
MLM Head
matched
mismatched
Class Set
yes
no
Tokens in the vocabulary
Learnable continuous tokens
Fine‐tuning
[CLS] serialize(e)  [SEP]  serialize(e')  [SEP]
CLS Head
Class Set
yes
no
Template
Label Words
[SEP]
Objective Forms
classification
generation
......
Figure 2: The illustration of fine-tuning and prompt-tuning. The blue rectangles in the figure are special prompt tokens, whose
parameters are initialized and learnable during prompt-tuning.
are serialized to sequences, and then, a pre-trained LM is fine-
tuned to solve the task. Existing methods are designed for EM over
structured data with homogeneous data, which are not suitable for
GEM. Following [50], we extend the serialization method presented
in Ditto [23] and introduce a reasonable way to fulfill this task.
Structured tables. For structured tables, an entity with𝑛attributes
can be denoted as 𝑒= {attr𝑖, val𝑖}𝑖∈[1,𝑛], where attr𝑖is the attribute
name and val𝑖is the corresponding attribute value. Then the serial-
ization is denoted as:
𝑠𝑒𝑟𝑖𝑎𝑙𝑖𝑧𝑒(𝑒) ::= [COL] attr1 [VAL] val1 . . . [COL] attrn [VAL] valn
where [COL] and [VAL] are two special tags indicating the start
of attribute names and values respectively. Taking the relational
entity in Figure 1 as an example, we serialize it as:
[COL] title [VAL] efficient similarity . . . [COL] authors [VAL]
renald . . . [COL] venue [VAL] SIGMOD [COL] year [VAL] 2003
Semi-structured tables. The semi-structured tables can be serial-
ized in a similar way. Specially, two differences exist: (i) For nested
attributes, we recursively add the [COL] and [VAL] tags along with
attribute names and values in each level of nests. (ii) To reduce the
length of the sequence while ensuring the amount of information,
we concatenate the elements in the list into one string for attributes
whose content is a list. As an example, given the semi-structured
entity in Figure 1, we serialize it as:
[COL] title [VAL] efficient similarity . . . [COL] year [VAL] 2003
[COL] authors [VAL] ronald fagin ravi kumar d. sivakumar
Unstructured tables. Unstructured textual entities are sequences
originally, and hence, there does not need to serialize them.
2.3
Vanilla Fine-tuning
Pre-trained language models (LMs) (e.g., BERT [6] and RoBERTa [26])
have demonstrated powerful semantic expression abilities, which
can support lots of downstream tasks (e.g., classification and ques-
tion answering). Formally, GEM can be treated as a sequence pair
classification task denoted as T = {X, Y}, where X is a candidate
pair set and Y is a class set. For each instance 𝑥∈X, it is seri-
alized by 𝑥::= [CLS] serialize(𝑒)[SEP] serialize(𝑒′)[SEP], and is
annotated with a label 𝑦∈Y. Here, [CLS] and [SEP] are special
tokens used to mark the beginning and end of a sequence.
Given a LM M, vanilla fine-tuning first coverts 𝑥to the input se-
quence {[CLS],𝑤1, . . . ,𝑤𝑛, [SEP],𝑤′1, . . . ,𝑤′𝑚, [SEP]}, and then,
it uses M to encode all tokens of the input sequence into corre-
sponding vectors
n
h[CLS], h𝑤1, . . . , h𝑤𝑛, h[SEP], h𝑤′
1, . . . , h𝑤′𝑚, h[SEP]
o
,
where 𝑤𝑖is the token and h𝑤𝑖is the corresponding embedding. For
a downstream classification task (e.g., GEM), a task-specific head
is trained to predict the probability distribution over the label set
𝑦with the softmax function 𝑝(· | 𝑥) = Softmax
 
W × h[CLS] + b
 
.
Here, h[CLS] is the embedding of special classification token [CLS],
𝑏is the bias for the layer, and W is a learnable matrix randomly
initialized before fine-tuning. The parameters of M, 𝑏, and W are
tuned to maximize
1
|X|
Í
𝑥∈X log𝑝(𝑦| 𝑥).
2.4
Prompt-based Tuning
Prompt-based tuning has been proposed to apply cloze-style tasks to
tune LMs. Formally, we define a label word set V𝑦= {𝑤1, . . . ,𝑤𝑚}.
V𝑦is a subset of the vocabulary V of the LM M, i.e., V𝑦⊆V.
We get an overall dictionary V∗by taking the union of the dictio-
nary corresponding to each label. Another primary component of
prompt-tuning is a prompt template 𝑇(·), which modifies the origi-
nal input 𝑥into a prompt input 𝑇(𝑥) by adding a set of additional
tokens in 𝑥. Generally, a token [MASK] is added for LMs to predict
the missing label word 𝑤∈V∗. Thus, in prompt-tuning, a classi-
fication problem is transferred into a masked language modeling
problem 𝑝(𝑦∈Y | 𝑥) = 𝑝 [MASK] = 𝑤∈V𝑦| 𝑇(𝑥) .
3
PROMPT TUNING FOR GEM
In this section, we detail how to utilize prompt-tuning to deal with
GEM. We first design GEM-specific prompt templates and label
words, and then, we describe the training and inference process.
3.1
Prompt Templates
To cast the GEM problem as a prompt-tuning one, we first design
suitable prompt templates (i.e., hard-encoding templates and con-
tinuous templates) and label words set (to consider general binary
relationship).
Hard-encoding templates. For the choice of hard-encoding tem-
plates, we do not use automatic searching methods for discrete
prompts since the GEM task is clearly defined and the prompts
are easily purposeful. Given each candidate pair 𝑥= (𝑒,𝑒′), we
construct the following templates:
T1(𝑥) = 𝑠𝑒𝑟𝑖𝑎𝑙𝑖𝑧𝑒(𝑒) 𝑠𝑒𝑟𝑖𝑎𝑙𝑖𝑧𝑒(𝑒′) They are [MASK]
T2(𝑥) = 𝑠𝑒𝑟𝑖𝑎𝑙𝑖𝑧𝑒(𝑒) is [MASK] to 𝑠𝑒𝑟𝑖𝑎𝑙𝑖𝑧𝑒(𝑒′)
Continuous templates. As prompt construction is to find a method
that allows a LM to effectively perform a task, rather than being
for human consumption, it is not necessary to limit the prompt
to human-interpretable natural language [24]. Thus, continuous
prompts are proposed to perform prompting directly in the embed-
ding space of the model. Here, we employ P-tuning [25], where
continuous prompt tokens are learned by inserting trainable vari-
ables into the embedding input. Specifically, trainable prompt to-
kens are initialized, and then, BiLSTM [15] is utilized to account
for interaction between prompt tokens. This enables the model to
3

[CAPTION] Figure 2: The illustration of fine-tuning and prompt-tuning. The blue rectangles in the figure are special prompt tokens, whose


<!-- page 4 -->
find better continuous prompts beyond the original vocabulary V
of M could express. We give an illustrative example in Figure 2.
Label words set. In addition to designing templates, another pri-
mary component is to design the label words set. Note that, tradi-
tional EM tasks find pairs of entities that are identical [50]. How-
ever, GEM might require finding out entity pairs satisfying a gen-
eral binary relationship. Taking paper matching as an example,
our goal is to find pairs between paper metadata and abstracts.
Indeed, the relationship between them is whether they are rele-
vant, which is more general beyond matching. Considering gen-
eral binary relationship, we map the label 𝑦= yes into a set
V𝑦= {matched, similar, relevant}. Similarly, the label set for label
𝑦= no is V𝑦= {mismatched, different, irrelevant}.
3.2
Training and Inference
A classification problem is transferred into a masked language
modeling problem via prompt-tuning. In masked language model-
ing, we use confidence scores of all the words in V𝑦to construct
the final score of the particular class 𝑦. Given a candidate pair 𝑥
(which is mapped to 𝑇(𝑥)) and its class 𝑦(which is mapped to
V𝑦= {𝑤1, . . . ,𝑤𝑚}), the conditional probability is computed as:
𝑃(𝑦| 𝑥) = 1
𝑚
𝑚
∑︁
𝑗
𝑃 [MASK] = 𝑤𝑗| 𝑇(𝑥) 
(1)
Training. The continuous prompt tokens can be parameterized by
𝜙and optimized along with M during training. We tune the pre-
trained model M (parameterized by 𝜃) along with the additional
prompt embeddings by using the cross-entropy loss function L =
−Í log 𝑃(𝑦| 𝑥;𝜃,𝜙). Here, we prompt-tune a pre-trained LM for
the GEM task as follows:
(1) Design task-specific prompt templates and label words set.
(2) Initialize the network with parameters from the pre-trained LM
and continuous prompt tokens.
(3) Train the network on the training set until convergence.
Different tasks require different prompt templates and label
words. Step (1) is specifically designed for GEM tasks. Continu-
ous prompt tokens in Step (2) are specifically designed to enable
the model to find better prompts beyond V of M could express.
Inference. For inference, we aim to assign a label for the input,
which can directly use Eq. 1 to predict the class of the current input
instance based on predicted words of the [MASK] position.
4
LIGHTWEIGHT SELF-TRAINING
With prompt-tuning, we can stimulate the rich knowledge dis-
tributed in LMs, which achieves considerable performance under
low-resource settings. To further improve the performance and
avoid expensive self-training, we develop a generic lightweight
self-training method.
4.1
Overview
Let 𝐷𝐿=
n 
𝑥(𝑖),𝑦(𝑖) o𝑁𝐿
𝑖=1 and 𝐷𝑈=
n
𝑥(𝑖)o𝑁𝑈
𝑖=1 be a labeled dataset
with 𝑁𝐿samples and an unlabeled dataset with 𝑁𝑈samples, respec-
tively. Our lightweight self-training aims to boost the performance
(i.e., effectiveness) using uncertainty meanwhile being more effi-
cient and lightweight than traditional self-training via dynamic
data pruning. We describe the lightweight self-training procedure,
with its pseudo code presented in Algorithm 1. Given a labeled
Algorithm 1: Lightweight Self-training
Input: the number Iter of iterations, a labeled train set 𝐷𝐿,
an unlabeled train set 𝐷𝑈
Output: a student model M𝜃,𝜙
1 for 𝑖←1 to Iter do
2
Initialize a new teacher model M𝑡,𝜃,𝜙,𝑖
3
for 𝑒𝑝𝑜𝑐ℎ←1 to Epochs of teacher do
4
Train M𝑡,𝜃,𝜙,𝑖using the train set 𝐷𝐿
5
⊲Uncertainty-aware Pseudo-label Selection
6
𝐷𝑃←Select pseudo-labels from 𝐷𝑈
7
𝐷𝑈←𝐷𝑈−𝐷𝑃
8
𝐷𝐿←𝐷𝐿∪𝐷𝑃
9
Initialize a new student model M𝑠,𝜃,𝜙,𝑖
10
for 𝑒𝑝𝑜𝑐ℎ←1 to Epochs of student do
11
Train M𝑠,𝜃,𝜙,𝑖using the train set 𝐷𝐿
12
⊲Dynamic Data Pruning
13
if (𝑒𝑝𝑜𝑐ℎmod frequency of pruning ) = 0 then
14
𝐷𝐷←Select useless samples from 𝐷𝐿
15
𝐷𝐿←𝐷𝐿−𝐷𝐷
16 return the best student model M𝜃,𝜙
dataset 𝐷𝐿, a teacher model M𝑡is initialized and trained on 𝐷𝐿
until convergence (Lines 2-4). Then the teacher model M𝑡produces
pseudo-labels on 𝐷𝑈. After that, we introduce an uncertainty-aware
pseudo-label selection strategy to select high-quality pseudo-labels
𝐷𝑃(Lines 5-6). Meanwhile, 𝐷𝑈and 𝐷𝐿are updated (Lines 7-8).
Next, a student model M𝑠is initialized and trained on the updated
𝐷𝐿(Lines 9-11). To make self-training more lightweight and ef-
ficient, we present a dynamic data pruning strategy, which can
prune useless samples and their labels in 𝐷𝐿every fixed number
of epochs (Lines 12-15). Finally, we choose the best student model
with the best performance on the validation set (Line 16). Since
LST is general enough to incorporate with other approaches, it is
possible to be widely used in practical low-resource applications.
4.2
Uncertainty-aware Pseudo-label Selection
Selecting high-quality pseudo-labels is a prerequisite for boost-
ing self-training performance. Therefore, we aim at reducing the
noise present in the selected samples to improve the overall perfor-
mance. A straightforward way to select pseudo-labels is by select-
ing samples with high-confidence predictions. However, incorrect
predictions can have high confidence scores in poorly calibrated
networks [37]. Besides, if the teacher model already predicts some
samples with high confidence, then these is little to gain for student
model with these samples [32]. Based on the observation that pre-
diction uncertainties can be leveraged to negate the effect of poor
calibration [37], we employ an uncertainty-aware pseudo-label se-
lection strategy. Formally, uncertainty can be divided into epistemic
uncertainty and aleatoric uncertainty [46]. The former comes from
uncertainty in the parameters of the model, and the latter is uncer-
tainty inherent in the data (e.g., two samples of different classes are
similar). We focus on quantifying epistemic uncertainty. Inspired by
[32, 37], we use MC-Dropout [12] to obtain an uncertainty measure
by calculating the standard deviation of a fixed number (e.g., 10 in
4


<!-- page 5 -->
Table 1: Statistics of the datasets used in our experiments.
"All" denotes the total number of labeled samples, and
"Train" represents the number of training samples used in
our default low-resource setting.
Datasets
Domain
Left Table
Right Table
Labeled Examples
#row #attr #row #attr
All
% rate Train
REL-HETER
restaurant
534
6.00
332
7.00
567
10%
57
SEMI-HOMO
citation
2,616
8.65 64,263 7.34 17,223
5%
861
SEMI-HETER
book
22,133 12.28 23,264 12.03 1,240
10%
124
SEMI-REL
movie
29,180 8.00 32,823 13.81 1,309
10%
131
SEMI-TEXT-w
product
9,234 10.00 9,234
1.00
5,540
10%
554
SEMI-TEXT-c
product
20,897 10.00 20,897 1.00 12,538
5%
627
REL-TEXT
citation
2,616
1.00
2,295
6.00
7,417
10%
742
GEO-HETER geo-spatial 2,469
5.00
2,788
4.00
2,500
10%
250
our experiments) of stochastic forward passes. To avoid carefully
chosen thresholds, we choose 𝑁𝑃samples with the least uncertainty
after calculating the uncertainties of 𝐷𝑈:
𝐷𝑃=
n 
𝑥(𝑖), ˜𝑦(𝑖) o𝑁𝑃
𝑖=1 = Top - 𝑁𝑃(𝐷𝑈| −𝑢(𝑖))
(2)
Here, 𝑁𝑃= 𝑁𝑈· 𝑢𝑟, 𝑢(𝑖) is the uncertainty of the sample, ˜𝑦(𝑖)
is the pseudo-label produced by the teacher model, and 𝑢𝑟is the
proportion of the unlabeled samples. The time complexity of the
uncertainty estimation is 𝑂(|𝑁𝑃| × 𝑙𝑜𝑔(|𝑁𝑈|)). Uncertainty-aware
pseudo-label selection makes the self-training process more effec-
tive, which will be analyzed in Section 5.5.
4.3
Dynamic Data Pruning
Traditional self-training can be expensive as the training set grows,
resulting in more training time. Recently, Paul et al. [35] show
that the Error L2-Norm (EL2N) score can identify important ex-
amples early in training. In other words, it can prune significant
fractions of useless training data without sacrificing test accuracy,
which can reduce the number of the training set and training time.
Inspired by EL2N [35] and MC-Dropout [12], we combine these
approaches and propose MC-EL2N, which is able to quantify im-
portance scores more stably. Formally, the MC-EL2N score of a
training sample (𝑥,𝑦) is defined as
Í𝑛
𝑖=1 ||M(𝑥)−𝑦||2
𝑛
, where 𝑛is the
number of stochastic forward passes. Similarly, to avoid carefully
chosen thresholds, we choose 𝑁𝐷samples with the least MC-EL2N
score after quantifying the importance of 𝐷𝐿:
𝐷𝐷=
n 
𝑥(𝑖),𝑦(𝑖) o𝑁𝐷
𝑖=1 = Top - 𝑁𝐷(𝐷𝐿| −𝑒(𝑖))
(3)
Here, 𝑁𝐷= 𝑁𝐿·𝑒𝑟, 𝑒(𝑖) is the MC-EL2N score of the sample, and 𝑒𝑟
is the proportion of the labeled samples. Similar to the uncertainty
estimation, this process can be computed efficiently in 𝑂(|𝑁𝐷| ×
𝑙𝑜𝑔(|𝑁𝐿|)) time. We prune the useless samples every fixed epochs
using dynamic data pruning, making the self-training process more
lightweight and efficient. We will confirm the efficiency of dynamic
data pruning, to be presented in Section 5.4.
5
EXPERIMENTS
In this section, we experimentally evaluate the proposed PromptEM
on eight real-world datasets. We aim at answering the following
research questions:
• RQ1: How does PromptEM perform compared with the state-of-
the-art methods under low-resource settings?
• RQ2: How does each module affect the overall performance of
the model?
• RQ3: How does PromptEM perform compared with state-of-the-
art approaches in terms of efficiency?
• RQ4: Why do we choose these key modules (i.e., continuous
templates and uncertainty-aware pseudo-label selection)?
5.1
Experimental Setup
Dataset. We use all the seven real-world benchmark datasets with
different structures from Machamp [50] and one geospatial dataset
(GEO-HETER) [2]. The detailed GEO-HETER construction can be
found in our online version1. The statistics of datasets are summa-
rized in Table 1. Each dataset consists of the left and right tables of
entities with possibly different formats (i.e., relational (REL) format,
semi-structured (SEMI) format, or textual (TEXT) format). When
they are of the same format, they can have a homogeneous (HOMO)
or heterogeneous (HETER) schema. We use rate% of labeled data
as training set (e.g., 57 labeled data for REL-HETER), and use the
same train/valid/test splits as Machamp.
Baselines. We compare PromptEM with eight SOTA EM methods,
among which three (i.e., Ditto, DADER, and Rotom) have made ef-
forts to low-resource EM, and TDmatch is an unsupervised matching
method for structural and textual data.
• DeepMatcher [31] is an entity matching framework that uses
RNN architecture to aggregate the attribute values and then align
the aggregated representations of the attributes.
• BERT [6] is fine-tuned to treat GEM as a sequence pair classifi-
cation task.
• SentenceBERT [36] proposes a siamese architecture for pre-
trained LMs for sentence matching, and could also be applied to
the task of GEM.
• Ditto [23] is the SOTA EM approach that fine-tunes a pre-trained
LM with three optimizations (i.e., domain knowledge, TF-IDF
summarization, and data augmentation).
• DADER [45] presents a transfer learning based EM framework
via domain adaptation.
• Rotom [30] proposes a meta-learning framework that selects
and weights the augmented data to better fine-tune the LMs.
• TDmatch [1] is an unsupervised approach to match textual and
structured data using graph creation and random walk. Further-
more, we build an MLP classifier on top of its embeddings to
perform in the supervised setting, called TDmatch*.
Implementation details. We implement PromptEM in PyTorch [34],
the Transformers library [51] and the OpenPrompt library [8]. We
use RoBERTa-base [26] as the backbone structure of our model in
all the experiments. Unless particularly specified, the experimen-
tal results are conducted under the low-resource setting shown in
Table 1. We further apply the half-precision floating-point (fp16)
optimization to save the GPU memory usage and running time.
In all the experiments, the max sequence length is set to 512; the
learning rate is set to 2e-5; the batch size is set to 32; the number of
iterations for self-training is set to 1; and the number of passes for
MC-Dropout is set to 10. We use AdamW as the optimizer for train-
ing, fix the epochs of training the teacher model to 20, and set the
epochs of training the student model to 30. We prune the train set
for every 8 epochs. We tune the hyper-parameters by doing a grid
search and selecting the one with the best performance. Specifically,
the continuous template is selected from {T1(·), T2(·)},𝑢𝑟is selected
1https://arxiv.org/pdf/2207.04802.pdf
5


**[Table p5.1]**
| Datasets | Domain | Left Table Right Table #row #attr #row #attr | Labeled Examples All % rate Train |
| --- | --- | --- | --- |
| REL-HETER SEMI-HOMO SEMI-HETER SEMI-REL SEMI-TEXT-w SEMI-TEXT-c REL-TEXT GEO-HETER | restaurant citation book movie product product citation geo-spatial | 534 6.00 332 7.00 2,616 8.65 64,263 7.34 22,133 12.28 23,264 12.03 29,180 8.00 32,823 13.81 9,234 10.00 9,234 1.00 20,897 10.00 20,897 1.00 2,616 1.00 2,295 6.00 2,469 5.00 2,788 4.00 | 567 10% 57 17,223 5% 861 1,240 10% 124 1,309 10% 131 5,540 10% 554 12,538 5% 627 7,417 10% 742 2,500 10% 250 |

[CAPTION] Table 1: Statistics of the datasets used in our experiments.

[CAPTION] Table 1. We further apply the half-precision floating-point (fp16)


<!-- page 6 -->
Table 2: Results of all the methods under the default low-resource setting.
Methods
REL-HETER
SEMI-HOMO SEMI-HETER
SEMI-REL
SEMI-TEXT-c SEMI-TEXT-w
REL-TEXT
GEO-HETER
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
DeepMatcher
0.0
0.0
0.0
74.6 72.9 73.8 78.3 22.6 35.1 70.1 44.8 54.7 23.0 39.6 29.1 23.5
1.9
3.5
36.9 17.1 23.4 28.9 90.9 43.8
BERT
100 90.9 95.2 90.1 93.2 91.6 63.6 17.6 27.6
92
94.5 93.3 56.9 47.2 51.6 19.6 20.9 20.2 27.1 26.6 26.9 70.8 90.4 79.4
SentenceBERT
100 90.9 95.2 92.6 93.6 93.1 81.5 13.8 23.7 83.2 100 90.8 60.0 51.3 55.3 26.2 21.3 23.5 36.4 52.0 42.9 74.5 72.1 73.3
Ditto
100 86.4 92.7 90.2 90.3 90.2 79.3 14.5 24.5 88.0 88.5 88.3 56.8 47.1 51.5 29.5 31.3 30.3 34.7 50.5 41.1 74.7 87.2 80.5
DADER
81.8 81.8 81.8 81.5 91.4 86.2 98.4 37.7 54.6 87.6 96.2 91.7 15.0 87.4 25.6 11.4 100
20.5 26.1 64.6 37.2 54.2 92.7 68.4
Rotom
100 77.3 87.2 89.2 94.3 91.7 83.3 15.7 26.5 95.8 88.0 91.7 68.0 54.6 60.5 43.6 34.1 38.3 51.9 45.5 48.5 76.7 78.5 77.6
TDmatch
56.4 100 72.1 93.7 42.0 58.0 97.2 88.1 92.4 97.5 85.8 91.3 69.0 10.4 18.0 42.3 14.2 21.3 80.2 47.3 59.5 72.8 73.0 72.9
TDmatch*
10.0
4.6
6.3
80.2 87.3 83.6 37.5 18.9 25.1 66.5 77.1 71.4 42.7 30.2 35.4 32.0 23.2 26.9 48.6 40.3 44.1 51.0 51.0 51.0
PromptEM
100 100 100 94.2 94.1 94.2 93.9 57.9 71.6 91.4 98.9 95.0 80.6 65.5 72.3 44.9 37.9 41.1 61.2 61.5 61.4 78.8 89.9 84.0
PromptEM w/o PT
100 95.5 97.7 90.5 94.8 93.3 39.5 52.2 45.0 83.8 53.6 65.3 56.3 55.3 55.8 31.5 19.0 23.7 22.1 55.9 31.7 79.3 85.1 82.1
PromptEM w/o LST
100 100 100 94.2 94.1 94.2 91.7 27.7 42.5 91.4 98.9 95.0 73.4 59.1 65.5 35.8 31.8 33.7 58.0 60.4 59.2 76.4 90.4 82.8
PromptEM w/o DDP 100 100 100 93.0 94.8 93.9 80.5 57.2 66.9 92.1 95.1 93.6 79.0 68.9 73.6 47.3 33.2 39.0 55.5 63.3 59.2 85.0 88.1 86.5
Table 3: Results of all the methods under the extremely challenging low-resource setting.
Methods
REL-HETER
SEMI-HOMO SEMI-HETER
SEMI-REL
SEMI-TEXT-c SEMI-TEXT-w
REL-TEXT
GEO-HETER
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
DeepMatcher
32.3 45.5 37.7 46.9 22.5 30.4 28.6 40.3 33.4 43.0 92.9 58.8
0.0
0.0
0.0
28.6
1.0
1.8
9.1
0.2
0.4
29.1 100 45.1
BERT
100 95.5 97.7 85.9 82.4 85.8 78.8 16.4 27.1 97.0 71.6 82.4 21.2 31.4 25.3 18.4
8.5
11.7 18.6
9.9
12.9 45.8 78.5 57.8
SentenceBERT
95.5 95.5 95.5 86.8 73.3 79.5 100 14.5 25.3 71.0 74.9 72.9 20.8 13.8 16.6 23.1
8.5
12.5 15.5
4.1
6.4
53.8 64.8 58.8
Ditto
100 81.8 90.0 80.9 81.8 81.4 95.2 12.6 22.2 78.7 95.1 86.1 13.9 100 24.3 12.6 69.7 21.4 18.0 99.3 30.4 33.0 85.4 47.6
DADER
88.9 72.7 80.0 75.9 86.6 80.9 47.1 65.0 54.6 94.0 86.3 90.0 41.7
0.9
1.7
12.4
7.1
9.0
60.2 11.9 19.9 63.0 87.9 73.4
Rotom
100 95.5 97.7 80.1 93.5 86.2 77.8 17.6 28.7 96.3 85.8 90.8 23.5 31.4 26.9 22.0
5.2
8.4
19.8 23.9 21.6 72.0 78.3 75.0
TDmatch
56.4 100 72.1 93.7 42.0 58.0 97.2 88.1 92.4 97.5 85.8 91.3 69.0 10.4 18.0 42.3 14.2 21.3 80.2 47.3 59.5 72.8 73.0 72.9
TDmatch*
11.1
9.1
10.0 37.8 27.8 32.0 37.8 17.6 24.0 47.8 75.4 58.5 16.3
5.2
7.9
19.0
7.1
10.3 20.0 14.4 16.8 36.7 33.6 35.1
PromptEM
100 100 100 86.1 92.2 89.0 93.9 28.9 44.2 94.0 94.5 94.3 40.8 29.0 33.9 15.7 46.9 23.6 26.5 50.2 34.7 78.0 81.9 79.9
from {0.05, 0.10, 0.15, 0.20, 0.25}, and 𝑒𝑟is selected from {0.1, 0.2, 0.3,
0.4, 0.5}. We select the epoch with the highest F1-score on the vali-
dation set, and report the values of precision, recall, and F1-score
on the test set. All the experiments are conducted on a machine
with an Intel Xeon Silver 4216 CPU, an NVIDIA A100 GPU and
512GB memory. We use the same serializing method as PromptEM
to implement each baseline method and report the performance
under their optimal settings. We present the implementation of
baselines in our online version.
Evaluation metrics. Following related studies [13, 23, 45], we
employ three widely-used classification metrics, namely, precision
(P), recall (R), F1-score (F).
5.2
Main Results (RQ1)
Results under the default low-resource setting. We first verify
the performance under low-resource setting of PromptEM using
the above eight baselines. The benchmark results of all methods
across the datasets are reported in Table 2. DeepMatcher achieves
the worst performance, since it does not leverage the recent ad-
vances in pre-trained LMs. Existing low-resource EM approaches
(i.e., Ditto, DADER, and Rotom) achieve relatively poor perfor-
mance, because the GEM problem is more intractable than EM (e.g.,
heterogeneous tables). In particular, TDmatch is not stable across
different datasets due to the absence of label guidance, which can
achieve the best F1-score on SEMI-HETER but only 18.0 F1-score on
SEMI-TEXT-c. TDmatch outperform other LM-based approaches
on SEMI-HETER. The reason is that SEMI-HETER has lots of nu-
meric attributes, i.e., 53% attribute values are digits. It is well known
that LMs are not good at understanding digits [47]. Also, we find
that the scalability of TDmatch is extremely poor, which will be
confirmed in Section 5.4. We can observe that TDmatch performs
better than TDmatch* in most cases. This is because TDmatch is
specifically designed for unsupervised learning, not necessarily
suitable for supervised learning. Besides, we have a similar finding
as those studies [21, 22]: BERT can be generalized to the specific
domain, and hence, BERT based methods (including PromptEM)
achieve better performance on GEO-HETER.
Effectiveness to different low-resource settings. We reduce
the training rate from 25% to 5% to see the performance under dif-
ferent low-resource settings. Experimental results are depicted in
Figure 3. We observe that PromptEM achieves SOTA performance
in most cases, while TDmatch and DADER achieve unstable results
across different datasets due to lacking the guidance of labels and
the heterogeneity of datasets. We also evaluate methods in a more
challenging setting, i.e., the number of available data for training is
only 80 for all the datasets. This setting is extremely challenging
for supervised methods, e.g., only using 0.46% labeled examples
on SEMI-HOMO. As shown in Table 3, PromptEM achieves SOTA
performance on most datasets, which demonstrates the great ro-
bustness of PromptEM compared to baselines. Moreover, it also
shows the outstanding scalability of PromptEM, which achieves
considerable performance only using a small number of labeled
data examples.
Overall, our PromptEM is superior to all the baselines in almost
all the cases under various low-resource settings. As mentioned
in Challenge I, there is a significant gap between objective forms
in pre-training and fine-tuning. This gap hinders the transfer and
adaptation of knowledge in LMs for GEM tasks, which restricts
taking full advantage of knowledge in LMs. Prompt-tuning is a
new promising paradigm in natural language processing, and is
able to bridge the gap of objective forms between pre-training and
fine-tuning [16, 24]. Thus, we can stimulate the rich knowledge
distributed in LMs through designing GEM-specific prompt-tuning
6


**[Table p6.1]**
| Methods | REL-HETER P R F | SEMI-HOMO P R F | SEMI-HETER P R F | SEMI-REL P R F | SEMI-TEXT-c P R F | SEMI-TEXT-w P R F | REL-TEXT P R F | GEO-HETER P R F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepMatcher BERT SentenceBERT | 0.0 0.0 0.0 100 90.9 95.2 100 90.9 95.2 | 74.6 72.9 73.8 90.1 93.2 91.6 92.6 93.6 93.1 | 78.3 22.6 35.1 63.6 17.6 27.6 81.5 13.8 23.7 | 70.1 44.8 54.7 92 94.5 93.3 83.2 100 90.8 | 23.0 39.6 29.1 56.9 47.2 51.6 60.0 51.3 55.3 | 23.5 1.9 3.5 19.6 20.9 20.2 26.2 21.3 23.5 | 36.9 17.1 23.4 27.1 26.6 26.9 36.4 52.0 42.9 | 28.9 90.9 43.8 70.8 90.4 79.4 74.5 72.1 73.3 |
| Ditto DADER Rotom | 100 86.4 92.7 81.8 81.8 81.8 100 77.3 87.2 | 90.2 90.3 90.2 81.5 91.4 86.2 89.2 94.3 91.7 | 79.3 14.5 24.5 98.4 37.7 54.6 83.3 15.7 26.5 | 88.0 88.5 88.3 87.6 96.2 91.7 95.8 88.0 91.7 | 56.8 47.1 51.5 15.0 87.4 25.6 68.0 54.6 60.5 | 29.5 31.3 30.3 11.4 100 20.5 43.6 34.1 38.3 | 34.7 50.5 41.1 26.1 64.6 37.2 51.9 45.5 48.5 | 74.7 87.2 80.5 54.2 92.7 68.4 76.7 78.5 77.6 |
| TDmatch TDmatch* | 56.4 100 72.1 10.0 4.6 6.3 | 93.7 42.0 58.0 80.2 87.3 83.6 | 97.2 88.1 92.4 37.5 18.9 25.1 | 97.5 85.8 91.3 66.5 77.1 71.4 | 69.0 10.4 18.0 42.7 30.2 35.4 | 42.3 14.2 21.3 32.0 23.2 26.9 | 80.2 47.3 59.5 48.6 40.3 44.1 | 72.8 73.0 72.9 51.0 51.0 51.0 |
| PromptEM PromptEM w/o PT PromptEM w/o LST PromptEM w/o DDP | 100 100 100 100 95.5 97.7 100 100 100 100 100 100 | 94.2 94.1 94.2 90.5 94.8 93.3 94.2 94.1 94.2 93.0 94.8 93.9 | 93.9 57.9 71.6 39.5 52.2 45.0 91.7 27.7 42.5 80.5 57.2 66.9 | 91.4 98.9 95.0 83.8 53.6 65.3 91.4 98.9 95.0 92.1 95.1 93.6 | 80.6 65.5 72.3 56.3 55.3 55.8 73.4 59.1 65.5 79.0 68.9 73.6 | 44.9 37.9 41.1 31.5 19.0 23.7 35.8 31.8 33.7 47.3 33.2 39.0 | 61.2 61.5 61.4 22.1 55.9 31.7 58.0 60.4 59.2 55.5 63.3 59.2 | 78.8 89.9 84.0 79.3 85.1 82.1 76.4 90.4 82.8 85.0 88.1 86.5 |


**[Table p6.2]**
| Methods | REL-HETER P R F | SEMI-HOMO P R F | SEMI-HETER P R F | SEMI-REL P R F | SEMI-TEXT-c P R F | SEMI-TEXT-w P R F | REL-TEXT P R F | GEO-HETER P R F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepMatcher BERT SentenceBERT | 32.3 45.5 37.7 100 95.5 97.7 95.5 95.5 95.5 | 46.9 22.5 30.4 85.9 82.4 85.8 86.8 73.3 79.5 | 28.6 40.3 33.4 78.8 16.4 27.1 100 14.5 25.3 | 43.0 92.9 58.8 97.0 71.6 82.4 71.0 74.9 72.9 | 0.0 0.0 0.0 21.2 31.4 25.3 20.8 13.8 16.6 | 28.6 1.0 1.8 18.4 8.5 11.7 23.1 8.5 12.5 | 9.1 0.2 0.4 18.6 9.9 12.9 15.5 4.1 6.4 | 29.1 100 45.1 45.8 78.5 57.8 53.8 64.8 58.8 |
| Ditto DADER Rotom | 100 81.8 90.0 88.9 72.7 80.0 100 95.5 97.7 | 80.9 81.8 81.4 75.9 86.6 80.9 80.1 93.5 86.2 | 95.2 12.6 22.2 47.1 65.0 54.6 77.8 17.6 28.7 | 78.7 95.1 86.1 94.0 86.3 90.0 96.3 85.8 90.8 | 13.9 100 24.3 41.7 0.9 1.7 23.5 31.4 26.9 | 12.6 69.7 21.4 12.4 7.1 9.0 22.0 5.2 8.4 | 18.0 99.3 30.4 60.2 11.9 19.9 19.8 23.9 21.6 | 33.0 85.4 47.6 63.0 87.9 73.4 72.0 78.3 75.0 |
| TDmatch TDmatch* | 56.4 100 72.1 11.1 9.1 10.0 | 93.7 42.0 58.0 37.8 27.8 32.0 | 97.2 88.1 92.4 37.8 17.6 24.0 | 97.5 85.8 91.3 47.8 75.4 58.5 | 69.0 10.4 18.0 16.3 5.2 7.9 | 42.3 14.2 21.3 19.0 7.1 10.3 | 80.2 47.3 59.5 20.0 14.4 16.8 | 72.8 73.0 72.9 36.7 33.6 35.1 |
| PromptEM | 100 100 100 | 86.1 92.2 89.0 | 93.9 28.9 44.2 | 94.0 94.5 94.3 | 40.8 29.0 33.9 | 15.7 46.9 23.6 | 26.5 50.2 34.7 | 78.0 81.9 79.9 |

[CAPTION] Table 2: Results of all the methods under the default low-resource setting.

[CAPTION] Table 3: Results of all the methods under the extremely challenging low-resource setting.

[CAPTION] Figure 3. We observe that PromptEM achieves SOTA performance


<!-- page 7 -->
M3-tree
 DeepMatcher   
 BERT   
 SentenceBERT   
 Ditto   
 DADER   
 Rotom   
 TDmatch   
 TDmatch*   
 PromptEM
5
10
15
20
25
0
25
50
75
100
F1-score (%)
rate (%)
(a) REL-HETER
5
10
15
20
25
55
70
85
100
F1-score (%)
rate (%)
(b) SEMI-HOMO
5
10
15
20
25
0
25
50
75
100
F1-score (%)
rate (%)
(c) SEMI-HETER
5
10
15
20
25
40
60
80
100
F1-score (%)
rate (%)
(d) SEMI-REL
5
10
15
20
25
15
40
65
90
F1-score (%)
rate (%)
(e) SEMI-TEXT-c
5
10
15
20
25
0
25
50
75
F1-score (%)
rate (%)
(f) SEMI-TEXT-w
5
10
15
20
25
5
25
45
65
F1-score (%)
rate (%)
(g) REL-TEXT
5
10
15
20
25
30
50
70
90
F1-score (%)
rate (%)
(h) GEO-HETER
Figure 3: Comparison results under different low-resource settings (%).
[16, 24]. Recently, prompt-tuning has been applied to lots of tasks
such as machine translation, information extraction, question an-
swering, and so on [24]. In summary, prompt-tuning has the po-
tential to outperform fine-tuning for those tasks based on LMs.
5.3
Ablation Study (RQ2)
Next, we study the effectiveness of each module in PromptEM (i.e.,
prompt-tuning (PT), lightweight self-training (LST), dynamic data
pruning (DDP)) by comparing PromptEM with its variants without
the key module. The results are listed in Tables 2.
PromptEM vs. PromptEM w/o PT. PromptEM w/o PT denotes
that we fine-tune the LM instead of prompt-tuning. It is observed
that the use of prompt-tuning contributes to a large portion of
the performance gain. The F1-score drops 15.7% on average under
the low-resource setting. This confirms that prompt-tuning greatly
helps to stimulate the rich knowledge distributed in the LM.
PromptEM vs. PromptEM w/o LST. We use LST to boost the
performance under low-resource settings. We can observe that LST
can bring performance improvement in most cases. For example,
LST brings 6.8% improvement on SEMI-TEXT-c. Also notice that
LST brings relatively low improvement on some datasets. This is
attributes to the nature of the datasets, as it is relatively much
easier for PromptEM to achieve the extremely high performance,
e.g., 100% F1-score on REL-HETER.
PromptEM vs. PromptEM w/o DDP. We can observe that DDP
can prune useless training data without sacrificing test accuracy.
It is worth noting that DDP can prune training data while slightly
improving test accuracy in some datasets. This is because DDP
makes the model focus on important and useful training data.
5.4
Efficiency Analysis (RQ3)
We further explore the efficiency of our proposed PromptEM in
terms of training time and memory usage, and the results are pre-
sented in Table 4. Since it is common for methods to use a similar
strategy for evaluating the GEM results in the test set, we do not
report the test time of every evaluated approach. SBERT denotes
SentenceBERT, and PromptEM- represents PromptEM without
dynamic data pruning.
PromptEM vs. best baselines. Due to the limitation of space, we
report PromptEM with the other evaluated approaches that achieve
the best quality of GEM results in the corresponding categories,
i.e., the normal EM method SBERT, the low-resource EM approach
Rotom, and the unsupervised matching method TDmatch. We re-
port the GPU memory for the methods running on GPU and the
CPU memory for the method (i.e., TDmatch) running on CPU, re-
spectively. As observed, PromptEM needs more training time than
SBERT to obtain the SOTA results. This demonstrates a trade-off be-
tween the effectiveness and efficiency of the GEM problem. To sum
up, it is significant that spending a relatively longer time in achiev-
ing better matching results. Rotom requires two-stage training,
incurring a long training process. SBERT, Rotom, and PromptEM
need similar memory usage since they are all based on LMs. We
would like to emphasize that TDmatch needs too much training
time and memory usage, especially on relatively large datasets (e.g.,
120.3 hours and 131.5 Gigabytes on SEMI-REL), which is very costly
in real-world applications.
PromptEM vs. PromptEM-. We also compare PromptEM with
PromptEM- to evaluate the efficiency of dynamic data pruning. It
is observed that DDP greatly helps to reduce the training time, i.e.,
reduce 26.1% time on average. This is because the proposed MC-
EL2N is able to quantify useless training data effectively. Meanwhile,
DDP does not bring extra memory usage as it does not require any
new model parameters. As analyzed in Section 5.3, DDP does not
hurt the performance. This further demonstrates that PromptEM is
effective and efficient in solving the GEM problem.
5.5
Model Variants (RQ4)
Finally, we investigate the performance of PromptEM using alter-
native modules by conducting the following experiments.
Effect of template choices. Designing prompt templates is a pri-
mary component of prompt-tuning. We verify the effect of differ-
ent templates, i.e., continuous T1(·), hard-encoding T1(·), continu-
ous T2(·) and hard-encoding T2(·). Their average F1-scores on all
datasets are 74.4, 67.8, 77.0, and 74.5, respectively. Continuous tem-
plates achieve better performance than hard-encoding templates.
This further validates the effectiveness of the proposed continuous
7


**[Table p7.1]**
| enceB | ERTM 3-tree D | itto | DADER 100 %) 75 |  |
| --- | --- | --- | --- | --- |
|  |  |  | ( 50 F1-score 25 0 |  |

[CAPTION] Figure 3: Comparison results under different low-resource settings (%).


<!-- page 8 -->
Table 4: Efficiency comparison between PromptEM and its
competitors, including the running time and memory usage.
"s" denotes seconds, "m" denotes minutes, "h" denotes hours,
and "G" represents gigabytes. Due to the limitation of space,
we use the abbreviations of datasets.
Datasets
SBERT
Rotom
TDmatch
PromptEM- PromptEM
T.
M.
T.
M.
T.
M.
T.
M.
T.
M.
R-H
28.8s 22.9G 35.4s 32.8G 14.0m
6.2G
38.3s 27.4G 26.6s 27.4G
S-HO
2.4m 29.5G 8.2m 32.8G 51.0h 41.4G 11.5m 29.2G 7.4m 29.2G
S-HE
43.4s 24.9G 1.8m 35.8G 102.8h 105.3G 1.6m 29.0G 1.5m 29.0G
S-R
50.1s 35.8G 2.6m 32.8G 120.3h 131.5G 1.4m 30.6G 1.1m 30.6G
S-T-c
2.0m 36.3G 20.2m 32.8G 10.7h 25.4G 20.8m 29.2G 11.5m 29.2G
S-T-w
1.8m 35.6G 11.2m 32.8G 2.2h
12.9G 6.1m 29.2G 5.3m 29.2G
R-T
2.3m 34.4G 11.1m 29.7G 5.6h
50.5G 8.1m 30.6G 5.6m 30.6G
G-H
49.2s 27.9G 1.3m 32.8G 36.2m 16.7G 6.2m 30.6G 4.6m 30.6G
Table 5: Results of pseudo-label selection strategies.
Datasets
Uncertainty
Confidence
Clustering
TPR
TNR
TPR
TNR
TPR
TNR
REL-HETER
1
1
0.250
0.864
0.250
0.881
SEMI-HOMO
1
0.998
0.197
0.803
0.193
0.815
SEMI-HETER
1
0.963
0.979
0.985
0.350
0
SEMI-REL
1
1
0.426
0.583
0.432
0.602
SEMI-TEXT-c
0.969
1
0.113
0.897
0.128
0.879
SEMI-TEXT-w
0.333
0.967
0.056
0.928
0.114
0.912
REL-TEXT
0.910
0.966
0.194
0.820
0.148
0.846
GEO-HETER
0.867
1
0.644
0.758
0.236
0.738
templates, which can find better continuous prompts beyond the
original vocabulary V of M could express.
Effect of label words choices. We compare our designed label
words with a simple one (i.e., matched and mismatched). Using
continuous T1 and T2, our designed label words achieve +5.2%
and +9.4% average F1-score improvements over the simple one,
respectively. This confirms the effectiveness of our designed label
words. In other words, considering the more general relationship
between entities is beneficial to the predictions.
Pseudo-label selection strategies. We consider several pseudo-
label selection strategies, including uncertainty [37], confidence,
and clustering [9]. We fix 𝑢𝑟to 0.1 on all datasets. Similarly, confi-
dence and clustering both select the samples whose scores are in
the top 10%. Following [13], we use true-positive rate (TPR) and
true-negative rate (TNR) to evaluate the quality of the pseudo-labels
generated by different strategies. Formally, TPR represents the pro-
portion of matched entity pairs that are correctly labeled, denoted
as
𝑇𝑃
𝑇𝑃+𝐹𝑁; TNR represents the proportion of mismatched pairs that
are correctly labeled, denoted as
𝑇𝑁
𝑇𝑁+𝐹𝑃. The results are reported
in Table 5. As expected, uncertainty can achieve state-of-the-art
performance when generating pseudo-labels, e.g., TPR and TNR are
0.88 and 0.99 on average, respectively. It confirms the effectiveness
of the uncertainty-aware pseudo-label selection strategy.
6
RELATED WORK
6.1
Entity Matching
Entity Matching (EM) is one of the fundamental and significant
tasks in data management. Many efforts have devoted to develop
effective approaches for EM, including rule-based methods [11,
41, 49], crowdsourcing-based methods [14, 29, 48], and traditional
ML-based methods [3, 5, 20, 38]. Recently, deep learning has been
used widely in EM, and achieved promising results. DeepER [10]
uses deep neural networks to extract features of entity pairs, and
then models EM as a binary classification task. DeepMatcher [31]
systematically describes a DL architecture, and designs a space of
DL solutions for EM. However, a lot of labeled training data are
still needed for those DL-based approaches, which is extremely
expensive in practice. To decrease the demand for high-quality
training data, Ditto [23] applies pre-trained language models to EM,
performing well with the help of some data augmentation (DA)
techniques. Rotom [30] effectively improves the performance of EM
tasks via combining multiple DA operators. DADER [45] develops
a framework that significantly advances EM by applying domain
adaptation. Some other attempts have also been made to enhance
the performance via information fusion [53], active learning [19, 33],
and transfer learning [27, 43, 54]. Nonetheless, these methods only
focus on EM tasks in low-resource scenarios but perform poorly
on GEM tasks. TDmatch [1] first attempts to match textual content
and structured data under an unsupervised setting. However, it has
one serious shortcoming: it is not scalable on large-scale datasets,
which makes it hard to be used in practical scenarios.
6.2
Prompt-tuning
Despite the success of fine-tuning pre-trained LMs [6, 26], the
huge objective form gap between pre-training and fine-tuning still
hinders the full use of pre-trained knowledge for downstream tasks
[24, 25]. The birth of GPT-3 [4] is the seminal work that stimulates
the development of prompt-tuning, which applies hand-encoding
prompts for tuning and achieves impressive performance on various
tasks, especially under the low-resource settings. Following GPT-3,
many hand-encoding prompts [7, 25] are widely explored. Recently,
automatic prompt search [40] and continuous prompts [17, 25] are
proposed to avoid labor-intensive prompt design and enhance the
expressiveness of the prompt. The burst of prompt-tuning has led
to significant advancement in many areas such as natural language
inference [25] and entity typing [7]. However, for the first time,
we introduce prompt-tuning in EM for the better usage of pre-
trained LMs. PromptEM provides a good connection to recent NLP
advancements with applications to the data management task.
7
CONCLUSIONS
In this paper, we study the problem of low-resource generalized
entity matching via our presented PromptEM. For the first time,
PromptEM introduces prompt-tuning to cast GEM as a cloze-style
task to bridge the gap between pre-training and fine-tuning. It is
non-trivial as prompt templates are not generic and need to be
specially designed for the task. To this end, we design GEM-specific
templates and label words set for the GEM problem. To further
improve the performance in low-resource settings, we develop a
generic lightweight self-training method using uncertainty, which is
a nice combination and can be efficient and lightweight by dynamic
data pruning. Extensive experimental results on eight real-world
datasets with different structures demonstrate the superiority of
PromptEM compared with the state-of-the-art approaches. In the
future, we plan to explore a general prompt-tuning method to sup-
port more data management tasks (e.g., data cleaning), advancing
the usage of LMs in the database community.
8


**[Table p8.1]**
| Datasets | SBERT T. M. | Rotom T. M. | TDmatch T. M. | PromptEM- T. M. | PromptEM T. M. |
| --- | --- | --- | --- | --- | --- |
| R-H S-HO S-HE S-R S-T-c S-T-w R-T G-H | 28.8s 22.9G 2.4m 29.5G 43.4s 24.9G 50.1s 35.8G 2.0m 36.3G 1.8m 35.6G 2.3m 34.4G 49.2s 27.9G | 35.4s 32.8G 8.2m 32.8G 1.8m 35.8G 2.6m 32.8G 20.2m 32.8G 11.2m 32.8G 11.1m 29.7G 1.3m 32.8G | 14.0m 6.2G 51.0h 41.4G 102.8h 105.3G 120.3h 131.5G 10.7h 25.4G 2.2h 12.9G 5.6h 50.5G 36.2m 16.7G | 38.3s 27.4G 11.5m 29.2G 1.6m 29.0G 1.4m 30.6G 20.8m 29.2G 6.1m 29.2G 8.1m 30.6G 6.2m 30.6G | 26.6s 27.4G 7.4m 29.2G 1.5m 29.0G 1.1m 30.6G 11.5m 29.2G 5.3m 29.2G 5.6m 30.6G 4.6m 30.6G |


**[Table p8.2]**
| Datasets | Uncertainty | Confidence | Clustering |
| --- | --- | --- | --- |
|  | TPR TNR | TPR TNR | TPR TNR |
| REL-HETER SEMI-HOMO SEMI-HETER SEMI-REL SEMI-TEXT-c SEMI-TEXT-w REL-TEXT GEO-HETER | 1 1 1 0.998 1 0.963 1 1 0.969 1 0.333 0.967 0.910 0.966 0.867 1 | 0.250 0.864 0.197 0.803 0.979 0.985 0.426 0.583 0.113 0.897 0.056 0.928 0.194 0.820 0.644 0.758 | 0.250 0.881 0.193 0.815 0.350 0 0.432 0.602 0.128 0.879 0.114 0.912 0.148 0.846 0.236 0.738 |

[CAPTION] Table 4: Efficiency comparison between PromptEM and its

[CAPTION] Table 5: Results of pseudo-label selection strategies.


<!-- page 9 -->
REFERENCES
[1] Naser Ahmadi, Hansjorg Sand, and Paolo Papotti. 2022. Unsupervised Matching
of Data and Text. In ICDE.
[2] Pasquale Balsebre, Dezhong Yao, Gao Cong, and Zhen Hai. 2022. Geospatial
Entity Resolution. In WWW. 3061–3070.
[3] Mikhail Bilenko and Raymond J Mooney. 2003. Adaptive duplicate detection
using learnable string similarity measures. In SIGKDD. 39–48.
[4] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language Models are Few-Shot Learners. In NeurIPS. 1877–
1901.
[5] William W Cohen and Jacob Richman. 2002. Learning to match and cluster large
high-dimensional data sets for data integration. In SIGKDD. 475–480.
[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert:
Pre-training of deep bidirectional transformers for language understanding. arXiv
preprint arXiv:1810.04805 (2018).
[7] Ning Ding, Yulin Chen, Xu Han, Guangwei Xu, Pengjun Xie, Hai-Tao Zheng,
Zhiyuan Liu, Juanzi Li, and Hong-Gee Kim. 2021. Prompt-learning for fine-
grained entity typing. arXiv preprint arXiv:2108.10604 (2021).
[8] Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen, Zhiyuan Liu, Hai-Tao Zheng,
and Maosong Sun. 2021. Openprompt: An open-source framework for prompt-
learning. arXiv preprint arXiv:2111.01998 (2021).
[9] Thomas Dopierre, Christophe Gravier, Julien Subercaze, and Wilfried Logerais.
2020. Few-shot pseudo-labeling for intent detection. In COLING. 4993–5003.
[10] Muhammad Ebraheem, Saravanan Thirumuruganathan, Shafiq Joty, Mourad
Ouzzani, and Nan Tang. 2018. Distributed representations of tuples for entity
resolution. PVLDB 11, 11 (2018), 1454–1467.
[11] Ahmed Elmagarmid, Ihab F Ilyas, Mourad Ouzzani, Jorge-Arnulfo Quiané-Ruiz,
Nan Tang, and Si Yin. 2014. NADEEF/ER: Generic and interactive entity resolution.
In SIGMOD. 1071–1074.
[12] Yarin Gal and Zoubin Ghahramani. 2016. Dropout as a bayesian approximation:
Representing model uncertainty in deep learning. In ICML. 1050–1059.
[13] Congcong Ge, Pengfei Wang, Lu Chen, Xiaoze Liu, Baihua Zheng, and Yunjun
Gao. 2021. CollaborEM: A Self-supervised Entity Matching Framework Using
Multi-features Collaboration. TKDE (2021).
[14] Chaitanya Gokhale, Sanjib Das, AnHai Doan, Jeffrey F Naughton, Narasimhan
Rampalli, Jude Shavlik, and Xiaojin Zhu. 2014. Corleone: Hands-off crowdsourc-
ing for entity matching. In SIGMOD. 601–612.
[15] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. 2013. Speech
recognition with deep recurrent neural networks. In ICASSP. 6645–6649.
[16] Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong
Qiu, Yuan Yao, Ao Zhang, Liang Zhang, et al. 2021. Pre-trained models: Past,
present and future. AI Open 2 (2021), 225–250.
[17] Xu Han, Weilin Zhao, Ning Ding, Zhiyuan Liu, and Maosong Sun. 2021. Ptr:
Prompt tuning with rules for text classification. arXiv preprint arXiv:2105.11259
(2021).
[18] Junxian He, Jiatao Gu, Jiajun Shen, and Marc’Aurelio Ranzato. 2019. Revisiting
Self-Training for Neural Sequence Generation. In ICLR.
[19] Jungo Kasai, Kun Qian, Sairam Gurajada, Yunyao Li, and Lucian Popa. 2019.
Low-resource Deep Entity Resolution with Transfer and Active Learning. In ACL.
5851–5861.
[20] Pradap Konda, Sanjib Das, AnHai Doan, Adel Ardalan, Jeffrey R Ballard, Han Li,
Fatemah Panahi, Haojun Zhang, Jeff Naughton, Shishir Prasad, et al. 2016. Mag-
ellan: toward building entity matching management systems over data science
stacks. PVLDB 9, 13 (2016), 1581–1584.
[21] David Krueger, Nicolas Ballas, Stanislaw Jastrzebski, Devansh Arpit, Maxinder S
Kanwal, Tegan Maharaj, Emmanuel Bengio, Asja Fischer, and Aaron C Courville.
2017. Deep Nets Don’t Learn via Memorization.. In ICLR (Workshop).
[22] Fei Li, Yonghao Jin, Weisong Liu, Bhanu Pratap Singh Rawat, Pengshan Cai,
Hong Yu, et al. 2019. Fine-tuning bidirectional encoder representations from
transformers (BERT)–based models on large-scale electronic health record notes:
an empirical study. JMIR medical informatics 7, 3 (2019), e14830.
[23] Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Tan.
2020. Deep entity matching with pre-trained language models. PVLDB 14, 1
(2020), 50–60.
[24] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Gra-
ham Neubig. 2021. Pre-train, prompt, and predict: A systematic survey of prompt-
ing methods in natural language processing. arXiv preprint arXiv:2107.13586
(2021).
[25] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and
Jie Tang. 2021. GPT understands, too. arXiv preprint arXiv:2103.10385 (2021).
[26] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A
robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692
(2019).
[27] Michael Loster, Ioannis Koumarelas, and Felix Naumann. 2021. Knowledge
transfer for entity resolution with siamese neural networks. JDIQ 13, 1 (2021),
1–25.
[28] Jayant Madhavan, Philip A Bernstein, and Erhard Rahm. 2001. Generic schema
matching with cupid. PVLDB 1, 2001 (2001), 49–58.
[29] Adam Marcus, Eugene Wu, David Karger, Samuel Madden, and Robert Miller.
2011. Human-powered sorts and joins. arXiv preprint arXiv:1109.6881 (2011).
[30] Zhengjie Miao, Yuliang Li, and Xiaolan Wang. 2021. Rotom: A meta-learned data
augmentation framework for entity matching, data cleaning, text classification,
and beyond. In SIGMOD. 1303–1316.
[31] Sidharth Mudgal, Han Li, Theodoros Rekatsinas, AnHai Doan, Youngchoon Park,
Ganesh Krishnan, Rohit Deep, Esteban Arcaute, and Vijay Raghavendra. 2018.
Deep learning for entity matching: A design space exploration. In SIGMOD.
19–34.
[32] Subhabrata Mukherjee and Ahmed Awadallah. 2020. Uncertainty-aware self-
training for few-shot text classification. NeurIPS, 21199–21212.
[33] Youcef Nafa, Qun Chen, Zhaoqiang Chen, Xingyu Lu, Haiyang He, Tianyi Duan,
and Zhanhuai Li. 2022. Active deep learning on entity resolution by risk sampling.
Knowledge-Based Systems 236 (2022), 107729.
[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019.
PyTorch: an imperative style, high-performance deep learning library. In NeurIPS.
8026–8037.
[35] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. 2021. Deep
Learning on a Data Diet: Finding Important Examples Early in Training. NeurIPS
34.
[36] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings
using siamese bert-networks. In EMNLP.
[37] Mamshad Nayeem Rizve, Kevin Duarte, Yogesh S Rawat, and Mubarak Shah.
2021. In defense of pseudo-labeling: An uncertainty-aware pseudo-label selection
framework for semi-supervised learning. In ICLR.
[38] Sunita Sarawagi and Anuradha Bhamidipaty. 2002. Interactive deduplication
using active learning. In SIGKDD. 269–278.
[39] Nikunj Saunshi, Sadhika Malladi, and Sanjeev Arora. 2020. A mathematical
exploration of why language models help solve downstream tasks. arXiv preprint
arXiv:2010.03648 (2020).
[40] Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer
Singh. 2020. AutoPrompt: Eliciting Knowledge from Language Models with
Automatically Generated Prompts. In EMNLP. 4222–4235.
[41] Rohit Singh, Venkata Vamsikrishna Meduri, Ahmed Elmagarmid, Samuel Madden,
Paolo Papotti, Jorge-Arnulfo Quiané-Ruiz, Armando Solar-Lezama, and Nan Tang.
2017. Synthesizing entity matching rules by examples. PVLDB 11, 2 (2017), 189–
202.
[42] Saravanan Thirumuruganathan, Han Li, Nan Tang, Mourad Ouzzani, Yash Govind,
Derek Paulsen, Glenn Fung, and AnHai Doan. 2021. Deep learning for blocking
in entity matching: a design space exploration. PVLDB 14, 11 (2021), 2459–2472.
[43] Saravanan Thirumuruganathan, Shameem A Puthiya Parambath, Mourad Ouz-
zani, Nan Tang, and Shafiq Joty. 2018. Reuse and Adaptation for Entity Resolution
through Transfer Learning. arXiv e-prints (2018), arXiv–1809.
[44] Hanghang Tong, Christos Faloutsos, and Jia-Yu Pan. 2006. Fast random walk
with restart and its applications. In ICDM. 613–622.
[45] Jianhong Tu, Ju Fan, Nan Tang, Peng Wang, Chengliang Chai, Guoliang Li, Ruixue
Fan, and Xiaoyong Du. 2022. Domain Adaptation for Deep Entity Resolution. In
SIGMOD. 443–457.
[46] Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. 2020. Un-
certainty estimation using a single deep deterministic neural network. In ICML.
9690–9700.
[47] Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, and Matt Gardner. 2019.
Do NLP Models Know Numbers? Probing Numeracy in Embeddings. In EMNLP.
5307–5315.
[48] Jiannan Wang, Tim Kraska, Michael J Franklin, and Jianhua Feng. 2012. CrowdER:
Crowdsourcing Entity Resolution. PVLDB 5, 11 (2012).
[49] Jiannan Wang, Guoliang Li, Jeffrey Xu Yu, and Jianhua Feng. 2011. Entity match-
ing: How similar is similar. PVLDB 4, 10 (2011), 622–633.
[50] Jin Wang, Yuliang Li, and Wataru Hirota. 2021. Machamp: A Generalized Entity
Matching Benchmark. In CIKM. 4633–4642.
[51] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al.
2019. Huggingface’s transformers: State-of-the-art natural language processing.
arXiv preprint arXiv:1910.03771 (2019).
[52] Qiantong Xu, Alexei Baevski, Tatiana Likhomanenko, Paden Tomasello, Alexis
Conneau, Ronan Collobert, Gabriel Synnaeve, and Michael Auli. 2021. Self-
training and pre-training are complementary for speech recognition. In ICASSP.
3030–3034.
[53] Zijun Yao, Chengjiang Li, Tiansi Dong, Xin Lv, Jifan Yu, Lei Hou, Juanzi Li, Yichi
Zhang, and Zelin Dai. 2021. Interpretable and Low-Resource Entity Matching
via Decoupling Feature Learning from Decision Making. In ACL. 2770–2781.
[54] Chen Zhao and Yeye He. 2019. Auto-em: End-to-end fuzzy entity-matching using
pre-trained deep models and transfer learning. In WWW. 2413–2424.
9


<!-- page 10 -->
APPENDIX
A
RESULTS UNDER SUFFICIENT RESOURCE
SETTING
As shown in Table 6, we observe that PromptEM achieves the best
F1-score for all datasets. PromptEM achieves an average 88.9% F1-
score, which is +8.0% relatively over the best baseline Rotom. This
attributes to the power of prompt-tuning. As mentioned in Section
1, prompt-based tuning is able to bridge the gap of objective forms
between pre-training and fine-tuning, which is good at stimulating
the rich knowledge distributed in LMs. PromptEM w/o PT denotes
that we fine-tune the LM instead of prompt-tuning. It is observed
that the use of prompt-tuning contributes to a large portion of the
performance gain. The F1-score drops 5.2% on average under the
sufficient setting when fine-tuning.
0
25
50
75
100
F1-score (%)
R-H
S-HO
S-HE
S-R
R-T
S-T-c
S-T-w
G-H
Figure 4: Effect of template choices. T1(·) and T2(·) denote
the continuous templates. T∗
1(·) and T∗
2(·) represent the hard-
encoding templates. We use the abbreviations of datasets.
T1(ꞏ) simple
T2(ꞏ)
T2(ꞏ) simple
T1(ꞏ)
0
25
50
75
100
F1-score (%)
R-H
S-HO
S-HE
S-R
R-T
S-T-c
S-T-w
G-H
Figure 5: Effect of label words choices. T1(·) simple and T2(·)
simple denote using the simple label words, i.e., matched for
"yes" and mismatched for "no". We use the abbreviations of
datasets.
B
MORE RESULTS OF PROMPT CHOICES
Effect of template choices. Figure 4 gives the experimental re-
sults of template choices. We have two findings: (i) T2(·) performs
better overall; (ii) As mentioned in Section 5.5, continuous templates
can bring performance improvement in most cases.
Effect of label words choices. The results of label words choices
are depicted in Figure 5. We compare our designed label words with
a simple one, i.e., matched for "yes" and mismatched for "no". Both
T1(·) and T2(·) use the continuous versions. We can observe that
the F1-score is significantly dropped using the simple label words.
This demonstrates the effectiveness of our designed label words. In
other words, considering the more general relationship between
entities is beneficial to the predictions.
C
ERROR ANALYSIS
We conduct an error analysis to verify the limitations of our pro-
posed PromptEM. Figure 6 plots the erroneous predictions, includ-
ing a false positive (FP) and a false negative (FN). For the FP case,
two entities have very similar titles and the same authors. Indeed,
we can find that they are different entities based on the publication
date and the ISBN. For the FN case, two entities have the same
titles but different authors and publishers. However, we can judge
they are the same based on the publication date and the ISBN. We
can find that some digital attributes are vital to making correct
predictions. Nonetheless, PromptEM and other LM-based methods
are not good at understanding digits, and hence, they cannot find
the importance of those digital attributes. Therefore, it is necessary
to improve the ability of LMs to understand and focus on important
digital attributes.
D
IMPLEMENTATION DETAILS OF
BASELINES
We implement each baseline method as follows and report its per-
formance under its optimal settings.
• DeepMatcher [31]: We implement DeepMatcher following the
original paper and public code2. As DeepMatcher only accepts
input with the same schema on two tables, we just make the
schemas of both tables have only one attribute, whose value is a
sentence consisting of all attribute values. And we use the default
setting of the hybrid model.
• BERT [6], SentenceBERT [36], Ditto [23] and Rotom [30]: These
methods are based on LMs. We implement these methods ac-
cording to the original papers, public code13 and public code24.
Following [50], we tune the hyper-parameters by doing a grid
search and select the one with the best performance. Specifically,
the learning rate is selected from {10−5, 3.0 × 10−5, 5.0 × 10−5}.
The maximum sequence length for LMs is selected from {128,
256, 384, 512}. The batch size and epoch are set to 32 and 40,
respectively.
• DADER [45]: We implement DADER following the original paper
and public code5. For the source dataset, we use all the training
samples. For the target dataset, we use the same low-resource
training samples as other supervised methods. Specifically, we
select the source and target datasets from a similar domain for
DADER. We use the same hyper-parameters setting according to
the paper. And we use the InvGAN+KD model.
• TDmatch [1]: We implement TDmatch according to the original
paper and public code6. All hyper-parameters are the same as the
original paper. To perform in the supervised setting, we extract
the embeddings from TDmatch. Then we build a classification
layer (i.e., MLP) upon the embeddings, whose name is TDmatch*.
Given two entity embeddings 𝑢and 𝑣, we use (𝑢, 𝑣, |𝑢−𝑣|,𝑢∗𝑣)
as input for the classifier. The batch size is set to 64; the number
of epoch is set to 100; and the learning rate is set to 5.0 × 10−3.
2https://github.com/anhaidgroup/deepmatcher
3https://github.com/megagonlabs/ditto
4https://github.com/megagonlabs/rotom
5https://github.com/ruc-datalab/DADER
6https://github.com/naserahmadi/TDmatch
10

[CAPTION] Figure 4: Effect of template choices. T1(·) and T2(·) denote

[CAPTION] Figure 5: Effect of label words choices. T1(·) simple and T2(·)


<!-- page 11 -->
Table 6: Results of all the methods under sufficient resource setting.
Methods
REL-HETER
SEMI-HOMO SEMI-HETER
SEMI-REL
SEMI-TEXT-c SEMI-TEXT-w
REL-TEXT
GEO-HETER
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
P
R
F
DeepMatcher
100 87.9 93.6 89.0 83.2 86.1 35.8 24.5 29.1 50.9 64.1 56.7 76.6 31.1 44.2 80.2 29.1 42.7 78.4 40.4 53.4 88.0 85.4 86.6
BERT
95.5 95.5 95.5 93.8 93.9 93.8 90.7 30.8 46.0 87.3 94.0 90.5 89.0 88.3 88.6 79.3 67.3 72.8 61.6 64.6 63.1 88.1 92.9 90.4
SentenceBERT
66.7 72.7 69.6 85.6 89.3 87.4 100 53.5 69.7 47.8 77.0 59.0 85.1 75.1 79.8 52.3 48.3 50.2 37.2 29.5 32.9 87.1 92.7 89.8
Ditto
100 100 100 94.7 91.6 93.1 84.6 48.4 61.6 95.8 86.9 91.1 82.2 81.3 81.8 63.6 66.3 64.9 65.6 60.1 62.7 92.1 88.1 90.1
DADER
87.0 90.9 88.9 87.4 87.4 87.4 98.5 40.3 57.1 87.6 96.7 92.0 13.9 100 24.4 11.4 100
20.5 28.3 54.1 37.2 80.7 79.6 80.2
Rotom
100 100 100 94.4 95.1 94.7 45.5 32.1 37.6 97.7 91.3 94.4 92.0 89.1 90.5 80.5 68.3 73.9 69.7 62.2 65.7 90.5 87.5 90.0
TDmatch
56.4 100 72.1 93.7 42.0 58.0 97.2 88.1 92.4 97.5 85.8 91.3 69.0 10.4 18.0 42.3 14.2 21.3 80.2 47.3 59.5 72.8 73.0 72.9
TDmatch*
50.0 50.0 50.0 86.5 91.2 88.8 57.9 41.5 48.4 77.6 98.4 86.8 77.7 72.7 75.1 74.7 58.8 65.8 60.3 52.9 56.4 72.9 78.3 75.5
PromptEM
100 100 100 96.5 96.0 96.3 99.3 87.4 93.0 96.7 95.6 96.2 93.4 90.9 92.1 80.4 70.1 74.9 66.7 66.0 66.4 92.5 92.7 92.6
PromptEM w/o PT 100 100 100 94.3 97.1 95.7 76.2 52.2 61.9 97.2 94.0 95.6 91.8 90.3 91.0 73.4 72.0 72.7 68.9 60.4 64.4 89.2 87.0 88.1
{
        "ID": "bn_2841",
        "Title": "Sams Teach Yourself SQL 
in 10 Minutes",
        "Author": "Ben Forta",
        "ISBN": 9780672336072,
        "Publisher": "Sams",
        "PublicationDate": "11/08/2012",
        "Pages": 288.0,
        "price": "$22.99",
        "ProductType": "Paperback"
}
{
        "id": "B860",
        "title": "Sams Teach Yourself SQL in 10 Minutes (3rd Edition)",
        "cover": "paperback",
        "pages": 256.0,
        "language": " English",
        "price": "$11.74",
        "authors": [ "Ben Forta"],
        "publication_info": {
            "publisher": " Sams Publishing; 3 edition (April 10, 2004)",
            "ISBN-10": "672325675",
            "ISBN13": 9780672325670.0
        }
}
False Positive
{
        "ID": "bn_1485",
        "Title": "Professional SQL Server 
2012 Internals and Troubleshooting",
        "Author": "Christian Bolton,  James 
Rowland-Jones,  Glenn Berry,  Justin 
Langford",
        "ISBN": 9781118177655,
        "Publisher": "Wiley",
        "PublicationDate": "11/06/2012",
        "Pages": 576.0,
        "price": "$44.78",
        "ProductType": "Paperback"
}
{
        "id": "B2573",
        "title": "Professional SQL Server 2012 Internals and 
Troubleshooting",
        "cover": "paperback",
        "pages": 576.0,
        "language": " English",
        "price": "$43.86",
        "authors": ["Christian Bolton", "Rob Farley", "Glenn Berry"],
        "publication_info": {
            "publisher": " Wrox; 1 edition (November 6, 2012)",
            "ISBN-10": "1118177657",
            "ISBN13": 9781118177655.0
        }
}
False Negative
Figure 6: An error analysis on SEMI-HETER.
E
DETAILS OF EXPERIMENTAL DATASETS
We use all seven real-world datasets from Machamp [50]. The two
tables of each dataset may have different formats (i.e., relational
(REL) format, semi-structured (SEMI) format, or textual (TEXT)
format) and different schemas (i.e., homogeneous (HOMO), or het-
erogeneous (HETER)).
Additionally, we use one real geo-spatial dataset GEO-HETER,
which is derived from OSM-FSQ-Pittsburgh [2]. For each entity,
there are textual attributes and geographical positions. Following
the previous study [50], the "latitude" and "longitude" of the right
table are combined into a single "position" attribute to convert the
dataset into a case of heterogeneous schema.
11


**[Table p11.1]**
| Methods | REL-HETER P R F | SEMI-HOMO P R F | SEMI-HETER P R F | SEMI-REL P R F | SEMI-TEXT-c P R F | SEMI-TEXT-w P R F | REL-TEXT P R F | GEO-HETER P R F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepMatcher BERT SentenceBERT | 100 87.9 93.6 95.5 95.5 95.5 66.7 72.7 69.6 | 89.0 83.2 86.1 93.8 93.9 93.8 85.6 89.3 87.4 | 35.8 24.5 29.1 90.7 30.8 46.0 100 53.5 69.7 | 50.9 64.1 56.7 87.3 94.0 90.5 47.8 77.0 59.0 | 76.6 31.1 44.2 89.0 88.3 88.6 85.1 75.1 79.8 | 80.2 29.1 42.7 79.3 67.3 72.8 52.3 48.3 50.2 | 78.4 40.4 53.4 61.6 64.6 63.1 37.2 29.5 32.9 | 88.0 85.4 86.6 88.1 92.9 90.4 87.1 92.7 89.8 |
| Ditto DADER Rotom | 100 100 100 87.0 90.9 88.9 100 100 100 | 94.7 91.6 93.1 87.4 87.4 87.4 94.4 95.1 94.7 | 84.6 48.4 61.6 98.5 40.3 57.1 45.5 32.1 37.6 | 95.8 86.9 91.1 87.6 96.7 92.0 97.7 91.3 94.4 | 82.2 81.3 81.8 13.9 100 24.4 92.0 89.1 90.5 | 63.6 66.3 64.9 11.4 100 20.5 80.5 68.3 73.9 | 65.6 60.1 62.7 28.3 54.1 37.2 69.7 62.2 65.7 | 92.1 88.1 90.1 80.7 79.6 80.2 90.5 87.5 90.0 |
| TDmatch TDmatch* | 56.4 100 72.1 50.0 50.0 50.0 | 93.7 42.0 58.0 86.5 91.2 88.8 | 97.2 88.1 92.4 57.9 41.5 48.4 | 97.5 85.8 91.3 77.6 98.4 86.8 | 69.0 10.4 18.0 77.7 72.7 75.1 | 42.3 14.2 21.3 74.7 58.8 65.8 | 80.2 47.3 59.5 60.3 52.9 56.4 | 72.8 73.0 72.9 72.9 78.3 75.5 |
| PromptEM PromptEM w/o PT | 100 100 100 100 100 100 | 96.5 96.0 96.3 94.3 97.1 95.7 | 99.3 87.4 93.0 76.2 52.2 61.9 | 96.7 95.6 96.2 97.2 94.0 95.6 | 93.4 90.9 92.1 91.8 90.3 91.0 | 80.4 70.1 74.9 73.4 72.0 72.7 | 66.7 66.0 66.4 68.9 60.4 64.4 | 92.5 92.7 92.6 89.2 87.0 88.1 |

[CAPTION] Table 6: Results of all the methods under sufficient resource setting.

[CAPTION] Figure 6: An error analysis on SEMI-HETER.


<!-- page 12 -->
Table 7: Recent research on task-specific prompt-tuning. Note that, prompt-tuning has not been introduced in the field of data
management, including the entity matching task.
Paper
Venue
Task
GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks
SIGKDD 2022
Graph Neural Network
Ontology-enhanced Prompt-tuning for Few-shot Learning
WWW 2022
Knowledge Graph Completion
Adversarial Soft Prompt Tuning for Cross-Domain Sentiment Analysis
ACL 2022
Sentiment Analysis
Continual Prompt Tuning for Dialog State Tracking
ACL 2022
Dialogue System
MSP: Multi-Stage Prompting for Making Pre-trained Language Models Better Translators
ACL 2022
Machine Translation
The Power of Prompt Tuning for Low-Resource Semantic Parsing
ACL 2022
Semantic Parsing
PTAU: Prompt Tuning for Attributing Unanswerable Questions
SIGIR 2022
Question Answering
Selective Fairness in Recommendation via Prompts
SIGIR 2022
Recommendation Sysytem
Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt Tuning
SIGIR 2022
Relation Extraction
Learning Transferable Visual Models From Natural Language Supervision
ICML 2021
Image Classification
F
SUMMARIZING LONG ENTRIES
When the textual data is an extremely long string, it is harder
for the LM to understand (e.g., the input to BERT can have at
most 512 sub-word tokens). A common practice is to truncate the
sequences. Nevertheless, the truncation strategy is not a wise choice
because the important information for matching is usually not at
the beginning of the sequences. Inspired by Ditto [23], we apply
a TF-IDF based summarization technique for textual data in our
implementation, which retains non-stopword tokens with high
TF-IDF scores.
G
INSIGHTS OF METHOD CHOICES
We would like to emphasize that some parts of the PromptEM are
nice combination of existing methods as each module has our in-
sights. We detail our insights for the method choices as follows: (i)
Why prompt-tuning? Existing SOTA EM methods based on LMs
achieve considerable performance. They fine-tune LMs to convert
EM to the sequence pair classification task while it is a suboptimal
design (Challenge I, Section 1, Page 2). To tune pre-trained LMs for
GEM better, we introduce the powerful prompt-tuning to bridge
the gap between the pre-training and the fine-tuning. (ii) Why
uncertainty? The quality of pseudo-labels determines whether
self-training can improve performance. A common strategy is using
confidence to select pseudo-labels. However, this strategy has some
serious drawbacks, e.g., incorrect predictions can have high confi-
dence scores in poorly calibrated networks (Challenge II, Section
1, Page 2). To select high-quality pseudo-labels, we employ recent
advances in Bayesian deep learning to obtain uncertainty estimates
of the teacher model for pseudo-labeling and boosting the self-
training process. (iii) Why dynamic data pruning? Uncertainty-
aware self-training is attractive since it can improve performance.
However, the self-training process can be costly. To be more spe-
cific, the labeled data is augmented by the pseudo-labels produced
by the teacher model, which may be beneficial to performance but
result in a longer training time (Challenge III, Section 1, Page 2).
Intuitively, maybe not all training data contribute to boosting the
performance of the student model. Thus, we propose MC-EL2N
to quantify the importance of training data, which can be used to
prune useless samples dynamically. Furthermore, prompt-tuning is
powerful and attractive, so recently many tasks introduce it to tune
the LMs better, as listed in Table 7. PromptEM is the first work in
entity matching that introduces powerful prompt-tuning, having
the potential to advance the usage of LMs for GEM (including EM).
12


**[Table p12.1]**
| Paper | Venue | Task |
| --- | --- | --- |
| GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks Ontology-enhanced Prompt-tuning for Few-shot Learning Adversarial Soft Prompt Tuning for Cross-Domain Sentiment Analysis Continual Prompt Tuning for Dialog State Tracking MSP: Multi-Stage Prompting for Making Pre-trained Language Models Better Translators The Power of Prompt Tuning for Low-Resource Semantic Parsing PTAU: Prompt Tuning for Attributing Unanswerable Questions Selective Fairness in Recommendation via Prompts Relation Extraction as Open-book Examination: Retrieval-enhanced Prompt Tuning Learning Transferable Visual Models From Natural Language Supervision | SIGKDD 2022 WWW 2022 ACL 2022 ACL 2022 ACL 2022 ACL 2022 SIGIR 2022 SIGIR 2022 SIGIR 2022 ICML 2021 | Graph Neural Network Knowledge Graph Completion Sentiment Analysis Dialogue System Machine Translation Semantic Parsing Question Answering Recommendation Sysytem Relation Extraction Image Classification |

[CAPTION] Table 7: Recent research on task-specific prompt-tuning. Note that, prompt-tuning has not been introduced in the field of data