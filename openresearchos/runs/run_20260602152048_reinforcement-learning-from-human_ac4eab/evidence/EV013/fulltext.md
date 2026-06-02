[2310.02743] Reward Model Ensembles Help Mitigate Overoptimization

Reward Model Ensembles Help Mitigate Overoptimization

Thomas Coste  ➀ , Usman Anwar  ➀ , Robert Kirk  ➁ , David Krueger  ➀

➀ University of Cambridge,
 ➁ University College London

{tc628,ua237,dsk30}@cam.ac.uk
 robert.kirk.20@ucl.ac.uk

Abstract

Reinforcement learning from human feedback (RLHF) is a standard approach for fine-tuning large language models to follow instructions. As part of this process, learned reward models are used to approximately model human preferences. However, as imperfect representations of the “true” reward, these learned reward models are susceptible to  overoptimization .  Gao et al. ( 2023 )  studied this phenomenon in a synthetic human feedback setup with a significantly larger “gold” reward model acting as the true reward (instead of humans) and showed that overoptimization remains a persistent problem regardless of the size of the proxy reward model and training data used. Using a similar setup, we conduct a systematic study to evaluate the efficacy of using ensemble-based conservative optimization objectives, specifically worst-case optimization (WCO) and uncertainty-weighted optimization (UWO), for mitigating reward model overoptimization when using two optimization methods: (a) best-of-n sampling (BoN) (b) proximal policy optimization (PPO). We additionally extend the setup of  Gao et al. ( 2023 )  to include

25  %

percent  25

25\%

label noise to better mirror real-world conditions. Both with and without label noise, we find that conservative optimization practically eliminates overoptimization and improves performance by up to

70  %

percent  70

70\%

for BoN sampling. For PPO, ensemble-based conservative optimization always reduces overoptimization and outperforms single reward model optimization. Moreover, combining it with a small KL penalty successfully prevents overoptimization at no performance cost. Overall, our results demonstrate that ensemble-based conservative optimization can effectively counter overoptimization.

1  Introduction

With the advent of large language models, reinforcement learning from human feedback (RLHF) has emerged as a powerful technique to fine-tune and enhance models’ behaviors  (Ziegler et al.,  2019 ; Ouyang et al.,  2022 ; Bai et al.,  2022a ) . However, despite its empirical success, RLHF remains a fickle method suffering from many failure modes  (Casper et al.,  2023 ) . One such failure mode is  overoptimization , a phenomenon in which policy optimization appears to be making progress according to the learned reward model, but in reality begins to regress with respect to the true reward function  (Ziegler et al.,  2019 ; Stiennon et al.,  2020 ; Gao et al.,  2023 ) . While many works on RLHF contain anecdotal evidence of overoptimization  (Ziegler et al.,  2019 ; Stiennon et al.,  2020 ; Dubois et al.,  2023 ) ,  Gao et al. ( 2023 )  is the only work that studies overoptimization in a systematic way. As working directly with human labelers is expensive,  Gao et al. ( 2023 )  introduce a synthetic setup to study overoptimization in which a much larger language model is first trained as a “gold” reward model and is then used to generate preference labels for training of proxy reward models.

In this work, we conduct a systematic study investigating whether combining ensembles with conservative optimization can help mitigate overoptimization. Our results indicate that not only does ensemble-based conservative optimization help mitigate overoptimization, it also results in improved performance. Our setup is identical to that of  Gao et al. ( 2023 )  with one modification: the addition of label noise.

Gao et al.

assume that the preference labels used to train the proxy reward model do not contain any noise. However, this does not mirror the real-world RLHF setup, in which agreement rates among human annotators are typically between

60  −

75  %

60

percent  75

60-75\%

(Ziegler et al.,  2019 ; Stiennon et al.,  2020 ; Dubois et al.,  2023 ) . To simulate that disagreement and to better reflect the real-world RLHF, we extend the setup to include

25  %

percent  25

25\%

label noise as well. In both the cases of no label noise and

25  %

percent  25

25\%

label noise, we provide strong evidence that ensemble-based conservative optimization methods are effective in mitigating overoptimization and improving performance.

Scaling laws for reward model overoptimization discovered by  Gao et al. ( 2023 )  indicate that increasing the size of the proxy reward model reduces overoptimization as well. However, reward models are derived from pretrained language models. Thus, acquiring a larger reward model would require significant  pretraining , which is not always feasible and can be very costly
 (Morgan,  2022 ; Venigalla &amp; Li,  2022 ) .
However, our approach, using ensembles of reward models, only requires fine-tuning multiple copies of an already pretrained reward model, which is relatively inexpensive. Moreover, our model and data scaling results (Figures

8

and

9  ) indicate that the gains provided by our method are orthogonal to the gains achieved by increasing the reward model size; thus, the two approaches can be combined seamlessly for even better results.

Our main contributions are as follows:

•

We conduct the first study of using ensembles to counter overoptimization in RLHF-based fine-tuning of language models.

•

Our results indicate that using ensembles and conservative optimization eliminates overoptimization for BoN and results in up to

70  %

percent  70

70\%

improvement in some cases.

•

For PPO, ensemble-based conservative optimization typically outperforms single reward model optimization, and when combined with a suitable KL penalty weight successfully eliminates overoptimization.

•

We further conduct studies to establish the robustness of the ensemble-based conservative optimization methods to any new hyperparameters it introduces (e.g. size of the ensemble).

Figure 1:  RLHF pipeline used in this work - our modifications on top of the standard RLHF setup used in  Gao et al. ( 2023 )  are highlighted in  green .

2  Background

In this section, we briefly review two commonly used policy optimization methods: best-of-n sampling (BoN) and proximal policy optimization (PPO), followed by a discussion of overoptimization.

2.1  Best-of-n Sampling (BoN)

Best-of-

n

𝑛

n

(BoN) sampling, also called rejection sampling, is a simple inference-time optimization method  (Ouyang et al.,  2022 ; Nakano et al.,  2021 ) . For a given prompt,

n

𝑛

n

responses are generated from the policy model, and the answer with the highest proxy reward model score is returned. To evaluate the degree of optimization, the KL distance is defined analytically as a function of

n

𝑛

n

:

KL  bon

=

log  ​  n

−

n  −  1

n

subscript

KL

bon

log

𝑛

𝑛  1

𝑛

\text{KL}_{\text{bon}}=\text{log}\,n-\frac{n-1}{n}

(1)

2.2  Proximal Policy Optimization (PPO)

Proximal Policy Optimization  (Schulman et al.,  2017 )  is a policy-gradient-based online reinforcement learning method that maximizes a given reward function by repeatedly performing small incremental updates to the policy.
PPO is the standard algorithm used in fine-tuning language models based on human feedback  (Ouyang et al.,  2022 ; Bai et al.,  2022a ; Stiennon et al.,  2020 ; Zheng et al.,  2023 ) . When using PPO to fine-tune a language model, a KL penalty term is added during the reward calculation to regularize the policy by preventing it from deviating far from the initial policy:

R  PPO

​

(  q  ,  a  )

=

R  ​

(  q  ,  a  )

−

β  ​

log  ⁡

[

π  PPO

​

(

a  |  q

)

π  init

​

(

a  |  q

)

]

superscript  𝑅

PPO

𝑞  𝑎

𝑅

𝑞  𝑎

𝛽

superscript  𝜋

PPO

conditional  𝑎  𝑞

superscript  𝜋

init

conditional  𝑎  𝑞

\displaystyle R^{\text{PPO}}(q,a)=R(q,a)-\beta\log\left[\frac{\pi^{\text{PPO}}(a|q)}{\pi