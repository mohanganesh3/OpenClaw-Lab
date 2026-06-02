[2210.10325] Improving Stability of Fine-Tuning Pretrained Language Models via Component-Wise Gradient Norm Clipping

Improving Stability of Fine-Tuning Pretrained Language Models

via Component-Wise Gradient Norm Clipping

Chenghao Yang 1 , Xuezhe Ma 2

1 University of Chicago

2 University of Southern California

yangalan1996@gmail.com ,  xuezhema@isi.edu

Abstract

Fine-tuning over large pretrained language models (PLMs) has established many state-of-the-art results.
Despite its superior performance, such fine-tuning can be unstable, resulting in significant variance in performance and potential risks for practical applications.
Previous works have attributed such instability to the catastrophic forgetting problem in the top layers of PLMs, which indicates iteratively fine-tuning layers in top-down manner is a promising solution.
In this paper, we first point out that this method does not always work out due to different convergence speeds of different layers/modules.
Inspired by this observation, we propose a simple component-wise gradient norm clipping method to adjust the convergence speed for different components.
Experiment results demonstrate that our method achieves consistent improvements in terms of generalization performance, convergence speed and training stability. The codebase can be found at  https://github.com/yangalan123/FineTuningStability .

1  Introduction

Fine-tuning over large pretrained language models (PLMs), which achieved remarkable performance over various benchmarks, has become the de facto paradigm for several current natural language processing (NLP) systems.
However, fine-tuning can be unstable in terms of significant variance in metrics, resulting in even worse-than-random failed models  (Devlin et al.,  2019 ; Lee et al.,  2019 ; Dodge et al.,  2020 ; Mosbach et al.,  2020 ) .

Catastrophic forgetting  (Kirkpatrick et al.,  2017 )  during fine-tuning of PLMs is one common explanation for this instability  (Lee et al.,  2019 ) , i.e., PLMs may lose their rich domain-agnostic knowledge acquired by language model pretraining in the process of fine-tuning.
Through layer-replacement experiments between pretrained models and fine-tuned models,  Mosbach et al. ( 2020 )  further connected the catastrophic forgetting problem to the optimization problem on top layers.

These findings give rise to a straightforward way to enhance the fine-tuning stability: how about fine-tuning the model from top to bottom to reduce the parameter changes and hence mitigate the catastrophic forgetting problem?
This is reminiscent of the gradual unfreezing  (Howard and Ruder,  2018 ) , which does layer-wise top-down fine-tuning and unfreezing new layers only when the layers above have been fine-tuned.
Therefore, the newly unfrozen layer would only be tuned for a slightly easier optimization problem at each iteration, leading to much fewer changes to the parameters.

Figure 1:  Fine-tuning performance over time on RTE datasets. Here we fine-tune over BERT-large-uncased model  (Devlin et al.,  2019 ) . In each iteration, we train the model for

3

3

3

epochs and the errorbar is plotted based on

5

5

5

different runs.

However, based on a comprehensive case study of the gradual unfreezing method, we obtained empirical observations beyond our expectations (§  2  ).
Our analysis further reveals a possible reason:  different components (e.g., feed-forward networks at different layers, fully connected matrices and biases at output layer) converge at varying speeds. 
Thus, components in upper layers, which have converged to local optima, cannot easily be fine-tuned with newly unfrozen parameters.

(a)

Compared with original BERT model

(b)  Compared with previous iteration

Figure 2:  Parameter update at each iteration for GU. Updates that are too small cannot be seen in this figure (e.g., for the

22

22

22

-th layer, or “layer.21” in the figure update is too small to plot out in

Fig.

2(b)  ). We only plot the first

4

4

4

iterations as we observe that the performance is almost stable by the fourth iteration. As different layers can have components with different dimensionalities, we show the component-wise maximum rooted mean squared difference for each layer.

Based on this observation, we propose a simple component-wise gradient clipping method to stabilize the fine-tuning process (§  3  ).
This method achieves significant empirical improvements of fine-tuning stability in terms of the variance and the failed run percentage over three tasks (§  4  ).
In summary we make the following contributions:

1.

We find that component-wise convergence speed divergences is the key challenge in fine-tuning stability, based on the case study of the gradual unfreezing method.

2.

Based on our observation, we propose a new simple component-wise gradient clipping method to help stabilize fine-tuning, which achieves empirical improvements of fine-tuning stability over previous methods.

2  A Bitter Case Study: Layer-wise Gradual Unfreezing

Mosbach et al. ( 2020 )  attributed the instability problem in fine-tuning process to the catastrophic forgetting in the top layers, through a layer replacement experiment between pretrained and fine-tuned models.
Following this empirical observation, the instability problem might be mitigated if we can minimize the edits to the pretrained model parameters, especially in top layers.
This inspires us to mitigate the instability problem via gradual unfreezing (GU,  Howard and Ruder ( 2018 ) ).

Specifically, suppose we are working with a model

M

𝑀

M

with

L

𝐿

L

layers parameterized by

{

θ

(  i  )

,  1

≤  i  ≤  L

}

superscript  𝜃  𝑖

1

𝑖

𝐿

\{\theta^{(i)},1\leq i\leq L\}

.
GU tunes

M

𝑀

M

for

L

𝐿

L

iterations.
At

k

𝑘

k

-th (

k

𝑘

k

start from

0

0

) iteration, we only tune a subset of parameters

R

(  k  )

=

{

θ

(  i  )

,

L  −  k

≤  i  ≤  L

}

superscript  𝑅  𝑘

superscript  𝜃  𝑖

𝐿  𝑘

𝑖

𝐿

R^{(k)}=\{\theta^{(i)},L-k\leq i\leq L\}

, where

θ

(  i  )

​

(

L  −  k

+  1

≤  i  ≤  L

)

superscript  𝜃  𝑖

𝐿  𝑘

1

𝑖

𝐿

\theta^{(i)}(L-k+1\leq i\leq L)

is also tuned in

k  −  1

𝑘  1

k-1

-th iteration.
In each iteration, we tune the parameter for

E

𝐸

E

epochs, where

E

𝐸

E

is large enough for convergence.  1

1  1 We select

E

𝐸

E

based on preliminary experiments.

Detailed algorithm is shown in

Algorithm

1

at

Appendix

A  .

(a)

Failed Run

(b)  Success Run

Figure 3:  Gradient norm across different parameters at layer 22 on RTE dataset fine-tuning. Here the left figure (“Failed Run”) refers to the case when the fine-tuned model cannot beat the majority classifier and the right figure (“Success Run”) otherwise. Different colors represent different parameters. Due to space limitation, legends are omitted and we cannot show results from all layers. But in our observation, most layers have similar phenomenon.

Failure of Gradual Unfreezing

From

Fig.

1  , the accuracy of gradual unfreezing is significantly worse than full fine-tuning,  2

2  2 As pointed out by  Mosbach et al. ( 2020 ) , without bias correction, the original results in  Devlin et al. ( 2019 )  paper can be pretty bad and unstable, so we add bias correction on top of  Devlin et al. ( 2019 )  to make a strong baseline.

although it indeed achieves smaller update to pretrained model parameters compared with full fine-tuning (  Fig.

2(a)  ).

Convergence Racing between Parameters

To investigate the reason behind this unsatisfying performance of GU, we plot the component-wise maximum update (measured by component-wise maximum rooted mean squared difference,  3

3  3 We also plot the parameter change measured by cosine distance and put the figures in

Appendix

C

as different layers can have multiple components with different dimensions) at each layer in

Fig.

2(b)  .
Clearly, from the very beginning, the parameter updates for both early-tuned parameters and ne