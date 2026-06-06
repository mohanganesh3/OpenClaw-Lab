[2006.16405] Unsupervised Calibration under Covariate Shift

\floatsetup

[table]capposition=top
 \floatsetup [figure]capposition=bottom

oddsidemargin has been altered.
 textheight has been altered.
 marginparsep has been altered.
 textwidth has been altered.
 marginparwidth has been altered.
 marginparpush has been altered.

The page layout violates the UAI style. 
Please do not change the page layout, or include packages like geometry,
savetrees, or fullpage, which change it for you.
We’re not able to reliably undo arbitrary changes to the style. Please remove
the offending package(s), or layout-changing commands and try again.

Unsupervised Calibration under Covariate Shift

Anusri Pampari

Computer Science Department

Stanford University
 &amp; Stefano Ermon

Computer Science Department

Stanford University

Abstract

A probabilistic model is said to be calibrated if its predicted probabilities match the corresponding empirical frequencies. Calibration is important for uncertainty quantification and decision making in safety-critical applications. While calibration of classifiers has been widely studied, we find that calibration is brittle and can be easily lost under minimal covariate shifts. Existing techniques, including domain adaptation ones, primarily focus on prediction accuracy and do not guarantee calibration neither in theory nor in practice. In this work, we formally introduce the problem of calibration under domain shift, and propose an importance sampling based approach to
address it. We evaluate and discuss the efficacy of our method on both real-world datasets and synthetic datasets.

1  INTRODUCTION

Machine learning models are increasingly being entrusted with complex decisions in many applications such as medical diagnosis  (Triantafyllidis &amp; Tsanas,  2019 ) , justice system  (Berk &amp; Hyatt,  2015 ) , financial decisions  (Heaton et al.,  2017 ) , human robot interaction  (Modares et al.,  2015 ) , etc. In all these applications, models must not only be accurate, but should also indicate confidence in their own predictions. Uncertainity quantification is important for safety-critical applications and in decision making. This will better inform when the model’s predictions are likely to be incorrect, and help in building trust with the user. For example, in medical diagnosis if the model is not confident about it’s prediction, then the decision making should be passed on to a doctor. Additionally, humans have a natural cognitive intuition for probabilities  (Cosmides &amp; Tooby,  1996 ) .  Calibrated probabilities provide an intuitive explanation to a model’s predictions, making them interpretable.

Ideally, the confidence or probability associated with the predicted class label should reflect its ground truth occurrence likelihood. For example, suppose a diabetes risk prediction model predicts a chance of 70% for a specific patient profile. Then, we expect that out of 100 similar patients, about 70 should have diabetes. Such a model is said to be  calibrated . Many existing machine learning models, such as SVMs, Gaussian processes, and Neural Networks, are not naturally calibrated  (Guo et al.,  2017 ; Bella et al.,  2010 ) , thus producing unreliable confidence estimates. This can, in turn, lead to bad decision making and reduce the trust in using these models.

Figure 1:  Reliability diagram for a LeNet-5 model trained using CDAN (SOTA domain adaptation technique) on MNIST and tested on USPS as target data.

Existing literature  (Platt et al.,  1999 ; Zadrozny &amp; Elkan,  2001 ;  2002 ; Bella et al.,  2010 ; Guo et al.,  2017 )  introduces many post processing techniques to correct these miscalibrated models. However, they assume the availability of labeled held-out validation data drawn from the same distribution as the test data to achieve calibration. This assumption is violated in many real world scenarios in the following two ways. Firstly, the test dataset can have a different distribution due to covariate shift. This can happen, e.g., when the operating conditions at test time are slightly different. Secondly, labelled test data is often unavailable if distribution shift occurs after training. While several unsupervised domain adaptation methods  (Chu &amp; Wang,  2018 ; Kouw &amp; Loog,  2019 )  propose solutions for correcting the accuracy of the models, there is no existing work to correct the effects of these circumstances on the confidence. For example, in Figure.

1  , we show how an existing calibration method temperature scaling (t-scaling)  (Guo et al.,  2017 )  can fail to calibrate a LeNet-5 model trained using CDAN (a SOTA domain adaptation model  (Long et al.,  2018 ) ) under domain shift. Here the model is trained on MNIST and has a prediction accuracy of 70% on USPS dataset.

In this work, we introduce and investigate the problem of miscalibration under covariate shift. We demonstrate that existing models learnt using domain adaptation are poorly calibrated, showing that while current domain adaptation techniques account for accuracy, they do not consider calibration of the models.  We then propose a modification to the calibration optimization objective used by existing techniques. Our solution employs importance sampling to account for the difference in the training and testing distributions, thereby overcoming the inherent assumptions of the existing methods.  Our proposed method can adapt any existing calibration method under covariate shift assumption without requiring any labeled data from the test distribution. In Figure.

1  , we show how our method (weighted t-scaling) adapts the use of t-scaling on source validation data to work under domain shift. We achieve close performance to perfect calibration or calibration obtained using labeled target data.

To summarize our contributions,

•

We introduce the problem of miscalibration under covariate shift, and show how existing domain adapted models such as CDAN  (Long et al.,  2018 )  remain uncalibrated in the target domain on using existing calibration methods;

•

We propose an importance sampling based solution to address the problem of calibration under covariate shift. Our method requests no additional labels from the test distribution and can be used to adapt any calibration method;

•

We use a discriminator trained on a domain-invariant feature layer of source and target to get density ratios for importance sampling.

2  RELATED WORK

Background and Notation  Calibration can be described mathematically as follows. Suppose that we have some data, comprising of inputs

X  ∈

R  d

𝑋

superscript  𝑅  𝑑

X\in R^{d}

and labels

Y  ∈

1  ,  …  ,  K

𝑌

1  …  𝐾

Y\in{1,\ldots,K}

, which follows the ground truth joint distribution

π  ​

(  X  ,  Y  )

=

π  ​

(

Y  |  X

)

​  π  ​

(  X  )

𝜋

𝑋  𝑌

𝜋

conditional  𝑌  𝑋

𝜋  𝑋

\pi(X,Y)=\pi(Y|X)\pi(X)

. Let

h

(  .  )

h(.)

be a classifier learned for this data, i.e.

h  :

X  →

[  0  ,  1  ]

K

:  ℎ

→  𝑋

superscript

0  1

𝐾

h:X\rightarrow[0,1]^{K}

which for an input

x  ∼  X

similar-to  𝑥  𝑋

x\sim X

, is the probability distribution over the

K

𝐾

K

classes in

Y

𝑌

Y

. The class with the maximum probability of occurrence is the prediction

Y  ^

^  𝑌

\hat{Y}

, and its corresponding probability is the confidence prediction

P  ^

^  𝑃

\hat{P}

. The classifier

h

ℎ

h

is said to be calibrated  (Guo et al.,  2017 )  when,

ℙ  ​

(

Y  ^

=

Y  |

P  ^

=  p

)

=

p

​

∀

p

∈

[  0  ,  1  ]

ℙ

^  𝑌

conditional  𝑌

^  𝑃

𝑝

p

for-all

p

0  1

\mathbb{P}(\hat{Y}=Y|\hat{P}=p)=\text{p }\forall\text{ p }\in[0,1]

(1)

Many existing classifiers do not naturally satisfy these requirements  (Guo et al.,  2017 ; Bella et al.,  2010 ) , and are therefore said to be miscalibrated.  This error is inevitable because of many reasons such as using finitely many samples to learn the classifier

h

ℎ

h

, model mismatch from the true distributi