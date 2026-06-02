[2305.08611] GeNAS: Neural Architecture Search with Better Generalization

GeNAS: Neural Architecture Search with Better Generalization

Joonhyun Jeong  1,2

Joonsang Yu  1,3

Geondo Park 2

Dongyoon Han 3 &amp;YoungJoon Yoo 1

1 NAVER Cloud, ImageVision

2 KAIST

3 NAVER AI Lab
 {joonhyun.jeong, joonsang.yu}@navercorp.com,
geondopark@kaist.ac.kr,

{dongyoon.han, youngjoon.yoo}@navercorp.com

Abstract

Neural Architecture Search (NAS) aims to automatically excavate the optimal network architecture with superior test performance. Recent neural architecture search (NAS) approaches rely on validation loss or accuracy to find the superior network for the target data.
In this paper, we investigate a new neural architecture search measure for excavating architectures with better generalization. 
We demonstrate that the flatness of the loss surface can be a promising proxy for predicting the generalization capability of neural network architectures.
We evaluate our proposed method on various search spaces, showing similar or even better performance compared to the state-of-the-art NAS methods.  Notably, the resultant architecture found by flatness measure generalizes robustly to various shifts in data distribution (e.g. ImageNet-V2,-A,-O), as well as various tasks such as object detection and semantic segmentation.

Code is available at  https://github.com/clovaai/GeNAS .

1  Introduction

Recently, Neural Architecture Search (NAS)  Liu  et al.  ( 2018b ,  a ); Hong  et al.  ( 2020 )  has evolved to achieve remarkable accuracy along with the development of human-designed networks  He  et al.  ( 2016 ); Dosovitskiy  et al.  ( 2020 )  on the image recognition task.
 Several NAS methods  Zoph  et al.  ( 2018 ); Chu  et al.  ( 2020 ); Zhang  et al.  ( 2021 ); Liu  et al.  ( 2018b ); Xu  et al.  ( 2019 ); Hong  et al.  ( 2020 )  further have demonstrated generalization ability (generalizability) of automatically designed networks with test accuracy and transfer performance onto the other datasets. For the widespread leverage of architectures found by NAS on the various tasks such as object detection  Lin  et al.  ( 2014 )  and segmentation  Cordts  et al.  ( 2016 )  (task-generalizability), investigating generalizability of each architecture candidate is prerequisite and indispensable.  Despite its importance, quantitative measurement of  generalizability  during the architecture search process is still under-explored. In this paper, we aim to find an optimal proxy measurement to discriminate generalizable architectures during the search process.

(a)

ABS

(b)

FBS (Ours)

Figure 1 :

Shape of local loss minima found by angle-based  search  (ABS) and flatness-based  search  (FBS). (a)  The  architecture found by ABS can not guarantee to be located within flat local minima. (b) FBS searches for architectures with flat local minima by inspecting loss values of local neighborhood weights.

Kendall’s Tau

CIFAR-10

CIFAR-100

ImageNet16-120

0.4302

0.4724

0.4097

Table 1 :

Low correlation of angle measure with flatness measure on NAS-Bench-201  Dong and Yang ( 2020 )  search space. We evaluated the angle and flatness of all architectures and compared Kendall’s Tau  Kendall ( 1938 )  rank correlation between these search metrics on CIFAR-10, CIFAR-100, and ImageNet16-120  Chrabaszcz  et al.  ( 2017 )  dataset.

Previous NAS algorithms including the pioneering differentiable search method, DARTS  Liu  et al.  ( 2018b )  and evolutionary search method, SPOS  Guo  et al.  ( 2020 )  use validation performance as a proxy measure for  the generalizability  as follows:

a  ∗

=

argmax

a  ∈  A

S

​

(  a  )

,

superscript  𝑎

subscript  argmax

𝑎  𝐴

𝑆

𝑎

a^{*}=\operatorname*{argmax}_{a\in A}S(a),

(1)

where

a

𝑎

a

and

A

𝐴

A

denote an architecture candidate and the entire search space, respectively, and

S  ​

(  ⋅  )

𝑆  ⋅

S(\cdot)

represents a measurement function which is broadly defined by accuracy

Guo  et al.  ( 2020 )  or negative of loss value  Liu  et al.  ( 2018b )  on a validation dataset.
Although these performance-based  search  (PBS) methods find the optimal architecture for generalization on the validation set, they show poor generalizability on the test set and other tasks, caused by overfitting on validation set  Zela  et al.  ( 2019 ); Oymak  et al.  ( 2021 ) .
In addition, PBS methods represent a large discrepancy between the validation accuracy and ground truth test accuracy provided by NAS benchmark  Dong and Yang ( 2020 )  as shown in  Guo  et al.  ( 2020 ); Zhang  et al.  ( 2021 ) .

To search generalizable architectures, several literatures  Shu  et al.  ( 2019 ); Zhang  et al.  ( 2021 )  empirically  observe  that architectures with fast convergence during training have a tendency to show better generalizability on test set.
Based on the empirical connection between convergence speed and generalization, RLNAS  Zhang  et al.  ( 2021 )  proposed an Angle-Based  Search  (ABS) method, which  estimates  angle  between initial and final network parameters after convergence of the model (i.e. convergence speed)  as a proxy performance measure during the search process.
However, we  argue  that ABS still has a large headroom for better generalization in terms of flat (wide) local minima, which has been considered as one of the key signals for inspecting generalizability of a trained network  Keskar  et al.  ( 2016 ); Zhang  et al.  ( 2018 ); Pereyra  et al.  ( 2017 ); Cha  et al.  ( 2020 ); He  et al.  ( 2019 ) .

Intuitively, since the architecture with flat loss minima has widely low loss values around the minimum, it can achieve a low generalization error even if the loss surface is shifted due to the distribution gap from the test dataset.

Since ABS only concerns the angle between initial model wights and trained ones in terms of convergence speed, found architectures can not be guaranteed to have flat local minima, as shown in Figure

1  . Specifically, architectures not  chosen  by ABS (i.e. small angle) might have better generalizability based on the flat property of loss minima. Table

1

corroborates that angle is indeed in short of correlation with flatness of local minima.

To explicitly design a search proxy measure that has a high correlation with the generalizability of the found model, we propose a flatness-based  search  method, namely FBS, which excavates a well-generalizable architecture by measuring the flatness of loss surface.
FBS can find out robust architecture with low generalization error on shifted data distribution (e.g. test data,  out-of-distribution  datasets, downstream tasks)  by  inspecting both the depth and flatness of loss curvature near local minima through injecting random noise. In addition, FBS can be either replaced or incorporated with other state-of-the-art search measures to enhance performance as well as generalizability.

Consequently, building upon our search method FBS, we propose a novel flatness-based NAS framework, namely GeNAS, to exactly discriminate generalizability of architectures during searching.  We show the effectiveness of the proposed GeNAS for both cases when using FBS solely or integrated into the conventional architecture score measurements such as PBS and ABS. 
Specifically, our GeNAS achieves comparable or even better performances on several NAS benchmarks compared to PBS- and ABS-based  search  methods  Liu  et al.  ( 2018b ); Zhang  et al.  ( 2021 ); Xu  et al.  ( 2019 ); Guo  et al.  ( 2020 ); Chu  et al.  ( 2020 ); Chen  et al.  ( 2019 ); Hong  et al.  ( 2020 ) .
Furthermore, we also show that the proposed FBS can be combined with conventional search metrics (e.g. PBS, ABS),
inducing significant performance gain.  Finally, we also demonstrate that our FBS can well-generalize on various data distribution shifts, as well as on multiple downstream tasks such as object detection and semantic segmentation.

Our contributions can be summarized as follows:

•

We