[2502.04807] Robust Conformal Outlier Detection under Contaminated Reference Data

Robust Conformal Outlier Detection under
 Contaminated Reference Data

Meshi Bashari

Department of Electrical and Computer Engineering, Technion IIT, Haifa, Israel

Matteo Sesia

Department of Data Sciences and Operations, University of Southern California, Los Angeles, California, USA

Department of Computer Science, University of Southern California, Los Angeles, California, USA

Yaniv Romano

Department of Electrical and Computer Engineering, Technion IIT, Haifa, Israel

Department of Computer Science, Technion IIT, Haifa, Israel

Abstract

Conformal prediction is a flexible framework for calibrating machine learning predictions, providing distribution-free statistical guarantees. In outlier detection, this calibration relies on a reference set of labeled inlier data to control the type-I error rate. However, obtaining a perfectly labeled inlier reference set is often unrealistic, and a more practical scenario involves access to a contaminated reference set containing a small fraction of outliers. This paper analyzes the impact of such contamination on the validity of conformal methods. We prove that under realistic, non-adversarial settings, calibration on contaminated data yields conservative type-I error control, shedding light on the inherent robustness of conformal methods. This conservativeness, however, typically results in a loss of power. To alleviate this limitation, we propose a novel, active data-cleaning framework that leverages a limited labeling budget and an outlier detection model to selectively annotate data points in the contaminated reference set that are suspected as outliers. By removing only the annotated outliers in this “suspicious” subset, we can effectively enhance power while mitigating the risk of inflating the type-I error rate, as supported by our theoretical analysis. Experiments on real datasets validate the conservative behavior of conformal methods under contamination and show that the proposed data-cleaning strategy improves power without sacrificing validity.

Keywords:  Conformal Prediction, Hypothesis Testing, Out-of-Distribution Detection, Contaminated Data

1  Introduction

1.1  Background and Motivation

This paper studies the problem of outlier detection: given a reference dataset (e.g., a collection of legitimate financial transactions) and an unlabeled test point (a new transaction), our goal is to determine whether the test point is an outlier (a fraudulent transaction) by assessing its deviation from the reference data distribution. Naturally, we aim to maximize the detection of outliers by harnessing the capabilities of complex machine learning (ML) models. However, these models typically lack type-I error rate control, potentially resulting in unreliable detections. In our running example, the type-I error is the probability of falsely flagging a legitimate transaction as fraudulent. As such, uncontrolled error rates can lead to costly unnecessary investigations of legitimate transactions and negatively impact customer experience.

The broad need for reliable ML systems has sparked a surge of interest in conformal prediction—a versatile framework that can provide statistical guarantees for any “black-box” predictive model  [ 38 ] . This framework formulates the outlier detection task as a statistical test, where the null hypothesis is that the new data point is not an outlier  [ 23 ,  5 ] . To derive a decision rule guaranteeing type-I error control, conformal inference relies on a reference (calibration) set of inlier data points. These points are assumed to be sampled i.i.d. from an unknown distribution, independent of the data used to train the outlier detection model.

In practice, however, it is often difficult to obtain a perfectly clean reference dataset that contains no outliers  [ 28 ,  44 ,  8 ,  20 ] .
In our example, a more realistic scenario would assume instead access to a slightly  contaminated  reference set, mostly legitimate transactions with a few unnoticed outliers  [ 44 ] .
But this setting poses new challenges for conformal prediction methods, potentially invalidating the error control guarantees or, as we shall see, often reducing the power to detect true outliers at test time.

1.2  Outline and Contributions

While type-I error control in conformal inference theoretically requires perfectly clean reference data, in practice contaminated data often makes these methods overly conservative, reducing the power to detect true outliers rather than inflating the type-I error rate. This empirical observation motivates the first question explored in this paper:

Q1:

When does conformal outlier detection with contaminated reference data yield valid type-I error control?

In Section

2.3  , we present the first contribution of this paper: a novel theoretical analysis that identifies common conditions under which this conservative behavior arises. Unfortunately, this conservativeness often comes at the cost of reduced detection power, particularly when targeting low type-I error rates. To address this issue, we investigate data-driven cleaning strategies aimed at mitigating the contamination in the reference dataset.

A straightforward approach to cleaning the contaminated set is to remove all data points flagged as likely outliers by the detection model. However, this method is unsatisfactory, as it risks inadvertently removing inliers along with outliers, resulting in an "overly clean" reference set. This, in turn, distorts the inlier distribution and inflates the type-I error rate above the desired nominal level.

This challenge motivates our second and main contribution. In Section

3.3  , we introduce an approach to clean the contaminated reference set by leveraging a limited labeling budget (e.g., 50 annotations). The outlier detection model is first used to identify suspected outliers within the contaminated reference set. The limited budget is then strategically allocated to annotate these points, thereby avoiding the unintended removal of inliers. While this is a practical and intuitive approach, it naturally prompts a critical question:

Q2:

How does the selective annotation and partial removal of outliers from a contaminated reference set affect the validity of conformal inferences?

We analyze the validity of this active labeling approach for trimming outliers in the contaminated set. Our theoretical results identify the conditions required to achieve approximate type-I error control, even when the data are selectively annotated and not all outliers are removed. This analysis also highlights key factors that can inflate the error rate, offering practitioners guiding principles to enhance the power of conformal methods in the presence of contaminated data.

Finally, in Section

4  , we empirically validate our theory and proposed data-cleaning approach through comprehensive experiments on real-world datasets. The experiments confirm that conformal inference with contaminated data tends to be conservative. Furthermore, they demonstrate that our method significantly boosts power, particularly when the target type-I error rate is low and the number of outliers in the contaminated set is small.
Software for reproducing the experiments is available at  https://github.com/Meshiba/robust-conformal-od .

1.3  Related Work

Recently, there has been growing interest in studying the statistical properties of conformal inference methods under more realistic scenarios, moving beyond the idealized assumption of perfectly clean and exchangeable observations to account for various types of  distribution shift

[ 36 ,  12 ,  33 ,  4 ,  17 ,  40 ,  16 ,  18 ,  30 ,  34 ,  31 ] . This paper draws inspiration from several prior works in this area.

Tibshirani et al. [ 36 ]  introduced a weighted conformal prediction approach to address covariate shift between calibration and test data, later extended by  Pod