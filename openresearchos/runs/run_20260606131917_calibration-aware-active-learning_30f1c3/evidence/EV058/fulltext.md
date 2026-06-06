[2407.14185] Achieving Well-Informed Decision-Making in Drug Discovery: A Comprehensive Calibration Study using Neural Network-Based Structure-Activity Models

Achieving Well-Informed Decision-Making in Drug Discovery: A Comprehensive Calibration Study using Neural Network-Based Structure-Activity Models

Hannah Rosa Friesacher

ESAT STADIUS
 KU Leuven
 Leuven, 3000, Belgium

and
 Molecular AI, Discovery Sciences, R&amp;D
 AstraZeneca Gothenburg
 Gothenburg, 431 83, Sweden

rosa.friesacher@kuleuven.be

&amp;Ola Engkvist

Department of Computer Science and Engineering
 Chalmers University of Technology
 Gothenburg, 412 96, Sweden
 and
 Molecular AI, Discovery Sciences, R&amp;D
 AstraZeneca Gothenburg
 Gothenburg, 431 83, Sweden

ola.engkvist@astrazeneca.com

&amp;Lewis Mervin

Molecular AI, Discovery Sciences, R&amp;D
 AstraZeneca Cambridge
 Cambridge, CB2 0AA, UK

lewis.mervin1@astrazeneca.com

&amp;Yves Moreau

ESAT STADIUS
 KU Leuven
 Leuven, 3000, Belgium

yves.moreau@esat.kuleuven.be

&amp;Adam Arany

ESAT STADIUS
 KU Leuven
 Leuven, 3000, Belgium

adam.arany@esat.kuleuven.be

Abstract

In the drug discovery process, where experiments can be costly and time-consuming, computational models that predict drug-target interactions are valuable tools to accelerate the development of new therapeutic agents.
Estimating the uncertainty inherent in these neural network predictions provides valuable information that facilitates optimal decision-making when risk assessment is crucial.
However, such models can be poorly calibrated, which results in unreliable uncertainty estimates that do not reflect the true predictive uncertainty.
In this study, we compare different metrics, including accuracy and calibration scores, used for model hyperparameter tuning to investigate which model selection strategy achieves well-calibrated models.
Furthermore, we propose to use a computationally efficient Bayesian uncertainty estimation method named Bayesian Linear Probing (BLP), which generates Hamiltonian Monte Carlo (HMC) trajectories to obtain samples for the parameters of a Bayesian Logistic Regression fitted to the hidden layer of the baseline neural network.
We report that BLP improves model calibration and achieves the performance of common uncertainty quantification methods by combining the benefits of uncertainty estimation and probability calibration methods.
Finally, we show that combining post hoc calibration method with well-performing uncertainty quantification approaches can boost model accuracy and calibration.

1  Introduction

The development of safe and effective drugs is a challenging task, which is associated with high development costs, a high risk of adverse effects or lack of efficacy leading to the failure of a drug candidate, and long approval processes until a drug can be brought to the market  [ 1 ,  2 ] .
Machine learning models have emerged as a valuable tool, revolutionizing the drug discovery and development process by shifting to a more time- and resource-efficient pipeline  [ 3 ,  4 ,  5 ] .

As a consequence of the increasing availability of computational resources and data, recent machine learning models perform well in prediction tasks, which is reflected in high accuracy scores and low classification errors.
Estimating the uncertainty inherent to such a prediction can provide a valuable source of information in various applications besides drug design  [ 6 ,  7 ,  8 ,  9 ,  10 ,  11 ,  12 ] . Moreover, accurate uncertainty estimates can be leveraged to improve decisions about which candidates to pursue across a candidate portfolio.

Even when prediction accuracy is strong, neural networks often fail to give realistic estimates of how uncertain they are about a prediction.
These models are called poorly calibrated, which implies that the predictive uncertainty does not reflect the true probability of making a prediction error.
However, the reliability of uncertainty estimates is crucial to guarantee the reliability of machine learning models.
This is particularly important for high-stakes decision processes like the drug discovery pipeline where experiments can be costly and poor decisions inevitably lead to an increase in required time and resources.

Predictive uncertainty can come from various sources.
While many different categorizations of these sources can be found in literature, a common one is the distinction between aleatoric and epistemic uncertainty  [ 13 ,  14 ] .
Aleatoric or data uncertainty is the uncertainty related to data and data acquisition, including systematic and unsystematic errors, such as measurement errors.
Aleatoric uncertainty is also often called irreducible uncertainty, as it cannot be decreased by adding more data samples to the current model.
By contrast, epistemic, or model uncertainty can be reduced by adding knowledge.
Epistemic uncertainty can have several causes, including model overfitting or distribution shifts between training and test data.

In classification, the model output is usually a probability-like score, reflecting the uncertainty of a prediction, if the network is well calibrated.
The predictive uncertainty should summarize the total uncertainty associated with the prediction, considering all sources of uncertainty.
However, these probabilities have been reported to diverge from their ground truth preventing a reliable risk assessment [ 15 ,  16 ] .
In

2017  ,

Guo et al.

[ 15 ]  drew attention to the lack in ability of modern neural networks to estimate uncertainties of predictions.
They reported that despite their high accuracy, large neural networks are poorly calibrated, resulting in inaccurate probability estimates.

In their paper, Guo and his colleagues linked poor probability calibration to model overfitting, leading to increased probabilistic errors rather than affecting the model’s ability to correctly classify test instances.
Furthermore, they concluded that model calibration and model accuracy are also likely to be optimized by different hyperparameter settings  [ 15 ] .
Wang lists three major factors diminishing the probability calibration of a model, including large model size and over-parametrization of models, lack of model regularization and data quality and quantity, as well as imbalanced label distribution in classification [ 17 ] .
In addition, the distribution of training and test data was reported to impact model calibration.
A calibrated model will be more uncertain the more the distribution of the test instances diverges from the distribution of the training data.
Current neural networks are often overconfident, so that probability calibration deteriorates with increasing distribution shift  [ 18 ,  19 ] .

This is particularly problematic when developing new therapeutic agents, which requires exploring the chemical space by shifting the focus during inference to chemical structures that are new and unknown to the model.
As a consequence, there is a pressing need for methods that can reliably support the drug discovery process by estimating the true risk associated with a decision.

This paper focuses specifically on drug-target interaction modeling to explore the effects of different model selection strategies and uncertainty estimation approaches to model calibration.
To our knowledge, there is no study investigating the impact of different hyperparameter (HP) optimization metrics on the calibration properties of bioactivity prediction models, and we are aiming to close this gap by contributing an analysis of how to train models when aiming for good uncertainty estimates.
Furthermore, we compare the uncalibrated baseline model with three common calibration methods and we propose a limited computational complexity Bayesian approach, which allows the retrieval of samples from the posterior distribution of the last layer weights.
Finally, we investigate if combining the post hoc calibration approach Platt scaling with other uncertainty quantification method