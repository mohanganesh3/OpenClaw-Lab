[2504.06915] An Analysis of Temporal Dropout in Earth Observation Time Series for Regression Tasks

1

1  institutetext:  University of Kaiserslautern-Landau, Kaiserslautern, Germany

2

2  institutetext:  German Research Center for Artificial Intelligence, Kaiserslautern, Germany

2

2  email:  {miro.miranda_lorenz,francisco.mena,andreas.dengel}@dfki.de

An Analysis of Temporal Dropout in Earth Observation Time Series for Regression Tasks

Miro Miranda

1122

0009-0002-8195-9776

Francisco Mena

1122

0000-0002-5004-6571

Andreas Dengel

1122

0000-0002-6100-8255

Abstract

Missing instances in time series data impose a significant challenge to deep learning models, particularly in regression tasks.
In the Earth Observation field, satellite failure or cloud occlusion frequently results in missing time-steps, introducing uncertainties in the predicted output and causing a decline in predictive performance.
While many studies address missing time-steps through data augmentation to improve model robustness, the uncertainty arising at the input level is commonly overlooked.
To address this gap, we introduce Monte Carlo Temporal Dropout (MC-TD), a method that explicitly accounts for input-level uncertainty by randomly dropping time-steps during inference using a predefined dropout ratio, thereby simulating the effect of missing data. To bypass the need for costly searches for the optimal dropout ratio, we extend this approach with Monte Carlo Concrete Temporal Dropout (MC-ConcTD), a method that learns the optimal dropout distribution directly.
Both MC-TD and MC-ConcTD are applied during inference, leveraging Monte Carlo sampling for uncertainty quantification.
Experiments on three EO time-series datasets demonstrate that MC-ConcTD improves predictive performance and uncertainty calibration compared to existing approaches. Additionally, we highlight the advantages of adaptive dropout tuning over manual selection, making uncertainty quantification more robust and accessible for EO applications.

Keywords:  Dropout Earth Observation Time Series Regression Uncertainty Quantification

1  Introduction

Recently,  Deep Learning (DL)  models have been widely used in the  Earth Observation (EO)  field to find optimal data-driven solutions in different applications  [ 3 ] .
 DL  effectively processes complex and heterogeneous sensor data, allowing for accurate analysis of environmental patterns  [ 27 ] .
For instance, predicting continuous values such as crop yield  [ 38 ] , water content in plants  [ 39 ] , and surface forecast  [ 40 ]  involves processing complex data. In the  EO  field, processing time series sensor data is essential for understanding the changes and dynamics of our planet  [ 29 ] .
However, sensors may experience anomalies and occlusions, leading to missing data over certain time-steps  [ 43 ,  15 ,  11 ] .
For instance, clouds obstruct the sunlight in optical images, as on average, 67% of Earth’s surface is covered by clouds  [ 22 ] , often resulting in uncertain predicted outputs.
Addressing missing data in time series is a prevalent challenge in the  DL  field, with various modeling solutions and pre-processing techniques developed to mitigate its impact and ensure accurate predictions.
While various techniques exist to mitigate missing data  [ 15 ,  26 ] , few explicitly quantify the uncertainty it introduces, particularly the impact of input-level uncertainty. Addressing this gap is crucial for improving model reliability in EO regression applications.

In this work, we analyze two models that simulate missing data across time series during training and inference for a dual purpose.
During training, it acts as an augmentation technique to increase model generalization to missing data in temporal  EO  data.
During inference, by generating multiple  Monte Carlo (MC)  samples with different missing patterns, it acts as an  Uncertainty Quantification (UQ)  mechanism, thereby increasing the trustworthiness of a prediction.
The first model is built on the dropout variational distribution applied during inference  [ 13 ]  and over time, referred to as  Monte Carlo Temporal Dropout (MC-TD) .
However, the optimal dropout value is a difficult hyperparameter to tune, requiring a resource-intensive process that depends on each dataset and missing data type  [ 26 ] .
Instead of manually searching, we propose using  Concrete Dropout (ConcD)

[ 14 ]  to learn the optimal dropout ratio using standard gradient descent. We refer to this method as  Monte Carlo Concrete Temporal Dropout (MC-ConcTD) , where  ConcD  is applied across time and during inference.
The  ConcD

[ 14 ]  approximates a Bernoulli distribution using a continuous relaxation, enabling the dropout probability to be optimized through gradient-based learning  [ 24 ] .
Unlike traditional dropout-based approaches, this model automatically learns an optimal dropout distribution, eliminating costly hyperparameter tuning.

We validate  MC-TD  and  MC-ConcTD  on regression tasks spanning three  EO  datasets with temporal sensor data.
Experimental results demonstrate that  MC-ConcTD  improves both predictive accuracy and uncertainty calibration, while also eliminating the need for manual dropout tuning. The  Concrete Temporal Dropout (ConcTD)  not only enhances predictive performance but also improves the accessibility and practicality of  UQ  in  EO  regression applications.
The code is publicly available at  https://github.com/mmiranda-l/Temporal-Dropout/ .

2  Related Work

Missing data in time series.

Irregular sampled time series are common in signal processing  [ 36 ] .
Temporal observations can easily suffer from problems in their collection, causing irregular observations with missing data.
Numerous research in  DL  has focused on learning to impute time series, such as BRITS  [ 4 ] , mTAN  [ 44 ] , and SAITS  [ 10 ] , while others focused on adapting models to ignore the missing data, such as D-GRU  [ 5 ]  and MissFormer  [ 1 ] .
In the  EO  field, missing spatial and temporal data are a common phenomenon due to real-world data collection constraints  [ 43 ] . This includes, sensor noise, sensor failure, and cloud occlusion, affecting observations and introducing uncertainty.
This problem negatively affects the performance of predictive models, where more missing data translates to worse predictions  [ 20 ,  12 ] .
However, some strategies in the  EO  field mitigate the negative effect of missing data, such as including features from different sensors or using dropout techniques  [ 35 ,  15 ] .
Recently, the  Temporal Dropout (TD)  technique, which involves randomly dropping time steps, has been used to enhance the model performance  [ 15 ,  47 ,  18 ] .
Furthermore, some studies leverage missing data as an augmentation technique to enhance generalization. For instance, Weitland et al.  [ 50 ]  randomly mask the end of a time-series for generalization to an early crop classification.
On the other hand, some studies have focused on reconstruction tasks that recover the missing time-steps.
For instance, Chen et al.  [ 6 ]  use a polynomial fit based on the Savitzky–Golay filter, while others use  DL  models based on  Multi-layer Perceptron (MLP)

[ 8 ] , or convolutions  [ 42 ] .

Regression with time series data.

Regression tasks have been lesser explored than classification tasks with time-series data  [ 32 ] .
One reason might be that  DL  models are not as effective for regression as for classification tasks, as shown by Tan et al.  [ 46 ] . The intrinsic challenge of predicting a continuous value instead of a categorical value is often overlooked. Nevertheless, in the  EO  field, there are numerous applications involving pixel-level regression with time series data.
For instance, Nguyen et al.  [ 33 ]  use multi-spectral data and weather time series for pixel-wise crop yield prediction. They use a  MLP  model for processing the time series data as individual features. However, it