# EV017: PM2.5 Concentration Prediction Method Based on Temporal Attention Mechanism and CNN-LSTM

URL: https://www.semanticscholar.org/paper/00967dd4f9b76200e594b4ef513052e613520c2d
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

Accurately predicting PM2.5 concentration can effectively avoid the harm caused by heavy pollution weather to human health. In view of the non-linearity, time series characteristics, and the problem of large multi-step prediction errors in PM2.5 concentration data, a method combining Long Short-term Memory Network and Convolutional Neural Network with Time Pattern Attention mechanism (TPA-CNN-LSTM) is proposed. The method uses historical PM2.5 concentration data, historical meteorological data, and surrounding station data to predict the future 6-hour PM2.5 concentration of air quality monitoring stations. Firstly, CNN is used to obtain the spatial characteristics between multiple stations, secondly, LSTM is added after CNN to extract the temporal changes of non-linear data, and finally, to capture the key features of temporal information, Temporal Pattern Attention mechanism (TPA) is added. TPA can automatically adjust weights based on the input of each time step, and select the most relevant time step for prediction, thereby improving the accuracy of the model. An example analysis is conducted on the measured data of Beijing's air quality stations in 2018, and compared with other mainstream algorithms. The results show that the proposed model has higher prediction accuracy and performance.
