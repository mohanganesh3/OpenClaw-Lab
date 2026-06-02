# EV008: Evaluating Lightweight Neural Models for Edge-Based Anomaly Detection: Performance and Efficiency Trade-offs

URL: https://www.semanticscholar.org/paper/001399bf0e198b2c74864ed786d55b4693ab9c5b
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

In edge computing situations where limited memory, slow response times, and low energy availability make it hard to use large models, smaller neural networks are becoming more popular for spotting unusual activities. However, uniform benchmarks for comparing commonly utilized lightweight models under these constraints remain absent. This research addresses the gap by evaluating three prominent lightweight neural architectures, pruned convolutional neural networks (CNNs), quantized long shortterm memory networks (LSTMs), and distilled transformers, across two established intrusion detection datasets: CIC-IoTDIAD 2024 and TON_IoT (TON_IoT_Modbus and TON_IoT_Thermostat). We assess each model using common detection measures (accuracy, precision, recall, and F1-score) and deployment metrics (model size, inference speed, and memory use) while simulating edge conditions. Our findings indicate significant trade-offs between model accuracy and efficiency, with performance varying based on the dataset utilized. Certain models perform more effectively with flowbased data compared to others with IoT telemetry. No single model excelled in all evaluation criteria. This research gives future studies on edge-optimized anomaly detection a solid and repeatable base, along with useful tips for choosing models for real-time edge use. The results also guide our attention in creating our forthcoming architecture, S3LiteNet, intended to enhance performance and deployment in information-centric networks.
