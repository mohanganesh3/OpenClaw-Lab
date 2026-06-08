# EV008: MetricCol: Metric Depth and Pose Estimation in Colonoscopy via Geometric Consistency and Domain Adaptation

URL: https://www.semanticscholar.org/paper/03c92aad2884e643d89ff03edd29aa173ccbf9c5
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Accurate metric depth and pose estimation are critical for colonoscopic navigation and lesion localization. However, existing methods often struggle with scale ambiguity and domain gaps between synthetic and real datasets. To address these issues, we propose a novel framework consisting of two stages: 1) a fully supervised depth estimation model utilizing synthetic data with anatomical priors to bridge the domain gap between synthetic and real datasets, and 2) a weakly supervised joint learning approach combining camera-aware depth scaling with uncertainty-driven pseudo-labeling to refine metric depth and pose estimation. We validate the framework on both synthetic and real colonoscopy datasets, achieving superior performance in metric depth ($\text{RMSE} =3.5408$) and pose estimation (average $\text{ATE} =0.6143$). Experimental results on both synthetic and real colonoscopy datasets show superior performance, robustness under challenging conditions, and demonstrate the clinical applicability of our method. Code is available at https://github.com/liuyq055/MetricCol.
