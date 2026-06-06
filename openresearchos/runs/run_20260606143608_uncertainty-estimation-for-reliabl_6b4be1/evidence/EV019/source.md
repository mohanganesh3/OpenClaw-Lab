# EV019: Aging States Estimation and Monitoring Strategies of Li-Ion Batteries Using Incremental Capacity Analysis and Gaussian Process Regression

URL: https://www.semanticscholar.org/paper/00446c3187c6e4673ad8166e0676c51047706d27
Year: 2026
Source: semantic_scholar
Arxiv: 2603.26155

## Abstract

Existing approaches for battery health forecasting often rely on extensive cycling histories and continuously monitored cells. In contrast, many real-world scenarios provide only sparse information, e.g. a single diagnostic cycle. In our study, we investigate state of health (SoH)- and remaining useful life (RUL) estimation of previously unseen lithium-ion cells, relying on cycling data from begin of life (BOL) to end of life (EOL) of multiple similar cells by using the publicly available Oxford battery aging dataset. The estimator applies incremental capacity analysis (ICA)-based feature extraction in combination with data-efficient regression methods. Particular emphasis is placed on a multi-model Gaussian process regression ensemble approach (GPRn), which also provides uncertainty quantification. Due to a rather cell invariant behaviour, the mapping of ICA features to SoH estimation is highly precise and points out a normalized mean absolute error (NMAE) of 1.3%. The more cell variant mapping to RUL estimation is challenging, reflecting in a NMAE of 5.3%. Using the estimation results, a RUL monitoring strategy is derived. The objective is to safely operate a battery cell from BOL to EOL by only taking sparse diagnostic measurements. On average, only four diagnostic measurements are required during a cell's lifetime of 3300 to 5000 cycles.
