# EV010: Annotation-Efficient Active Test-Time Adaptation with Conformal Prediction

URL: https://www.semanticscholar.org/paper/049c493fd9c3a85269806a44d94a0d43eb4b53fa
Year: 2025
Source: semantic_scholar
Arxiv: 2509.25692

## Abstract

Active Test-Time Adaptation (ATTA) improves model robustness under domain shift by selectively querying human annotations at deployment, but existing methods use heuristic uncertainty measures and suffer from low data selection efficiency, wasting human annotation budget. We propose Conformal Prediction Active TTA (CPATTA), which first brings principled, coverage-guaranteed uncertainty into ATTA. CPATTA employs smoothed conformal scores with a top-K certainty measure, an online weight-update algorithm driven by pseudo coverage, a domain-shift detector that adapts human supervision, and a staged update scheme balances human-labeled and model-labeled data. Extensive experiments demonstrate that CPATTA consistently outperforms the state-of-the-art ATTA methods by around 5% in accuracy. Our code and datasets are available at https://github.com/tingyushi/CPATTA.
