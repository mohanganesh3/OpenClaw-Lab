# EV019: AI Hydrologic Nowcasting in HEC-HMS: Graph Neural Nets under Nonstationarity

URL: https://www.semanticscholar.org/paper/6cc20a83bf984214bce419ba2b25209d511f2f31
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

Abstract:
Operational river nowcasting is challenged by climate- and land-use–driven nonstationarity that degrades purely process-based or purely data-driven models. This study proposes an AI hydrologic nowcasting framework that couples HEC-HMS process states with graph neural networks (GNNs) defined on a directed river–subbasin topology. The approach represents each subbasin/reach as a node/edge; GNN encoders ingest HEC-HMS internal states, precipitation and temperature forcings, and antecedent flows to produce deterministic and probabilistic discharge nowcasts. Methods include a diffusion convolutional recurrent network (DCRNN) with physics-aware inputs, Bayesian calibration for reliability (isotonic regression with temperature scaling), and drift diagnostics based on covariate shift tests.

In synthetic–to–realistic HIL-style studies configured from observed basins, the DCRNN+HMS hybrid improved Nash–Sutcliffe Efficiency from 0.62 to 0.84 and reduced Continuous Ranked Probability Score from 0.205 to 0.135 at near–real-time runtimes; additional metrics included peak timing error, exceedance probability coverage, and reliability diagrams. Implications are a reproducible path to robust nowcasting that preserves physical interpretability while adapting to evolving hydroclimate regimes. Limitations include sensitivity to twin fidelity, sparse gauge networks, and potential extrapolation error under unprecedented events, motivating adaptive re-calibration and active learning. Why it matters: reliable short-horizon discharge forecasts underpin flood early warning and reservoir operations under accelerating nonstationarity. What’s inside: a reference graph-physics architecture, drift-aware training and calibration, and governance-ready evaluation protocols.

Keywords: HEC-HMS, hydrologic nowcasting, graph neural networks, nonstationarity, domain adaptation, Bayesian calibration, river discharge forecasting, DCRNN, NSE, CRPS, uncertainty quantification, data assimilation.
