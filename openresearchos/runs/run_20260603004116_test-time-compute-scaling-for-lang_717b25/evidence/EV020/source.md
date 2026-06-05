# EV020: Historical Knowledge Graphs for Global Maritime Estimated Time of Arrival

URL: https://www.semanticscholar.org/paper/0023c2709e2609407951dab4397c6a928dea3567
Year: 2026
Source: semantic_scholar
Arxiv: 2605.18408

## Abstract

Accurate vessel estimated-time-of-arrival forecasts are critical for port operations and decarbonization, yet global-scale travel-time prediction remains difficult without costly contextual data. Herein, I present a methodology for constructing a historical maritime knowledge graph using only Automatic Identification System (AIS) data. First, segmented trajectories are extracted from noisy AIS data using a Gaussian-mixture-model-based preprocessing pipeline. The graph is then constructed by iteratively processing the trajectories and storing speed distributions stratified by vessel type, time of travel, and direction of travel; the resulting global graph comprises 5,433 geohash-3 nodes and 12,334 edges. The graph can be queried to retrieve travel-time predictions between any two location via a hierarchical, priority-based system that uses historical statistics with principled fallback. On a temporally held-out test set, median RMSE is 22.75 min (segment-level) and 30.90 min (trajectory-level), with 69.1% of trajectories within 20% of actual arrival time. On a second external test set, median RMSE is 27.36 min (segment-level) and 37.46 min (trajectory-level), with 62.1% of trajectories within 20%. These results corroborate the promise of our method, enabling global travel-time prediction and providing a strong foundation for just-in-time arrival planning and emissions reduction.
