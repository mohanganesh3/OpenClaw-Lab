# EV002: CoBiTS: Single-detector discrimination of binary black hole signals from glitches using deep learning

URL: https://www.semanticscholar.org/paper/001754e0cc23dc724fcbc5c06dc271c48b657841
Year: 2025
Source: semantic_scholar
Arxiv: 2512.17975

## Abstract

We develop a Conformer neural network, called Conformer Binary neTwork Search, or CoBiTS, for distinguishing binary black hole (BBH) gravitational wave (GW) signals from non-Gaussian and non-stationary noise artifacts in the data from current generation LIGO-Virgo-KAGRA detectors. A large subset of these transient noise artifacts, termed as ``glitches''for short, trigger BBH search templates. Some of them go on to produce detection candidates and require human vetting, supported by data quality tools, to be correctly identified and vetoed. In its current version, CoBiTS takes as inputs single-detector strain timeseries snippets, claimed by other search pipelines to be containing GW candidates, and outputs the significance of each snippet to contain a BBH signal and a glitch. CoBiTS is shown to be particularly effective in discriminating high-mass BBH signals from blips and scattered light glitches, even when a signal is near concurrent or overlapping with a glitch. The performance of CoBiTS gains from employing Conformer, which is a specialized model that combines convolutional layers and Transformer architecture for sequence modeling tasks. Conformer is especially good at leveraging the strengths of both convolutional layers -- for local feature extraction -- and self-attention layers -- for capturing long-range dependencies.
