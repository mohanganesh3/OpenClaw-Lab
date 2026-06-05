# EV020: Augmenting sparse spaceflight mass spectra datasets for machine learning applications

URL: https://www.semanticscholar.org/paper/0e41d752e3e2447d3c00f0819e80a67af792b9bc
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract


 Mass spectrometers are powerful instruments that aim to identify unknown compounds via their mass-to-charge ratio and perform quantitative and semi-quantitative analysis. These instruments have been essential to space missions over the past several decades (
 e.g.
 , Pioneer Venus, Viking, Galileo, Cassini, Mars Science Laboratory) with several more en route (
 e.g.
 , JUpiter ICy moons Explorer (JUICE), Europa Clipper) or under development (
 e.g.
 , Rosalind Franklin, Dragonfly). However, future missions targeting remote planetary bodies increasingly face limited data transmission rates and volumes, which limit the amount of information that can be sent back to Earth. These challenges highlight the need for onboard science autonomy to optimize science return. Machine learning (ML) and data science tools can significantly contribute to the development of science autonomy by enabling rapid interpretation and prioritization of science data. Yet, these efforts for planetary science applications are hindered by the scarcity of representative datasets for training models, especially for complex flight instruments. In this work, we build on our earlier science autonomy work using the Mars Organic Molecule Analyzer (MOMA) instrument for the Rosalind Franklin (ExoMars) mission as a proof-of-concept. We investigate the generation of artificial mass spectra through “manual” augmentation techniques and evaluate their performance on mass spectrometer (MS) data using the laser desorption/ionization mass spectrometry (LDMS) mode of the flight-like MOMA engineering test unit (ETU). We implement basic transformation-based augmentation methods such as peak intensity randomization, peak shifting (by limited and realistic
 m/z
 values), etc. We assess their scientific integrity in collaboration with instrument experts and investigate how the inclusion of generated data affects the performance of ML algorithms for mass spectral analysis. We compare the performance of supervised learning models on predicting the chemical categories of new input mass spectra, both with and without augmented data, to evaluate the impact of these techniques. Our work provides guidelines for developing realistic augmented mass spectra without compromising scientific validity, while also contributing to the development of a mature framework for ML tools in MS data analysis, advancing science autonomy for existing and future planetary missions.

