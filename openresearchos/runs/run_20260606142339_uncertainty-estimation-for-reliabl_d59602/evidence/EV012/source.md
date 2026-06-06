# EV012: High-fidelity retrieval from instantaneous line-of-sight returns of nacelle-mounted lidar including supervised machine learning

URL: https://www.semanticscholar.org/paper/0022e50d1226a335827d9e68d76564f32e44edf7
Year: 2022
Source: semantic_scholar
Arxiv: n/a

## Abstract

Abstract. Wind turbine applications that leverage nacelle-mounted
Doppler lidar are hampered by several sources of uncertainty in the lidar
measurement, affecting both bias and random errors. Two problems encountered
especially for nacelle-mounted lidar are solid interference due to
intersection of the line of sight with solid objects behind, within, or in
front of the measurement volume and spectral noise due primarily to
limited photon capture. These two uncertainties, especially that due to
solid interference, can be reduced with high-fidelity retrieval techniques
(i.e., including both quality assurance/quality control and subsequent
parameter estimation). Our work compares three such techniques, including
conventional thresholding, advanced filtering, and a novel application of
supervised machine learning with ensemble neural networks, based on their
ability to reduce uncertainty introduced by the two observed nonideal
spectral features while keeping data availability high. The approach
leverages data from a field experiment involving a continuous-wave (CW)
SpinnerLidar from the Technical University of Denmark (DTU) that provided
scans of a wide range of flows both unwaked and waked by a field turbine.
Independent measurements from an adjacent meteorological tower within the
sampling volume permit experimental validation of the instantaneous velocity
uncertainty remaining after retrieval that stems from solid interference and
strong spectral noise, which is a validation that has not been performed
previously. All three methods perform similarly for non-interfered returns,
but the advanced filtering and machine learning techniques perform better
when solid interference is present, which allows them to produce overall
standard deviations of error between 0.2 and 0.3 m s−1, or a 1 %–22 %
improvement versus the conventional thresholding technique, over the rotor
height for the unwaked cases. Between the two improved techniques, the
advanced filtering produces 3.5 % higher overall data availability, while
the machine learning offers a faster runtime (i.e., ∼ 1 s
to evaluate) that is therefore more commensurate with the requirements of
real-time turbine control. The retrieval techniques are described in terms
of application to CW lidar, though they are also relevant to pulsed lidar.
Previous work by the authors (Brown and Herges, 2020) explored a novel
attempt to quantify uncertainty in the output of a high-fidelity lidar
retrieval technique using simulated lidar returns; this article provides
true uncertainty quantification versus independent measurement and does so
for three techniques rather than one.

