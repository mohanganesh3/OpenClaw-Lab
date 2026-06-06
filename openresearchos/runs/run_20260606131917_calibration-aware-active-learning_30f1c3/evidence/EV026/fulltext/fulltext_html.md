[2411.18721] A Machine Learning Approach Capturing Hidden Parameters in Autonomous Thin-Film Deposition

A Machine Learning Approach Capturing Hidden Parameters in Autonomous Thin-Film Deposition

Yuanlong Zheng  1,2

Connor Blake  2

Layla Mravac  2

Fengxue Zhang  3

Yuxin Chen  3

Shuolong Yang  2

1  Department of Physics, University of Chicago, Chicago, IL 60637, USA

2  Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637, USA

3  Department of Computer Science, University of Chicago, Chicago, IL 60637, USA

Abstract

Abstract

The integration of machine learning and robotics into thin film deposition is transforming material discovery and optimization. However, challenges remain in achieving a fully autonomous cycle of deposition, characterization, and decision-making. Additionally, the inherent sensitivity of thin film growth to hidden parameters such as substrate conditions and chamber conditions can compromise the performance of machine learning models. In this work, we demonstrate a fully autonomous physical vapor deposition system that combines in-situ optical spectroscopy, a high-throughput robotic sample handling system, and Gaussian Process Regression models. By employing a calibration layer to account for hidden parameter variations and an active learning algorithm to optimize the exploration of the parameter space, the system fabricates silver thin films with optical reflected power ratios within 2.5% of the target in an average of 2.3 attempts. This approach significantly reduces the time and labor required for thin film deposition, showcasing the potential of machine learning-driven automation in accelerating material development.

I  Introduction

In physical vapor deposition (PVD) of thin film materials, the traditional human-led process encompasses numerous cycles of selecting deposition parameters, performing deposition, characterizing film properties, and re-adjusting the parameters accordingly. The recent development of machine learning (ML)  [ 1 ] , coupled with advancements in robotics  [ 2 ] , now potentially enables fully automating this process, liberating researchers from the repetitive cycles and accelerating the optimization of material properties  [ 3 ] .

Several studies have sought to integrate ML with the PVD process. Typically, these approaches involved training ML models that map deposition parameters, such as substrate temperature, deposition rate, and flux ratio, to material properties, such as stoichiometry  [ 4 ,  5 ,  6 ] , electrical conductivity  [ 7 ,  8 ,  9 ] , surface morphology  [ 10 ,  11 ] , crystallinity  [ 12 ,  13 ,  14 ,  15 ,  16 ,  17 ] , and superconducting critical temperatures  [ 18 ] . The trained models are then used to predict material properties, with Bayesian optimization (BO) frequently employed to autonomously determine the deposition parameters for subsequent samples  [ 9 ,  19 ,  20 ] . Such optimization has the potential to replace human decision-making, efficiently exploring the deposition parameter space to optimize the material properties  [ 21 ] .

However, ML models are known for their sensitivity to noise in the training data. In the meantime, the thin film deposition process can be prone to irreproducibility due to factors such as different initial substrate conditions and chamber environments  [ 5 ,  8 ,  18 ] . The inconsistency in the growth outcome may be treated by the model as noise which undermines the model’s performance. These “hidden parameters” that lead to the irrepeatability of thin film properties are often difficult to capture and incorporate into the model. The field of ML-assisted thin film deposition currently lacks a systematic approach to effectively account for these hidden parameters.

Figure 1 :

A self-learning autonomous physical vapor deposition system for silver thin film growth. (a) The autonomous learning cycle incorporates (i) identification of the growth condition with the highest model uncertainty, (ii) sample growth, and (iii) updating the model with new data. (b) The autonomous testing cycle incorporating (i) prediction of the optimal growth condition that minimizes the loss function, (ii) sample growth, and (iii) assessment of success. (c) Schematic illustration of the autonomous deposition setup featuring robotic sample handling and in-situ optical characterization.

Beyond algorithmic challenges, the full potential of ML-assisted PVD hinges on the complete automation of hardware systems. PVD systems require high vacuum (HV) or ultra-high vacuum (UHV) environments, which present significant challenges for fully automating sample transfer and characterization processes. As such, most studies in the field of ML-assisted thin film deposition still rely on traditional manual handling of samples, which limits sample throughput and hinders the realization of the fully autonomous ML-assisted PVD  [ 9 ,  18 ,  22 ] . Shimizu et al. demonstrated a system fully automating the deposition of Nb-doped

TiO  2

subscript  TiO  2

\mathrm{TiO}_{2}

films and the minimization of its resistance  [ 23 ] . However, the system requires complex multi-chamber systems with sophisticated transfer mechanisms, thereby increasing the complexity of the setup and limiting its scalability. Achieving a scalable method to fully automate PVD systems is a critical hurdle to unlocking the full promise of ML integration in thin film deposition.

II  Methods and Results

In this work, we address inefficiencies in current ML-driven thin film deposition by developing a fully automated system with high sample throughput and a mechanism to account for hidden parameter variations. Our system integrates a UHV chamber with a 72-slot robotic sample handling system optimized for maximizing throughput, in-situ optical characterization, and ML-powered prediction of characterization outputs and optimal growth conditions. To account for the varying hidden parameters, we introduce a calibration layer that provides critical information about the initial condition of each sample and the chamber environment. Furthermore, we employ an autonomous learning approach using Gaussian Process Regression (GPR) models to efficiently explore the multidimensional input space and accurately predict the target properties of the films. By integrating these techniques, we have engineered an ML-driven, closed-loop deposition system, thereby eliminating the need for human intervention at any stage of the growth cycle. We demonstrate the autonomous synthesis of silver thin films with optical properties within 2.5% of the user-specified target in an average of 2.3 attempts, underscoring the potential of our ML-driven system for significantly accelerating the optimization of material properties.

1. Introduction of the System

To demonstrate the principles of ML-driven autonomous PVD, we seek to fabricate silver thin films with user-specified optical properties. The growth conditions of silver films such as the growth rate, substrate temperature, and film thickness  [ 24 ]  impact the films’ porosity  [ 25 ] , grain size  [ 26 ] , and electron-phonon interactions  [ 27 ] . All of these aspects are correlated with the real and imaginary parts of the optical constants for silver films. It is difficult to model all of these mechanisms using simple physics laws, which warrants an ML-driven material optimization.

Our autonomous PVD system incorporates a shadow mask beneath a 72-slot sample handling system, ensuring that only one sample is exposed to the molecular beam at a time (Figure 1(c)). Silver (99.999%, Thermo Fisher) is deposited onto double-side polished BK7 substrates using an effusion cell (MBE-Komponenten) at a base pressure of

&lt;

5  ×

10

−  9

absent

5

superscript  10

9

&lt;5\times{10}^{-9}

mbar and a deposition pressure of

1  ×

10

−  8

1

superscript  10

8

{1\times 10}^{-8}

mbar. The reflectivity and absorptivity of the silver thin films are character