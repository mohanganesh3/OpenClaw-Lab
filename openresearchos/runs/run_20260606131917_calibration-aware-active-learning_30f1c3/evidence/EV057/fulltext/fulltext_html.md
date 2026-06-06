[2604.20707] Generative Flow Networks for Model Adaptation in Digital Twins of Natural Systems

[1] \fnm Pascal  \sur Archambault

1] \fnm Houari  \sur Sahraoui

1] \fnm Eugene  \sur Syriani

[1] \orgdiv  DIRO ,  \orgname Université de Montréal,  \state Quebec,  \country Canada

Generative Flow Networks for Model Adaptation in Digital Twins of Natural Systems

[

Email:

syriani@iro.umontreal.ca

Affiliation:  *

Abstract

Digital twins of natural systems must remain aligned with physical systems that evolve over time, are only partially observed, and are typically modeled by mechanistic simulators whose parameters cannot be measured directly. In such settings, model adaptation is naturally posed as a simulation-based inference problem. However, sparse and indirect observations often fail to identify a unique and optimal calibration, leaving several simulator parameterizations compatible with the available evidence. This article presents a GFlowNet-based approach to model adaptation for digital twins of natural systems. We formulate adaptation as a generative modeling problem over complete simulator configurations, so that plausible parameterizations can be sampled with probability proportional to a reward derived from agreement between simulated and observed behavior. Using a controlled environment agriculture case study based on a mechanistic tomato model, we show that the learned policy recovers dominant regions of the adaptation landscape, retrieves strong calibration hypotheses, and preserves multiple plausible configurations under uncertainty.

keywords digital twins, model-driven engineering, simulation-based inference, mechanistic simulation, model adaptation, generative modeling, generative flow networks, natural systems, cyber-biophysical systems, controlled environment agriculture

1  Introduction

Digital twins have become an important technology for the monitoring, prediction, and control of complex systems, particularly in industrial and cyber-physical settings  [ Zhou2022Revisiting ,  Javaid2023DTIndustry40 ] . More recently, the concept has also been extended to natural systems in domains such as smart farming  [ david2023digital ]  and health sciences  [ katsoulakis2024digital ] . This extension raises distinct modeling challenges. In natural systems, observations are often sparse, noisy, delayed, destructive, or costly to acquire, while uncertainty remains central throughout model updating and downstream use  [ lin2021uncertainty ] .

In such settings, digital twin services rely on mechanistic simulators that encode domain knowledge about the underlying physical, chemical, or biological processes. Their usefulness depends on whether they remain consistent with the physical system being twinned. This raises a model-adaptation problem: simulator parameterizations must be updated so that simulated behavior remains credible in the target context, for example under local environmental conditions, cultivar-dependent traits, or configuration-specific responses. Because the objective is to infer parameter settings from observed outputs rather than predict outputs from known parameters, this is naturally an inverse problem. In digital twins of natural systems, however, sparse and indirect observations often fail to identify a unique calibration. Several parameterizations may remain compatible with the same observed behavior, especially when many parameters are not directly measurable  [ gallo2022lack ] .

This motivates the use of simulation-based inference, which addresses parameter inference in simulator-based models when the likelihood is unavailable or intractable. Candidate parameterizations can be proposed, instantiated in the simulator, and evaluated through their discrepancy with observed trajectories without requiring an explicit analytical likelihood. Within this setting, Generative Flow Networks  [ bengio2023foundations ]  (GFlowNets) are appealing because they learn reusable stochastic policies that generate complete configurations with probability proportional to a reward. This makes them well suited to adaptation settings in which several high-quality parameterizations may remain plausible and should be preserved explicitly rather than collapsed into a single fitted solution.

This article studies GFlowNet-based model adaptation in a digital twin setting for greenhouse crop modeling, where a mechanistic tomato growth simulator  [ vanthoor2011methodology ]  is adapted to new observational contexts from sparse trajectories. The greenhouse setting provides a useful testbed because it allows the approach to be analyzed first in a fully enumerable regime, where all candidate parameterizations in the considered search space can be evaluated exactly, and then in a larger regime where exhaustive characterization is no longer feasible. The main contributions of this work are as follows:

•

We formulate mechanistic model adaptation in digital twins of natural systems as a simulation-based inference problem, and position it as a dedicated service within the digital twin lifecycle under sparse observations and structural non-identifiability.

•

We introduce a generative adaptation mechanism based on GFlowNets that preserves and samples multiple plausible simulator parameterizations, rather than reducing adaptation to a single fitted calibration.

•

We report an empirical study on a smart farming model, examining candidate retrieval, alignment with an enumerable target landscape, and the behavior of the approach as the adaptation space becomes larger and less tractable.

The remainder of this article is organized as follows. Section

2

reviews digital twins of natural systems and GFlowNets. Section

3

formulates model adaptation as a simulation-based service within the digital twin lifecycle and introduces the requirements that guide our approach. Section

4

presents the proposed generative formulation and its GFlowNet-based implementation. Section

5

showcases our results and discusses their implications for the model adaptation service, Section

6

presents the related works, and Section

7

concludes the paper.

2  Background

This section reviews the two main concepts framing our approach. We first introduce digital twins of natural systems and the particular challenges they raise for model adaptation under uncertainty. We then present GFlowNets as generative models for structured spaces.

2.1  Digital twins of natural systems

Digital twins are virtual representations of physical systems that remain linked to their real-world counterparts through data exchange and model updating, enabling monitoring, prediction, and decision support  [ kapteyn2021probabilistic ] . Recent work in model-driven engineering further distinguishes digital twins by the nature of the system being twinned, separating engineered from non-engineered contexts and explicitly including biological systems within the latter  [ michael2025mde ] . This distinction matters because moving from artifacts whose structure is largely known by design to natural systems introduces forms of variability, partial observability, and uncertainty that are not controlled in the same way. Natural systems may evolve through phases such as plan, growth, life, and end-of-life  [ michael2025mde ] . Across these phases, the digital twin must coordinate model artifacts, contextual data, and downstream services while the underlying biological or environmental system continues to change. Unlike many engineered systems whose relevant structure is largely fixed by design, natural systems may vary substantially over time as a result of development, environmental exposure, or management conditions. The models embedded in the twin then cannot be treated as static artifacts calibrated once and reused unchanged. They must remain suitable for the current lifecycle phase and operating context of the physical system.

Digital twins for natural systems have recently gained attention in domains