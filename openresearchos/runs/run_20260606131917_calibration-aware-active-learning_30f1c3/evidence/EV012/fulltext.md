[2011.14307] Active Output Selection Strategies for Multiple Learning Regression Models

\SelectInputMappings

adieresis=ä,germandbls=ß

Active Output Selection Strategies for Multiple Learning Regression Models

Adrian Prochaska  1

, Julien Pillas  1

and Bernard Bäker  2

1

Mercedes-Benz AG, 71059 Sindelfingen, Germany

2

TU Dresden, Chair of Vehicle Mechatronics, 01062 Dresden, Germany

{ adrian.prochaska, julien.pillas } @daimler.com, bernard.baeker@tu-dresden.de

https://orcid.org/0000-0003-2707-1266

Abstract

Active learning shows promise to decrease test bench time for model-based drivability calibration.
This paper presents a new strategy for  active output selection , which suits the needs of calibration tasks.
The strategy is actively learning multiple outputs in the same input space.
It chooses the output model with the highest cross-validation error as leading.
The presented method is applied to three different toy examples with noise in a real world range and to a benchmark dataset.
The results are analyzed and compared to other existing strategies.
In a best case scenario, the presented strategy is able to decrease the number of points by up to 30 % compared to a sequential space-filling design while outperforming other existing active learning strategies.
The results are promising but also show that the algorithm has to be improved to increase robustness for noisy environments.
Further research will focus on improving the algorithm and applying it to a real-world example.

1  Introduction

Active learning methods – sometimes called  online design of experiments  or  optimal experimental design  – increase the capabilities of algorithms taking part in test design and execution  [ Cohn, 1996 ] .
They reduce the required number of measurements significantly, while guaranteeing adequate model qualities  [ Klein et al., 2013 ] .
However, most methods aim at optimally identifying only one model.
In most real-world applications, there are not one but multiple outputs.
That leaves the test engineer with a question:
Should all models be learned sequentially or simultaneously?
And if they learn simultaneously, how to decide which model is the leading one?
Drivability calibration applications can be further distinguished from other active learning tasks because

•

the goal is to identify all measured outputs equally well and

•

pulling one query reveals the values of all outputs of interest.

[ Dursun et al., 2015 ]  showed a comparison of a sequential and a round-robin learning strategy for a drivability calibration task.
To the authors knowledge, no other publication analyses more sophisticated strategies for multiple learning regression models, which follow the conditions described above.
This paper proposes a new concept of learning strategy, which decides on the leading output by evaluating a cross validation error.
This new strategy is compared to other existing strategies.
Multiple toy examples are used to create a noisy but reproducible test environment with different complexities.
At last, the strategy is also applied to a benchmark dataset.

The paper is structured in six sections.

Section

2

of this paper introduces previous works in context of active learning in general and in particular for regression tasks.

Section

3

focuses on describing the specialties of active learning in the calibration context.
A new active learning task called  active output selection  (AOS) is introduced there.

Section

4

describes the analyzed approaches.
Furthermore, a new approach for AOS is presented.
The approaches are evaluated using a toy example and a benchmark dataset.
Experimental details and a discussion of results are shown in

section

5  .
At the end,

section

6

concludes the results and presents fields of possible future works.

2  Previous work

The field of active learning is a growing branch of the very present machine learning domain.
It is also referred to as  optimal experimental design

[ Cohn, 1996 ] .
 [ Settles, 2009 ]  shows a broad overview of the current state of the art in this discipline and gives an outlook to multiple possible future work fields.
Recent methodological advances in the scientific community mainly focused on classification problems.
The main application domains are speech recognition and text information extraction  [ Settles, 2009 ] .

While regression tasks in the context of active learning have not been as popular, the methodological development is relevant as well.
 [ Sugiyama and Rubens, 2008 ]  propose an approach which actively learns multiple models for the same task and picks the best one to query new points.
 [ Cai et al., 2013 ]  introduced an approach which uses expected model change maximization (EMCM) to improve the active learning progress for gradient boosted decision trees, which was later extended to choose a set of informative queries and to gaussian process regression models (GPs) by  [ Cai et al., 2017 ] .
 [ Park and Kim, 2020 ]  propose a learning algorithm based on the EMCM, which handles outliers more robustly than before.
Those publications focus on new criteria for single output regression models to improve the active learning process.
 [ Zhang et al., 2016 ]  present a learning algorithm for multiple-output gaussian processes (MOGP) which outperforms multiple single-output gaussian processes (SOGP).
However, this publication focuses on improving the prediction accuracy of one target output with the help of several correlated auxiliary outputs. The experiments indicate that a global consideration is beneficial.

There were also advances in active learning for automotive calibration tasks for which the identification of multiple process outputs in the same experiment is more relevant to the application.
 [ Klein et al., 2013 ]  applied a design of experiments for hierarchical local model trees (HiLoMoT-DoE), which was presented by  [ Hartmann and Nelles, 2013 ] , successfully to an engine calibration task.
They presented two application examples with two outputs each and five respectively seven inputs.
The two outputs were modeled with a sequential strategy, which identifies an output model completely before moving to the next one  [ Klein et al., 2013 ] .

[ Dursun et al., 2015 ]  applied the HiLoMoT-DoE active learning algorithm to a drivability calibration example characterized by multiple static regression tasks with identical input spaces.
They analyzed the sequential strategy already shown by  [ Klein et al., 2013 ]  and compared it to a round-robin strategy, which switches the leading model after each iteration/measurement  [ Dursun et al., 2015 ] .
The authors show that the round-robin strategy outperforms offline methods and the online sequential strategy in this experiment.
It might indicate, that round-robin is preferably used in general, but further experiments are necessary.
Since then, no efforts have been made to analyze active learning strategies for multiple outputs.

3  Problem definition

The analyses of this paper are motivated by the field of model-based drivability calibration.
For this application, an active learning algorithm learns a number of

M

𝑀

M

different outputs, which are possibly non-correlated.
Their models are equally important for succeeding optimizations, so the goal is to identify adequate models for all outputs.
The input dimensions of all models are the same.
Querying a new instance corresponds to conducting a measurement on powertrain test benches.
Therefore, a measurement point is cost-sensitive, which is inherent to active learning problems.
Contrary to other applications, every single measurement provides values for all

M

𝑀

M

outputs  1

1  1 This is in contrast to e. g. geostatistics, where measuring any individual output, even at the same place (i. e. model inputs), has its own costs  [ Zhang et al., 2016 ] .

.
Tasks of simultaneously learning

M  &gt;  1

𝑀  1

M&gt;1

process outputs with equal priority and mul