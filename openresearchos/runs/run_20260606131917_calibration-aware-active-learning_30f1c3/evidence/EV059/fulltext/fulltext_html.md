[2403.15959] Risk-Calibrated Human-Robot Interaction via Set-Valued Intent Prediction

\DeclareBoldMathCommand  \vpi

π

Risk-Calibrated Human-Robot Interaction

via Set-Valued Intent Prediction

Justin Lidard, Hang Pham, Ariel Bachman, Bryan Boateng, Anirudha Majumdar

Department of Mechanical and Aerospace Engineering
 Princeton University, Princeton, New Jersey 08540
 Email:  jlidard@princeton.edu

Abstract

Tasks where robots must cooperate with humans, such as navigating around a cluttered home or sorting everyday items, are challenging because they exhibit a wide range of valid actions that lead to similar outcomes. Moreover,  zero-shot  cooperation between human-robot partners is an especially challenging problem because it requires the robot to infer and adapt  on the fly  to a latent human intent, which could vary significantly from human to human. Recently, deep learned motion prediction models have shown promising results in predicting human intent but are prone to being  confidently incorrect.  In this work, we present Risk-Calibrated Interactive Planning (RCIP), which is a framework for measuring and calibrating risk associated with uncertain action selection in human-robot cooperation, with the fundamental idea that the robot should ask for human clarification when unertainty in the human’s intent may adversely affect task performance. RCIP builds on the theory of  set-valued risk calibration  to provide a finite-sample statistical guarantee on the cumulative loss incurred by the robot while minimizing the cost of human clarification in complex multi-step settings. Our main insight is to frame the risk control problem as a  sequence-level  multi-hypothesis testing problem, allowing efficient calibration using a low-dimensional parameter that controls a pre-trained risk-aware policy. Experiments across a variety of simulated and real-world environments demonstrate RCIP’s ability to predict and adapt to a diverse set of dynamic human intents.  1

1  1 Website with additional information, videos, and code:  https://risk-calibrated-planning.github.io/

I

Introduction

Predicting and understanding human intent is a critical task for robotics, specifically for safe interaction with humans in cluttered, close-quarters environments. However, human intent prediction is challenging because no two humans may have the same preferences, and intents may differ depending on the specific environment. As an example, a robot is tasked with sorting items into three bins based on an example provided by the human (see Fig.  1 ). While the bins have a ground-truth sorting criterion known by the human (vegetables, children’s toys, and miscellaneous orange items), the robot must infer the human’s intent in order to sort new items. Given the provided context, the robot should be able to sort some unambiguous items (e.g. the crab) autonomously, while other items (e.g. the carrot) may be placed into multiple bins, resulting in  situational ambiguity . If asked to operate fully autonomously, the robot must take a  risk  and guess the correct placement for the carrot. However, the robot may also  ask for help  if it is unsure, guaranteeing the correct action but potentially burdening the human.
In this work, we study the tradeoff between risk and autonomy governing optimal action selection in the face of situational ambiguity.

Recently, calibrated predict-then-plan (also known as contingency planning)  [ 1 ,  2 ]  approaches have demonstrated the ability to generate provably safe plans by first using confidence-aware prediction models to generate a set of possible futures and then constructing a safe plan that accommodates for the future uncertainty. These approaches enable synthesis of large amounts of scene-specific context (such as image or map information) while simultaneously providing a guarantee on the plan success rate by calibrating the coverage of the prediction. However, one of the major challenges of predict-then-plan approaches comes from  multi-modal human behavior : if the distribution of human actions contains multiple high-level behaviors, a single robot plan may become overly conservative in trying to accommodate all possible human intents. Moreover, environments themselves may generate additional sources of ambiguity that may result in unsafe behavior from the robot if misinterpreted. In such cases, if possible, the robot should ask for help in order to clarify the human’s intent instead of committing to a potentially unsafe action.

Our approach utilizes deep-learned human intent prediction models (e.g.  [ 3 ,  4 ] ) for understanding interactivity, and rigorously quantifies the uncertainty of these models in order to decide when to ask for help. As shown in Fig.  1  (middle), we produce a limited set of human intents based on the prediction model’s confidence scores. For each predicted intent, we plan a sequence of actions that satisfy an environment objective, such as placing the item in the correct bin. To accommodate different levels of robot autonomy, we assume that the predictor has a small number of highly flexible hyperparameters (such as the temperature), which allow the end-user to specify high-level behaviors (more or less confident predictions). We use a small calibration dataset of human-robot interactions to choose a set of valid hyperparameters that provide a level of risk and autonomy set in advance by the user. By leveraging recent advances in distribution-free risk control  [ 5 ] , we show that the robot’s behavior can simultaneously limit several notions of risk. We formalize this challenge via two objectives: (i)  statistical risk calibration : the robot should seek sufficient help from the human when necessary to ensure a statistically guaranteed level risk specified by the user, and (ii)  flexible autonomy : the robot should ask for a minimal amount of help as specified by the user by by narrowing down situational ambiguities through planning. We refer to these simultaneous objectives, with help from the human when necessary, as Risk-Calibrated Interactive Planning (RCIP).

Statement of contributions.  In this work, we introduce RCIP, a framework for measuring and calibrating risk in situations that involve interactions with humans with potentially ambiguous action choices. By reasoning about the human’s desired task outcome in the space of  intents , we efficiently plan safe actions in the face of diverse, multi-modal human behavior, and ask for help when necessary. We make the following contributions:  (1)  We demonstrate how to use statistical risk control (SCR) to control the planning error rate across a set of model hyper-parameters, allowing flexible but provably safe levels of autonomy.  (2)  We prove theoretical guarantees for multi-dimensional risk control for both single-step and multi-step planning problems: with a set of user-specified risk budgets

(

α  1

,  …  ,

α  K

)

subscript  𝛼  1

…

subscript  𝛼  𝐾

(\alpha_{1},...,\alpha_{K})

for different measures of risk (e.g., probability of failure and probability that the robot asks for help) and the robot performs the task correctly (with high probability) by asking for help if any of the

K

𝐾

K

risk budgets will be violated.  (3)  We evaluate RCIP in both simulation and hardware with a suite of human-robot interactive planning tasks with various styles of situational ambiguity (spatial, contextual, semantic). Experiments across multiple platforms and human uncertainty showcase the ability of RCIP to provide statistically guaranteed task success rates while providing more flexible autonomy levels than baseline approaches. RCIP reduces the amount of human help by

5  −

30  %

5

percent  30

5-30\%

versus baseline approaches.

II

Related Work

RCIP brings together techniques from contingency planning, human intent prediction, and conformal prediction and empirical risk control. We discuss related work in each area below.

II-A

Contingency Pla