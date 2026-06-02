[2507.20443] Provable In-Context Learning of Nonlinear Regression with Transformers

Provable In-Context Learning of Nonlinear Regression with Transformers

Hongbo Li

Department of ECE, The Ohio State University

Lingjie Duan

ESD Pillar, Singapore University of Technology and Design

Yingbin Liang

Department of ECE, The Ohio State University

Abstract

The transformer architecture, which processes sequences of input tokens to produce outputs for query tokens, has revolutionized numerous areas of machine learning. A defining feature of transformers is their ability to perform previously unseen tasks using task-specific prompts without updating parameters—a phenomenon known as in-context learning (ICL). Recent research has actively explored the training dynamics behind ICL, with much of the focus on relatively simple tasks such as linear regression and binary classification.
To advance the theoretical understanding of ICL, this paper investigates more complex nonlinear regression tasks, aiming to uncover how transformers acquire in-context learning capabilities in these settings.
We analyze the stage-wise dynamics of attention during training: attention scores between a query token and its target features grow rapidly in the early phase, then gradually converge to one, while attention to irrelevant features decays more slowly and exhibits oscillatory behavior.
Our analysis introduces new proof techniques that explicitly characterize how the nature of general non-degenerate

L  L

-Lipschitz task functions affects attention weights. Specifically, we identify that the Lipschitz constant

L  L

of nonlinear function classes as a key factor governing the convergence dynamics of transformers in ICL.
Leveraging these insights, for two distinct regimes depending on whether

L  L

is below or above a threshold, we derive different time bounds to guarantee near-zero prediction error.
Notably, despite the convergence time depending on the underlying task functions, we prove that query tokens consistently attend to prompt tokens with highly relevant features at convergence, demonstrating the ICL capability of transformers for unseen functions.

1  Introduction

The transformer architecture  [ 1 ]  has driven transformative advances across a wide spectrum of machine learning domains, including computer vision  [ 2 ,  3 ,  4 ] , natural language processing  [ 5 ,  6 ] , and speech processing  [ 7 ,  8 ] . A salient feature of transformers is their ability to perform new tasks without updating parameters, simply by conditioning on a few input-output examples—known as prompts. This capability, referred to as  in-context learning (ICL) , enables models to generalize to unseen tasks purely through inference  [ 9 ] .

ICL has attracted growing interest in recent years, with numerous empirical studies aiming to understand the conditions under which transformers succeed or fail at in-context generalization  [ 10 ,  11 ,  12 ,  13 ,  14 ,  15 ,  16 ] .
Notably,  [ 11 ]  offered preliminary theoretical evidence that transformers trained on specific function classes (e.g., linear functions) can accurately infer the function of a query sample based on prompts of a sequence of variable and their corresponding function value pairs.
This highlights the surprising ability of transformers to "learn" within their forward pass, mimicking classical function approximation.

Building on this foundation, a few works have provided  theoretical  understandings of ICL by characterizing the dynamics of transformer models with one attention layer during the training process  [ 17 ,  18 ,  19 ,  20 ,  21 ,  22 ,  23 ] . For instance,  [ 18 ]  investigates convergence dynamics under gradient flow, but focuses solely on a single linear attention layer applied to linear regression.  [ 19 ]  extends this line of analysis to softmax attention, revealing how attention weights evolve during training. More recent studies consider nonlinear function classes, such as binary classification and low-degree polynomial regression  [ 20 ,  22 ,  23 ] , and characterize how transformers adapt to these tasks during training.

Despite recent advances, most  theoretical  studies of ICL remain limited to simplified function classes, such as linear or low-complexity mappings. However, understanding how transformers learn in practice demands moving beyond these idealized settings—insights derived from such simplified classes often fail to generalize to nonlinear functions commonly encountered in real-world tasks. While extending the analysis to more general function classes is crucial, it presents significant technical challenges due to the inherent complexity of nonlinear functions even for  one-layer  transformers.

In this work, we take a step toward closing this gap by developing a theoretical framework for ICL with more general  nonlinear  regression functions. We investigate two fundamental questions: (1) What salient geometric properties of the target function govern the convergence behavior of transformer-based learning? and (2) Despite the nonlinearity and generality of these functions, can the transformer still learn effectively in context and achieve globally small prediction error? We answer both questions by analyzing the training dynamics of transformers under gradient descent.

We summarize our key contributions as follows.

∙  \bullet

Broad Class of Nonlinear Functions and More General Feature Sets:  Our analysis generalizes previous studies in two folds. (i) In contrast to prior works that focus on linear mappings  [ 18 ,  19 ] , binary classification  [ 20 ] , or low-degree polynomials  [ 23 ] , our analysis applies to much broader non-degenerate

L  L

-Lipschitz task functions without assuming low complexity. This class is satisfied by almost any nontrivial

L  L

-Lipschitz function (e.g., all non-constant linear functions) and excludes only degenerate cases. (ii) Our results also apply to more general feature embeddings without the restrictive orthonormality assumption used in prior work  [ 19 ,  20 ,  25 ,  24 ] .

∙  \bullet

Phase Transition of Training Dynamics between Flat and Sharp Curvature Regimes:  We identify the Lipschitz constant

L  L

of nonlinear function classes as a key factor governing the convergence dynamics of transformers in ICL. An interesting phase transition emerges between the flat and sharp curvature regimes, depending on whether

L  L

is below or above a threshold of order

Θ  ​

(

1

Δ  ​  δ

)

\Theta\left(\frac{1}{\Delta\delta}\right)

, with each regime exhibiting distinct training behavior. In the  flat curvature regime ,

L  L

is below the threshold and induces smaller gradients, which allow for larger step sizes to convergence. In contrast, in the  sharp curvature regime ,

L  L

is above the threshold and incurs large gradients, which necessitate smaller step sizes to converge. Comparing the two, the flat regime may yield faster convergence when a high level of accuracy is required. However, if

L  L

is significantly larger relative to the desired accuracy, the increased curvature enhances feature distinguishability, enabling faster training in the sharp regime.

∙  \bullet

Convergence Guarantee for ICL with Nonlinear Functions:  We provide the formal convergence guarantees for a one-layer softmax attention model trained via gradient descent to perform ICL over  nonlinear regression functions  determined by the Lipschitz constant

L  L

. For both flat and sharp

L  L

-regimes, we show that gradient descent converges to near-zero training loss in polynomial time. We further characterize the distinct convergence rates in the two regimes by analyzing a two-phase training dynamic: in the early phase, attention scores between query tokens and their target features grow rapidly; in the later phase, these scores gradually converge to one, while attention to irrelevant features decays more slowly with oscillations. Further, we prove that query tokens