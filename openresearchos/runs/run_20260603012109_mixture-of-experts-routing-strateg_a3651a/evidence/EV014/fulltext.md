[2311.02187] An algorithmic framework for synthetic cost-aware decision making in molecular design

An algorithmic framework for synthetic cost-aware decision making in molecular design

Jenna C. Fromer

Connor W. Coley

ccoley@mit.edu

[

Abstract

Small molecules exhibiting desirable property profiles are often discovered through an iterative process of designing, synthesizing, and testing sets of molecules. The selection of molecules to synthesize from all possible candidates is a complex decision-making process that typically relies on expert chemist intuition. We propose a quantitative decision-making framework, SPARROW, that prioritizes molecules for evaluation by balancing expected information gain and synthetic cost. SPARROW integrates molecular design, property prediction, and retrosynthetic planning to balance the utility of testing a molecule with the cost of batch synthesis. We demonstrate through three case studies that the developed algorithm captures the non-additive costs inherent to batch synthesis, leverages common reaction steps and intermediates, and scales to hundreds of molecules. SPARROW is open source and can be found at  \url github.com/coleygroup/sparrow.

keywords:  American Chemical Society,  L a T e X

\SectionsOn  \SectionNumbersOn

Unknown University]
Department of Chemical Engineering, MIT, Cambridge, MA 02139
 \alsoaffiliation [Second University]
Department of Electrical Engineering and Computer Science, MIT, Cambridge, MA
02139

\abbreviations IR,NMR,UV

1  Introduction

Small molecules exhibiting desirable property profiles, such as therapeutic candidates, are often optimized through an iterative process of designing, synthesizing, and testing sets of molecules to better understand the relationship between structure and function. The key challenge in each design iteration is to downselect and prioritize, among all possible molecules that  could  be made and tested, which candidates are worth pursuing.

A myriad of computational workflows can aid in the prioritization of molecules, but they each make simplifying assumptions to the overarching goal of molecular design.
Generative models, for example, often propose molecules that are
impractical to synthesize

1 ,  2

and therefore costly to evaluate. Beyond manual inspection of molecules

3

, candidates can be evaluated for synthetic accessibility with synthetic complexity or accessibility score filters

4 ,  5 ,  6 ,  7

and/or retrosynthetic software

8 ,  9 ,  10

. These approaches, however, do not capture the non-additive costs of synthesizing a batch of molecules. The consideration of synthetic cost may be better described as an art than a science at present, explaining the lack of quantitative decision-making frameworks that we feel are suitable for automatically selecting molecules, for example, in a lead optimization campaign.

The framework of Bayesian optimization

11 ,  12 ,  13 ,  14

partially captures the complexity of iterative design cycles and the challenge of prioritizing molecules for testing.  Cost-aware  Bayesian optimization selects experiments based on acquisition scores that aim to balance experimental cost and expected  utility , which may measure the predicted information gain or the likelihood of finding a compound with superior properties to prior observations

15 ,  16 ,  17 ,  18

. However, cost-aware approaches presume a specific numerical cost for each experiment and cannot capture the non-additivity of synthetic costs for a batch of multiple molecules. The use of common intermediates and starting materials, parallel library chemistry, and laboratory automation can significantly influence the cost of molecular synthesis. Methods that appropriately accommodate the value and cost of a hypothetical set of experiments could both accelerate molecular design campaigns and expand the adoption of computer-aided molecular design tools.

In this work, we propose
Synthesis Planning And Rewards-based Route Optimization Workflow (SPARROW): an algorithmic decision-making framework for driving design cycles (Figure

1  ). This work builds upon prior problem formulations for the simultaneous selection of synthetic routes to multiple molecules

19 ,  20 ,  21 ,  22

and the integration of product and process systems design

23 ,  24 ,  25 ,  26

. SPARROW downselects molecules and their hypothetical synthetic routes from a pool of candidates using a multi-objective optimization criterion. It will benefit from advances in generative modeling for design ideation, computer-aided synthesis planning, and structure-property relationship modeling and uncertainty quantification.
An open source implementation of SPARROW is made available at  \url github.com/coleygroup/sparrow.

We demonstrate SPARROW’s ability to orchestrate molecular design cycles through three case studies.
These applications illustrate how SPARROW
(1) successfully balances information gain and synthetic cost,
(2) captures the non-additivity of synthetic costs for a batch of molecules,
and (3) scales to candidate libraries of hundreds of molecules. Importantly,
SPARROW provides a unified framework for simultaneously evaluating suggestions from virtual libraries, de novo design algorithms, and human experts.

Figure 1:  Overview of SPARROW and its role within the molecular design cycle. Each molecule in a candidate set, comprising molecular ideas from any combination of algorithmic or expert sources, is annotated with its anticipated properties and potential synthetic routes. These annotations can make use of quantitative structure-property relationship models with or without uncertainty quantification as well as computer-aided synthesis planning tools or human experts. SPARROW then weighs the utility of every candidate against their synthetic costs, not one-by-one but as a batch, and selects an optimal subset of candidates for synthesis and testing.

2  Methodology

Optimization of reward per unit cost

SPARROW aims to select molecules and synthetic routes that maximize the utility of synthesized compounds while minimizing synthetic cost. Such an optimization necessitates quantifying the utility of synthesizing a molecule. The utility, or value, associated with a candidate may be understood as an acquisition score that quantifies information gain in the context of Bayesian optimization. It can be a function of the predicted properties of the molecule, uncertainty in those predictions, or the potential for a new datapoint to improve a structure-property relationship. An appropriate measure of utility will vary across molecular design applications and at different stages of design. A candidate library must be provided to SPARROW with corresponding  rewards , which indicate the utility or
information gain associated with evaluating the properties of each molecule. Hereafter, we use utility and reward interchangeably.

The reward that is achieved by synthesizing a molecule also depends on the success of the reaction steps used to synthesize it. If a reaction step in the route to a candidate fails, no information can be gained. We formalize this by aiming to maximize the  expected  reward of selecting a candidate molecule, which can be represented by its reward multiplied by the probability that it is successfully synthesized (Figure

2  ). Balancing cost and utility, the objective of SPARROW may be formalized as the expected reward for all selected targets (i.e. candidates) divided by the cost of synthesizing all selected targets using selected routes (Figure

2  ).

Figure 2:  SPARROW’s problem formulation. A nonlinear objective function is defined that maximizes the expected reward per unit cost of selected candidates and routes. We currently simplify this into a tractable objective function that balances utility and cost through a weighted sum. Three constraints are included to ensure that selected compounds have reactions to produce them, selected reactions have reactants to run them, a