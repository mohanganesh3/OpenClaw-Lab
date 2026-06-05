# EV012: Surrogate-Assisted Evolutionary Multiobjective Neural Architecture Search Based on Transfer Stacking and Knowledge Distillation

URL: https://www.semanticscholar.org/paper/007ec99ea06849f5b9713fdd0ce150865c7fb84c
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract

Multiobjective neural architecture search (MONAS) methods based on evolutionary algorithms (EAs) are inefficient when the evaluation of each architecture incorporates parameter learning from scratch. A surrogate-assisted multiobjective neural architecture search (MONAS) problem can be tough considering cold-start in surrogate construction, and the evaluation of predicted promising architectures could still be cumbersome. Previously solved MONAS problems are likely to convey useful knowledge that could assist solving the current MONAS problem. To take the benefit from knowledge of these previous practices, a framework tackling large-scale knowledge transfer is proposed. Through sparse-constraint transfer stacking, the surrogate for the current problem gets informative easily. With knee-region knowledge distillation from previously learned parameters of nondominated architectures, evaluation of current architectures could be efficient and credible. To avoid transferring knowledge from irrelevant problems, an iterative source selection algorithm is designed to avoid negative transfer. The proposed framework is analyzed under different source and target MONAS problem combinations. Results show that with the help of this framework, architectures with competitive performance could be found under limited evaluation budget.
