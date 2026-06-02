[2405.02517] Mothman at SemEval-2024 Task 9: An Iterative System for Chain-of-Thought Prompt Optimization

Mothman at SemEval-2024 Task 9:
 An Iterative System for Chain-of-Thought Prompt Optimization

Alvin Po-Chun Chen

Ray Groshan

Sean von Bayern
 University of Colorado Boulder

{alvin.chen, ray.groshan, sean.vonbayern}@colorado.edu

Abstract

Extensive research exists on the performance of large language models on logic-based tasks, whereas relatively little has been done on their ability to generate creative solutions on lateral thinking tasks. The  BrainTeaser  shared task tests lateral thinking and uses adversarial datasets to prevent memorization, resulting in poor performance for out-of-the-box models. We propose a system for iterative, chain-of-thought prompt engineering which optimizes prompts using human evaluation. Using this shared task, we demonstrate our system’s ability to significantly improve model performance by optimizing prompts and evaluate the input dataset.  1

1  1 Our code can be found at  https://github.com/alvin-pc-chen/semeval_brainteaser .

Mothman at SemEval-2024 Task 9:
 An Iterative System for Chain-of-Thought Prompt Optimization

Alvin Po-Chun Chen

and  Ray Groshan

and  Sean von Bayern

University of Colorado Boulder

{alvin.chen, ray.groshan, sean.vonbayern}@colorado.edu

1  Introduction

The ability for language models to reason or possess common sense knowledge has become a controversial topic with far-reaching implications  (Bender and Koller,  2020 ) . Large language models (LLMs) show remarkable results on  vertical thinking  tasks that require sequential logical inference  (Liu et al.,  2019 )  but there have been relatively few studies done on  lateral thinking  puzzles—tasks that require more creative, “outside the box" problem-solving processes. As larger LLMs with the ability to memorize large corpora  (Hartmann et al.,  2023 )  are developed, lateral thinking tasks become an increasingly important benchmark for analyzing and evaluating their reasoning capacities. The  BrainTeaser  shared task  (Jiang et al.,  2023 )  (Jiang et al.,  2024 )  is designed to elicit and evaluate lateral thinking through two English-language subtasks, using sentence puzzles and word puzzles respectively.

In this paper, we propose a novel method for optimizing chain-of-thought (CoT) prompting  (Wei et al.,  2023 )  on the  GPT-4  model which we use to tackle the sentence puzzle subtask. Our system iteratively optimizes CoT prompting by systematically evaluating input data and model output using human performance as a benchmark. We identify question types that are difficult for humans, informing the next iteration of prompt engineering. Not only does this process optimize CoT prompting for a specific task, our system also provides insights for improving future data collection and synthesis.

Our main contribution is the novel approach for identifying reasoning challenges to optimize prompting. For the sentence-based task, we develop a prompt engineering method which requires the model to reason over all answer choices and provide explanations for both correct and incorrect choices. In doing so, the model is more likely to refute choices that are semantically related to the question but logically incorrect. Our methodology significantly improves performance for adversarial datasets and achieves more consistent results, which suggests that the model relies less on memorization when using these CoT prompts.

As part of our evaluation of the data, we also identify several questions in the adversarial datasets that are difficult to solve due to having multiple logical options or are unanswerable with the provided premises. By combining model reasoning with human evaluation, we can quickly identify and evaluate problematic questions. This process can further explain model performance and provide guidance for future data collection/generation.

2  Background

Question Answering (QA) is a well-established task in natural language processing with broad applications both in academia and in industry  (Hirschman and Gaizauskas,  2001 ) . Recent work such as CommonSenseQA (CSQA)  (Talmor et al.,  2018 )  and StrategyQA  (Geva et al.,  2021 )  focus on reasoning questions that require logical inference in the form of vertical thinking.  BrainTeaser  questions instead require lateral thinking to answer, much like questions in the traditional "brainteaser" style (Jiang et al.,  2023 )  (Jiang et al.,  2024 ) :

Base:

Samuel was out for a walk when it started to rain. He did not have an umbrella and he wasn’t wearing a hat. His clothes were soaked, yet not a single hair on his head got wet. How could this happen?

1.

His hair is dyed.

2.

This man is bald.

3.

This man walk upside down to avoid rain.

4.

None of above.

SR:

Rain began to fall as Samuel was taking a stroll. He wasn’t wearing a hat, and he didn’t have an umbrella. Even though his clothes were completely drenched, not a single hair on his head was moist. How is this even possible?

1.

This man walk upside down to avoid rain.

2.

His hair is dyed.

3.

This man is bald.

4.

None of above.

CR:

Tom is a clean freak but he never dries his hair after a shower. How is this possible?

1.

His hair is dyed.

2.

He tries to stand upside down during shower to avoid rain.

3.

This man is bald.

4.

None of above.

The data for this subtask is drawn from online English-language riddles and brainteasers, with incorrect choices created by handpicking entailments generated by  COMET

(Bosselut et al.,  2019 )  using incorrect premises. Each question has three unique answers, as well as a shared fourth option, "None of above". To counter memorization from LLMs trained on web crawls, the task authors generated two synthetic datasets using  semantic reconstruction  (SR) and  context reconstruction  (CR). The SR dataset rephrases the original question without changing the answer or premises while the CR dataset changes the situational context without changing the misleading premise. The SR dataset was generated with an open-source rephrasing tool while the CR dataset was generated using  GPT-4 ; both sets were manually refined by human annotators. In total, 208 question/answer pairs were sampled for the base set resulting in 624 questions after SR and CR augmentation. The training set was split with 81.25% of the data with the same base/SR/CR questions kept together in the split.

Although the task is designed to elicit lateral thinking, we consider an alternative understanding of the task by thinking of the questions as  noisy . Questions are loaded with irrelevant, contradictory, or misleading information to distract the respondent. Since transformers generally learn meaning by scoring tokens across the sentence or sentence pair, they are biased against long-tail knowledge  (Li et al.,  2023 ) , which is knowledge that occurs infrequently in the training set.

Brainteasers, by their nature, rely on the unconventional interpretation of the question to stump the respondent. This same property can trick the model into selecting a semantically similar answer choice that is logically incorrect. Chain-of-Thought (CoT) prompt engineering  (Wei et al.,  2023 )  is a recent method that has been shown to not only improve outcomes on similar problems, but also to provide an interpretable window for human review. CoT prompts provide example questions with related reasoning to the model, which induces the model to provide reasoning for a given answer in the output. We utilize both of these properties to introduce an iterative method that optimizes CoT prompting for a given task.

3  System Overview

We propose an iterative system for optimizing the CoT prompt engineering process:

1.

Randomly sample the training data and naively engineer CoT prompts.

2.

Identify distinct categories in output reasoning to partition training data.

3.

Perform independent human evaluation to isolate speci