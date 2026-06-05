# EV015: LeMo-NADe: Multi-Parameter Neural Architecture Discovery with LLMs

URL: https://www.semanticscholar.org/paper/008ad76bc4531486062169270584faeab4da1ec2
Year: 2024
Source: semantic_scholar
Arxiv: 2402.18443

## Abstract

Building efficient neural network architectures can be a time-consuming task requiring extensive expert knowledge. This task becomes particularly challenging for edge devices because one has to consider parameters such as power consumption during inferencing, model size, inferencing speed, and CO2 emissions. In this article, we introduce a novel framework designed to automatically discover new neural network architectures based on user-defined parameters, an expert system, and an LLM trained on a large amount of open-domain knowledge. The introduced framework (LeMo-NADe) is tailored to be used by non-AI experts, does not require a predetermined neural architecture search space, and considers a large set of edge device-specific parameters. We implement and validate this proposed neural architecture discovery framework using CIFAR-10, CIFAR-100, and ImageNet16-120 datasets while using GPT-4 Turbo and Gemini as the LLM component. We observe that the proposed framework can rapidly (within hours) discover intricate neural network models that perform extremely well across a diverse set of application settings defined by the user.
