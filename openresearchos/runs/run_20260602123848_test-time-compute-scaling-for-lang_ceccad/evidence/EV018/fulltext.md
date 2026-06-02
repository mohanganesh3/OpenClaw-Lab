[2512.23412] MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning

MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning

MindGPT-ov Team

Li Auto Inc

Traditional workflow-based agents exhibit limited intelligence when addressing real-world problems requiring tool invocation. Tool-integrated reasoning (TIR) agents capable of autonomous reasoning and tool invocation are rapidly emerging as a powerful approach for complex decision-making tasks involving multi-step interactions with external environments.
In this work, we introduce MindWatcher, a TIR agent integrating interleaved thinking and multimodal chain-of-thought (CoT) reasoning. MindWatcher can autonomously decide whether and how to invoke diverse tools and coordinate their use, without relying on human prompts or workflows. The interleaved thinking paradigm enables the model to switch between thinking and tool calling at any intermediate stage, while its multimodal CoT capability allows manipulation of images during reasoning to yield more precise search results.
We implement automated data auditing and evaluation pipelines, complemented by manually curated high-quality datasets for training, and we construct a benchmark, called MindWatcher-Evaluate Bench (MWE-Bench), to evaluate its performance. MindWatcher is equipped with a comprehensive suite of auxiliary reasoning tools, enabling it to address broad-domain multimodal problems. A large-scale, high-quality local image retrieval database, covering eight categories including cars, animals, and plants, endows model with robust object recognition despite its small size. Finally, we design a more efficient training infrastructure for MindWatcher, enhancing training speed and hardware utilization.
Experiments not only demonstrate that MindWatcher matches or exceeds the performance of larger or more recent models through superior tool invocation, but also uncover critical insights for agent training, such as the genetic inheritance phenomenon in agentic RL.
The agent reasoning framework, MWE-Bench, three smaller-scale agent models (2B, 3B, and 4B) distilled from MindWatcher 32B, and related resources will be open-sourced.

–

Last Update Date:

December 29, 2025

#

Correspondence:

chenwei10@lixiang.com

Code:

https://github.com/TIMMY-CHAN/MindWatcher

Hugging Face:

https://huggingface.co/datasets/Lost-Cloud/MWE-Bench

1  Introduction

Figure 1 :

MWE-Bench Performance of MindWatcher.

Large language models (LLMs)  [ 27 ,  1 ,  14 ,  42 ,  9 ,  37 ,  25 ]  have achieved remarkable progress in recent years, demonstrating strong capabilities in language understanding, knowledge acquisition, and complex reasoning tasks. However, despite the powerful world knowledge and multimodal capabilities of the latest models such as Gemini 2.5 Pro  [ 9 ] , most LLMs remain fundamentally constrained by the limits of their parametric knowledge: they struggle to cover long‑tail information and fine‑grained domain‑specific knowledge  [ 7 ] , and they cannot directly access real‑time information that emerges after training. These structural bottlenecks hinder their reliability in many real‑world applications, especially those requiring external knowledge, multi‑step information integration, or cross‑modal reasoning. Equipping LLMs with external tools has therefore become a key strategy to overcome these limitations. By connecting models with retrieval engines  [ 19 ,  29 ] , computation tools, or code interpreters, the boundary of problem‑solving capabilities can be substantially extended.

Traditional tool‑augmented approaches typically rely on manually designed workflows  [ 17 ,  33 ]  to orchestrate tool invocation, yet such methods exhibit limited adaptability when confronted with the diversity and uncertainty inherent in open-domain environments, which become even more fragile when handling cross‑modal demands. Multi‑agent systems  [ 41 ,  31 ,  22 ,  21 ]  partially alleviate these issues: a powerful planner agent is responsible for decision‑making, while tool‑specialized agents execute designated subtasks. This architecture has become highly popular in the industry and significantly improves system flexibility and scalability. However, it also introduces new complexity and overhead, including redundant model deployment and latency caused by chained interactions, which limits its further expansion. With the emergence of thought‑augmented models  [ 38 ,  14 ] , the research community increasingly recognizes that intelligent systems need not rely on multi‑component designs: a single unified language model can assume both planning and acting roles. This has led to the rise of Tool‑Integrated Reasoning (TIR) methods  [ 26 ] , exemplified by the ReAct  [ 43 ]  paradigm. The core idea is to let the model explicitly generate intermediate thoughts, autonomously invoke tools, and iteratively make decisions based on environmental feedback. TIR agents can dynamically plan multi‑step operations in open‑world tasks and achieve end‑to‑end problem solving, making them a promising path toward more general‑purpose agents.

However, current TIR systems still fall short of being truly practical and general intelligent agents, with significant limitations across several key dimensions. From an application perspective, existing TIR agents  [ 26 ,  22 ,  11 ]  are predominantly focused on text‑based tasks, particularly DeepSearch‑style reasoning centered on retrieval. Only a small number of works  [ 13 ,  44 ]  attempt to introduce visual capabilities, and most rely solely on image search tools without enabling the agent to directly manipulate images or perform fine‑grained cross‑modal reasoning to support problem solving. This severely limits their performance on multimodal tasks and prevents them from tackling the many vision‑driven decision‑making scenarios found in real‑world environments.

From a training methodology perspective, TIR agents face a triple challenge across data, algorithms, and training frameworks. High‑quality reasoning trajectories involving multiple tools and multi‑step interactions are extremely difficult to construct manually. SFT‑based training  [ 24 ,  6 ]  often causes models to “imitate” the thought‑action format rather than truly “learn” the underlying strategy—manifested in excessive, redundant tool calls on simple problems and substantial performance degradation on general benchmarks. Moreover, existing training frameworks lack fine‑grained supervision over the interleaved process of thinking, tool invocation, and subsequent reasoning, preventing models from forming stable and reliable tool‑use behaviors and exacerbating issues such as tool misuse and unnecessary calls. From the perspective of tool ecosystems, many core retrieval capabilities, especially visual retrieval, rely on expensive external APIs. Their high cost under frequent invocation further constrains the practical deployment of TIR agents in local or enterprise settings.

To address the challenges outlined above, we introduce MindWatcher, a TIR agent capable of autonomous planning and execution, multimodal perception, and robust tool coordination. Leveraging an interleaved thinking paradigm and a multimodal Chain‑of‑Thought (CoT) mechanism, MindWatcher can flexibly alternate between internal thinking and external tool invocation at any stage of the reasoning process. By integrating fine‑grained visual operations into the reasoning chain, the agent achieves precise region‑level visual perception and more accurate cross‑modal information retrieval.

To avoid the drawbacks of conventional SFT, such as rigid imitation of reasoning formats and redundant tool calls on simple tasks, MindWatcher abandons standard SFT and instead adopts a continuous reinforcement learning (RL) strategy conducted in both real and offline environments. We develop two automated image–text pair construction pipelines to reduce data generation costs. In parallel, we equip MindWatcher with