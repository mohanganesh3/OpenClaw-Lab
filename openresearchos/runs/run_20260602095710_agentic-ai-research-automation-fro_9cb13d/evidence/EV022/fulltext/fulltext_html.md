[2601.11903] AEMA: Verifiable Evaluation Framework for Trustworthy and Controlled Agentic LLM Systems

\correspondingauthor

Keerthi Koneru ( keerthi.koneru@accenture.com ). All work was done during the authors’ tenure at Accenture.

AEMA: Verifiable Evaluation Framework for Trustworthy and Controlled Agentic LLM Systems

Yen-Ting Lee

University of California, San Diego

Keerthi Koneru

Center for Advanced AI, Accenture

Zahra Moslemi

University of California, Irvine

Sheethal Kumar

Center for Advanced AI, Accenture

Ramesh Radhakrishnan

Center for Advanced AI, Accenture

( February 5, 2026 )

Abstract

Evaluating large language model (LLM)-based multi-agent systems remains a critical challenge, as these systems must exhibit reliable coordination, transparent decision-making, and verifiable performance across evolving tasks. Existing evaluation approaches often limit themselves to single-response scoring or narrow benchmarks, which lack stability, extensibility, and automation when deployed in enterprise settings at multi-agent scale. We present  AEMA (Adaptive Evaluation Multi-Agent) , a process-aware and auditable framework that plans, executes, and aggregates multi-step evaluations across heterogeneous agentic workflows under human oversight. Compared to a single LLM-as-a-Judge, AEMA achieves greater stability, human alignment, and traceable records that support accountable automation. Our results on enterprise-style agent workflows simulated using realistic business scenarios demonstrate that AEMA provides a transparent and reproducible pathway toward responsible evaluation of LLM-based multi-agent systems.

Keywords:  Agentic AI, Multi-Agent Systems, Trustworthy AI, Verifiable Evaluation, Human Oversight

1  Introduction

Agentic LLM systems are moving from static prediction to autonomous reasoning, tool use, and collaboration. As they operate in dynamic environments, their trustworthiness depends on ensuring alignment with human intent, robustness under real-world complexity, and verifiability of their internal decisions. These properties motivate evaluation frameworks that provide auditable transparency and bounded autonomy under human oversight.

Evaluating such systems is difficult because reasoning and coordination unfold over many steps; traditional benchmarks cannot verify process-level reliability or alignment with human expectations.
In high-stakes domains where agents operate under human oversight, evaluation must function as a mechanism of trust and control rather than a mere scoring.

We introduce AEMA (Adaptive Evaluation Multi-Agent), a framework for verifiable evaluation of agentic LLM systems. AEMA adapts to diverse tasks and records traceable evaluation logs for oversight and accountability. Unlike single-model evaluators, AEMA operates as a coordinated multi-agent evaluator that plans, debates, and aggregates judgments across steps to produce consistent and explainable assessments under human control.

To demonstrate feasibility and trust reliability, we evaluate AEMA on enterprise-style agent workflows that simulate realistic multi-agent co-ordination under controlled conditions.

AEMA shows lower score dispersion, stronger human alignment, and verifiable consistency compared to a single LLM-as-a-Judge.

In summary, our contributions advance trust and control in agentic AI through the following dimensions:

1.

AEMA introduces a process-aware, verifiable framework that unifies step-level and end-to-end assessment for multi-agent evaluation;

2.

Provides an adaptive methodology that offers support for human-in-the-loop oversight and continuous refinement of evaluation criteria;

3.

Demonstrates verifiable evaluation that enables alignment, robustness, and trust in autonomous LLM systems in enterprise-style environments.

2  Related Work: From Single-Turn Evaluation to Process-Level Assessment of LLM Multi-Agent Systems

LLMs are increasingly used to automate the evaluation, offering greater scalability and reproducibility than human evaluation. The  LLM-as-a-Judge  paradigm has been adopted for a wide range of tasks such as instruction

InstructionFollowing  , summarization

Summarization  , and machine translation

MachineTranslation  . Frameworks like G-Eval

G-Eval

use a single GPT-4 model with a  chain-of-thought  urging the text to score on coherence, relevance and factual accuracy, while ChatEval

ChatEval

assembles multiple LLM “referees”, having diverse expertise, which debate to reach consensus. CollabEval

CollabEval

focuses on building consensus among multiple LLM evaluators using different models rather than adversarial debate. These methods improve consistency, but remain limited to single-turn responses and do not capture multi-step agent reasoning.

Recent research focuses on the evaluation of agentic processes that involve planning, tool use, and interaction. Mind2Web 2

Mind2Web2

benchmarks web agents through an Agent-as-a-Judge rubric that assesses correctness and source attribution in 130 challenging search tasks. WebCanvas

WebCanvas

measures intermediate actions in realistic interface states using an online evaluation framework, and BEARCUBS

BEARCUBS

provides a multimodal benchmark of browsing tasks to test the effectiveness of reasoning.

Another line of work provides broad benchmark suites to measure how well modern LLM-driven agents perform in specialized domains. MLAgentBench

MLAgentBench

introduces a benchmark of 13 end-to-end machine learning experimentation tasks that evaluate agents ability to plan, write code, run experiments, and analyze results. Similarly, ITBench

ITBench

targets real-world IT automation tasks in Site Reliability Engineering (SRE), Compliance/Security (CISO), and Financial Operations (FinOps).

Beyond outcome accuracy, several studies assess behavior quality and collaboration dynamics.
AutoLibra

AutoLibra

derives metrics from clustered human feedback for fine-grained behavior assessment. GEMMAS

GEMMAS

introduces graph-based metrics that measure
collaboration efficiency and redundancy in multi-agent reasoning. WorFEval

WorFEval

compares
workflow graphs against ground-truth structures using subsequence and subgraph matching, while
DevAI

DevAI

monitors the progress of another agent to provide step level feedback on coding tasks.

These studies collectively advance LLM-based evaluation, but primarily assess single-response quality, benchmarked outcomes, or isolated collaboration patterns. In contrast, our AEMA evaluation framework unifies process-level planning, adaptive scoring, and historical trace learning into a reproducible evaluation workflow suited for enterprise-style multi-agent systems.

3  Method

3.1  AEMA Framework

Figure 1 :

Overview of AEMA (Adaptive Evaluation Multi-Agent):  Planning Agent builds the plan; Prompt-Refinement Agent retrieves and prepares examples; Evaluation Agents score intermediate actions; Final Report Agent aggregates results into an auditable, reproducible report.

AEMA  (Adaptive Evaluation Multi-Agent) is a verifiable multi-agent evaluation loop with four roles: Planning, Prompt-Refinement, Evaluation, and Final Report.
It analyzes execution traces and criteria to plan, parameterize, and score step-wise behaviors, then aggregates auditable reports under human oversight Figure

1  . This  multi role workflow creates a controlled evaluation loop  that records every decision for auditability and post-hoc verification.

3.2  Planning Agent

The Planning Agent determines which tasks to evaluate based on the evaluation prompt specified by the human evaluator. By default, it evaluates all actions, but prioritizes those that are semantically meaningful within the workflow. In complex multi-agent systems, agents often engage in extended discussions or reasoning to clarify objectives. As shown in GEMMAS

GEMMAS  , naive agent pipelines tend to generate high redundancy and low diversity in such internal communication. To avoid