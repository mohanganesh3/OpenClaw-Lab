# EV009: AgentClinic: a multimodal benchmark for tool-using clinical AI agents

URL: https://agentclinic.github.io/

Source: openclaw_web_search

Year: unknown

Accessed: 2026-06-02T05:45:17.748Z

## Abstract Or Metadata

Here, we introduce AgentClinic, a multimodal agent benchmark for evaluating LLMs in simulated clinical environments that include patient interactions, multimodal data collection under incomplete information, and the usage of various tools, resulting in an in -depth evaluation across nine medical specialties and seven languages.

## Fetched Snapshot (via OpenClaw infer web fetch)



AgentClinic: a multimodal benchmark for tool-using clinical AI agents

More Research

AgentClinic

Published in npj Digital Medicine (2026)

AgentClinic: a multimodal benchmark for tool-using clinical AI agents

Samuel Schmidgall  1 ,

Rojin Ziaei  2 ,

Carl Harris  3 ,

Ji Woong Kim  4 ,

Eduardo Pontes Reis  5,6 ,

Jeffrey Jopling  7 ,

Michael Moor  8

1 Dept. of ECE, Johns Hopkins University

2 Dept. of CS, Johns Hopkins University

3 Dept. of BME, Johns Hopkins University

4 Dept. of CS, Stanford University

5 Dept. of Radiology, Stanford University

6 Hospital Israelita Albert Einstein

7 Dept. of Surgery, Johns Hopkins Hospital

8 Dept. of Biosystems Science and Engineering, ETH Zürich

Paper (npj Digital Medicine)

arXiv

Code and Dataset

Figure 1.

AgentClinic  simulates clinical encounters with four interacting agents (doctor, patient, measurement, moderator) and equips the doctor agent with clinical tools. The right panel shows an example simulation where the doctor agent diagnoses cavernous sinus thrombosis through history-taking, scan requests, and multimodal data interpretation.

Abstract

Evaluating large language models (LLM) in clinical scenarios is crucial to assessing their potential clinical utility. Existing benchmarks rely heavily on static question-answering, which does not accurately depict the complex, sequential nature of clinical decision-making. Here, we introduce AgentClinic, a multimodal agent benchmark for evaluating LLMs in simulated clinical environments that include patient interactions, multimodal data collection under incomplete information, and the usage of various tools, resulting in an in-depth evaluation across nine medical specialties and seven languages. We find that solving MedQA problems in the sequential decision-making format of AgentClinic is considerably more challenging, resulting in diagnostic accuracies that can drop to below a tenth of the original accuracy. Overall, we observe that agents sourced from Claude-3.5 outperform other LLM backbones in most settings. Nevertheless, we see stark differences in the LLMs' ability to make use of tools, such as experiential learning, adaptive retrieval, and reflection cycles. Strikingly, Llama-3 shows up to 92% relative improvements with the notebook tool that allows for writing and editing notes that persist across cases. To further scrutinize our clinical simulations, we leverage real-world electronic health records, perform a clinical reader study, perturb agents with biases, and explore patient-centric metrics that this interactive environment enables.

LLM Benchmarking

AgentClinic enables comprehensive benchmarking of LLMs as clinical agents. We evaluate a suite of state-of-the-art models across three settings: AgentClinic-MedQA performance based on doctor LLM, patient LLM, and AgentClinic-MIMIC-IV with real-world electronic health records. Claude-3.5 Sonnet achieves the highest diagnostic accuracy across settings, while human physicians perform comparably. The choice of patient agent LLM is a significant factor: performance varies substantially depending on which model simulates the patient.

Figure 2.  Diagnostic accuracy across AgentClinic-MedQA (varying doctor and patient LLMs) and AgentClinic-MIMIC-IV (real-world EHR). Claude-3.5 Sonnet leads in most settings; the patient agent LLM significantly impacts doctor performance.

Static QA vs. Sequential Decision-Making

A core finding of AgentClinic is that static medical QA benchmarks dramatically overestimate clinical competence. When the same MedQA problems are presented in AgentClinic's sequential decision-making format, diagnostic accuracies drop substantially across all models, in some cases to below a tenth of the original accuracy. This gap highlights the importance of evaluating clinical AI in interactive, agent-based settings rather than relying on multiple-choice benchmarks alone.

Figure 3.  Comparison of static MedQA accuracy (dashed) vs. AgentClinic-MedQA agentic a
