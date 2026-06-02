# EV012: Stanford Researchers Introduced MedAgentBench: A Real-World Benchmark ...

URL: https://www.marktechpost.com/2025/09/16/stanford-researchers-introduced-medagentbench-a-real-world-benchmark-for-healthcare-ai-agents/

Source: openclaw_web_search

Year: unknown

Accessed: 2026-06-02T05:45:17.748Z

## Abstract Or Metadata

A team of Stanford University researchers have released MedAgentBench, a new benchmark suite designed to evaluate large language model (LLM) agents in healthcare contexts. Unlike prior question-answering datasets, MedAgentBench provides a virtual electronic health record (EHR) environment where AI systems must interact, plan, and execute multi-step clinical tasks. This marks a significant ...

## Fetched Snapshot (via OpenClaw infer web fetch)



Stanford Researchers Introduced MedAgentBench: A Real-World Benchmark for Healthcare AI Agents - MarkTechPost

Discord

Linkedin

Reddit

X

Home

Open Source/Weights

AI Agents

Tutorials

Voice AI

Robotics

Newsletter

→ Partner with Us

Search

News  Hub

News  Hub

Premium Content

Read our exclusive articles

Facebook

Instagram

X

Home

Open Source/Weights

AI Agents

Tutorials

Voice AI

Robotics

Newsletter

→ Partner with Us

News  Hub

Search

Home

Open Source/Weights

AI Agents

Tutorials

Voice AI

Robotics

Newsletter

→ Partner with Us

Home

Editors Pick

Agentic AI

Stanford Researchers Introduced MedAgentBench: A Real-World Benchmark for Healthcare AI Agents

Editors Pick

Agentic AI

AI Agents

Tech News

AI Paper Summary

Technology

AI Shorts

Artificial Intelligence

Applications

Language Model

New Releases

Staff

Stanford Researchers Introduced MedAgentBench: A Real-World Benchmark for Healthcare AI Agents

By

Michal Sutter

-

September 16, 2025

A team of Stanford University researchers have released  MedAgentBench , a new benchmark suite designed to evaluate large language model (LLM) agents in healthcare contexts. Unlike prior question-answering datasets, MedAgentBench provides a  virtual electronic health record (EHR) environment  where AI systems must interact, plan, and execute multi-step clinical tasks. This marks a significant shift from testing static reasoning to assessing agentic capabilities in  live, tool-based medical workflows .

https://ai.nejm.org/doi/full/10.1056/AIdbp2500144

Why Do We Need Agentic Benchmarks in Healthcare?

Recent LLMs have moved beyond static chat-based interactions toward  agentic behavior —interpreting high-level instructions, calling APIs, integrating patient data, and automating complex processes. In medicine, this evolution could help address  staff shortages, documentation burden, and administrative inefficiencies .

While general-purpose agent benchmarks (e.g., AgentBench, AgentBoard, tau-bench) exist,  healthcare lacked a standardized benchmark  that captures the complexity of medical data, FHIR interoperability, and longitudinal patient records. MedAgentBench fills this gap by offering a reproducible, clinically relevant evaluation framework.

What Does MedAgentBench Contain?

How Are the Tasks Structured?

MedAgentBench consists of  300 tasks across 10 categories , written by licensed physicians. These tasks include patient information retrieval, lab result tracking, documentation, test ordering, referrals, and medication management. Tasks average 2–3 steps and mirror workflows encountered in inpatient and outpatient care.

What Patient Data Supports the Benchmark?

The benchmark leverages  100 realistic patient profiles  extracted from Stanford’s STARR data repository, comprising over  700,000 records  including labs, vitals, diagnoses, procedures, and medication orders. Data was de-identified and jittered for privacy while preserving clinical validity.

How Is the Environment Built?

The environment is  FHIR-compliant , supporting both retrieval (GET) and modification (POST) of EHR data. AI systems can simulate realistic clinical interactions such as documenting vitals or placing medication orders. This design makes the benchmark directly translatable to live EHR systems.

How Are Models Evaluated?

Metric : Task success rate (SR), measured with strict  pass@1  to reflect real-world safety requirements.

Models Tested : 12 leading LLMs including GPT-4o, Claude 3.5 Sonnet, Gemini 2.0, DeepSeek-V3, Qwen2.5, and Llama 3.3.

Agent Orchestrator : A baseline orchestration setup with nine FHIR functions, limited to  eight interaction rounds per task .

Which Models Performed Best?

Claude 3.5 Sonnet v2 : Best overall with  69.67% success , especially strong in retrieval tasks (85.33%).

GPT-4o : 64.0% success, showing balanced retrieval and action performance.

DeepSeek-V3 : 62.67% success, leading among open-weight models.

Observation : Most models ex
