[2601.11816] POLARIS: Typed Planning and Governed Execution for Agentic AI in Back-Office Automation

\correspondingauthor

Keerthi Koneru ( keerthi.koneru@accenture.com ). All work was done during the authors’ tenure at Accenture.

POLARIS: Typed Planning and Governed Execution for Agentic AI in Back-Office Automation

Zahra Moslemi

University of California, Irvine

Keerthi Koneru

Center for Advanced AI, Accenture

Yen-Ting Lee

University of California, San Diego

Sheethal Kumar

Center for Advanced AI, Accenture

Ramesh Radhakrishnan

Center for Advanced AI, Accenture

( February 5, 2026 )

Abstract

Enterprise back-office workflows require agentic systems that are auditable, policy-aligned, and operationally predictable, capabilities that generic multi-agent setups often fail to deliver. We present  POLARIS  (Policy-Aware LLM Agentic Reasoning for Integrated Systems), a governed orchestration framework that treats automation as typed plan synthesis and validated execution over LLM agents. A planner proposes structurally diverse, type-checked directed acyclic graphs (DAGs); a rubric-guided reasoning module selects a single compliant plan; and execution is guarded by validator-gated checks, a bounded repair loop, and compiled policy guardrails that block or route side effects before they occur. Applied to document-centric finance tasks,  POLARIS  produces decision-grade artifacts and full execution traces while reducing human intervention. Empirically,  POLARIS  achieves a micro-F1 of 0.81 on the SROIE dataset and, on a controlled synthetic suite, achieves 0.95–1.00 precision for anomaly routing with preserved audit trails. These evaluations constitute an initial benchmark for governed Agentic AI. POLARIS provides a methodological and benchmark reference for policy-aligned Agentic AI.

Keywords:  Agentic AI; Enterprise Automation; Back-Office Tasks; Benchmarks; Governance; Typed Planning; Evaluation

1  Introduction

Enterprise back-office automation (e.g., accounts payable, contract checks) imposes requirements that generic multi-agent LLM stacks often fail to meet: actions must be auditable, operationally predictable, policy-aligned, and governed before any side effects occur, and reliably evaluable against reproducible metrics.
In practice, untyped tool calls, best-of-

N  N

prompting, and open-ended retries yield brittle pipelines with unclear provenance and no predictable service-level agreement (SLA) guarantees. We argue that meeting enterprise guarantees requires re-casting orchestration as typed, governed planning and execution where plans are type-checked and structurally diverse; selection is rubric-based and policy-aware; and execution is guarded by validators and compiled policy checks that gate side effects.

Recent progress in Agentic AI has created powerful but ungoverned multi-agent stacks. Modern LLMs now support precise language understanding and the use of controlled tools  ( achiam2023gpt4 ) , enabling agentic systems for operational work.  LLM-based agents  combine core reasoning with domain tools  ( qin2024agentsurvey ;  wang2024agentbench ) ; multi-agent systems assign specialized roles across workflows  ( gaoutil2023autogen ;  hong2024metagpt ) ; and planning-centric frameworks cast orchestration as  agent-oriented planning (AOP)  with typed sub-tasks  ( li2024aop ) . In parallel, workflow-centric approaches represent execution as a directed acyclic graph (DAG) of tool calls with explicit interfaces  ( gaoutil2023autogen ;  hong2024metagpt ) , often following a  plan-then-act  paradigm that synthesizes candidate workflows before selection  ( gentask2025 ;  gaoutil2023autogen ) . However, enterprise deployment of such Agentic AI remains limited by the lack of typed planning, evaluation metrics, and auditable execution paths.

Applying these ideas in regulated settings is challenging: inputs are heterogeneous and legally constrained, and errors carry financial risk.
Common stacks (a) pass untyped input/output (I/O) in free-form messages, (b) treat plan generation as best-of-

N  N

prompting without structural diversity guarantees, (c) lack  compiled  policy checks that gate side effects, and (d) don’t use past experience to guide future decisions; making governed, auditable behavior difficult to guarantee.

To overcome these inherent drawbacks, we introduce  POLARIS , a modular orchestration framework that treats back-office automation as planning over  typed  agents. A Chain-of-Action planner proposes a small set of structurally diverse type-checked DAGs; a lightweight reasoning selector chooses one plan using a rubric over compliance, safe sequencing, and parsimony. Execution is then  guarded : the validator checks gate side effects and drives a targeted parser–validator repair loop, while compiled policy guardrails (e.g., thresholds, currency rules, segregation-of-duties (SoD)) block, route, or annotate actions prior to external effects. The result is an auditable decision object and complete execution trace suitable for regulated environments (Figure

1  ).

Relative to prior multi-agent DAG systems, POLARIS (i) enforces type-sound composition with first-class diversity constraints at plan generation (not just prompt sampling), (ii) uses a rubric-based selector that is policy-aware and emits auditable JSON decisions, and (iii) bounds latency and cost via a validator-gated repair loop that targets only failing fields before any external side-effects. In synthetic invoice suites and the SROIE benchmark, the same plan–select–act loop shows strong extraction accuracy and high-precision policy/anomaly routing while preserving end-to-end lineage.

This work contributes to the broader effort to establish reproducible benchmarks and evaluable frameworks for Agentic AI in enterprise settings. POLARIS provides a reference implementation for typed policy-aware orchestration that can be quantitatively assessed in controlled enterprise scenarios.

The remainder of this paper is organized as follows: Section

2

reviews prior advances in agentic process automation, benchmarking, and programmatic prompting. Section

3

introduces the enterprise task setting, notation, and agent taxonomy used throughout this work. Section

4

formalizes the POLARIS framework, covering typed plan synthesis, rubric-based reasoning, and dependency-aware execution. Sections

4.6  –  4.7

describe the validator-gated repair loop, policy guardrails, and risk control mechanisms. Section

5

presents empirical evaluations and benchmark comparisons, while Section

6

concludes with implications for governed, auditable agentic automation and future benchmark development.

2  Related Work

LLM-based back-office automation spans general orchestration frameworks and domain-specific systems.  ProAgent  reframes Robotic Process Automation as Agentic Process Automation by having an LLM synthesize executable workflows and coordinate sub-agents for data-dependent branches  ( ye2023proagent ) .  SmartFlow  complements this with vision and LLM planning to operate changing enterprise UIs without fragile selectors  ( jain2024smartflow ) . Domain-aligned approaches harden governance:  FinRobot  introduces finance-focused Generative Business Process AI Agents that map intents to ERP tasks via rule-constrained planning, Chain-of-Actions execution, and human-in-the-loop for auditability  ( yang2025finrobot ) ;  Agent-S  formalizes the execution of the SOP with a state-decision planner, an action executor for API/user steps, and shared execution memory for fault recovery  ( kulkarni2025agents ) ; and an  RL-guided multi-agent parser  learns document classification, schema induction, and iterative extraction to handle layout drift in invoices and POs  ( amjad2025agentic ) . Although not enterprise-specific,  Generative Agents  contribute reusable patterns for memory, retrieval, and reflection that stabilize multi-step behaviors  ( park2023generative ) .

Benchmarks and evalu