[2512.09458] Chapter 3: Architectures for Building Agentic AI

1

1  institutetext:  Slawomir Nowaczyk

2

2  institutetext:  Center for Applied Intelligent Systems Research, Halmstad University, Sweden

2

2  email:  slawomir.nowaczyk@hh.se

Chapter 3: Architectures for Building Agentic AI

Slawomir Nowaczyk \orcidID 0000-0002-7796-5201

1  Introduction: purpose, scope, and architecture reliability

This chapter surveys architectural choices for building agentic AI systems and analyses how those choices shape reliability. Our central claim is straightforward:  reliability is, first and foremost, an architectural property . It emerges from how we decompose a system into components, how we specify and enforce interfaces between them, and how we embed control and assurance loops around the parts that reason, remember, and act. Individual models matter, but without the right architectural scaffolding, even state-of-the-art models will behave inconsistently, be impossible to audit, and prove fragile in the face of novelty.

This is a preprint of a chapter accepted for publication in  Generative and Agentic AI Reliability: Architectures, Challenges, and Trust for Autonomous Systems , published by Springer Nature.

Agentic AI  in this book denotes systems that pursue goals over time by  deciding  what to do next,  selecting and using tools, consulting and updating memory , and  interacting with their environment  under constraints. An agent is not merely a predictor; it is a decision-maker in a closed loop. It observes, plans (or at least chooses), acts, and learns, typically under uncertainty and partial observability.  Generative AI  refers to models that synthesise content—text, code, images, plans, or intermediate representations—often serving as the reasoning substrate inside the agent, or providing artefacts (queries, programs, simulations, explanations) that other components execute or verify. In modern systems,  generative models  supply the  policy  (how to reason and propose actions), while the  agentic architecture  supplies the  machinery  (how proposals are validated, enacted, bounded, and recorded).

Understanding the relation of  Agentic GenAI  with  classic autonomous agents  is crucially important to avoid reinventing the wheel: many key concepts have been studied for a long time and are relatively well-understood today; however, the nature of GenAI also brings up challenges that are completely novel and require rethinking of what was believed to be known. Traditional reactive, deliberative, or BDI (belief-desire-intention) architectures offer theoretically-founded and crisp notions of concepts such as beliefs, goals, plans, and intentions, with clear control loops and explicit world models. Modern agentic systems often replace hand-engineered reasoning with neural-network-based foundation models. These models, trained on huge amounts of diverse data, vastly increase the flexibility and breadth of competence, but also introduce uncertainty in reasoning steps and tool usage. In this chapter, we retain the useful discipline of the classic view—explicit state, goals, plans, commitment strategies, and monitoring—while acknowledging that parts of the pipeline (e.g., plan generation or hypothesis formation) may be implemented by generative models. That reconciliation is precisely where architecture earns its keep.

This book is not intended as yet another broad introduction to Agentic GenAI; instead, we put these recent developments in the specific context of  reliability . By reliability, we mean  the consistent achievement of intended outcomes under stated conditions, within acceptable bounds of safety, security, data protection, and resource usage, and with evidence that failure modes are known, contained, and recoverable . For agentic AI, this encompasses much more than just model accuracy. It includes correct tool invocation, bounded action sequences, resistance to manipulation, predictable latency and cost, graceful degradation, auditability, and human-override paths. Architectures make these properties tractable by: (i) isolating  responsibilities  in modules whose contracts we can reason about; (ii) interposing  validators  and  verifiers  between reasoning and action; (iii)  constraining  authority and side-effects through permissioned tool interfaces; and (iv)  instrumenting  the system so that internal state, decisions, and outcomes are observable and replayable.

Rather than hinging on a single mechanism, system-level reliability is shaped by the interaction of a few foundational architectural choices. In practice, three mutually reinforcing design choices determine how agentic systems behave under stress: how we decompose functionality, how the parts communicate and are constrained, and how their behaviour is supervised at run-time.

Componentisation . Separating the functionality, such as perception, memory, planning, tool routing, execution, verification, and oversight, confines faults to well-defined boundaries and limits their blast radius. Clear responsibilities make defects diagnosable and upgrades safe: a verifier can be strengthened without disturbing the planner; an execution gateway can be hardened without touching memory logic. Componentisation also enables staged deployment (mock tools, simulators, or read-only modes first) and offers natural choke points for safety checks.

Interfaces and contracts . Primary means to tame open-ended model behaviour are typed and schema-validated messages; explicit capability scopes for tools; idempotent and (where feasible) transactional semantics; rate/authority limits. All of these convert free-form model outputs into predictable, auditable actions. Interfaces extend to memory: retrieval must carry provenance and freshness guarantees; long-term stores need retention, compaction, and contamination controls. Good contracts enable the system to act deterministically when safe and refuse when not, transforming ambiguous proposals into either valid commands or actionable error reports.

Control and assurance loops . Monitors compare planned with observed behaviour; critics and verifiers check factuality, policy compliance, and safety invariants; supervisors enforce budgets, escalation rules, and termination criteria; fallbacks provide safe modes of operation when assumptions fail. These loops supply the governing feedback around generative components, preventing small reasoning slips from cascading into hazardous sequences and ensuring graceful degradation under uncertainty.

Taken together, these choices turn a powerful but free-form reasoning engine into a bounded, observable, and governable system. In the remainder of this section, to make these ideas concrete, we illustrate how they play out in the  running example  of a tool-using diagnosis agent operating in a safety-critical environment. Imagine a fleet operator responsible for electric power systems in autonomous service vehicles. The agent’s mission is to triage anomalies, recommend mitigations, and, within a narrow envelope, execute pre-approved actions that reduce risk and downtime.

The agent comprises: a  Goal Manager  (ingesting alerts and operator intents), a  Perception and Retrieval  layer (querying telemetry stores and maintenance logs), a  Planner  (often a generative model producing hypotheses, tests, and action candidates), a  Tool Router  (mapping abstract actions to concrete, permissioned tools: telemetry queries, digital-twin simulation, firmware status, dispatch scheduling), an  Execution Gateway  (schema validation, pre-condition checks, simulators-before-actuators, idempotency tokens), a  Verifier/Critic  (analysing proposed explanations and commands against rules, limits, and known hazards), a  Memory subsystem  (short-term scratchpads for the current case, long-term episodic/semantic stores with provenance), and a  Safety Supervisor  (budgets, escalation, and safe-halt rules). All interactions generate stru