[2604.24153] 1 Introduction

The Pre-Action Legitimacy Gap in AI Systems

A Deterministic, Non-Compensatory Decision Boundary for Execution

Gadi Lavi

Independent Researcher, DECIMAG

https://decimag-protocol.netlify.app/

April 2026

Abstract. 
AI systems are increasingly capable of initiating real-world actions, yet current architectures lack a formal mechanism to determine whether a decision should be allowed to execute in the first place. Existing paradigms such as authorization, safety alignment, policy enforcement, runtime governance, and statistical certification evaluate identity, compliance, risk, or observed behavior. They often assume that once a decision is permitted, safe enough, or policy-compliant, it may proceed to execution.

This paper defines the  pre-action legitimacy problem : the missing computational step that determines whether an AI-generated decision has the right to exist as an executable event prior to execution. We introduce a minimal formalization of a  Right-to-Act protocol : a deterministic, non-compensatory boundary that treats legitimacy as a feasibility condition rather than a score. A decision is allowed to execute only if all required structural constraints are satisfied; no positive signal may compensate for a failed required condition.

We prove a non-equivalence result showing that compensatory scoring systems cannot guarantee pre-action legitimacy. We then position this boundary relative to authorization, safety, runtime governance, and statistical certification, and present a case study in AI-driven account suspension. The contribution is intentionally architectural and formal: it defines a missing decision primitive without prescribing or disclosing any proprietary implementation.

Preprint. This paper introduces a formal framing for pre-action legitimacy and non-compensatory execution boundaries in AI systems.

1  Introduction

AI systems are moving from advisory outputs toward executable decisions. In agentic and tool-using systems, model outputs may trigger fund transfers, account restrictions, database operations, infrastructure changes, procurement workflows, or other actions with legal, financial, operational, or reputational consequences.

A common architecture can be summarized as follows:

AI decision

→  \rightarrow

validation

→  \rightarrow

execution .

The validation step may include access control, safety filters, policy checks, risk scoring, compliance rules, or post-hoc audit. These mechanisms are necessary. However, they do not fully answer a different question:

Should this decision be allowed to exist as an executable event at all?

This paper argues that this question defines a distinct architectural gap. Existing paradigms evaluate whether a user is allowed, whether an output is harmful, whether a policy is violated, or whether observed system behavior falls within certified risk bounds. A pre-action legitimacy boundary instead evaluates whether the AI-generated decision has earned the right to proceed toward execution before ordinary validation or certification is treated as sufficient.

The goal is not to replace authorization, safety, runtime governance, or certification. The goal is to formalize a missing condition before execution:  the right to act . We call this gap the  Pre-Action Legitimacy Gap .

1.1  Contributions

This paper makes four contributions:

1.

It defines the pre-action legitimacy problem as a distinct computational and architectural problem for AI systems that act in the world.

2.

It formalizes a non-compensatory Right-to-Act decision boundary using minimal constraint notation that avoids implementation-specific details.

3.

It proves a non-equivalence theorem: compensatory scoring systems cannot guarantee pre-action legitimacy when required constraints must hold individually.

4.

It provides a case study showing how the proposed boundary changes the outcome of a high-confidence, policy-compliant account suspension decision.

2  Background and Related Work

2.1  Authorization and access control

Authorization systems determine whether an identity, role, process, or agent is permitted to access a resource or perform a class of operations. These systems ask:  who is allowed to act?  They do not necessarily determine whether a specific AI-generated decision is structurally legitimate in its context. An authorized actor may still produce an action that should not proceed.

2.2  AI safety and alignment

AI safety and alignment research focuses on reducing harmful, unintended, or misaligned behavior in AI systems. This includes model training, reward design, refusal behavior, monitoring, and evaluation. These methods are essential but often probabilistic, model-dependent, or focused on generated content and behavior rather than per-decision execution legitimacy.

2.3  Runtime governance for AI agents

Recent work on runtime governance treats agent execution paths as the object of governance. Kaptein, Khan, and Podstavnychy formalize policies over paths, arguing that prompts and static access control cannot fully govern path-dependent behavior in agentic systems  [ 6 ] . This line of work is close to the present paper in its focus on runtime boundaries, but it remains oriented around governance and policy over paths. The present paper isolates a narrower primitive: whether a proposed decision has the right to proceed before execution.

2.4  Pre-action authorization

Uchibeke characterizes the pre-action authorization problem for autonomous agents, emphasizing deterministic enforcement before tool calls and signed audit records  [ 5 ] . This is directly relevant to the present work. The distinction here is conceptual: pre-action authorization asks whether an action is permitted by policy or authority. Pre-action legitimacy asks whether the decision satisfies required structural conditions such that execution is legitimate in the first place.

2.5  Statistical certification and AI risk regulation

Levy and Perl propose a statistical certification framework for black-box AI risk regulation, focusing on bounding system behavior under uncertainty  [ 7 ] . Their framing is valuable for regulation and certification of systems whose internals are opaque. The present paper complements this approach by addressing the per-decision boundary before execution. Certification can state that a system behaves within acceptable bounds over a distribution; a Right-to-Act layer asks whether a particular decision should become executable.

2.6  Risk management frameworks

The NIST AI Risk Management Framework provides guidance for managing AI risks across design, deployment, and use  [ 3 ] . The EU AI Act establishes a risk-based regulatory framework for AI systems  [ 4 ] . OWASP has also documented risks associated with agentic AI execution layers and autonomous skills  [ 8 ] . These frameworks motivate the need for systematic controls, but they do not by themselves define the non-compensatory computational boundary proposed here.

3  The Pre-Action Legitimacy Gap

Definition 1

(Pre-Action Legitimacy Gap) .

The Pre-Action Legitimacy Gap is the absence of a formal mechanism that determines, prior to execution, whether an AI-generated decision has the right to exist as an executable event.

The gap is easiest to see in cases where a decision is:

•

authorized by identity or role;

•

compliant with an explicit policy;

•

scored as low risk or high confidence;

•

still inappropriate, premature, unjustified, or structurally invalid in context.

Examples include disabling an account without sufficient contextual verification, executing a payment when authority exists but justification is incomplete, or allowing an AI agent to take irreversible infrastructure action under ambiguous state. In such cases, the failure is not merely a safety failure, a policy failure, or an authorization failure. It is a legitimacy failure at the execution boundary.

4  Formal Model

This section prese