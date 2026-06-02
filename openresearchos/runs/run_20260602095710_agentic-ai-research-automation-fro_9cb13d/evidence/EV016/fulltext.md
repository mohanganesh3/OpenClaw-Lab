[2601.07880] Sola-Visibility-ISPM: Benchmarking Agentic AI for Identity Security Posture Management Visibility

Sola-Visibility-ISPM:

Benchmarking Agentic AI for Identity Security Posture Management Visibility

Gal Engelberg ,
 Konstantin Koutsyi ,
 Leon Goldberg ,
 Reuven Elezra ,
 Idan Pinto
 Tal Moalem ,
 Shmuel Cohen ,
 Yoni Weintrob

Sola Security, Tel Aviv, Israel

Corresponding author: gal.e@sola.security

Abstract

Identity Security Posture Management (ISPM) is a core challenge for modern enterprises operating across cloud and SaaS environments. Answering basic ISPM visibility questions, such as understanding identity inventory and configuration hygiene, requires interpreting complex identity data, motivating growing interest in agentic AI systems. Despite this interest, there is currently no standardized way to evaluate how well such systems perform ISPM visibility tasks on real enterprise data. We introduce the  Sola Visibility ISPM Benchmark , the first benchmark designed to evaluate agentic AI systems on foundational ISPM visibility tasks using a live, production-grade identity environment spanning AWS, Okta, and Google Workspace. The benchmark focuses on identity inventory and hygiene questions and is accompanied by the  Sola AI Agent , a tool-using agent that translates natural-language queries into executable data exploration steps and produces verifiable, evidence-backed answers. Across 77 benchmark questions, the agent achieves strong overall performance, with an expert accuracy of 0.84 and a strict success rate of 0.77. Performance is highest on AWS hygiene tasks, where expert accuracy reaches 0.94, while results on Google Workspace and Okta hygiene tasks are more moderate, yet competitive. Overall, this work provides a practical and reproducible benchmark for evaluating agentic AI systems in identity security and establishes a foundation for future ISPM benchmarks covering more advanced identity analysis and governance tasks.

K  eywords  Identity Security Posture Management

⋅  \cdot

AI For Security

⋅  \cdot

Benchmarking

1  Introduction

Identity now defines the practical security boundary of the modern enterprise. As organizations operate across multi-cloud, SaaS, and hybrid environments, the integrity of identities, entitlements, authentication methods, and access paths determines the real perimeter of control. Industry guidance, such as RSA’s ISPM Framework  [ 26 ]  - highlights that identity misconfigurations, excessive privileges, unmanaged lifecycle states, and weak authentication posture are at the core of many security failures. ISPM has therefore become a foundational discipline for modern cybersecurity programs.

At the same time, agentic AI is reshaping how security operations are performed. Google Cloud’s 2025 Return on AI in Security report  [ 11 ]  shows that organizations expect their largest operational gains from AI systems capable of autonomous reasoning, cross-system understanding, and end-to-end workflow execution. These systems can interpret complex telemetry, propose decisions, and accelerate response, capabilities that align directly with the data-intensive, multi-step nature of ISPM. As a result, agentic AI is increasingly viewed as a powerful accelerator for identity-centric security operations.

Yet despite this convergence, a significant gap remains: no standardized benchmark exists to evaluate how well agentic AI systems perform ISPM tasks, particularly in the context of question-answering agents operating over enterprise-grade identity and access data sources, a setup analogous to the data-grounded evaluation paradigm introduced in Spider-2.0 for SQL reasoning  [ 15 ] . Existing cybersecurity benchmarks have advanced AI evaluation across several domains: interactive SOC investigation workflows  [ 31 ] , applied SOC reasoning over incident reports  [ 8 ] , cyber threat intelligence reasoning  [ 1 ,  13 ] , adversarial validation and robustness evaluation  [ 32 ] , foundational NLP tasks in cybersecurity  [ 18 ,  7 ] , ICS-focused knowledge evaluation  [ 3 ] , vulnerability detection in code  [ 29 ] , secured code generation  [ 16 ] , and RBAC rule-following in synthetic access-control hierarchies  [ 27 ] .

However, none of these benchmarks evaluate the core tasks required for identity security posture management, such as interpreting identity inventories, parsing entitlements and privileges, validating lifecycle states, assessing configuration hygiene, performing cross-platform identity correlation, or generating posture-aware answers grounded in real organizational datasets. In other words, while the broader cybersecurity community has made progress benchmarking AI for investigations, CTI reasoning, and offensive capabilities, identity security, the control plane responsible for many modern breaches, remains unmeasured in the context of agentic AI systems reasoning over real identity data.

We outline nine operational ISPM dimensions that together represent the full lifecycle of identity security posture management, and describe the capabilities an agentic AI system must demonstrate within each:

•

Visibility &amp; Hygiene  – Maintain complete, accurate identity inventories; detect misconfigurations and hygiene drift; evaluate MFA posture; and surface risky or stale identities.

•

Cross-System Correlation  – Connect identity signals across IdPs, cloud IAM, productivity suites, directories, and application layers; interpret federated trust; and reason end-to-end about identity exposure.

•

Behavioral Analytics  – Interpret authentication logs, privilege-use patterns, and other audit signals to detect suspicious or anomalous behavior.

•

Risk Assessment &amp; Scoring  – Rank identity risks based on posture, privilege, behavior, and potential business impact; identify identities that disproportionately increase organizational exposure.

•

Mitigation &amp; Recommendations  – Produce actionable, least-privilege-aligned remediation steps that consider operational constraints and security governance.

•

Framework Alignment &amp; Governance  – Map findings to standards such as NIST  [ 14 ] , CIS  [ 4 ] , and ISO  [ 12 ] , and answer framework-level questions directly (e.g., “Am I compliant with the CIS AWS Benchmark?”). Support audit, control validation, and policy-aligned reporting.

•

Contextual Threat Awareness  – Interpret identity exposures using known adversarial techniques, threat trends, and attack paths, and enrich reasoning with signals from CTI feeds to prioritize issues based on active or relevant threats.

•

Organizational Context Awareness  – Incorporate regulatory requirements, business-critical roles, sensitivity of assets, and organizational structures into reasoning.

•

Advanced Analytics  – Apply graph analysis, anomaly detection, multi-signal correlation, and other analytical techniques to detect emergent or latent identity risks beyond rule-based logic.

Across these dimensions, an agentic AI system must not only answer identity-security questions accurately, it must demonstrate correct data usage, multi-step reasoning, prioritization, and governance alignment consistent with real-world ISPM workflows.

In this paper, we introduce the Sola Visibility ISPM Benchmark, a focused benchmark covering the foundational ISPM tasks of  identity inventory, hygiene and misconfiguration detection . It provides a curated set of security questions, and reproducible evaluation metrics for assessing agentic ISPM performance across environments built on AWS

1

1  1  https://aws.amazon.com/what-is-aws/

, Okta

2

2  2  https://www.okta.com/products/workforce-identity/

, and Google Workspace

3

3  3  https://workspace.google.com/

. This benchmark establishes the core evaluation layer and lays the foundation for a broader, multi-dimensional Agentic ISPM Benchmark suite.

2  Related Work

AI evaluation in security and data systems has expanded significantly, driven by the need to measure