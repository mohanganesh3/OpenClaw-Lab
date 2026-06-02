[2603.17419] Contents

Caging the Agents: A Zero Trust Security

Architecture for Autonomous AI in Healthcare

Saikat Maiti

VP of Trust, Commure

Founder &amp; CEO, nFactor Technologies

saikat@nfactor.ai

March 2026

Version 1.0

Abstract

Autonomous AI agents powered by large language models are being deployed in production environments with capabilities that include shell execution, file system access, database queries, HTTP requests, and multi party communication. Recent empirical research has demonstrated that these agents exhibit critical security vulnerabilities when deployed in realistic settings, including unauthorized compliance with non owner instructions, disclosure of sensitive information, identity spoofing, cross agent propagation of unsafe practices, and susceptibility to indirect prompt injection through external editable resources  [ 7 ] . When these agents operate within healthcare infrastructure processing Protected Health Information (PHI), every documented vulnerability becomes a potential HIPAA violation.

This paper presents a comprehensive security architecture developed and deployed for a fleet of nine autonomous AI agents running in production at a healthcare technology company. The architecture addresses the six domain threat model we developed for agentic AI in healthcare: credential exposure, execution capability abuse, network egress exfiltration, prompt integrity failures, database access risks, and fleet configuration drift. We implement a four layer defense in depth approach: (1) kernel level workload isolation using gVisor sandboxed containers on Kubernetes, (2) credential proxy sidecars that prevent agent containers from accessing raw secrets, (3) network egress policies enforced at the Kubernetes NetworkPolicy layer restricting each agent to allowlisted destinations, and (4) a prompt integrity framework with cryptographically structured metadata envelopes and explicit untrusted content labeling.

We report empirical results from a 90 day deployment, including four HIGH severity findings discovered and remediated by an automated security audit agent, the progressive hardening of the fleet from an unhardened baseline to the target architecture, and the security posture metrics before and after control deployment. We map each documented vulnerability from recent red teaming research to the specific defensive control that addresses it, demonstrating coverage across all eleven attack patterns identified in the literature. All architecture specifications, Kubernetes configurations, audit tooling, and the prompt integrity framework are released as open source.

Keywords:  agentic AI security, autonomous agents, healthcare cybersecurity, zero trust, prompt injection, HIPAA, Kubernetes security, OpenClaw

Contents

1  Introduction

2  Background and Related Work

2.1  Autonomous AI Agent Architectures

2.2  Documented Vulnerabilities in Agentic AI Systems

2.3  Regulatory Context for Healthcare Agentic AI

3  Threat Model: Six Domains of Agentic AI Risk in Healthcare

3.1  Domain 1: Credential Exposure

3.2  Domain 2: Execution Capability Abuse

3.3  Domain 3: Network Egress Exfiltration

3.4  Domain 4: Prompt Integrity and Indirect Injection

3.5  Domain 5: Database Access and PHI Exposure

3.6  Domain 6: Fleet Configuration Drift

3.7  Threat Model to HIPAA Mapping

4  Defense Architecture: Four Layers of Agent Containment

4.1  Layer 1: Kernel Level Workload Isolation (gVisor)

4.2  Layer 2: Credential Proxy Sidecar

4.3  Layer 3: Network Egress Policy Enforcement

4.4  Layer 4: Prompt Integrity Framework

4.4.1  Trusted Metadata Envelopes

4.4.2  Untrusted Content Labeling

4.4.3  Anti Injection Rules

5  Automated Fleet Security Audit System

5.1  Audit Agent Architecture

5.2  Findings and Remediation

5.3  Meta Security: Constraining the Audit Agent

6  VM Image Hardening Progression

6.1  Generation 1: openclaw-base (February 3, 2026)

6.2  Generation 2: openclaw-hardened (February 16, 2026)

6.3  Generation 3: openclaw-hardened-v2 (March 9, 2026)

6.4  Target Architecture: Kubernetes with Four Layer Defense

7  Mapping Defenses to Documented Attack Patterns

8  Discussion

8.1  Limitations of the Prompt Integrity Layer

8.2  The Audit Agent Paradox

8.3  Regulatory Implications

9  Conclusion

A  Responsible Disclosure

B  Open Source Release

References

1  Introduction

The deployment of autonomous AI agents in production environments represents a qualitative shift in the security landscape. Unlike conventional software that processes inputs through well defined interfaces, autonomous agents powered by large language models (LLMs) operate with capabilities that blur the boundary between tool and operator: they execute shell commands, read and write files, query databases, make HTTP requests to external services, spawn sub agents, and maintain persistent memory across sessions  [ 5 ,  7 ] . These capabilities, combined with natural language instruction processing from multiple communication channels, create an attack surface that existing security frameworks were not designed to address.

The urgency of this challenge is underscored by recent empirical research. Shapira et al.  [ 7 ]  conducted a two week red teaming study of autonomous agents deployed in a live laboratory environment using the OpenClaw framework, documenting eleven representative failure modes including unauthorized compliance with non owner instructions, disclosure of 124 email records to an unauthorized party, identity spoofing through display name manipulation, agent corruption via indirect prompt injection through external editable resources, cross agent propagation of unsafe practices, and denial of service through uncontrolled resource consumption. Their findings establish that these are not theoretical risks but empirically demonstrated vulnerabilities in realistic deployment settings.

When autonomous agents with these capabilities operate within healthcare infrastructure, the stakes are fundamentally different. Every vulnerability documented by Shapira et al. maps to a potential HIPAA violation: an agent that discloses email records containing Protected Health Information to an unauthorized party triggers breach notification obligations; an agent that accepts instructions from a spoofed identity may execute operations on clinical data systems; an agent corrupted through indirect prompt injection may exfiltrate patient data to attacker controlled destinations. The NIST AI Agent Standards Initiative, announced in February 2026, identifies agent identity, authorization, and security as priority areas for standardization  [ 4 ] , but provides no implementation guidance for healthcare deployments.

This paper addresses the gap between documented vulnerabilities and deployed defenses. We present the complete security architecture developed for a fleet of nine autonomous AI agents operating in production at a healthcare technology company whose subsidiaries serve major hospital networks including clinical AI, ambient documentation, and patient engagement systems. The fleet uses the OpenClaw framework  [ 5 ]  with Claude via AWS Bedrock for model inference, deployed on Google Cloud Platform Compute Engine infrastructure.

Our contributions are as follows:

1.

A six domain threat model for autonomous AI agents in healthcare that maps every attack vector to specific HIPAA Security Rule provisions, incorporating the empirical findings from Shapira et al.  [ 7 ]  as validated threat scenarios.

2.

A four layer defense in depth architecture (kernel isolation, credential proxy, network egress policy, prompt integrity framework) designed specifically for agentic AI workloads on Kubernetes with Temporal workflow orchestration.

3.

An automated fleet security audit system (itself an AI agent) that continuously scans for credential exposure, permission drift, and configuration divergence, with empirical results from production operation including