# EV012: Poster: Automated SIEM Detection Rule Translation System

URL: https://www.semanticscholar.org/paper/005d37d2a1b4993805be20e8b0c455b04eb9914d
Year: 2026
Source: semantic_scholar
Arxiv: n/a

## Abstract

Security Operations Centers (SOCs) depend on SIEM detection rules to identify malicious activity, but detection logic is tightly bound to platform-specific query languages such as SPL (Splunk), KQL (Microsoft Sentinel), and YARA-L (Chronicle). This coupling complicates SIEM migration and multi-platform detection engineering, increasing cost, slowing adaptation, and raising the risk of errors. Detection content becomes fragmented, difficult to standardize or reuse, and prone to semantic inconsistencies and incorrect field mappings, which can lead to detection gaps and false negatives. These issues are intensified by the growing volume and diversity of security telemetry, which increases rule complexity and limits the feasibility of manual cross-platform alignment. Although Large Language Models (LLMs) show promise for automated query translation, unguided generation often produces invalid syntax or semantically incorrect rules, which is unacceptable in security-critical contexts. This paper presents an automated architecture for cross-platform SIEM rule translation that integrates Retrieval-Augmented Generation, deterministic workflow orchestration, and SIEM-native validation to enable stable, accurate, and scalable end-to-end detection rule translation.
