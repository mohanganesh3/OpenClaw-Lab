[2511.11347] Privacy Challenges and Solutions in RAG-Enhanced LLMs for Healthcare Chatbots: A Review of Applications, Risks, and Future Directions

Privacy Challenges and Solutions in RAG-Enhanced LLMs for Healthcare Chatbots: A Review of Applications, Risks, and Future Directions

Shaowei Guan, Hin Chi Kwok

Centre for Smart Health, School of Nursing

The Hong Kong Polytechnic University
 Hong Kong, China

Ngai Fong Law

Department of Electrical and Electronic Engineering
 The Hong Kong Polytechnic University
 Hong Kong, China

email@email

Gregor Stiglic

Faculty of Health Sciences
 University of Maribor
 Maribor, Slovenia

Harry QIN, Vivian Hui

Centre for Smart Health, School of Nursing

The Hong Kong Polytechnic University
 Hong Kong, China

Abstract

Retrieval-augmented generation (RAG) has rapidly emerged as a transformative approach for integrating large language models into clinical and biomedical workflows. However, privacy risks, such as protected health information (PHI) exposure, remain inconsistently mitigated. This review provides a thorough analysis of the current landscape of RAG applications in healthcare, including (i) sensitive data type across clinical scenarios, (ii) the associated privacy risks, (iii) current and emerging data-privacy protection mechanisms and (iv) future direction for patient data privacy protection. We synthesize 23 articles on RAG applications in healthcare and systematically analyze privacy challenges through a pipeline-structured framework encompassing data storage, transmission, retrieval and generation stages, delineating potential failure modes, their underlying causes in threat models and system mechanisms, and their practical implications. Building on this analysis, we critically review 17 articles on privacy-preserving strategies for RAG systems. Our evaluation reveals critical gaps, including insufficient clinical validation, absence of standardized evaluation frameworks, and lack of automated assessment tools. We propose actionable directions based on these limitations and conclude with a call to action. This review provides researchers and practitioners with a structured framework for understanding privacy vulnerabilities in healthcare RAG and offers a roadmap toward developing systems that achieve both clinical effectiveness and robust privacy preservation.

K  eywords  Data Privacy

⋅  \cdot

Retrieval-Augmented Generation

⋅  \cdot

Large Language Models

⋅  \cdot

Privacy-Preserving

⋅  \cdot

Privacy Risks

⋅  \cdot

Healthcare Informatics

⋅  \cdot

Clinical Applications

1  Introduction

Large Language Models (LLMs) are rapidly transforming healthcare delivery, supporting diverse applications from patient care and education to clinical documentation and decision support

ho2025development  . However, their deployment in high-stakes clinical environments is constrained by well-documented limitations, such as hallucinations, inherent data biases, and the lack of explainability, which compromise safe and accountable use of LLMs. Retrieval-Augmented Generation (RAG) has emerged as a promising solution to this challenge by conditioning generation on curated information from authoritative sources, including clinical guidelines, textbooks, and medical records. RAG mitigates LLM limitations through a three-step process: (i) embedding and indexing vetted medical knowledge, (ii) retrieving context relevant to user queries, and (iii) conditioning the generator on that context

mahapatra2025storage  . Early deployments in medical education, chronic disease counselling, and workflow assistance demonstrate substantial improvements in answer reliability and user trust compared with purely parametric LLMs

xu2025development ;  kelly2025effectiveness ;  son2025development  .

Despite these advantages, the integration of RAG into healthcare workflows introduces unique and complex privacy challenges that extend beyond those encountered in traditional health information technology or standalone LLMs

neha2025retrieval  . Healthcare RAG deployments are being integrated into high-stakes workflows, including triage chatbots, discharge counselling, guideline lookup, radiology consultation, and rare disease support, that routinely process patient-linked text, institution-specific documents, and population-level datasets

busch2025evaluation ;  kreimeyer2024using  . Even when data repositories are considered "de-identified," they remain vulnerable to privacy breaches. Techniques such as embedding inversion attacks, prompt injection, and membership inference can re-identify individuals or re-expose sensitive information

huang2024transferable ;  anderson2024my  . Consequently, privacy risk in healthcare RAG is not merely a narrow information technology concern but rather a system-wide property that directly affects clinical reliability, clinician adoption, and patient willingness to disclose information.

The incorporation of retrieval mechanisms into LLM pipelines fundamentally reshapes the privacy risk surface. Healthcare RAG systems may handle sensitive health information (SHI), including protected health information (PHI) and institution-specific knowledge (i.e., internal clinical protocols, supplier contracts, operational staffing models and metrics), across multiple architectural components (indexers, vector stores, re-rankers, generators) and computational boundaries (on-device, on-premises, cloud). Each processing stage, cache layer, and data transformation introduces potential leakage channels. Beyond classical threats such as database compromise and insecure APIs, RAG introduces content-driven attack vectors: prompt injection that coerces unsafe context disclosure

zeng-etal-2024-good ;  qi2024follow  , embedding inversion against vector indexes

huang2024transferable  , backdoor triggers that extract confidential information

peng2024data  , and membership-inference techniques that exploit retrieval and generation behavior to infer whether individuals’ data were included in the database

anderson2024my ;  li2025generating  . This creates a critical conflict: the same features that make RAG reliable and transparent, such as its use of retrieved evidence and source citations, can also expand its vulnerability to privacy attacks if the system is not designed with security in mind.

Additionally, the deployment of healthcare RAG systems operates within an increasingly complex and stringent regulatory landscape that mandates rigorous data protection measures. The Health Insurance Portability and Accountability Act (HIPAA) serves as the foundational framework, requiring all healthcare AI systems to implement comprehensive technical, administrative, and physical safeguards for PHI

act1996health  . Similarly, the European Union’s General Data Protection Regulation (GDPR) classifies health data as a "special category" under Article 9, demanding explicit consent or legitimate medical necessity for processing

GDPR_2016  . The recently enacted EU Artificial Intelligence Act (2024) further elevates these requirements by categorizing AI-enabled medical devices as "high-risk" systems, mandating fundamental rights impact assessments, algorithmic transparency, and enhanced human oversight mechanisms

EUAIAct  . Moreover, emerging state-level regulations, such as the California Consumer Privacy Act (CCPA), introduce supplementary privacy rights that healthcare organizations should navigate alongside federal requirements. This multi-layered regulatory framework creates a compliance imperative where healthcare RAG systems should not only demonstrate clinical efficacy but also provide verifiable evidence of data protection, audit trails, and risk mitigation strategies—making privacy preservation not merely a technical consideration but a legal prerequisite for market deployment and operational sustainability

mulgund2021implications  .

Furthermore, healthcare RAG systems designed for diagnostic or therapeutic purposes m