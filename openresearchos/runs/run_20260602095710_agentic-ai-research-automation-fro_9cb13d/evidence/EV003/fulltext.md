[2603.18914] Security, privacy, and agentic AI in a regulatory view: From definitions and distinctions to provisions and reflections

Security, privacy, and agentic AI in a regulatory view:

From definitions and distinctions to provisions and reflections

Shiliang Zhang ∗

Sabita Maharjan

Shiliang Zhang and Sabita Maharjan are with Department of Informatics, University of Oslo, Norway (e-mail: {shilianz, sabita}@uio.no).

Abstract

The rapid proliferation of artificial intelligence (AI) technologies has led to a dynamic regulatory landscape, where legislative frameworks strive to keep pace with technical advancements. As AI paradigms shift towards greater autonomy, specifically in the form of agentic AI, it becomes increasingly challenging to precisely articulate regulatory stipulations. This challenge is even more acute in the domains of security and privacy, where the capabilities of autonomous agents often blur traditional legal and technical boundaries. This paper reviews the evolving European Union (EU) AI regulatory provisions via analyzing 24 relevant documents published between 2024 and 2025. From this review, we provide a clarification of critical definitions. We deconstruct the regulatory interpretations of security, privacy, and agentic AI, distinguishing them from closely related concepts to resolve ambiguity. We synthesize the reviewed documents to articulate the current state of regulatory provisions targeting different types of AI, particularly those related to security and privacy aspects. We analyze and reflect on the existing provisions in the regulatory dimension to better align security and privacy obligations with AI and agentic behaviors. These insights serve to inform policymakers, developers, and researchers on the compliance and AI governance in the society with increasing algorithmic agencies.

1  Introduction

Artificial intelligence (AI) has undergone rapid evolution across scientific, industrial, and societal domains, shifting from pretrained predictive models to powerful systems that can plan, act, and adapt in open-ended environments  [ radanliev2025artificial ] . AI mechanisms, models, systems, and applications now permeate energy  [ melguizo2026can ] , healthcare  [ kim2025understanding ] , finance  [ vukovic2025ai ] , manufacturing  [ fosso2026building ] , transportation  [ yan2025generative ] , automation  [ spencer2025ai ]  and vehicles  [ sharma2026generative ] , government services  [ chen2025uncovering ] , and knowledge work  [ tan2025artificial ] . More recently, the rise of autonomous agentic AI  [ hasselwander2026toward ]  has expanded the scope of applications, enabling complex workflows through autonomous orchestration of APIs, software tools, and data sources. Unlike their predecessors, which passively respond to human prompts, agentic AI possesses the capability to perceive, reason, and proactively execute complex, multi-step goals with minimal human intervention  [ biswas2025building ] . This transition from AI as a tool to AI as an active agent is rapidly reshaping various domains in human life.

However, this increasing AI autonomy introduces challenges, particularly regarding security and privacy  [ leo2026threat ,  hosseini2025role ,  zhang2025data ] . As AI agents are increasingly granted direct access to external tools, databases, and APIs to fulfill their objectives, the attack surface expands significantly  [ deng2025ai ] . Agentic behaviors introduce security vulnerabilities, such as prompt injection propagating across multi-agent ecosystems  [ ferrag2025prompt ] , unauthorized data exfiltration during autonomous collaboration  [ huang2025securing ] , and the potential for agents to bypass safety guardrails in pursuit of misaligned sub-goals  [ huang2025commercial ] . Supply chain risks intensify when agents rely on third-party libraries, APIs, and model components  [ jannelli2025agentic ] . Poisoning or dependency confusion can compromise agent outputs and actions  [ esen2025risks ] . The privacy implications are equally critical. Privacy risks extend beyond model-centric leakage to pipeline-centric exposure  [ ray2026comprehensive ] . Agents that persistently act on behalf of users might require access to personal context and sensitive data  [ zhang2025towards ] , raising issues about data minimization, consent management, and the black-box nature of autonomous decision-making. Long-horizon tasks and memory components increase the likelihood of retaining sensitive information beyond legitimate use  [ dwivedi2025agentic ] . Tool calls can cross jurisdictional boundaries and contractual frameworks, raising compliance questions under data protection regimes  [ hughes2025ai ] .

In response to these disruptions, the regulatory landscape is striving to adapt. The European Union (EU), a global forerunner in digital governance, has introduced frameworks like the EU AI Act  [ nolte2025robustness ]  to categorize and mitigate AI risks. Complementary EU instruments and frameworks reinforce privacy and security baselines that are applicable to AI systems. Such examples are the GDPR for data protection  [ beltran2025ai ] , the NIS2 Directive for network and information system security  [ kianpour2025digital ] , the Cyber Resilience Act for secure-by-design products with digital elements  [ mueck2025introduction ] , and the Data Act for data access and sharing  [ hohmann2025reflections ] . Between 2024 and 2025, EU has issued a range of regulatory documents aiming at operationalizing the AI Act and harmonizing it with existing privacy and security law  [ calvano2026building ] . These materials explore definitions of AI systems, clarify risk categories, propose conformity assessment procedures, and outline cybersecurity and data governance requirements.

However, conceptual ambiguity persists around key terms - security, privacy, personal data, generative AI, general-purpose AI, large language models, agentic AI - especially when applied to AI systems that can act in unanticipated ways. This ambiguity makes it difficult for practitioners to interpret obligations  [ nizza2026ais ] . A lack of distinctions can lead to inconsistent compliance practices and regulatory gaps. Furthermore, while high-level principles for privacy and security exist, specific regulatory provisions that address the unique risks of agentic AI remain fragmented and less articulated. There is a need to clarify how existing privacy and security mandates apply to systems that can act independently of real-time human oversight. Articulating the regulatory provisions that pertain to privacy and security in the context of agentic AI is necessary, which is anticipate to bridge the gap between abstract mandates and implementable controls.

To address this gap, this paper provides a review and regulatory analysis of the evolving EU landscape concerning agentic AI. We focus on the intersection of security, privacy, and agentic AI. Our contributions are summarized as follows:

(i) We analyze 24 relevant EU AI regulatory documents published between 2024 and 2025. Based on this review, we deconstruct and clarify the critical definitions of privacy, security, and agentic AI, and distinguish them from closely related concepts to resolve regulatory ambiguities.

(ii) We synthesize the regulatory documents to articulate the provisions targeting agentic AI. We map these provisions against the technical capabilities of agents to identify where the law is robust and where it remains porous.

(iii) We reflect and discuss the existing provisions. We extract regulatory recommendations to help policymakers, developers, and researchers align security and privacy obligations with the reality of increasing algorithmic agency.

The reminder of this paper is as follows. Section

2

overviews the reviewed regulatory documents. Section

3

provides the definitions and distinctions of key regulatory concepts related to privacy, security, and agentic AI. Section