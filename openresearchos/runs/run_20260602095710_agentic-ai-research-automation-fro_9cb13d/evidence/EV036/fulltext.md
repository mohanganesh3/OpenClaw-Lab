[2604.00187] Explainable AI for Blind and Low-Vision Users: Navigating Trust, Modality, and Interpretability in the Agentic Era

Explainable AI for Blind and Low-Vision Users: Navigating Trust, Modality, and Interpretability in the Agentic Era

Conference:  ; April 13, 2016; Barcelona, Spain

CCS:  Human-centered computing Accessibility systems and tools

CCS:  Human-centered computing Empirical studies in accessibility

CCS:  Computing methodologies Philosophical/theoretical foundations of artificial intelligence

CCS:  Human-centered computing Empirical studies in HCI

Abu Noman Md Sakib

email:  abunomanmd.sakib@utsa.edu

OrcID:  0000-0002-0761-035X

Affiliation:

University of Texas at San Antonio 
,  San Antonio 
,  Texas 
,  USA

,

Protik Dey

email:  protik.dey@utsa.edu

OrcID:  0009-0006-4089-5111

Affiliation:

University of Texas at San Antonio 
,  San Antonio 
,  Texas 
,  USA

,

Zijie Zhang

email:  zijie.zhang@utsa.edu

OrcID:  0000-0003-1254-098X

Affiliation:

University of Texas at San Antonio 
,  San Antonio 
,  Texas 
,  USA

and

Taslima Akter

email:  taslima.akter@utsa.edu

OrcID:  0000-0002-0686-1448

Affiliation:

University of Texas at San Antonio 
,  San Antonio 
,  Texas 
,  USA

(© none)

Abstract.

Explainable Artificial Intelligence (XAI) is critical for ensuring trust and accountability, yet its development remains predominantly visual. For blind and low-vision (BLV) users, the lack of accessible explanations creates a fundamental barrier to the independent use of AI-driven assistive technologies. This problem intensifies as AI systems shift from single-query tools into autonomous agents that take multi-step actions and make consequential decisions across extended task horizons, where a single undetected error can propagate irreversibly before any feedback is available. This paper investigates the unique XAI requirements of the BLV community through a comprehensive analysis of user interviews and contemporary research. By examining usage patterns across environmental perception and decision support, we identify a significant modality gap. Empirical evidence suggests that while BLV users highly value conversational explanations, they frequently experience “self-blame” for AI failures. The paper concludes with a research agenda for accessible Explainable AI in agentic systems, advocating for multimodal interfaces, blame-aware explanation design, and participatory development.

Keywords:  Explainable AI, Blind and Low-Vision, HCXAI, Accessibility, Agentic AI

1.  Introduction

Integrating Artificial Intelligence (AI) into daily life offers incredible autonomy for the blind and low-vision (BLV) community  ( misfitAI ;  humanError ) . With the introduction of vision models and large language models (LLMs), AI systems can now describe scenes, recognize objects, and interpret complex environments in real time. For many BLV individuals, these tools are no longer just convenient. They have become essential aids that change how people navigate the physical world  ( MAIDR ) . However, these systems are rapidly evolving from simple description tools into autonomous agents that plan multi-step task sequences, invoke external tools, and take actions with real-world consequences. Because their inner workings remain a “black box,” this shift creates an especially acute challenge for the BLV community  ( teachableAI ) . A sighted user who sees an agent draft a message or confirm a purchase can interrupt if something looks wrong. A BLV user has no equivalent monitoring channel. An error in Step 2 of a 10-step task can propagate silently through all subsequent steps and result in an irreversible outcome before any feedback is received. An AI system’s output is often the primary source of environmental truth for a BLV user. Sometimes, it is the only source. Without accessible ways to inspect how the agent made its decisions, users cannot detect hallucinations, erroneous tool invocations, or cascading misinterpretations  ( pieceIt ) . The current approach to Explainable AI (XAI) makes this problem worse. The Human-Centered Explainable AI (HCXAI) movement tries to look beyond math and understand how people actually use explanations  ( HCXAI ) . Yet, mainstream XAI methods like SHAP  ( lundberg2017unified )  and GradCAM  ( selvaraju2017grad )  still rely heavily on visuals. By using color-coded heatmaps and visual bounding boxes, traditional XAI shuts out the exact users who rely on AI the most for sensory translation.

This paper addresses that gap by exploring accessible XAI and user trust within the BLV community. We look at how trust changes depending on the stakes of the situation. We also highlight a common “Self-Blame Bias.” This occurs when BLV users blame themselves for taking a bad photo rather than realizing the AI made a mistake. We then evaluate which types of explanations work best for verifying information and supporting decisions, comparing formats like system rules against conversational interactions. We argue that real digital inclusion for the BLV community requires legitimacy across the entire AI pipeline. This means everything from using representative training data to building clear, non-visual ways for users to understand AI decisions.

2.  Related Work

Recent work in HCXAI highlights that explanations must be tailored to the people actually using them  ( HCXAI ;  HCXAIGynae ;  ehsan2023human ) . However, most mainstream methods for computer vision still rely on visual cues like heatmaps, which completely exclude BLV users  ( treemap ) . To fix this, accessible XAI focuses on translating model logic into non-visual formats like text or conversation. This is especially important because BLV users face unique challenges with trust. As AI moves from simple object recognition to complex, multi-step autonomous agents, this overreliance creates serious epistemological vulnerability  ( AIreliance ;  AIRelianceSurvey ) . If a system confidently hallucinates early in a task, the user has no easy way to catch the error  ( hallucinationHCXAI ) . Researchers have suggested using conversational AI to introduce “cognitive forcing,” encouraging users to ask questions and double-check outputs rather than passively accepting them  ( conversationalAI ;  relationalAI ) . Some studies indicate that BLV users develop complex, often flawed, mental models of Generative AI, sometimes viewing these tools as authoritative rather than probabilistic systems prone to hallucination  ( kingofknowledge ) . Furthermore, trust in these systems is highly contextual. Xinru et al. demonstrated in their framework that users’ acceptance of technical failures is not binary but contingent on the stakes of the task and the social environment  ( EverydayUncertainty ) . Accessible XAI must support explorable interactions that allow users to interrogate these stakes  ( explorablexai ) . We view all of these challenges which argues that an AI system is only valid if it centers the people impacted by its decisions. For the BLV community, this means building legitimacy directly into the pipeline, starting with training data that actually reflects how they take photos, all the way to providing transparent, non-visual ways to verify what the AI is doing.

The emergence of LLM-based agents introduces explainability demands that go beyond explaining a single model prediction. Agentic systems orchestrate chains of reasoning steps, invoke external tools, and produce execution traces that unfold over time  ( hallucinationHCXAI ) . The relevant unit of explanation is no longer a feature attribution score but a process trace: which sub-goals were set, which tools were called, and where the agent chose to act rather than verify. The proxy verification strategies BLV users rely on today (barcode cross-checking, multi-shot photography, human fallback) are workarounds for the absence of accessible process transparency. These workarounds do not scale to agentic pipelines. A