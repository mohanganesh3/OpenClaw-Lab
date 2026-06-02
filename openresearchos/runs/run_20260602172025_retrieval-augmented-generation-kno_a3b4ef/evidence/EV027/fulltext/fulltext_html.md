[2604.06710] ATANT: An Evaluation Framework for AI Continuity

ATANT: An Evaluation Framework for AI Continuity

Samuel Sameer Tanguturi

Affiliation:  Kenotic Labs

Affiliation:

sam@kenoticlabs.com

Abstract

We present ATANT (Automated Test for Acceptance of Narrative Truth), an open evaluation framework for measuring  continuity  in AI systems: the ability to persist, update, disambiguate, and reconstruct meaningful context across time. While the AI industry has produced memory components (RAG pipelines, vector databases, long context windows, profile layers), no published framework formally defines or measures whether these components produce genuine continuity. We define continuity as a system property with 7 required properties, introduce a 10-checkpoint evaluation methodology that operates without an LLM in the evaluation loop, and present a narrative test corpus of 250 stories comprising 1,835 verification questions across 6 life domains. We evaluate a reference implementation across 5 test suite iterations, progressing from 58% (legacy architecture) to 100% in isolated mode (250 stories) and 100% in 50-story cumulative mode, with 96% at 250-story cumulative scale. The cumulative result is the primary measure: when 250 distinct life narratives coexist in the same database, the system must retrieve the correct fact for the correct context without cross-contamination. ATANT is system-agnostic, model-independent, and designed as a sequenced methodology for building and validating continuity systems. The framework specification, example stories, and evaluation protocol are available at  https://github.com/Kenotic-Labs/ATANT . The full 250-story corpus will be released incrementally.

Resources: 
Project:  https://kenoticlabs.com

Thesis:  https://kenoticlabs.com/thesis

Dataset:  https://huggingface.co/datasets/Kenotic-Labs/ATANTV1.0-corpus

Code:  https://github.com/Kenotic-Labs/ATANT

1  Introduction

Most AI systems today are session-based. A user provides input, the system responds, and the moment ends. Whatever persists is typically prompt context, conversation history, or retrieved notes. This is adequate for single-turn tasks but insufficient for any system that claims to maintain a meaningful relationship with a user over time.

Human interaction with AI extends beyond isolated prompts. Users have unfinished situations, changing states, recurring concerns, and ongoing commitments. AI systems that serve users across time need more than single-session intelligence. They need structured persistence.

The industry has produced partial solutions: long context windows keep recent material alive temporarily; RAG pipelines retrieve semantically similar text from storage; profile layers hold static preferences; vector databases store embeddings for similarity search. None of these, individually or combined, produce what we term  continuity : the system property that determines what persists, in what form, what has changed, what still matters, and how to reconstruct it.

Despite growing recognition of this gap  [ logan2026cma ,  natangelo2025nct ,  packer2023memgpt ,  chhikara2025mem0 ] , the field lacks three things:

1.

A formal definition of what continuity means as a system property.

2.

A set of testable requirements that any continuity system must satisfy.

3.

A benchmark that measures continuity rather than retrieval accuracy alone.

ATANT addresses all three. Our contributions are:

•

A formal definition of  continuity  as an architectural layer, distinct from memory, retrieval, and context, with 7 required properties.

•

A  model-independent evaluation methodology  with 10 checkpoints that tests the write path and read path of a continuity system without an LLM in the evaluation loop.

•

A  narrative test corpus  of 250 stories (1,835 questions) across 6 life domains, designed for progressive evaluation from isolated correctness to disambiguation at scale.

•

4 compliance levels  (Core, Stress, Cumulative, Scale) that provide a sequenced roadmap for building and validating continuity systems.

•

Reference implementation results  across 5 test suite iterations, including honest reporting of failures, limitations, and the active frontier at 96% cumulative-scale accuracy.

2  Related Work

2.1  Memory Systems for AI

Several systems have been proposed to give AI persistent memory. MemGPT  [ packer2023memgpt ]  introduces an operating system metaphor with tiered memory (core memory always in context, archival memory stored separately). Mem0  [ chhikara2025mem0 ]  provides a production-oriented memory layer that extracts facts, stores them, and manages updates across user, session, and agent scopes. A-MEM  [ xu2025amem ]  proposes agentic memory with self-organizing capabilities for LLM agents.

These systems address components of persistence but do not formally define or test for continuity as a system property. A system can store and retrieve facts without satisfying disambiguation, reconstruction, or temporal ordering.

2.2  Architectural Frameworks

The Continuum Memory Architecture (CMA)  [ logan2026cma ]  defines 6 behavioral properties for long-horizon LLM agents: persistence, selective retention, retrieval-driven mutation, associative routing, temporal continuity, and consolidation. This is the closest prior work to our formalization. However, CMA focuses on memory mechanisms rather than the higher-level logic of what persists and reconstructs, and does not provide an evaluation corpus.

The Narrative Continuity Test  [ natangelo2025nct ]  proposes a conceptual framework for evaluating identity persistence across 5 dimensions. It provides theoretical grounding but no implementation or test corpus.

2.3  Evaluation Benchmarks

Existing benchmarks evaluate specific capabilities: MemoryBench  [ ai2025memorybench ]  tests memory and continual learning; BEAM  [ tavakoli2025beam ]  benchmarks long-term memory beyond a million tokens. These evaluate memory retrieval in isolation. ATANT differs in three ways: (1) it tests the full write-path-to-read-path pipeline, not retrieval alone; (2) it uses naturalistic multi-turn narratives, not synthetic fact pairs; (3) its cumulative mode tests disambiguation under memory load, a property no existing benchmark measures.

2.4  Positioning

ATANT is not a replacement for memory systems or retrieval benchmarks. It is a framework for evaluating whether the combination of components in a system produces  continuity , a higher-order property that emerges from correct persistence, update handling, temporal ordering, disambiguation, and reconstruction working together.

3  Defining Continuity

Continuity  is the system property that enables an AI to carry forward what still matters from prior interactions, update it when reality changes, and reconstruct useful context later in the appropriate form for the current situation.

3.1  Continuity vs. Memory

Memory stores the past. Continuity keeps the right parts of the past alive in the present. A database can store  (user, partner_name, Mia) . A continuity system can answer “Tell me about my relationship” with a reconstruction that includes Mia, where she lives, how the user feels about her, what changed last week, and what remains unresolved.

3.2  Continuity vs. Retrieval

Retrieval returns text similar to a query. Continuity reconstructs the current state of a situation, including what changed, what remains active, and what was superseded. The distinction is between similarity search and state reconstruction.

3.3  The 7 Required Properties

We define 7 properties that any system claiming continuity must satisfy. These properties were derived empirically by building a continuity system, running it against hundreds of real-world narratives, and identifying what breaks when each property is absent.

Table 1:  The 7 Required Properties of Continuity

#

Property

Testable Requirement

1

Persistence Beyond Session

After ingesting facts, terminate an