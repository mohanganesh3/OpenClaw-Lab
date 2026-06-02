[2509.06917] : Reimagining Research Papers As Interactive and Reliable AI Agents

\keepXColumns

:  Reimagining Research Papers As Interactive and Reliable AI Agents

Jiacheng Miao

Department of Genetics, Stanford University

Department of Biomedical Data Science, Stanford University

Joe R. Davis

Department of Genetics, Stanford University

Jonathan K. Pritchard

Department of Genetics, Stanford University

Department of Biology, Stanford University

James Zou

Department of Biomedical Data Science, Stanford University

Department of Computer Science, Stanford University

Abstract

We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from  passive artifacts  into  active systems  that can accelerate downstream use, adoption, and discovery.
Conventional research papers require readers to invest substantial effort to understand and adapt a paper’s code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent’s effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper’s results and can correctly carry out novel user queries. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists.

†

†  footnotetext:

Email: {jcmiao, jamesz}@stanford.edu

Repository

Demo

1  Introduction

The research paper is the traditional unit of scientific communication. It remains the norm for documenting methods, results, and insights, and is the primary way research is shared with the broader community. However, papers are fundamentally passive objects: a reader must discover the paper (not an easy task given the flood of publications), parse its contributions, and manually determine how to apply them to their own work. In particular, when a paper describes a new computational method, significant technical barriers often remain before the method can be used on new data  [ 1 ] . A reader might need to locate the corresponding code repository, install dependencies, configure environments, and interpret the correct inputs and outputs  [ 2 ] . Even with well-maintained repositories, this process is often non-trivial.

For instance, consider AlphaGenome, which provides a powerful framework for genome-scale foundation modeling  [ 3 ] . Despite its utility, this system requires substantial technical expertise to set up and deploy, limiting accessibility for biologists who could otherwise benefit. Using AlphaGenome in code involves installing the environment, importing multiple modules, creating client objects with API keys, and constructing inputs such as chromosomes, variant objects, and selecting desired output modalities. Users must understand the API hierarchy and parameter semantics, which imposes a learning curve for biologists unfamiliar with these abstractions.

This illustrates a broader challenge: research outputs are passively siloed behind technical barriers.  Paper2Agent re-imagines research dissemination by turning static papers into active AI agents. Each agent serves as an interactive expert on the corresponding paper, capable of demonstrating, applying, and adapting its methods to new projects.

Figure 1 :

Overview of the Paper2Agent. 
(A) Paper2Agent turns research papers into interactive AI agents by building remote MCP servers with tools, resources, and prompts. Connecting an AI agent to the server creates a paper-specific agent for diverse tasks. (B) Workflow of Paper2Agent. It starts with codebase extraction and automated environment setup for reproducibility. Core analytical features are wrapped as MCP tools, then validated through iterative testing. The resulting MCP server is deployed remotely and integrated with an AI agent, enabling natural-language interaction with the paper’s methods and analyses.

AI agents are autonomous systems that can reason about tasks and act to achieve goals by leveraging external tools and resources  [ 4 ] . Modern AI agents are typically powered by large language models (LLMs) connected to external tools or APIs. They can perform reasoning, invoke specialized models, and adapt based on feedback  [ 5 ] . Agents differ from static models in that they are interactive and adaptive. Rather than returning fixed outputs, they can take multi-step actions, integrate context, and support iterative human–AI collaboration. Importantly, because agents are built on top of LLMs, users can interact with agents through human language, substantially reducing usage barriers for scientists.

Recent advances highlight the promise of agents for accelerating discovery. For example, the Virtual Lab framework organizes teams of AI scientist agents that collaboratively design and execute research projects across biology and chemistry  [ 6 ] . Similarly, Google’s AI co-scientist serves as a virtual collaborator, assisting with hypothesis generation and research proposal development  [ 7 ] . Sakana AI’s co-scientist aims for automation of the research lifecycle—from ideation to publication  [ 8 ] . FutureHouse provides an AI scientist platform designed for diverse scientific tasks  [ 9 ] . Alongside these general-purpose platforms, specialized agents are also emerging for specific domains  [ 10 ] . For example, CellVoyager introduces an agentic system for autonomous analysis of single-cell omics data  [ 11 ] . Biomni is an AI agent for diverse biological tasks  [ 12 ] . These systems demonstrate that agents can not only execute code, but also generate hypotheses, evaluate uncertainty, and adapt methods to new datasets. Paper2Agent complements this emerging paradigm by generalizing the concept: any research paper can be converted into an agent that embodies the knowledge and methods described in the publication.

Paper2Agent provides an automated workflow for converting a scientific paper into an agent. The core idea is to represent the paper as a Model Context Protocol (MCP) server  [ 13 ] . MCP is a standardized protocol that allows structured APIs and tools to be exposed in a way that is directly accessible to LLMs and agent frameworks. The conversion process involves: (i) identifying the key contributions of the paper (datasets, methods, models, or workflows); (ii) encapsulating these contributions through an MCP server, defining the inputs, outputs, and usage instructions; (iii) linking the MCP server to LLM-based agents, enabling natural language querying and autonomous execution. Users can then interact with the paper by asking questions, requesting demonstrations, or applying the method to new data.

As an illustration, applying Paper2Agent to AlphaGenome would expose its genome foundation model as an MCP. Instead of requiring users to clone repositories and configure dependencies, they could simply ask: “ Generate AlphaGenome predictions for these variants. ”, " Interpret the expected effect of this variant on chromatin accessibility in muscle cells. " or “ Visually compare the AlphaGenome predicted expression changes for a sp