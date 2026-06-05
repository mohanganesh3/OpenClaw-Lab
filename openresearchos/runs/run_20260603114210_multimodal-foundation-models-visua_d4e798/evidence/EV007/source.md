# EV007: A Proposal-Free Query-Guided Network for Grounded Multimodal Named Entity Recognition

URL: https://www.semanticscholar.org/paper/02d7c863caa9c76a330baa89420ec3dcba9f84cf
Year: 2026
Source: semantic_scholar
Arxiv: 2603.17314

## Abstract

Grounded Multimodal Named Entity Recognition (GMNER) identifies named entities, including their spans and types, in natural language text and grounds them to the corresponding regions in associated images. Most existing approaches split this task into two steps: they first detect objects using a pre-trained general-purpose detector and then match named entities to the detected objects. However, these methods face a major limitation. Because pre-trained general-purpose object detectors operate independently of textual entities, they tend to detect common objects and frequently overlook specific fine-grained regions required by named entities. This misalignment between object detectors and entities introduces imprecision and can impair overall system performance. In this paper, we propose a proposal-free Query-Guided Network (QGN) that unifies multimodal reasoning and decoding through text guidance and cross- modal interaction. QGN enables accurate grounding and robust performance in open-domain scenarios. Extensive experiments demonstrate that QGN achieves top performance among compared GMNER models on widely used benchmarks.
