[2605.05628] Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.05628

Help  |  Advanced Search

All fields

Title

Author

Abstract

Comments

Journal reference

ACM classification

MSC classification

Report number

arXiv identifier

DOI

ORCID

arXiv author ID

Help pages

Full text

Search

GO

quick links

Login

Help Pages

About

-->

Computer Science > Hardware Architecture

arXiv:2605.05628  (cs)

[Submitted on 7 May 2026]

Title: Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems

Authors:  Chen Zhang ,  Qijun Zhang ,  Zhuoshan Zhou ,  Yijia Diao ,  Haibo Wang ,  Zhe Zhou ,  Zhipeng Tu ,  Zhiyao Li ,  Guangyu Sun ,  Zhuoran Song ,  Zhigang Ji ,  Jingwen Leng ,  Minyi Guo

View a PDF of the paper titled Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems, by Chen Zhang and 12 other authors

View PDF

HTML (experimental)

Abstract: Tensor parallelism (TP) in large-scale LLM inference and training introduces frequent collective operations that dominate inter-GPU communication. While in-switch computing, exemplified by NVLink SHARP (NVLS), accelerates collective operations by reducing redundant data transfer, its communication-centric design philosophy introduces the mismatch between its communication mode and the memory semantic requirement of LLM&#39;s computation kernel. Such a mismatch isolates the compute and communication phases, resulting in underutilized resources and limited overlap in multi-GPU systems. To address the limitation, we propose CAIS, the first Compute-Aware In-Switch computing framework that aligns communication modes with computation&#39;s memory semantics requirement. CAIS consists of three integral techniques: (1) compute-aware ISA and microarchitecture extension to enable compute-aware in-switch computing. (2) merging-aware TB (Thread Block) coordination to improve the temporal alignment for efficient request merging. (3) graph-level dataflow optimizer to achieve a tight cross-kernel overlap. Evaluations on LLM workloads show that CAIS achieves 1.38$\times$ average end-to-end training speedup over the SOTA NVLS-enabled solution, and 1.61$\times$ over T3, the SOTA compute-communicate overlap solutions but do not leverage NVLS, demonstrating its effectiveness in accelerating TP on multi-GPU systems.

Comments:

15 pages, 18 figures, HPCA 2026

Subjects:

Hardware Architecture (cs.AR) ; Distributed, Parallel, and Cluster Computing (cs.DC)

Cite as:

arXiv:2605.05628  [cs.AR]

&nbsp;

(or

arXiv:2605.05628v1  [cs.AR]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.05628

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Qijun Zhang [ view email ]

[v1]

Thu, 7 May 2026 03:29:51 UTC (3,691 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems, by Chen Zhang and 12 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.AR

&lt;&nbsp;prev

&nbsp; | &nbsp;

next&nbsp;&gt;

new

|

recent

|

2026-05

Change to browse by:

cs

cs.DC

References &amp; Citations

NASA ADS

Google Scholar

Semantic Scholar

export BibTeX citation

Loading...

BibTeX formatted citation

&times;

loading...

Data provided by:

Bookmark

Bibliographic Tools

Bibliographic and Citation Tools

Bibliographic Explorer Toggle

Bibliographic Explorer

( What is the Explorer? )

Connected Papers Toggle

Connected Papers

( What is Connected Papers? )

Litmaps Toggle

Litmaps

( What is Litmaps? )

scite.ai Toggle

scite Smart Citations

( What are Smart Citations? )

Code, Data, Media

Code, Data and Media Associated with this Article

alphaXiv Toggle

alphaXiv

( What is alphaXiv? )

Links to Code Toggle

CatalyzeX Code Finder for Papers

( What is CatalyzeX? )

DagsHub Toggle

DagsHub

( What is DagsHub? )

GotitPub Toggle

Gotit.pub

( What is GotitPub? )

Huggingface Toggle

Hugging Face

( What is Huggingface? )

ScienceCast Toggle

ScienceCast

( What is ScienceCast? )

Demos

Demos

Replicate Toggle

Replicate

( What is Replicate? )

Spaces Toggle

Hugging Face Spaces

( What is Spaces? )

Spaces Toggle

TXYZ.AI

( What is TXYZ.AI? )

Related Papers

Recommenders and Search Tools

Link to Influence Flower

Influence Flower

( What are Influence Flowers? )

Core recommender toggle

CORE Recommender

( What is CORE? )

Author

Venue

Institution

Topic

About arXivLabs

arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community?

Learn more about arXivLabs  .

Which authors of this paper are endorsers?  |

Disable MathJax  ( What is MathJax? )

About

Help

contact arXiv  Click here to contact arXiv

Contact

subscribe to arXiv mailings  Click here to subscribe

Subscribe

Copyright

Privacy Policy

Web Accessibility Assistance

arXiv Operational Status