[2605.23945] Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.23945

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

Computer Science > Artificial Intelligence

arXiv:2605.23945  (cs)

[Submitted on 3 May 2026]

Title: Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism

Authors:  Long Zhao ,  Qinghe Wang ,  Jiaan Zhu ,  Youhui Bai ,  Zewen Jin ,  Chaoyi Ruan ,  Shengnan Wang ,  Cheng Li

View a PDF of the paper titled Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism, by Long Zhao and 7 other authors

View PDF

HTML (experimental)

Abstract: Reinforcement Learning from Human Feedback (RLHF) has become a key post-training paradigm for improving model quality. However, the synchronous three-stage RLHF pipeline is often bottlenecked by the generation stage, where response-length skew causes the effective batch size to shrink rapidly during decoding, leaving GPUs underutilized while a few long responses remain unfinished. Mainstream frameworks employ a static tensor parallelism (TP) configuration that cannot adapt to changing batch characteristics, leaving substantial performance headroom unexplored. We propose PAT, an adaptive TP method that dynamically reconfigures TP during the generation stage of each RLHF iteration. PAT introduces two key techniques. First, a predictor-guided online reconfiguration method decides both the reconfiguration point and the target TP configuration based on offline profiling, triggering reconfiguration only when the predicted latency benefit outweighs the reconfiguration overhead. Second, a lightweight online reconfiguration mechanism updates only the states and layouts affected by TP changes: it adapts unfinished decoding states through a cost-model-based choice between KV-cache migration and recomputation, performs in-place weight resharding, and reuses cached communication groups. We implement PAT on top of SGLang and integrate it with the VeRL framework. Evaluations on LLaMA3.1-8B and Qwen3-14B using DeepScaleR show that PAT reduces generation latency by up to 34.6% and end-to-end RLHF training iteration latency by up to 27.2% compared to the original VeRL setup.

Comments:

11page, 14 figures

Subjects:

Artificial Intelligence (cs.AI) ; Distributed, Parallel, and Cluster Computing (cs.DC)

Cite as:

arXiv:2605.23945  [cs.AI]

&nbsp;

(or

arXiv:2605.23945v1  [cs.AI]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.23945

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Long Zhao [ view email ]

[v1]

Sun, 3 May 2026 05:53:32 UTC (753 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism, by Long Zhao and 7 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.AI

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