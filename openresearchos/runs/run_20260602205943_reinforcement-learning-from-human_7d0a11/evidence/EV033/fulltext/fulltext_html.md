[2604.06916] FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2604.06916

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

Computer Science > Machine Learning

arXiv:2604.06916  (cs)

[Submitted on 8 Apr 2026]

Title: FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling

Authors:  Yitong Li ,  Junsong Chen ,  Shuchen Xue ,  Pengcuo Zeren ,  Siyuan Fu ,  Dinghao Yang ,  Yangyang Tang ,  Junjie Bai ,  Ping Luo ,  Song Han ,  Enze Xie

View a PDF of the paper titled FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling, by Yitong Li and 10 other authors

View PDF

HTML (experimental)

Abstract: Reinforcement-Learning-based post-training has recently emerged as a promising paradigm for aligning text-to-image diffusion models with human preferences. In recent studies, increasing the rollout group size yields pronounced performance improvements, indicating substantial room for further alignment gains. However, scaling rollouts on large-scale foundational diffusion models (e.g., FLUX.1-12B) imposes a heavy computational burden. To alleviate this bottleneck, we explore the integration of FP4 quantization into Diffusion RL rollouts. Yet, we identify that naive quantized pipelines inherently introduce risks of performance degradation. To overcome this dilemma between efficiency and training integrity, we propose Sol-RL (Speed-of-light RL), a novel FP4-empowered Two-stage Reinforcement Learning framework. First, we utilize high-throughput NVFP4 rollouts to generate a massive candidate pool and extract a highly contrastive subset. Second, we regenerate these selected samples in BF16 precision and optimize the policy exclusively on them. By decoupling candidate exploration from policy optimization, Sol-RL integrates the algorithmic mechanisms of rollout scaling with the system-level throughput gains of NVFP4. This synergistic algorithm-hardware design effectively accelerates the rollout phase while reserving high-fidelity samples for optimization. We empirically demonstrate that our framework maintains the training integrity of BF16 precision pipeline while fully exploiting the throughput gains enabled by FP4 arithmetic. Extensive experiments across SANA, FLUX.1, and SD3.5-L substantiate that our approach delivers superior alignment performance across multiple metrics while accelerating training convergence by up to $4.64\times$, unlocking the power of massive rollout scaling at a fraction of the cost.

Subjects:

Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV)

Cite as:

arXiv:2604.06916  [cs.LG]

&nbsp;

(or

arXiv:2604.06916v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2604.06916

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Yitong Li [ view email ]

[v1]

Wed, 8 Apr 2026 10:14:47 UTC (5,176 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling, by Yitong Li and 10 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.LG

&lt;&nbsp;prev

&nbsp; | &nbsp;

next&nbsp;&gt;

new

|

recent

|

2026-04

Change to browse by:

cs

cs.AI

cs.CV

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

IArxiv recommender toggle

IArxiv Recommender

( What is IArxiv? )

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