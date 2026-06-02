[2605.20659] RoPeSLR: 3D RoPE-driven Sparse-LowRank Attention for Efficient Diffusion Transformers

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.20659

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

Computer Science > Computer Vision and Pattern Recognition

arXiv:2605.20659  (cs)

[Submitted on 20 May 2026]

Title: RoPeSLR: 3D RoPE-driven Sparse-LowRank Attention for Efficient Diffusion Transformers

Authors:  Yuxi Liu ,  Zekun Zhang ,  Yixiang Cai ,  Renjia Deng ,  Yutong He ,  Kun Yuan

View a PDF of the paper titled RoPeSLR: 3D RoPE-driven Sparse-LowRank Attention for Efficient Diffusion Transformers, by Yuxi Liu and Zekun Zhang and Yixiang Cai and Renjia Deng and Yutong He and Kun Yuan

View PDF

HTML (experimental)

Abstract: Diffusion Transformers (DiTs) have revolutionized high-fidelity video generation, yet their $\mathcal{O}(L^2)$ attention complexity poses a formidable bottleneck for long-sequence synthesis. While recent sparse-linear attention hybrids aim to mitigate this, their performance severely degrades at extreme sparsity due to the &#34;RoPE Dilemma&#34;: standard linear attention fails to preserve the orthogonal relative-position structure of 3D Rotary Position Embeddings (RoPE), neutralizing vital distance awareness. To address this, we propose \textbf{RoPeSLR}, a 3D RoPE-guided Sparse-LowRank attention framework. We establish that under empirically validated assumptions, the DiT attention manifold admits a decoupling into a high-frequency semantic spike set (bounded by $\mathcal{O}(L^{3/2})$ sparsity) and an extreme low-rank ($\mathcal{O}(d_h \log L)$) background continuum. Guided by this structural prior, RoPeSLR eschews standard linear attention for a head-wise low-rank parameterization equipped with a learnable 3D Absolute Positional Embedding (PE) injection, seamlessly synthesizing long-range relative distance decay. By guaranteeing sub-quadratic sparsity and sub-linear rank growth, RoPeSLR is exceptionally suited for scaling to ultra-long video inference. Extensive evaluations validate this scalable superiority: at 90\% sparsity, RoPeSLR achieves up to $10\times$ fewer FLOPs on Wan2.1-1.3B and delivers a $2.26\times$ end-to-end inference speedup on the ultra-long 100K+ token sequences of HunyuanVideo-13B, all while maintaining near-lossless generation fidelity (less than 1.3\% average VBench degradation).

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Machine Learning (cs.LG)

Cite as:

arXiv:2605.20659  [cs.CV]

&nbsp;

(or

arXiv:2605.20659v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.20659

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Yuxi Liu [ view email ]

[v1]

Wed, 20 May 2026 03:24:50 UTC (11,837 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled RoPeSLR: 3D RoPE-driven Sparse-LowRank Attention for Efficient Diffusion Transformers, by Yuxi Liu and Zekun Zhang and Yixiang Cai and Renjia Deng and Yutong He and Kun Yuan

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.CV

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

cs.LG

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